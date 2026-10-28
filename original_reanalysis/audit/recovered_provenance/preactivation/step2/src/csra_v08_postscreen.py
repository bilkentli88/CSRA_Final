from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from csra_v08_common import *

ALLOWED_REVIEW_STATUS={"APPROVE","EXCLUDE"}


def common_eligible(screen_summaries: dict[str,pd.DataFrame]) -> list[str]:
    sets=[]
    for m in PRIMARY_MODELS:
        s=screen_summaries[m]; sets.append(set(s.loc[s["eligible"],"item_id"]))
    return sorted(set.intersection(*sets))


def make_blinded_review(candidates: list[dict], common_ids: list[str], out_csv):
    allowed=["item_id","relation","subject","question","parametric_answer","counterfactual_answer","irrelevant_answer","source_subject_uri","source_object_uri","counterfactual_source_object_uri","counterfactual_source_subject_uri","irrelevant_source_object_uri","irrelevant_source_subject_uri","source_dataset","source_record_sha256"]
    lookup={x["item_id"]:x for x in candidates}; rows=[]
    for iid in sorted(common_ids):
        r={k:lookup[iid].get(k,"") for k in allowed}; r["review_status"]=""; r["review_reason"]=""; rows.append(r)
    frame=pd.DataFrame(rows); atomic_write_csv(frame,out_csv); return frame


def seal_review(review_csv, seal_json):
    f=pd.read_csv(review_csv,dtype=str).fillna("")
    if set(f["review_status"])-ALLOWED_REVIEW_STATUS: raise RuntimeError("Review contains blank/invalid status")
    if f["item_id"].duplicated().any(): raise RuntimeError("Duplicate item IDs in review")
    if (f.loc[f["review_status"].eq("EXCLUDE"),"review_reason"].str.strip()=="").any(): raise RuntimeError("Every EXCLUDE row requires a reason")
    rec={"protocol_id":PROTOCOL_ID,"review_file":Path(review_csv).name,"review_sha256":sha256_file(review_csv),"rows":int(len(f)),"status_counts_sealed":True,"note":"Review completed before approval counts, gate progress, or selection ranks were revealed."}
    atomic_write_json(rec,seal_json); return rec


def select_after_review(candidates: list[dict], review_csv, seal_json, analysis_spec: dict):
    seal=load_json(seal_json)
    if seal["review_sha256"]!=sha256_file(review_csv): raise RuntimeError("Sealed review hash mismatch")
    review=pd.read_csv(review_csv,dtype=str).fillna(""); approved=review[review["review_status"].eq("APPROVE")][["item_id","relation"]].copy(); approved["selection_hash"]=approved.apply(lambda r:selection_hash(r["relation"],r["item_id"]),axis=1)
    cap=int(analysis_spec["selection"]["cap_per_original_relation"]); chosen=[]
    for rel,fr in approved.groupby("relation",sort=True): chosen.append(fr.sort_values(["selection_hash","item_id"]).head(cap))
    selected=pd.concat(chosen,ignore_index=True) if chosen else approved.iloc[0:0].copy(); selected=selected.sort_values(["relation","selection_hash","item_id"]).reset_index(drop=True)
    lookup={x["item_id"]:x for x in candidates}; full=pd.DataFrame([lookup[i] for i in selected["item_id"]]); full=full.merge(selected[["item_id","selection_hash"]],on="item_id",how="left")
    gate=len(full)>=int(analysis_spec["selection"]["minimum_common_items"])
    return full,gate


def freeze_postreview_design(selected: pd.DataFrame, review_csv, seal_json, out_dir, analysis_spec: dict, analysis_bundle_sha256: str):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    min_items=int(analysis_spec["selection"]["minimum_common_items"])
    if len(selected)<min_items: raise RuntimeError(f"Primary information gate failed: {len(selected)} < {min_items}; no answer scoring permitted")
    relation_map=relation_map_for_selected(selected,int(analysis_spec["relation_collapse"]["threshold"]))
    folds={"primary":make_fold_assignments(selected,relation_map,"primary",5)}
    for i in range(1,6): folds[f"sensitivity_{i}"]=make_fold_assignments(selected,relation_map,str(i),5)
    alt=alternate_subset(selected,float(analysis_spec["alternate_wording"]["fraction"]))
    selected_path=out/'final_common_items.csv'; atomic_write_csv(selected,selected_path)
    relation_map_path=out/'analysis_relation_map.json'; atomic_write_json(relation_map,relation_map_path)
    folds_path=out/'fold_assignments.json'; atomic_write_json(folds,folds_path)
    alt_path=out/'alternate_wording_item_ids.json'; atomic_write_json(alt,alt_path)
    empty={"primary_prediction":None,"primary_bootstrap":None,"peer_prediction":None,"alignment":None,"wording":None,"fictional":None,"classification":None}; empty_path=out/'empty_result_shells.json'; atomic_write_json(empty,empty_path)
    lock={"protocol_id":PROTOCOL_ID,"status":"POSTREVIEW_ANALYSIS_DESIGN_LOCKED_BEFORE_ANSWERS","selected_items":len(selected),"selected_items_sha256":sha256_file(selected_path),"sealed_review_sha256":sha256_file(review_csv),"review_seal_sha256":sha256_file(seal_json),"analysis_bundle_sha256":analysis_bundle_sha256,"analysis_relation_map_sha256":sha256_file(relation_map_path),"fold_assignments_sha256":sha256_file(folds_path),"alternate_wording_ids_sha256":sha256_file(alt_path),"empty_result_shells_sha256":sha256_file(empty_path)}
    lock_path=out/'postreview_analysis_lock.json'; atomic_write_json(lock,lock_path); return lock_path
