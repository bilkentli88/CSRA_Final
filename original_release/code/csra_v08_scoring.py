from __future__ import annotations
import gc, json, math, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from csra_v08_common import *


def render_chat(tokenizer, user_text: str, model_key: str, prompt_bundle: dict, model_lock: dict) -> str:
    messages = [{"role":"system","content":prompt_bundle["system_text"]},{"role":"user","content":user_text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    kwargs.update(model_lock["models"][model_key].get("chat_template_kwargs", {}))
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_model_bundle(model_key: str, model_lock: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    spec = model_lock["models"][model_key]; q = model_lock["quantization"]
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer_id"], revision=spec["tokenizer_revision"], use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["revision"], device_map="auto", quantization_config=quant, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    return {"key":model_key,"model":model,"tokenizer":tokenizer,"model_revision":spec["revision"],"tokenizer_revision":spec["tokenizer_revision"],"precision_label":q["precision_label"]}


def unload_model_bundle(bundle):
    if bundle is not None:
        bundle.pop("model", None); bundle.pop("tokenizer", None)
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass


def resolve_label_encoding(tokenizer, labels, prompt_bundle):
    prefixes = prompt_bundle.get("label_prefix_search", ["", " ", "\n"])
    for prefix in prefixes:
        enc = {lab: tokenizer.encode(prefix + lab, add_special_tokens=False) for lab in labels}
        if all(len(ids)==1 for ids in enc.values()) and len({ids[0] for ids in enc.values()})==len(labels):
            return {"mode":"single_token","prefix":prefix,"ids":{k:v[0] for k,v in enc.items()}}
    enc = {lab: tokenizer.encode(lab, add_special_tokens=False) for lab in labels}
    if any(not ids for ids in enc.values()): raise RuntimeError("A response label has no tokenizer IDs")
    return {"mode":"sequence","prefix":"","ids":enc}


def _score_batch(bundle, batch, labels, label_encoding, prompt_bundle):
    import torch
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    rendered = [render_chat(tokenizer, spec["user_prompt"], bundle["key"], prompt_bundle, spec["_model_lock"]) for spec in batch]
    scored=[]
    if label_encoding["mode"]=="single_token":
        inputs=tokenizer(rendered,return_tensors="pt",padding=True,truncation=False)
        inputs={k:v.to(model.device) for k,v in inputs.items()}
        logits=model(**inputs,use_cache=False).logits[:,-1,:].float(); lp=torch.log_softmax(logits,dim=-1).cpu()
        for row in range(len(batch)):
            scored.append({lab:float(lp[row,label_encoding["ids"][lab]]) for lab in labels})
        del inputs,logits,lp
    else:
        for prompt in rendered:
            prompt_ids=tokenizer(prompt,return_tensors="pt",add_special_tokens=False).input_ids.to(model.device)
            vals={}
            for lab in labels:
                cont_ids=label_encoding["ids"][lab]
                continuation=torch.tensor([cont_ids],device=model.device,dtype=prompt_ids.dtype)
                full=torch.cat([prompt_ids,continuation],dim=1)
                logits=model(input_ids=full,use_cache=False).logits[:,:-1,:].float(); lps=torch.log_softmax(logits,dim=-1)
                start=prompt_ids.shape[1]-1; total=0.0
                for off,tid in enumerate(cont_ids): total += float(lps[0,start+off,tid].cpu())
                vals[lab]=total
            scored.append(vals)
    for vals in scored:
        if not np.isfinite(np.asarray([vals[x] for x in labels],float)).all(): raise FloatingPointError("Non-finite label score")
    return scored


def score_specs(bundle, specs, labels, checkpoint_path, phase_name, prompt_bundle, model_lock, analysis_spec, require_complete=False):
    import torch
    checkpoint_path=Path(checkpoint_path); checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    expected={s["job_id"]:sha256_text(s["user_prompt"]) for s in specs}
    if len(expected)!=len(specs): raise RuntimeError(f"Duplicate job IDs in {phase_name}")
    if checkpoint_path.exists():
        saved=pd.read_csv(checkpoint_path,dtype={"job_id":str}).drop_duplicates("job_id",keep="last")
        overlap=saved[saved["job_id"].isin(expected)]
        bad=overlap[overlap.apply(lambda r:r.get("prompt_sha256")!=expected[r["job_id"]],axis=1)]
        if len(bad): raise RuntimeError(f"Prompt-hash mismatch in {phase_name}")
    else: saved=pd.DataFrame()
    prior_attempts={} if saved.empty else saved.set_index("job_id")["attempt_count"].fillna(0).astype(int).to_dict()
    completed=set() if saved.empty else set(saved.loc[saved["status"].eq("ok"),"job_id"])
    pending=[dict(s) for s in specs if s["job_id"] not in completed]
    for s in pending: s["_model_lock"]=model_lock
    failed_attempts={s["job_id"]:0 for s in pending}
    label_encoding=resolve_label_encoding(bundle["tokenizer"],labels,prompt_bundle)
    preferred=int(model_lock["batch_size"]); oom_sizes=list(model_lock["oom_fallback_batch_sizes"])
    max_same=int(analysis_spec["technical"]["identical_retries"])
    cursor=0
    while cursor<len(pending):
        size=preferred; batch=pending[cursor:cursor+size]; same_error_retries=0; rows=None; last_error=None
        while True:
            t0=time.time()
            try:
                vals=_score_batch(bundle,batch,labels,label_encoding,prompt_bundle)
                now=pd.Timestamp.utcnow().isoformat(); out=[]
                for spec,v in zip(batch,vals):
                    r={k:x for k,x in spec.items() if k not in {"user_prompt","_model_lock"}}
                    r.update({"prompt_text":spec["user_prompt"],"prompt_sha256":sha256_text(spec["user_prompt"]),"model_revision":bundle["model_revision"],"tokenizer_revision":bundle["tokenizer_revision"],"inference_precision":bundle["precision_label"],"label_scoring_mode":label_encoding["mode"],"label_encoding_json":json.dumps(label_encoding,sort_keys=True),"status":"ok","attempt_count":prior_attempts.get(spec["job_id"],0)+failed_attempts.get(spec["job_id"],0)+1,"batch_size_used":len(batch),"elapsed_seconds":time.time()-t0,"checkpointed_utc":now})
                    for lab,val in v.items(): r[f"logp_{lab}"]=val
                    out.append(r)
                rows=pd.DataFrame(out); break
            except torch.cuda.OutOfMemoryError as exc:
                last_error=exc
                for spec in batch: failed_attempts[spec["job_id"]]=failed_attempts.get(spec["job_id"],0)+1
                gc.collect(); torch.cuda.empty_cache()
                next_sizes=[x for x in oom_sizes if x < len(batch)]
                if next_sizes:
                    size=max(next_sizes); batch=pending[cursor:cursor+size]; continue
                break
            except (RuntimeError,FloatingPointError) as exc:
                last_error=exc
                for spec in batch: failed_attempts[spec["job_id"]]=failed_attempts.get(spec["job_id"],0)+1
                same_error_retries+=1; gc.collect(); torch.cuda.empty_cache()
                if same_error_retries<=max_same: continue
                break
        if rows is None:
            now=pd.Timestamp.utcnow().isoformat(); out=[]
            for spec in batch:
                r={k:x for k,x in spec.items() if k not in {"user_prompt","_model_lock"}}
                r.update({"prompt_text":spec["user_prompt"],"prompt_sha256":sha256_text(spec["user_prompt"]),"model_revision":bundle["model_revision"],"tokenizer_revision":bundle["tokenizer_revision"],"inference_precision":bundle["precision_label"],"label_scoring_mode":label_encoding["mode"],"label_encoding_json":json.dumps(label_encoding,sort_keys=True),"status":"terminal_error","attempt_count":prior_attempts.get(spec["job_id"],0)+failed_attempts.get(spec["job_id"],0)+1,"batch_size_used":len(batch),"elapsed_seconds":0.0,"error":repr(last_error),"checkpointed_utc":now})
                out.append(r)
            rows=pd.DataFrame(out)
        combined=rows if saved.empty else pd.concat([saved,rows],ignore_index=True); saved=combined.drop_duplicates("job_id",keep="last")
        atomic_write_csv(saved,checkpoint_path); cursor += len(batch)
    result=saved[saved["job_id"].isin(expected)].copy()
    if set(result["job_id"])!=set(expected): raise RuntimeError(f"Missing job IDs in {phase_name}")
    extra=float(np.maximum(result["attempt_count"].fillna(1).to_numpy(float)-1,0).sum()/len(specs))
    if extra>float(analysis_spec["technical"]["additional_attempt_ceiling"]): raise RuntimeError(f"Additional-attempt ceiling exceeded in {phase_name}: {extra:.3%}")
    if require_complete and not (result["status"]=="ok").all(): raise RuntimeError(f"Unresolved terminal primary report failure in {phase_name}")
    return result.sort_values("job_id").reset_index(drop=True)


def answer_options(item, mapping_id, prompt_bundle):
    label_to_role=prompt_bundle["answer_mappings"][int(mapping_id)]; role_to_label={r:l for l,r in label_to_role.items()}
    lines="\n".join(f"{lab}. {answer_for_role(item,label_to_role[lab])}" for lab in prompt_bundle["answer_labels"])
    return lines,role_to_label


def add_answer_semantics(frame):
    frame=frame.copy()
    for role in ANSWER_ROLES: frame[f"logp_{role}"]=frame.apply(lambda r,role=role:r["logp_"+r["label_"+role]],axis=1)
    vals=frame[[f"logp_{r}" for r in ANSWER_ROLES]].to_numpy(float); norm=np.logaddexp.reduce(vals,axis=1,keepdims=True); probs=np.exp(vals-norm)
    for j,role in enumerate(ANSWER_ROLES): frame[f"prob_{role}"]=probs[:,j]
    frame["L_C_minus_P"]=frame["logp_C"]-frame["logp_P"]; frame["selected_role"]=np.array(ANSWER_ROLES)[np.argmax(probs,axis=1)]
    return frame


def build_screen_specs(candidates, model_key, prompt_bundle):
    specs=[]
    for item in candidates:
        for pid,mid in enumerate(screening_mapping_ids(item["item_id"],prompt_bundle)):
            opts,r2l=answer_options(item,mid,prompt_bundle)
            prompt=f"Question:\n{item['question']}\n\nOptions:\n{opts}\n\n{prompt_bundle['screen_prompt_endings'][pid]}"
            s={"job_id":f"v08|{model_key}|screen|{item['item_id']}|p{pid}|m{mid}","model_key":model_key,"phase":"screen","item_id":item["item_id"],"relation":item["relation"],"prompt_id":pid,"mapping_id":mid,"user_prompt":prompt,"label_role_map_json":json.dumps(r2l,sort_keys=True)}
            for role,lab in r2l.items(): s[f"label_{role}"]=lab
            specs.append(s)
    return specs


def summarize_screen(frame):
    x=add_answer_semantics(frame[frame["status"].eq("ok")].copy()); x["best_alt_logp"]=x[["logp_C","logp_I"]].max(axis=1); x["P_vs_best_alt"]=x["logp_P"]-x["best_alt_logp"]
    s=x.groupby(["model_key","item_id","relation"],as_index=False).agg(p_wins=("selected_role",lambda v:int((v=="P").sum())),mean_margin=("P_vs_best_alt","mean"),min_margin=("P_vs_best_alt","min"),max_margin=("P_vs_best_alt","max"),screen_jobs=("job_id","size"))
    s["mapping_spread"]=s["max_margin"]-s["min_margin"]; s["eligible"]=(s["p_wins"]==3)&(s["screen_jobs"]==3)
    return s


def make_document(item, condition, prompt_bundle):
    if condition=="NO_DOC": return "","NONE","NONE"
    role,strength=condition.split("_",1); claim=item["claim_template"].format(answer=answer_for_role(item,role))
    if strength=="BARE" and claim.startswith("The original language"): claim="the"+claim[3:]
    return prompt_bundle["strength_text"][strength].format(claim=claim),role,strength


def build_answer_specs(items, model_key, prompt_bundle):
    specs=[]
    for item in items:
        for cond in prompt_bundle["answer_conditions"]:
            doc,docrole,strength=make_document(item,cond,prompt_bundle)
            for mid in range(6):
                opts,r2l=answer_options(item,mid,prompt_bundle)
                if cond=="NO_DOC": prompt=prompt_bundle["answer_prompt_no_doc"].format(question=item["question"],options=opts)
                else: prompt=prompt_bundle["answer_prompt_with_doc"].format(document=doc,question=item["question"],options=opts)
                s={"job_id":f"v08|{model_key}|answer|{item['item_id']}|{cond}|m{mid}","model_key":model_key,"phase":"answer","item_id":item["item_id"],"relation":item["relation"],"condition":cond,"document_role":docrole,"strength":strength,"mapping_id":mid,"document_text":doc,"user_prompt":prompt,"label_role_map_json":json.dumps(r2l,sort_keys=True)}
                for role,lab in r2l.items(): s[f"label_{role}"]=lab
                specs.append(s)
    return specs


def summarize_answers(frame, candidate_lookup):
    ok=frame[frame["status"].eq("ok")].copy(); scored=add_answer_semantics(ok)
    agg=scored.groupby(["model_key","item_id","relation","condition","document_role","strength","document_text"],as_index=False).agg(mean_logp_P=("logp_P","mean"),mean_logp_C=("logp_C","mean"),mean_logp_I=("logp_I","mean"),mean_prob_P=("prob_P","mean"),mean_prob_C=("prob_C","mean"),mean_prob_I=("prob_I","mean"),mean_L_C_minus_P=("L_C_minus_P","mean"),mapping_jobs=("job_id","size"))
    vals=agg[["mean_logp_P","mean_logp_C","mean_logp_I"]].to_numpy(float); agg["frozen_answer_role"]=np.array(ANSWER_ROLES)[np.argmax(vals,axis=1)]
    agg["frozen_answer"]=agg.apply(lambda r:answer_for_role(candidate_lookup[r["item_id"]],r["frozen_answer_role"]),axis=1)
    base=agg[agg["condition"].eq("NO_DOC")].copy(); base["baseline_margin"]=base["mean_logp_P"]-base[["mean_logp_C","mean_logp_I"]].max(axis=1); base=base[["item_id","baseline_margin"]]
    agg=agg.merge(base,on="item_id",how="left",validate="many_to_one")
    cie=[]
    for strength in ("SKEPTICAL","PROVISIONAL","BARE"):
        c=agg[agg["condition"].eq(f"C_{strength}")][["item_id","model_key","mean_L_C_minus_P"]].rename(columns={"mean_L_C_minus_P":"c_value"})
        p=agg[agg["condition"].eq(f"P_{strength}")][["item_id","model_key","mean_L_C_minus_P"]].rename(columns={"mean_L_C_minus_P":"p_value"})
        j=c.merge(p,on=["model_key","item_id"],validate="one_to_one"); j["strength"]=strength; j["CIE"]=j["c_value"]-j["p_value"]; cie.append(j[["model_key","item_id","strength","CIE"]])
    cie=pd.concat(cie,ignore_index=True); agg=agg.merge(cie,on=["model_key","item_id","strength"],how="left")
    return scored,agg,cie


def report_options(order_id, prompt_bundle):
    l2r,r2l=report_order_map(order_id,prompt_bundle); txt=prompt_bundle["report_role_text"]; lines="\n".join(f"{lab}. {txt[l2r[lab]]}" for lab in prompt_bundle["report_labels"])
    return lines,r2l


def build_report_specs(items, answer_summary, model_key, prompt_bundle, alternate=False, selected_ids=None):
    lookup={x["item_id"]:x for x in items}; rows=answer_summary[answer_summary["condition"].isin(prompt_bundle["report_conditions"])].copy()
    if selected_ids is not None: rows=rows[rows["item_id"].isin(set(selected_ids))]
    specs=[]; wording="alternate" if alternate else "principal"
    for a in rows.to_dict("records"):
        item=lookup[a["item_id"]]
        for frame in ("SELF","PEER"):
            stem=prompt_bundle["report_stems"][wording][frame].format(answer=a["frozen_answer"])
            for oid in range(4):
                opts,r2l=report_options(oid,prompt_bundle); prompt=prompt_bundle["report_prompt"].format(document=a["document_text"],question=item["question"],answer=a["frozen_answer"],stem=stem,options=opts)
                phase="report_alt" if alternate else "report_main"; s={"job_id":f"v08|{model_key}|{phase}|{item['item_id']}|{a['condition']}|{frame}|o{oid}","model_key":model_key,"phase":phase,"item_id":item["item_id"],"relation":item["relation"],"condition":a["condition"],"document_role":a["document_role"],"strength":a["strength"],"frozen_answer_role":a["frozen_answer_role"],"frozen_answer":a["frozen_answer"],"frame":frame,"report_order_id":oid,"alternate_wording":int(alternate),"user_prompt":prompt,"label_role_map_json":json.dumps(r2l,sort_keys=True)}
                for role,lab in r2l.items(): s[f"label_report_{role}"]=lab
                specs.append(s)
    return specs


def add_report_semantics(frame):
    x=frame.copy()
    for role in REPORT_ROLES: x[f"logp_report_{role}"]=x.apply(lambda r,role=role:r["logp_"+r["label_report_"+role]],axis=1)
    return x


def aggregate_report_rotations(frame):
    ok=add_report_semantics(frame[frame["status"].eq("ok")].copy())
    keys=["model_key","item_id","relation","condition","document_role","strength","frozen_answer_role","frozen_answer","frame","alternate_wording"]
    aggdict={f"logp_report_{role}":("logp_report_"+role,"mean") for role in REPORT_ROLES}; aggdict["report_order_jobs"]=("job_id","size")
    out=ok.groupby(keys,as_index=False).agg(**aggdict); out["VSA"]=out["logp_report_document"]-out["logp_report_prior"]
    vals=out[[f"logp_report_{r}" for r in REPORT_ROLES]].to_numpy(float); out["modal_report"]=np.array(REPORT_ROLES)[np.argmax(vals,axis=1)]
    return out


def build_fiction_answer_specs(controls, model_key, prompt_bundle):
    specs=[]
    for item in controls:
        semantic={"P":item["distractor_1"],"C":item["contextual_answer"],"I":item["distractor_2"]}
        claim=item["claim_template"].format(answer=item["contextual_answer"])
        if claim.startswith("The original language"): claim="the"+claim[3:]
        doc=prompt_bundle["fiction_document_prefix"]+claim
        for mid,l2r in enumerate(prompt_bundle["answer_mappings"]):
            r2l={r:l for l,r in l2r.items()}; opts="\n".join(f"{lab}. {semantic[l2r[lab]]}" for lab in prompt_bundle["answer_labels"]); prompt=prompt_bundle["answer_prompt_with_doc"].format(document=doc,question=item["question"],options=opts)
            s={"job_id":f"v08|{model_key}|fiction_answer|{item['item_id']}|m{mid}","model_key":model_key,"phase":"fiction_answer","item_id":item["item_id"],"relation":item["relation"],"mapping_id":mid,"document_text":doc,"user_prompt":prompt,"label_role_map_json":json.dumps(r2l,sort_keys=True)}
            for role,lab in r2l.items(): s[f"label_{role}"]=lab
            specs.append(s)
    return specs


def summarize_fiction_answers(frame, controls):
    x=add_answer_semantics(frame[frame["status"].eq("ok")].copy()); out=x.groupby(["model_key","item_id","relation","document_text"],as_index=False).agg(mean_logp_P=("logp_P","mean"),mean_logp_C=("logp_C","mean"),mean_logp_I=("logp_I","mean"),mapping_jobs=("job_id","size")); vals=out[["mean_logp_P","mean_logp_C","mean_logp_I"]].to_numpy(float); out["frozen_answer_role"]=np.array(ANSWER_ROLES)[np.argmax(vals,axis=1)]; lookup={i["item_id"]:i for i in controls}; out["frozen_answer"]=out.apply(lambda r:{"P":lookup[r["item_id"]]["distractor_1"],"C":lookup[r["item_id"]]["contextual_answer"],"I":lookup[r["item_id"]]["distractor_2"]}[r["frozen_answer_role"]],axis=1); return out


def build_fiction_report_specs(controls, answer_summary, model_key, prompt_bundle):
    lookup={i["item_id"]:i for i in controls}; specs=[]
    for a in answer_summary.to_dict("records"):
        item=lookup[a["item_id"]]; stem=prompt_bundle["report_stems"]["principal"]
        for frame in ("SELF","PEER"):
            st=stem[frame].format(answer=a["frozen_answer"])
            for oid in range(4):
                opts,r2l=report_options(oid,prompt_bundle); prompt=prompt_bundle["report_prompt"].format(document=a["document_text"],question=item["question"],answer=a["frozen_answer"],stem=st,options=opts)
                s={"job_id":f"v08|{model_key}|fiction_report|{item['item_id']}|{frame}|o{oid}","model_key":model_key,"phase":"fiction_report","item_id":item["item_id"],"relation":item["relation"],"condition":"FICTION","document_role":"C","strength":"FICTION","frozen_answer_role":a["frozen_answer_role"],"frozen_answer":a["frozen_answer"],"frame":frame,"report_order_id":oid,"alternate_wording":0,"user_prompt":prompt,"label_role_map_json":json.dumps(r2l,sort_keys=True)}
                for role,lab in r2l.items(): s[f"label_report_{role}"]=lab
                specs.append(s)
    return specs
