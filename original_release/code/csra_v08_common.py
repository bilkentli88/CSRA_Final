from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

PROTOCOL_ID = "CSRA-0.8"
DESIGN_SEED = 20260812
BOOTSTRAP_SEED = 20260813
PRIMARY_MODELS = ("qwen", "granite", "phi")
RELATIONS = ("country_of_origin", "developer", "formation_location", "original_language", "original_network")
ANSWER_ROLES = ("P", "C", "I")
ANSWER_LABELS = ("A", "B", "C")
REPORT_ROLES = ("document", "prior", "both", "uncertain")
REPORT_LABELS = ("A", "B", "C", "D")
PRIMARY_CONDITIONS = ("C_SKEPTICAL", "C_PROVISIONAL", "C_BARE")
REPORT_CONDITIONS = ("C_SKEPTICAL", "C_PROVISIONAL", "C_BARE", "P_PROVISIONAL", "I_PROVISIONAL")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(text: str) -> str:
    return sha256_text(text)


def stable_index(text: str, modulo: int) -> int:
    return int(stable_hash(f"{DESIGN_SEED}|{text}"), 16) % modulo


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write_json(value, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def answer_for_role(item: dict, role: str) -> str:
    return {"P": item["parametric_answer"], "C": item["counterfactual_answer"], "I": item["irrelevant_answer"]}[role]


def screening_mapping_ids(item_id: str, prompt_bundle: dict) -> tuple[int, int, int]:
    family_id = stable_index(item_id + "|family", 2)
    family = tuple(prompt_bundle["screen_mapping_schedule"][f"family_{family_id}"])
    rotation = stable_index(item_id + "|rotation", 3)
    return family[rotation:] + family[:rotation]


def report_order_map(order_id: int, prompt_bundle: dict):
    roles = prompt_bundle["report_orders"][int(order_id)]
    label_to_role = dict(zip(prompt_bundle["report_labels"], roles))
    role_to_label = {role: label for label, role in label_to_role.items()}
    return label_to_role, role_to_label


def selection_hash(relation: str, item_id: str) -> str:
    return stable_hash(f"20260812|v08|select|{relation}|{item_id}")


def alternate_hash(relation: str, item_id: str) -> str:
    return stable_hash(f"20260812|v08|alternate|{relation}|{item_id}")


def fold_hash(analysis_relation: str, item_id: str, salt: str = "primary") -> str:
    if salt == "primary":
        prefix = "20260812|v08|fold|"
    else:
        prefix = f"20260812|v08|fold_sens{int(salt)}|"
    return stable_hash(prefix + analysis_relation + "|" + item_id)


def relation_map_for_selected(selected: pd.DataFrame, threshold: int = 5) -> dict[str, str]:
    counts = selected.groupby("relation")["item_id"].nunique().to_dict()
    return {rel: ("OTHER_RARE" if counts.get(rel, 0) < threshold else rel) for rel in RELATIONS}


def make_fold_assignments(selected: pd.DataFrame, relation_map: dict[str, str], salt: str = "primary", n_folds: int = 5) -> dict[str, int]:
    work = selected[["item_id", "relation"]].drop_duplicates().copy()
    work["analysis_relation"] = work["relation"].map(relation_map)
    out = {}
    for rel, frame in work.groupby("analysis_relation", sort=True):
        ordered = frame.assign(_rank=frame["item_id"].map(lambda x: fold_hash(rel, x, salt))).sort_values(["_rank", "item_id"])
        for pos, item_id in enumerate(ordered["item_id"].tolist()):
            out[item_id] = int(pos % n_folds)
    return out


def alternate_subset(selected: pd.DataFrame, fraction: float = 0.20) -> list[str]:
    work = selected[["item_id", "relation"]].drop_duplicates().copy()
    n = len(work); target = int(np.floor(fraction * n + 0.5))
    counts = work.groupby("relation").size().sort_index()
    raw = counts.astype(float) * fraction
    quotas = np.floor(raw).astype(int)
    remaining = target - int(quotas.sum())
    if remaining > 0:
        order = sorted(counts.index, key=lambda rel: (-(raw.loc[rel] - quotas.loc[rel]), rel))
        for rel in order[:remaining]: quotas.loc[rel] += 1
    chosen = []
    for rel, frame in work.groupby("relation", sort=True):
        ranked = frame.assign(_rank=frame["item_id"].map(lambda x: alternate_hash(rel, x))).sort_values(["_rank", "item_id"])
        chosen.extend(ranked.head(int(quotas.loc[rel]))["item_id"].tolist())
    assert len(chosen) == target
    return sorted(chosen)
