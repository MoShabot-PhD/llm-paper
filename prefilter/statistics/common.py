import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
HUMAN_ANNOTATIONS_CSV = ROOT / "data" / "human_annotations.csv"
ANNOTATIONS_DIR = ROOT / "results" / "annotations"
OUT_DIR = ROOT / "results" / "statistics"

SEED = 42
N_BOOT = 1000
N_SPLITS = 5

RATERS = ["A", "B", "C", "D"]
VIDEO_RATERS = ["A", "B", "C"]

MODELS = {
    "GPT-5.2": ("jsonl", "openai_gpt-5_2_batch_youtube_batch_results.jsonl"),
    "Claude Sonnet 4.6": ("jsonl", "anthropic_claude-sonnet-4_6_batch_youtube_batch_results.jsonl"),
    "QWEN3.6-36B": ("csv", "annotations_metadata_qwen3_6.csv"),
    "Gemma4:31B": ("csv", "annotations_metadata_gemma4_31b.csv"),
}


def normalize(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    return s if s in {"KEEP", "DISCARD"} else None


def load_model_outputs(name: str) -> pd.DataFrame:
    fmt, filename = MODELS[name]
    path = ANNOTATIONS_DIR / filename
    rows = []
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                r = o.get("result") or {}
                rows.append((str(o["custom_id"]).strip(), normalize(r.get("decision")), r.get("confidence")))
    else:
        df = pd.read_csv(path)
        rows = list(zip(df["url"].astype(str).str.strip(), df["DECISIONS_AI"].map(normalize), df["CONFIDENCE_AI"]))

    out = pd.DataFrame(rows, columns=["url", "decision", "confidence"])
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out = out.dropna().drop_duplicates(subset=["url"], keep="first")
    out["keep_score"] = np.where(out["decision"] == "KEEP", out["confidence"], 1.0 - out["confidence"])
    return out.set_index("url")


def load_evaluation_set() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = pd.read_csv(HUMAN_ANNOTATIONS_CSV)
    df["url"] = df["url"].astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()
    for r in RATERS:
        df[f"meta_{r}"] = df[f"meta_{r}"].map(normalize)
    df = df.dropna(subset=[f"meta_{r}" for r in RATERS]).drop_duplicates(subset=["url"])

    outputs = {name: load_model_outputs(name) for name in MODELS}
    scored = set(df["url"])
    for out in outputs.values():
        scored &= set(out.index)
    df = df[df["url"].isin(scored)].reset_index(drop=True)

    keep_votes = sum((df[f"meta_{r}"] == "KEEP").astype(int) for r in RATERS)
    df["y"] = (keep_votes > len(RATERS) / 2).astype(int)
    return df, {name: out.loc[df["url"]] for name, out in outputs.items()}


def metrics(y, pred) -> tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f1


def cluster_indices(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(groups)
    return uniq, {g: np.where(groups == g)[0] for g in uniq}


def resample(rng, uniq: np.ndarray, idx_by_group: dict) -> np.ndarray:
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([idx_by_group[g] for g in picked])
