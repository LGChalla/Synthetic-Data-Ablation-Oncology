"""
preprocessing/mtsamples_prep.py
Extracts TNM labels from MTSamples clinical dictation notes.
Produces ground-truth CSVs for the TSTR benchmark (Phase 3).

Outputs:
  data_splits/mtsamples_lung_gold.csv
  data_splits/mtsamples_all_cancer_gold.csv

Usage:
  python preprocessing/mtsamples_prep.py
"""

import os
import re
import sys

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# ── Keyword-based filters (case-insensitive, robust to spacing variations) ───
ONCOLOGY_KEYWORDS = re.compile(
    r"(oncolog|hematolog|cancer|carcinoma|tumor|tumour|malign|neoplasm|"
    r"chemotherapy|radiation|staging|sarcoma|lymphoma|leukemia|melanoma|"
    r"urology|gynecolog|gastroenterol|nephrolog|neurosurg|orthoped|radiol|surgery)",
    re.IGNORECASE,
)
LUNG_KEYWORDS = re.compile(
    r"\b(lung|pulmon|bronch|lobar|adenocarcinoma|squamous|nsclc|sclc|mesotheliom)\b",
    re.IGNORECASE,
)

# ── TNM extraction patterns ───────────────────────────────────────────────────
TNM_COMPACT  = re.compile(r"\b(T[0-4isx][a-z]?)(N[0-3x]?)(M[01x]?)\b", re.IGNORECASE)
TNM_SPACED   = re.compile(r"\b(T[0-4isx][a-z]?)\s+(N[0-3x]?)\s+(M[01x]?)\b", re.IGNORECASE)
T_STANDALONE = re.compile(r"\bT([0-4])[a-z]?\b")


def extract_tnm(text: str) -> tuple:
    for pat in (TNM_COMPACT, TNM_SPACED):
        m = pat.search(text)
        if m:
            return m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
    m = T_STANDALONE.search(text)
    if m:
        return f"T{m.group(1)}", "Unknown", "Unknown"
    return "Unknown", "Unknown", "Unknown"


def is_valid_t(t: str) -> bool:
    return bool(re.match(r"T[0-4][a-z]?$", t, re.IGNORECASE))


def label_quality(t: str, n: str, m: str) -> str:
    if t != "Unknown" and n != "Unknown" and m != "Unknown":
        return "complete"
    if t != "Unknown":
        return "T_only"
    return "sparse"


def is_oncology(row: pd.Series) -> bool:
    """Keyword-based oncology filter — robust to specialty name variations."""
    spec = str(row.get("medical_specialty", "")).lower()
    text = str(row.get("transcription", "")).lower()
    keys = str(row.get("keywords", "")).lower()
    combined = f"{spec} {text} {keys}"
    return bool(ONCOLOGY_KEYWORDS.search(combined))


def is_lung(row: pd.Series) -> bool:
    text = str(row.get("transcription", "")).lower()
    keys = str(row.get("keywords", "")).lower()
    spec = str(row.get("medical_specialty", "")).lower()
    return bool(LUNG_KEYWORDS.search(f"{spec} {text} {keys}"))


def main():
    os.makedirs(cfg.DATA_SPLITS_DIR, exist_ok=True)
    print("Loading MTSamples from HuggingFace...")
    ds = load_dataset(cfg.MTSAMPLES_DATASET, split="train")
    df = ds.to_pandas()
    print(f"  Total records : {len(df)}")
    print(f"  Columns       : {list(df.columns)}")

    # ── Show unique specialties so we know what we're working with ────────────
    if "medical_specialty" in df.columns:
        specs = df["medical_specialty"].value_counts()
        print(f"\n  Top specialties:\n{specs.head(20).to_string()}\n")

    # ── Oncology filter (keyword-based) ───────────────────────────────────────
    onco_mask = df.apply(is_oncology, axis=1)
    onco      = df[onco_mask].copy().reset_index(drop=True)
    print(f"  After oncology filter : {len(onco)}")

    if len(onco) == 0:
        print("[ERROR] No oncology records found. Check dataset structure above.")
        return

    # ── Extract TNM ───────────────────────────────────────────────────────────
    print("  Extracting TNM labels...")
    tnm_results = onco["transcription"].fillna("").apply(
        lambda x: pd.Series(extract_tnm(x), index=["T_label", "N_label", "M_label"])
    )
    onco = pd.concat([onco, tnm_results], axis=1)
    onco["label_quality"] = onco.apply(
        lambda r: label_quality(r["T_label"], r["N_label"], r["M_label"]), axis=1
    )
    onco["free_text"] = onco["transcription"].fillna("")

    t_found = onco["T_label"].apply(is_valid_t).sum()
    print(f"  T-stage found in {t_found}/{len(onco)} records")
    print(f"  T distribution: {onco['T_label'].value_counts().to_dict()}")

    # ── Lung subset ───────────────────────────────────────────────────────────
    lung_mask = onco.apply(is_lung, axis=1) & onco["T_label"].apply(is_valid_t)
    lung      = onco[lung_mask].copy()
    print(f"\n  Lung cancer (T annotated): {len(lung)}")

    out_cols = ["free_text", "T_label", "N_label", "M_label",
                "label_quality", "medical_specialty"]
    lung[out_cols].to_csv(cfg.MTSAMPLES_LUNG_CSV, index=False)
    print(f"  Saved -> {cfg.MTSAMPLES_LUNG_CSV}")

    # ── All-cancer subset ─────────────────────────────────────────────────────
    all_onco = onco[onco["T_label"].apply(is_valid_t)].copy()
    print(f"  All-cancer (T annotated): {len(all_onco)}")
    all_onco[out_cols].to_csv(cfg.MTSAMPLES_ALL_CSV, index=False)
    print(f"  Saved -> {cfg.MTSAMPLES_ALL_CSV}")

    print(f"\nMTSamples prep complete.")
    print(f"  Lung: {len(lung)} | All-cancer: {len(all_onco)}")


if __name__ == "__main__":
    main()
