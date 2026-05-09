"""
preprocessing/mtsamples_prep.py
Extracts TNM labels from MTSamples clinical dictation notes.
Produces ground-truth CSVs for the TSTR benchmark (Phase 3).

Outputs:
  data_splits/mtsamples_lung_gold.csv      (cancer_type=lung, T annotated)
  data_splits/mtsamples_all_cancer_gold.csv (all oncology specialties, T annotated)

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

ONCOLOGY_SPECIALTIES = {
    "Oncology","Hematology - Oncology","Radiology","Urology",
    "Obstetrics / Gynecology","Gastroenterology","Nephrology",
    "Neurosurgery","Surgery","Orthopedic",
}
LUNG_KEYWORDS = re.compile(
    r"\b(lung|pulmon|bronch|lobar|adenocarcinoma|squamous|nsclc|sclc|mesotheliom)\b",
    re.IGNORECASE,
)
TNM_COMPACT   = re.compile(r"\b(T[0-4isx][a-z]?)(N[0-3x]?)(M[01x]?)\b", re.IGNORECASE)
TNM_SPACED    = re.compile(r"\b(T[0-4isx][a-z]?)\s+(N[0-3x]?)\s+(M[01x]?)\b", re.IGNORECASE)
T_STANDALONE  = re.compile(r"\bT([0-4])[a-z]?\b")


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


def main():
    os.makedirs(cfg.DATA_SPLITS_DIR, exist_ok=True)
    print("Loading MTSamples from HuggingFace...")
    ds  = load_dataset(cfg.MTSAMPLES_DATASET, split="train")
    df  = ds.to_pandas()
    print(f"  Total records: {len(df)}")

    # Filter oncology specialties
    onco = df[df["medical_specialty"].isin(ONCOLOGY_SPECIALTIES)].copy()
    print(f"  After specialty filter: {len(onco)}")

    # Extract TNM from transcription
    onco[["T_label","N_label","M_label"]] = onco["transcription"].fillna("").apply(
        lambda x: pd.Series(extract_tnm(x))
    )
    onco["label_quality"] = onco.apply(
        lambda r: label_quality(r["T_label"], r["N_label"], r["M_label"]), axis=1)
    onco["free_text"]     = onco["transcription"].fillna("")

    # Lung subset — needs at least T annotation
    lung = onco[
        onco["transcription"].fillna("").apply(lambda x: bool(LUNG_KEYWORDS.search(x)))
        & onco["T_label"].apply(is_valid_t)
    ].copy()
    print(f"  Lung cancer (T annotated): {len(lung)}")
    lung[["free_text","T_label","N_label","M_label","label_quality","medical_specialty"]
        ].to_csv(cfg.MTSAMPLES_LUNG_CSV, index=False)
    print(f"  Saved -> {cfg.MTSAMPLES_LUNG_CSV}")

    # All-cancer subset — any specialty, needs T annotation
    all_onco = onco[onco["T_label"].apply(is_valid_t)].copy()
    print(f"  All-cancer (T annotated): {len(all_onco)}")
    all_onco[["free_text","T_label","N_label","M_label","label_quality","medical_specialty"]
            ].to_csv(cfg.MTSAMPLES_ALL_CSV, index=False)
    print(f"  Saved -> {cfg.MTSAMPLES_ALL_CSV}")

    print(f"\nMTSamples prep complete.")
    print(f"  Lung: {len(lung)} | All-cancer: {len(all_onco)}")


if __name__ == "__main__":
    main()
