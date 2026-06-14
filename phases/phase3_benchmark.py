"""
Phase 3 — TSTR Benchmark Construction and Evaluation  (T / N / M)
================================================================
Loads trained adapters from Phase 4, evaluates them on held-out synthetic
data and real TCGA pathology reports under the train-on-synthetic,
test-on-real (TSTR) protocol.

WHAT CHANGED FROM THE T-ONLY VERSION
------------------------------------
The previous version called `t_pred, _, _ = extract_tnm(...)` and threw the
N and M predictions away, then scored only T_label. `extract_tnm` already
returns all three axes, so:

  * evaluation now runs ONE inference per note and scores T, N, AND M from
    that single call (no extra GPU cost);
  * gold N/M label columns are auto-detected per test set and an axis is
    skipped (with a clear warning) if its gold is absent;
  * labels are normalized (substages collapsed: pT2a->T2, N2c->N2; TX/NX/MX
    and the junk "3 CM (T2)" handled), which also fixes the bogus
    acc_1/acc_3 CM (T2) columns in the old synthetic results;
  * two files are written:
        analysis/phase3_tstr_results.csv   — one row per (set, adapter, AXIS)
        analysis/per_note_predictions.csv  — one row per note: gold+pred T/N/M

Adapters evaluated (one per ablation condition):
  adapter_A_ungated / B_schema / C_schema_onto / D_full_norag / E_full_rag

Test sets (increasing distributional distance from training):
  1. Synthetic held-out   — in-distribution
  2. TCGA Lung            — real lung pathology reports
  3. TCGA Cross-tumor     — cross-tumor transfer

Usage:
  python phases/phase3_benchmark.py
  python phases/phase3_benchmark.py --results-dir /path --adapters-dir /path
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from core.generation    import load_hf_model, generate_one, parse_output
from core.logging_utils import load_jsonl
from core.schemas       import EXTRACTION_PROMPT

ADAPTER_ORDER = [
    "adapter_A_ungated",
    "adapter_B_schema",
    "adapter_C_schema_onto",
    "adapter_D_full_norag",
    "adapter_E_full_rag",
]

# Evaluable classes per axis. Gold outside these (TX/NX/MX/Unknown/blank) is
# excluded from that axis, matching the constant-classifier ceiling assumption.
VALID = {
    "T": ["T1", "T2", "T3", "T4"],
    "N": ["N0", "N1", "N2", "N3"],
    "M": ["M0", "M1"],
}

# Predicted "Unknown"/unparseable handling: "wrong" counts it incorrect
# (honest default); "drop" abstains (inflates scores).
UNKNOWN_PRED_POLICY = "wrong"


# ── Label normalization ───────────────────────────────────────────────────────

def normalize(axis, value):
    """Map a raw label to a canonical class in VALID[axis], or 'Unknown'.
    Handles pathologic/clinical prefixes (pT2a, cN1), substages (T1c->T1),
    explicit unknowns (TX/NX/MX), and junk gold like '3 CM (T2)'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    s = str(value).strip().upper()
    if s in ("", "NAN", "NONE", "NA", "UNKNOWN"):
        return "Unknown"
    # parenthetical canonical form first: "3 CM (T2)" -> T2
    m = re.search(rf"\(({axis}\s*\d)", s)
    if m:
        s = m.group(1).replace(" ", "")
    if re.search(rf"{axis}\s*X", s):           # TX / NX / MX -> unknown
        return "Unknown"
    m = re.search(rf"{axis}\s*([0-4])", s)      # pT2A / T2 / cN1 -> canonical
    if m:
        return f"{axis}{m.group(1)}"
    m = re.fullmatch(r"\s*([0-4])\s*(?:CM)?\s*", s)   # bare "2" -> T2
    if m:
        return f"{axis}{m.group(1)}"
    return "Unknown"


def find_label_col(df, axis):
    """Find the gold column for an axis among common naming conventions."""
    for cand in (f"{axis}_label", f"{axis}_target", axis,
                 f"{axis.lower()}_label", f"{axis.lower()}_target", axis.lower()):
        if cand in df.columns:
            return cand
    return None


# ── Loaders (unchanged) ───────────────────────────────────────────────────────

def load_tcga_sets(lung_path: str, cross_path: str) -> dict:
    sets = {}
    for name, path in [("lung", lung_path), ("crosstumor", cross_path)]:
        if os.path.exists(path):
            sets[name] = pd.read_csv(path)
            print(f"  Real-world {name}: {len(sets[name])} records")
        else:
            print(f"  [SKIP] {path} not found.")
    return sets


def load_synthetic_holdout(results_dir: str, condition: str = "full_rag") -> pd.DataFrame:
    """Uses the gated corpus from Phase 1 as in-distribution held-out test."""
    path = os.path.join(results_dir, f"phase1_{condition}.jsonl")
    recs = load_jsonl(path)
    admitted = [r for r in recs if r.get("gate_pass")]
    n_holdout = max(int(len(admitted) * 0.2), 1)
    holdout   = admitted[-n_holdout:]
    rows = []
    for r in holdout:
        pj = r.get("parsed_json") or {}
        try:
            stg = pj.get("notes", [{}])[0].get("staging", {})
        except (IndexError, AttributeError):
            stg = {}
        rows.append({
            "free_text": pj.get("notes", [{}])[0].get("free_text", r.get("raw_output", "")),
            "T_label":   str(stg.get("T", "Unknown")).upper(),
            "N_label":   str(stg.get("N", "Unknown")).upper(),
            "M_label":   str(stg.get("M", "Unknown")).upper(),
            "source":    "synthetic",
        })
    return pd.DataFrame(rows)


# ── Adapter loading (unchanged) ───────────────────────────────────────────────

def load_adapter(adapter_name: str, adapters_dir: str):
    from peft import PeftModel
    adapter_path = os.path.join(adapters_dir, adapter_name, "final_adapter")
    if not os.path.exists(adapter_path):
        print(f"  [SKIP] Adapter not found: {adapter_path}")
        return None, None
    base_model, tokenizer = load_hf_model(cfg.FINETUNE_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"  Loaded adapter: {adapter_name}")
    return model, tokenizer


# ── Extraction (unchanged — already returns all three axes) ───────────────────

def extract_tnm(model, tokenizer, text: str, model_id: str) -> tuple:
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])
    raw    = generate_one(model, tokenizer, prompt,
                          {"temperature": 0.1, "do_sample": False, "max_new_tokens": 64},
                          model_id=model_id)
    parsed, _ = parse_output(raw)
    if not isinstance(parsed, dict):
        return "Unknown", "Unknown", "Unknown"
    return (
        str(parsed.get("T", "Unknown")).upper(),
        str(parsed.get("N", "Unknown")).upper(),
        str(parsed.get("M", "Unknown")).upper(),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def bootstrap_accuracy(y_true, y_pred, n=1000) -> tuple:
    if len(y_true) < 2:
        acc = accuracy_score(y_true, y_pred) if y_true else float("nan")
        return acc, acc, acc
    arr = np.array([(yt == yp) for yt, yp in zip(y_true, y_pred)], dtype=float)
    res = bootstrap((arr,), np.mean, n_resamples=n, confidence_level=0.95,
                    random_state=42, method="percentile")
    return (float(np.mean(arr)),
            float(res.confidence_interval.low),
            float(res.confidence_interval.high))


def per_class_accuracy(y_true, y_pred) -> dict:
    out = {}
    for cls in sorted(set(y_true)):
        idxs = [i for i, y in enumerate(y_true) if y == cls]
        correct = sum(1 for i in idxs if y_pred[i] == cls)
        out[cls] = round(correct / len(idxs), 3)
    return out


def score_axis(gold_raw, pred_raw, axis, set_name):
    """Normalize, restrict to evaluable gold, and score one axis."""
    g = [normalize(axis, x) for x in gold_raw]
    p = [normalize(axis, x) for x in pred_raw]
    pairs = [(gg, pp) for gg, pp in zip(g, p) if gg in VALID[axis]]
    if UNKNOWN_PRED_POLICY == "drop":
        pairs = [(gg, pp) for gg, pp in pairs if pp in VALID[axis]]
    if not pairs:
        return None
    y_true = [gg for gg, _ in pairs]
    y_pred = [pp for _, pp in pairs]

    acc, lo, hi = bootstrap_accuracy(y_true, y_pred)
    const_cls   = pd.Series(y_true).mode()[0]
    const_acc   = sum(1 for y in y_true if y == const_cls) / len(y_true)
    present     = sorted(set(y_true))
    macro_f1    = (f1_score(y_true, y_pred, labels=present, average="macro",
                            zero_division=0) if len(present) > 1 else None)
    per_cls     = per_class_accuracy(y_true, y_pred)

    row = {
        "set": set_name, "axis": axis, "n": len(y_true),
        "accuracy": round(acc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
        "constant_clf": round(const_acc, 3),
        "macro_f1": round(macro_f1, 3) if macro_f1 is not None else None,
    }
    for cls, a in per_cls.items():
        row[f"acc_{cls}"] = a
    return row


def evaluate_set_all_axes(model, tokenizer, df, model_id, set_name, adapter_name):
    """One inference per note; score T, N, M; also collect per-note rows."""
    label_cols = {a: find_label_col(df, a) for a in ("T", "N", "M")}
    avail = [a for a in ("T", "N", "M") if label_cols[a]]
    missing = [a for a in ("T", "N", "M") if not label_cols[a]]
    print(f"    {set_name}: gold available for {avail}"
          + (f"; MISSING gold for {missing} (axis skipped)" if missing else ""))

    raw_gold = {a: [] for a in ("T", "N", "M")}
    raw_pred = {a: [] for a in ("T", "N", "M")}
    per_note = []

    for _, row in df.iterrows():
        text = str(row.get("free_text", ""))
        t, n, m = extract_tnm(model, tokenizer, text, model_id)
        preds = {"T": t, "N": n, "M": m}
        rec = {"set": set_name, "adapter": adapter_name}
        for a in ("T", "N", "M"):
            col = label_cols[a]
            gold = str(row[col]) if (col and pd.notna(row.get(col))) else "Unknown"
            raw_gold[a].append(gold)
            raw_pred[a].append(preds[a])
            rec[f"gold_{a}"] = gold
            rec[f"pred_{a}"] = preds[a]
        per_note.append(rec)

    metrics = []
    for a in avail:
        r = score_axis(raw_gold[a], raw_pred[a], a, set_name)
        if r:
            r["adapter"] = adapter_name
            metrics.append(r)
            extra = "  ".join(f"{k}={r[k]:.3f}" for k in ("accuracy", "constant_clf")
                              if r.get(k) is not None)
            mf1 = r.get("macro_f1")
            print(f"      [{a}] n={r['n']}  {extra}"
                  + (f"  macroF1={mf1:.3f}" if mf1 is not None else ""))
    return metrics, per_note


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: TSTR Benchmark (T/N/M)")
    parser.add_argument("--results-dir",  default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir", default=cfg.ADAPTERS_DIR)
    parser.add_argument("--export-dir",   default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()
    os.makedirs(args.export_dir, exist_ok=True)

    print("Loading test sets...")
    rw_sets  = load_tcga_sets(cfg.TCGA_LUNG_CSV, cfg.TCGA_CROSS_CSV)
    synth_df = load_synthetic_holdout(args.results_dir)
    print(f"  Synthetic held-out: {len(synth_df)} records")

    all_metrics, all_per_note = [], []
    for adapter_name in ADAPTER_ORDER:
        print(f"\n--- Adapter: {adapter_name} ---")
        model, tokenizer = load_adapter(adapter_name, args.adapters_dir)
        if model is None:
            continue

        for set_name, df in [
            ("synthetic_holdout", synth_df),
            ("tcga_lung",         rw_sets.get("lung",       pd.DataFrame())),
            ("tcga_crosstumor",   rw_sets.get("crosstumor", pd.DataFrame())),
        ]:
            if df.empty:
                continue
            mets, pn = evaluate_set_all_axes(model, tokenizer, df,
                                             cfg.FINETUNE_MODEL, set_name, adapter_name)
            all_metrics.extend(mets)
            all_per_note.extend(pn)

        from core.generation import unload_model
        unload_model(model, tokenizer)

    # Per-note predictions — the durable artifact; never re-run inference again.
    if all_per_note:
        pn_path = os.path.join(args.export_dir, "per_note_predictions.csv")
        pd.DataFrame(all_per_note).to_csv(pn_path, index=False)
        print(f"\nPer-note predictions saved -> {pn_path}")

    # Aggregate results: one row per (set, adapter, axis), per-class acc columns.
    if all_metrics:
        df_out = pd.DataFrame(all_metrics)
        front = ["set", "adapter", "axis", "n", "accuracy", "ci_low", "ci_high",
                 "constant_clf", "macro_f1"]
        acc_cols = sorted([c for c in df_out.columns if c.startswith("acc_")])
        df_out = df_out[[c for c in front if c in df_out.columns] + acc_cols]
        path = os.path.join(args.export_dir, "phase3_tstr_results.csv")
        df_out.to_csv(path, index=False)
        print(f"TSTR results (T/N/M) saved -> {path}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
