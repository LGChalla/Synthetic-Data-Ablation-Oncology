"""
Ablation 3 — RAG vs No-RAG
Comparison Analysis
====================
Loads results from run_norag.py and run_rag.py and produces the
SNOMED density comparison with Mann-Whitney U significance test.

Run AFTER both generation scripts have completed.

Shows:
  - SNOMED density (avg, median) per condition
  - Unique SNOMED concept counts
  - Schema compliance and gate pass rates
  - Label diversity
  - Per-record density delta (matched by TNM grid index)
  - Statistical test: one-sided Mann-Whitney U (RAG > No-RAG)

Outputs:
  results/analysis/ablation3_rag_vs_norag.csv
  results/analysis/ablation3_rag_vs_norag_detail.csv

Usage:
  python ablations/3_rag_vs_norag/compare.py
  python ablations/3_rag_vs_norag/compare.py --results-dir /path/to/results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import compute_entropy, ENTROPY_FLOORS


def condition_summary(recs: list, label: str) -> dict:
    n          = len(recs)
    n_valid    = sum(1 for r in recs if r.get("parsed_json_valid"))
    n_gate     = sum(1 for r in recs if r.get("gate_pass"))
    densities  = [r.get("snomed_density", 0.0) for r in recs]
    unique_ids = {
        a.get("snomed_id", "")
        for r in recs
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    }
    valid      = [r for r in recs if r.get("parsed_json_valid")]
    t_lbl      = [r.get("T","Unknown") for r in valid]
    n_lbl      = [r.get("N","Unknown") for r in valid]
    m_lbl      = [r.get("M","Unknown") for r in valid]

    return {
        "condition":              label,
        "n_records":              n,
        "schema_compliance":      round(n_valid / n, 3) if n else 0,
        "gate_pass_rate":         round(n_gate / n, 3) if n else 0,
        "avg_snomed_per_100w":    round(float(np.mean(densities)), 2) if densities else 0,
        "median_snomed_per_100w": round(float(np.median(densities)), 2) if densities else 0,
        "unique_snomed_concepts": len(unique_ids),
        "T_entropy":              round(compute_entropy(t_lbl), 3),
        "N_entropy":              round(compute_entropy(n_lbl), 3),
        "M_entropy":              round(compute_entropy(m_lbl), 3),
        "_densities":             densities,
    }


def print_comparison(norag: dict, rag: dict, stat: float, p: float, lift: float):
    metrics = [
        ("n_records",              "Records"),
        ("schema_compliance",      "Schema compliance"),
        ("gate_pass_rate",         "Gate pass rate"),
        ("avg_snomed_per_100w",    "Avg SNOMED density (/100w)"),
        ("median_snomed_per_100w", "Median SNOMED density (/100w)"),
        ("unique_snomed_concepts", "Unique SNOMED concepts"),
        ("T_entropy",              f"T entropy  (floor {ENTROPY_FLOORS['T']})"),
        ("N_entropy",              f"N entropy  (floor {ENTROPY_FLOORS['N']})"),
        ("M_entropy",              f"M entropy  (floor {ENTROPY_FLOORS['M']})"),
    ]
    print("\n" + "="*75)
    print("ABLATION 3: RAG vs NO-RAG")
    print("="*75)
    print(f"  {'Metric':<38} {'No-RAG (D)':>14} {'RAG (E)':>14} {'Delta':>10}")
    print("  " + "-"*79)
    for key, label in metrics:
        nr  = norag.get(key, "-")
        rg  = rag.get(key, "-")
        try:
            delta = f"+{rg-nr:.3f}" if rg >= nr else f"{rg-nr:.3f}"
        except TypeError:
            delta = "-"
        print(f"  {label:<38} {str(nr):>14} {str(rg):>14} {delta:>10}")

    print(f"\n  SNOMED density lift (RAG over No-RAG): {lift:+.1f}%")
    print(f"  Mann-Whitney U (one-sided, RAG > No-RAG): U={stat:.0f}, p={p:.4f} "
          f"{'*significant*' if p < 0.05 else '(not significant)'}")
    print("="*75)


def main():
    parser = argparse.ArgumentParser(description="Ablation 3: RAG vs No-RAG comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    norag_path = os.path.join(args.results_dir, "phase1_full_norag.jsonl")
    rag_path   = os.path.join(args.results_dir, "phase1_full_rag.jsonl")

    for p, name in [(norag_path, "run_norag.py"), (rag_path, "run_rag.py")]:
        if not os.path.exists(p):
            print(f"[SKIP] {p} not found — run {name} first")
            return

    norag_recs = load_jsonl(norag_path)
    rag_recs   = load_jsonl(rag_path)

    norag = condition_summary(norag_recs, "No-RAG (Adapter D)")
    rag   = condition_summary(rag_recs,   "RAG (Adapter E)")

    stat, p = mannwhitneyu(rag["_densities"], norag["_densities"], alternative="greater")
    lift    = (
        (rag["avg_snomed_per_100w"] - norag["avg_snomed_per_100w"])
        / max(norag["avg_snomed_per_100w"], 1e-9) * 100
    )

    print_comparison(norag, rag, stat, p, lift)

    for d in (norag, rag):
        d.pop("_densities", None)

    df = pd.DataFrame([norag, rag])
    df.to_csv(os.path.join(args.export_dir, "ablation3_rag_vs_norag.csv"), index=False)

    # Per-record delta
    min_len = min(len(norag_recs), len(rag_recs))
    detail  = [{
        "run_index":     i,
        "T_target":      norag_recs[i].get("T_target"),
        "N_target":      norag_recs[i].get("N_target"),
        "M_target":      norag_recs[i].get("M_target"),
        "norag_density": norag_recs[i].get("snomed_density"),
        "rag_density":   rag_recs[i].get("snomed_density"),
        "density_delta": round((rag_recs[i].get("snomed_density") or 0)
                               - (norag_recs[i].get("snomed_density") or 0), 2),
        "norag_gate":    norag_recs[i].get("gate_pass"),
        "rag_gate":      rag_recs[i].get("gate_pass"),
        "rag_retriever": rag_recs[i].get("rag_retriever"),
    } for i in range(min_len)]
    pd.DataFrame(detail).to_csv(
        os.path.join(args.export_dir, "ablation3_rag_vs_norag_detail.csv"), index=False)

    print(f"\nSaved -> {args.export_dir}")


if __name__ == "__main__":
    main()
