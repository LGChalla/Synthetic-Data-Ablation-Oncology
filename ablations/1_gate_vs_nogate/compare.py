"""
Ablation 1 — Gate vs No-Gate
Comparison Analysis
====================
Loads results from run_ungated.py and run_gated.py and produces the
side-by-side quality comparison table for Ablation Study 1.

Run AFTER both generation scripts have completed.

Metrics compared:
  Schema compliance  ·  Ontology coverage  ·  AJCC logic pass rate
  SNOMED density  ·  Label diversity (T/N/M entropy)  ·  Corpus yield

Outputs:
  results/analysis/ablation1_gate_vs_nogate.csv
  results/analysis/ablation1_gate_vs_nogate_detail.csv

Usage:
  python ablations/1_gate_vs_nogate/compare.py
  python ablations/1_gate_vs_nogate/compare.py --results-dir /path/to/results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import compute_entropy, ENTROPY_FLOORS

CONDITIONS = {
    "ungated":    "Adapter A — Ungated",
    "full_norag": "Adapter D — Full Gate",
}


def condition_summary(recs: list, label: str) -> dict:
    admitted   = [r for r in recs if r.get("admitted")]
    n_total    = len(recs)
    n_admitted = len(admitted)
    densities  = [r.get("snomed_density", 0.0) for r in recs]
    unique_ids = {
        a.get("snomed_id", "")
        for r in recs
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    }
    valid = [r for r in recs if r.get("parsed_json_valid")]
    t_lbl = [r.get("T","Unknown") for r in valid]
    n_lbl = [r.get("N","Unknown") for r in valid]
    m_lbl = [r.get("M","Unknown") for r in valid]

    return {
        "condition":           label,
        "total_generated":     n_total,
        "corpus_yield":        round(n_admitted / n_total, 3) if n_total else 0,
        "schema_compliance":   round(sum(1 for r in recs if r.get("gate_schema")) / max(n_total,1), 3),
        "ontology_coverage":   round(sum(1 for r in recs if r.get("gate_ontology")) / max(n_total,1), 3),
        "logic_pass_rate":     round(sum(1 for r in recs if r.get("gate_logic")) / max(n_total,1), 3),
        "avg_snomed_per_100w": round(float(np.mean(densities)), 2) if densities else 0,
        "unique_snomed":       len(unique_ids),
        "T_entropy":           round(compute_entropy(t_lbl), 3),
        "N_entropy":           round(compute_entropy(n_lbl), 3),
        "M_entropy":           round(compute_entropy(m_lbl), 3),
        "T_floor_pass":        compute_entropy(t_lbl) >= ENTROPY_FLOORS["T"],
        "N_floor_pass":        compute_entropy(n_lbl) >= ENTROPY_FLOORS["N"],
        "M_floor_pass":        compute_entropy(m_lbl) >= ENTROPY_FLOORS["M"],
    }


def print_table(rows: list):
    metrics = [
        ("total_generated",     "Records generated"),
        ("corpus_yield",        "Corpus yield"),
        ("schema_compliance",   "Schema compliance"),
        ("ontology_coverage",   "Ontology coverage"),
        ("logic_pass_rate",     "AJCC logic pass rate"),
        ("avg_snomed_per_100w", "SNOMED density (/100w)"),
        ("unique_snomed",       "Unique SNOMED concepts"),
        ("T_entropy",           f"T entropy  (floor {ENTROPY_FLOORS['T']})"),
        ("N_entropy",           f"N entropy  (floor {ENTROPY_FLOORS['N']})"),
        ("M_entropy",           f"M entropy  (floor {ENTROPY_FLOORS['M']})"),
        ("T_floor_pass",        "T diversity PASS"),
        ("N_floor_pass",        "N diversity PASS"),
        ("M_floor_pass",        "M diversity PASS"),
    ]
    by_cond = {r["condition"]: r for r in rows}
    labels  = list(by_cond.keys())

    print("\n" + "="*75)
    print("ABLATION 1: GATE vs NO-GATE")
    print("="*75)
    header = f"  {'Metric':<35}" + "".join(f" {l[:20]:>20}" for l in labels)
    print(header)
    print("  " + "-"*73)
    for key, label in metrics:
        line = f"  {label:<35}"
        for lbl in labels:
            val = str(by_cond[lbl].get(key, "-"))
            line += f" {val:>20}"
        print(line)
    print("="*75)


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Gate vs No-Gate comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    rows = []
    for cond, label in CONDITIONS.items():
        path = os.path.join(args.results_dir, f"phase1_{cond}.jsonl")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run run_{cond.replace('full_norag','gated')}.py first")
            continue
        recs = load_jsonl(path)
        rows.append(condition_summary(recs, label))

    if not rows:
        return

    print_table(rows)
    df = pd.DataFrame(rows)
    out = os.path.join(args.export_dir, "ablation1_gate_vs_nogate.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
