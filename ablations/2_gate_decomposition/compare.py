"""
Ablation 2 — Gate Decomposition
Comparison Analysis
====================
Loads results from run_schema_only.py, run_schema_onto.py, and
run_full_gate.py and produces the sequential gate decomposition table.

Run AFTER all three generation scripts have completed.

Shows:
  - Admission rate at each checkpoint
  - Records rejected by each additional gate component
  - Quality improvement per gate layer (SNOMED density, entropy)

Outputs:
  results/analysis/ablation2_gate_decomposition.csv
  results/analysis/ablation2_gate_decomposition_detail.csv

Usage:
  python ablations/2_gate_decomposition/compare.py
  python ablations/2_gate_decomposition/compare.py --results-dir /path/to/results
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

CONDITIONS = [
    ("schema_only", "B — Schema (C1)"),
    ("schema_onto", "C — Schema + Ontology (C2)"),
    ("full_norag",  "D — Full G(x) (C3)"),
]


def checkpoint_summary(cond: str, label: str, recs: list, n_prev_admitted: int) -> dict:
    admitted   = [r for r in recs if r.get("admitted")]
    n_total    = len(recs)
    n_admitted = len(admitted)
    rejected_by_gate = n_prev_admitted - n_admitted if n_prev_admitted is not None else "-"
    densities  = [r.get("snomed_density", 0.0) for r in admitted]
    unique_ids = {
        a.get("snomed_id", "")
        for r in admitted
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    }
    t_lbl = [r.get("T","Unknown") for r in admitted if r.get("parsed_json_valid")]
    n_lbl = [r.get("N","Unknown") for r in admitted if r.get("parsed_json_valid")]
    m_lbl = [r.get("M","Unknown") for r in admitted if r.get("parsed_json_valid")]

    return {
        "adapter":             label,
        "total_generated":     n_total,
        "n_admitted":          n_admitted,
        "corpus_yield":        round(n_admitted / n_total, 3) if n_total else 0,
        "rejected_by_gate":    rejected_by_gate,
        "schema_pass_rate":    round(sum(1 for r in recs if r.get("gate_schema")) / max(n_total,1), 3),
        "onto_pass_rate":      round(sum(1 for r in recs if r.get("gate_ontology")) / max(n_total,1), 3),
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


def print_decomposition(rows: list):
    print("\n" + "="*85)
    print("ABLATION 2: GATE DECOMPOSITION  —  Schema -> Ontology -> AJCC Logic")
    print("="*85)
    print(f"\n  {'Adapter':<35} {'Total':>8} {'Admitted':>9} {'Yield':>7} {'Rejected':>10} {'SNOMED/100w':>12}")
    print("  " + "-"*83)
    for r in rows:
        rej = str(r["rejected_by_gate"])
        print(f"  {r['adapter']:<35} {r['total_generated']:>8} {r['n_admitted']:>9} "
              f"{r['corpus_yield']:>7.1%} {rej:>10} {r['avg_snomed_per_100w']:>12.2f}")

    print(f"\n  Entropy per checkpoint (floors: T/N >= {ENTROPY_FLOORS['T']}, M >= {ENTROPY_FLOORS['M']})")
    print(f"  {'Adapter':<35} {'T ent':>8} {'T':>5} {'N ent':>8} {'N':>5} {'M ent':>8} {'M':>5}")
    print("  " + "-"*83)
    for r in rows:
        tp = "PASS" if r["T_floor_pass"] else "FAIL"
        np_ = "PASS" if r["N_floor_pass"] else "FAIL"
        mp = "PASS" if r["M_floor_pass"] else "FAIL"
        print(f"  {r['adapter']:<35} {r['T_entropy']:>8.3f} {tp:>5} "
              f"{r['N_entropy']:>8.3f} {np_:>5} {r['M_entropy']:>8.3f} {mp:>5}")
    print("="*85)


def main():
    parser = argparse.ArgumentParser(description="Ablation 2: Gate decomposition comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    rows, prev_admitted = [], None

    for cond, label in CONDITIONS:
        path = os.path.join(args.results_dir, f"phase1_{cond}.jsonl")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run run_{cond}.py first")
            continue
        recs = load_jsonl(path)
        row  = checkpoint_summary(cond, label, recs, prev_admitted)
        prev_admitted = row["n_admitted"]
        rows.append(row)

    if not rows:
        return

    print_decomposition(rows)
    df = pd.DataFrame(rows)
    out = os.path.join(args.export_dir, "ablation2_gate_decomposition.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
