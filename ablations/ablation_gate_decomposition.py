"""
Ablation 2: Gate Decomposition
================================
Runs Phase 1 generation under three sequential gate conditions,
trains one adapter per condition, and evaluates all three on MTSamples.

Conditions (cumulative gate tightening):
  schema_only   — C1: JSON schema completeness required
  schema_onto   — C2: C1 + at least one SNOMED CT term
  full_norag    — C3: C2 + no AJCC clinical-logic violations (full G(x))

Central question:
  Which gate component is the binding constraint?
  Does adding ontology grounding on top of schema help?
  Does the AJCC logic check add meaningful value beyond schema+ontology?

Ablation table produced:
  | Condition    | Yield | Schema% | Onto% | Logic% | T entropy | TSTR T4 | TSTR T3 |
  |--------------|-------|---------|-------|--------|-----------|---------|---------|
  | Schema only  |       |         |       |        |           |         |         |
  | +Ontology    |       |         |       |        |           |         |         |
  | +Logic (G(x))|       |         |       |        |           |         |         |

Usage:
  python ablations/ablation_gate_decomposition.py
  python ablations/ablation_gate_decomposition.py --runs 64 --skip-training
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from phases import phase1_generate, phase4_finetune


def main():
    parser = argparse.ArgumentParser(description="Ablation 2: Gate Decomposition")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.GATE_ABLATION_RUNS)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# ABLATION 2: GATE DECOMPOSITION")
    print("#" * 70)

    conditions = ["schema_only", "schema_onto", "full_norag"]

    # Phase 1 — generate under all three conditions
    for condition in conditions:
        print(f"\n[Phase 1] Condition: {condition}")
        phase1_generate.run(
            condition=condition, model_id=args.model,
            n_runs=args.runs, results_dir=args.results_dir,
        )

    # Phase 2 — gate decomposition audit
    print("\n[Phase 2] Gate decomposition analysis...")
    from core.logging_utils import load_jsonl
    from phases.phase2_audit import (
        schema_compliance, gate_decomposition, diversity_audit,
        snomed_density_table, ajcc_violations_table, summary_table, print_summary
    )
    import pandas as pd

    export_dir = os.path.join(args.results_dir, "analysis")
    os.makedirs(export_dir, exist_ok=True)

    data = {}
    for c in conditions:
        path = os.path.join(args.results_dir, f"phase1_{c}.jsonl")
        if os.path.exists(path):
            data[c] = load_jsonl(path)

    # Incremental rejection table
    print("\n  Gate decomposition — incremental rejections:")
    print(f"  {'Condition':<20} {'Total':>8} {'Admitted':>10} {'Yield':>8} {'Rejected by this gate':>22}")
    print("  " + "-" * 72)
    prev_admitted = None
    for c, recs in data.items():
        total    = len(recs)
        admitted = sum(1 for r in recs if r.get("admitted"))
        yield_r  = round(admitted / total, 3) if total else 0
        rejected = (prev_admitted - admitted) if prev_admitted is not None else "-"
        prev_admitted = admitted
        print(f"  {c:<20} {total:>8} {admitted:>10} {yield_r:>8.1%} {str(rejected):>22}")

    schema_df = schema_compliance(data)
    gate_df   = gate_decomposition(data)
    div_df    = diversity_audit(data)
    snomed_df = snomed_density_table(data)
    vio_df    = ajcc_violations_table(data)
    summ_df   = summary_table(schema_df, gate_df, div_df, snomed_df, vio_df)
    print_summary(summ_df)
    summ_df.to_csv(os.path.join(export_dir, "ablation2_gate_decomposition_summary.csv"), index=False)
    gate_df.to_csv(os.path.join(export_dir, "ablation2_gate_decomposition_detail.csv"), index=False)

    # Phase 4 — train adapters
    if not args.skip_training:
        print("\n[Phase 4] Training adapters...")
        for condition in conditions:
            try:
                phase4_finetune.train_adapter(condition, args.results_dir, args.adapters_dir)
            except RuntimeError as e:
                print(f"  [ERROR] {e}")

    print("\n[Done] Ablation 2 complete.")
    print("  Next: run phases/phase3_benchmark.py to compare adapters B, C, D on MTSamples.")


if __name__ == "__main__":
    main()
