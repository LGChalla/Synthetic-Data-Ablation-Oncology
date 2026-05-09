"""
Ablation 1: Gate vs No-Gate
============================
Runs Phase 1 generation under two conditions, then Phase 2 audit,
then trains adapters (Phase 4), and reports the TSTR delta (Phase 3).

Conditions:
  ungated    — all records admitted; no validation applied
  full_norag — full G(x) gate; schema + ontology + AJCC logic required

Central question:
  Does the neuro-symbolic gate produce a corpus that trains a
  meaningfully better extraction adapter than no gate at all?

Ablation table produced:
  | Condition | Schema% | Ontology% | LogicPass% | T entropy | TSTR T4 | TSTR T3 |
  |-----------|---------|-----------|------------|-----------|---------|---------|
  | Ungated   |         |           |            |           |         |         |
  | Full gate |         |           |            |           |         |         |

Usage:
  python ablations/ablation_gate_vs_nogate.py
  python ablations/ablation_gate_vs_nogate.py --runs 64 --skip-training
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from phases import phase1_generate, phase2_audit, phase4_finetune


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Gate vs No-Gate")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.GATE_ABLATION_RUNS)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Phase 4 — only generate and audit")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# ABLATION 1: GATE vs NO-GATE")
    print("#" * 70)

    # Phase 1 — generate under both conditions
    for condition in ["ungated", "full_norag"]:
        print(f"\n[Phase 1] Condition: {condition}")
        phase1_generate.run(
            condition=condition, model_id=args.model,
            n_runs=args.runs, results_dir=args.results_dir,
        )

    # Phase 2 — quality audit
    print("\n[Phase 2] Quality audit across conditions...")
    from phases.phase2_audit import load_all, schema_compliance, gate_decomposition, \
        diversity_audit, snomed_density_table, ajcc_violations_table, summary_table, print_summary
    import os as _os
    export_dir = _os.path.join(args.results_dir, "analysis")
    _os.makedirs(export_dir, exist_ok=True)

    data = {c: [] for c in ["ungated","full_norag"]}
    from core.logging_utils import load_jsonl
    for c in ["ungated","full_norag"]:
        path = _os.path.join(args.results_dir, f"phase1_{c}.jsonl")
        if _os.path.exists(path):
            data[c] = load_jsonl(path)

    schema_df = schema_compliance(data)
    gate_df   = gate_decomposition(data)
    div_df    = diversity_audit(data)
    snomed_df = snomed_density_table(data)
    vio_df    = ajcc_violations_table(data)
    summ_df   = summary_table(schema_df, gate_df, div_df, snomed_df, vio_df)
    print_summary(summ_df)
    summ_df.to_csv(_os.path.join(export_dir, "ablation1_gate_vs_nogate_summary.csv"), index=False)

    # Phase 4 — train adapters
    if not args.skip_training:
        print("\n[Phase 4] Training adapters...")
        for condition in ["ungated", "full_norag"]:
            try:
                phase4_finetune.train_adapter(condition, args.results_dir, args.adapters_dir)
            except RuntimeError as e:
                print(f"  [ERROR] {e}")

    print("\n[Done] Ablation 1 complete.")
    print("  Next: run phases/phase3_benchmark.py to evaluate adapters on MTSamples.")


if __name__ == "__main__":
    main()
