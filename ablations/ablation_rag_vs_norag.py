"""
Ablation 3: RAG vs No-RAG
===========================
Runs Phase 1 generation under the full G(x) gate, with and without
MedCPT RAG grounding. Trains one adapter per condition and evaluates
on MTSamples. Both conditions see the same TNM cells in the same order.

Conditions:
  full_norag — full G(x), no retrieval context
  full_rag   — full G(x), MedCPT-retrieved PubMed abstracts injected per cell

Central question:
  Does RAG grounding produce a clinically richer corpus (higher SNOMED
  density, more diverse vocabulary) that translates to better adapter
  performance on real clinical notes?

If FAISS index is unavailable, keyword_context() is used as fallback —
results are still meaningful but the RAG vocabulary effect is understated.

Ablation table produced:
  | Condition  | SNOMED/100w | Unique terms | T entropy | TSTR T4 | TSTR T3 |
  |------------|-------------|--------------|-----------|---------|---------|
  | No-RAG     |             |              |           |         |         |
  | RAG        |             |              |           |         |         |
  | Delta      |             |              |           |         |         |

Usage:
  python ablations/ablation_rag_vs_norag.py
  python ablations/ablation_rag_vs_norag.py --faiss-index /path/to/index.faiss
  python ablations/ablation_rag_vs_norag.py --runs 32 --skip-training
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from phases import phase1_generate, phase4_finetune
from core.logging_utils import load_jsonl


def rag_comparison_table(norag_recs: list, rag_recs: list) -> pd.DataFrame:
    rows = []
    for label, recs in [("No-RAG", norag_recs), ("RAG", rag_recs)]:
        densities = [r.get("snomed_density", 0.0) for r in recs]
        unique_ids = {
            a.get("snomed_id","")
            for r in recs
            for a in (r.get("snomed_codes") or [])
            if a.get("snomed_id")
        }
        rows.append({
            "condition":              label,
            "n_records":              len(recs),
            "n_admitted":             sum(1 for r in recs if r.get("admitted")),
            "schema_compliance":      round(sum(1 for r in recs if r.get("parsed_json_valid"))
                                           / max(len(recs),1), 3),
            "gate_pass_rate":         round(sum(1 for r in recs if r.get("gate_pass"))
                                           / max(len(recs),1), 3),
            "avg_snomed_per_100w":    round(float(np.mean(densities)), 2) if densities else 0,
            "median_snomed_per_100w": round(float(np.median(densities)), 2) if densities else 0,
            "unique_snomed_concepts": len(unique_ids),
        })
    df = pd.DataFrame(rows)

    # Delta row
    delta = {"condition": "Delta (RAG - No-RAG)"}
    for col in ("avg_snomed_per_100w","median_snomed_per_100w","unique_snomed_concepts"):
        rag_val   = df[df["condition"]=="RAG"][col].values[0]
        norag_val = df[df["condition"]=="No-RAG"][col].values[0]
        delta[col] = round(rag_val - norag_val, 2)
    df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)
    return df


def print_rag_table(df: pd.DataFrame, stat: float, p: float, lift: float):
    print("\n" + "=" * 70)
    print("ABLATION 3: RAG-GROUNDED vs NON-RAG")
    print("=" * 70)
    metrics = [
        ("n_records",              "Records generated"),
        ("n_admitted",             "Records admitted"),
        ("schema_compliance",      "Schema compliance"),
        ("gate_pass_rate",         "Gate pass rate"),
        ("avg_snomed_per_100w",    "Avg SNOMED density (/100w)"),
        ("median_snomed_per_100w", "Median SNOMED density (/100w)"),
        ("unique_snomed_concepts", "Unique SNOMED concepts"),
    ]
    rows_dict = {row["condition"]: row for row in df.to_dict("records")}
    print(f"  {'Metric':<38} {'No-RAG':>12} {'RAG':>12} {'Delta':>10}")
    print("  " + "-" * 76)
    for key, label in metrics:
        nr  = rows_dict.get("No-RAG", {}).get(key, "-")
        rg  = rows_dict.get("RAG", {}).get(key, "-")
        dlt = rows_dict.get("Delta (RAG - No-RAG)", {}).get(key, "-")
        print(f"  {label:<38} {str(nr):>12} {str(rg):>12} {str(dlt):>10}")
    print(f"\n  SNOMED density lift: {lift:+.1f}%")
    print(f"  Mann-Whitney U (RAG > No-RAG): U={stat:.0f}, p={p:.4f} "
          f"{'*significant*' if p < 0.05 else '(not significant)'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ablation 3: RAG vs No-RAG")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.RAG_ABLATION_RUNS)
    parser.add_argument("--faiss-index",   default=cfg.FAISS_INDEX_PATH)
    parser.add_argument("--faiss-texts",   default=cfg.FAISS_TEXTS_PATH)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# ABLATION 3: RAG vs NO-RAG")
    print(f"#   FAISS index: {args.faiss_index or 'not provided (keyword fallback)'}")
    print("#" * 70)

    # Phase 1 — generate both conditions
    for condition in ["full_norag", "full_rag"]:
        print(f"\n[Phase 1] Condition: {condition}")
        phase1_generate.run(
            condition=condition, model_id=args.model,
            n_runs=args.runs, results_dir=args.results_dir,
            faiss_index=args.faiss_index, faiss_texts=args.faiss_texts,
        )

    # Load and compare
    norag_path = os.path.join(args.results_dir, "phase1_full_norag.jsonl")
    rag_path   = os.path.join(args.results_dir, "phase1_full_rag.jsonl")
    norag_recs = load_jsonl(norag_path) if os.path.exists(norag_path) else []
    rag_recs   = load_jsonl(rag_path)   if os.path.exists(rag_path)   else []

    if norag_recs and rag_recs:
        norag_d = [r.get("snomed_density", 0.0) for r in norag_recs]
        rag_d   = [r.get("snomed_density", 0.0) for r in rag_recs]
        stat, p = mannwhitneyu(rag_d, norag_d, alternative="greater")
        lift    = (
            (float(np.mean(rag_d)) - float(np.mean(norag_d)))
            / max(float(np.mean(norag_d)), 1e-9) * 100
        )
        df_cmp = rag_comparison_table(norag_recs, rag_recs)
        print_rag_table(df_cmp, stat, p, lift)

        export_dir = os.path.join(args.results_dir, "analysis")
        os.makedirs(export_dir, exist_ok=True)
        df_cmp.to_csv(os.path.join(export_dir, "ablation3_rag_vs_norag.csv"), index=False)

        # Per-record delta export
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
        } for i in range(min_len)]
        pd.DataFrame(detail).to_csv(
            os.path.join(export_dir, "ablation3_rag_vs_norag_detail.csv"), index=False)

    # Phase 4 — train adapters D and E
    if not args.skip_training:
        print("\n[Phase 4] Training adapters D (full_norag) and E (full_rag)...")
        for condition in ["full_norag", "full_rag"]:
            try:
                phase4_finetune.train_adapter(condition, args.results_dir, args.adapters_dir)
            except RuntimeError as e:
                print(f"  [ERROR] {e}")

    print("\n[Done] Ablation 3 complete.")
    print("  Next: run phases/phase3_benchmark.py to compare adapters D and E on MTSamples.")


if __name__ == "__main__":
    main()
