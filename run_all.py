"""
run_all.py — Full pipeline orchestrator
=========================================
Runs all three ablation studies end-to-end, then evaluates all five
adapters on MTSamples under the TSTR protocol.

Execution order:
  1. preprocessing/mtsamples_prep.py     prepare real-world test sets
  2. ablations/ablation_gate_vs_nogate   ungated vs full_norag
  3. ablations/ablation_gate_decomposition  schema -> schema+onto -> full G(x)
  4. ablations/ablation_rag_vs_norag     full_norag vs full_rag
  5. phases/phase3_benchmark.py          TSTR evaluation: all 5 adapters x 3 test sets
  6. analysis/figures.py                 generate paper figures

Individual ablations can be run independently — see each script's usage string.

Usage:
  python run_all.py
  python run_all.py --runs 32 --skip-training --skip-figures
  python run_all.py --model gpt-4o --faiss-index /path/to/index.faiss
"""

import argparse
import os
import subprocess
import sys


def run_script(script: str, extra_args: list = None, cwd: str = None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*65}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, cwd=cwd or os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[WARN] {script} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: all ablations")
    parser.add_argument("--model",         default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--runs",          type=int, default=64)
    parser.add_argument("--faiss-index",   default="")
    parser.add_argument("--faiss-texts",   default="")
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--adapters-dir",  default="adapters")
    parser.add_argument("--skip-prep",     action="store_true", help="Skip MTSamples prep")
    parser.add_argument("--skip-training", action="store_true", help="Skip Phase 4 training")
    parser.add_argument("--skip-figures",  action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    base_args = [
        "--model",       args.model,
        "--results-dir", args.results_dir,
        "--adapters-dir",args.adapters_dir,
    ]
    if args.skip_training:
        base_args.append("--skip-training")

    # Step 1 — MTSamples prep
    if not args.skip_prep:
        run_script("preprocessing/mtsamples_prep.py")

    # Step 2 — Ablation 1: Gate vs No-Gate
    run_script("ablations/ablation_gate_vs_nogate.py",
               base_args + ["--runs", str(args.runs)])

    # Step 3 — Ablation 2: Gate Decomposition
    run_script("ablations/ablation_gate_decomposition.py",
               base_args + ["--runs", str(args.runs)])

    # Step 4 — Ablation 3: RAG vs No-RAG
    rag_args = base_args + ["--runs", str(args.runs // 2)]
    if args.faiss_index:
        rag_args += ["--faiss-index", args.faiss_index]
    if args.faiss_texts:
        rag_args += ["--faiss-texts", args.faiss_texts]
    run_script("ablations/ablation_rag_vs_norag.py", rag_args)

    # Step 5 — TSTR Benchmark (all 5 adapters)
    if not args.skip_training:
        run_script("phases/phase3_benchmark.py",
                   ["--results-dir", args.results_dir,
                    "--adapters-dir", args.adapters_dir])

    # Step 6 — Figures
    if not args.skip_figures:
        figures_script = os.path.join("analysis", "figures.py")
        if os.path.exists(figures_script):
            run_script(figures_script, ["--results-dir", args.results_dir])

    print("\n" + "="*65)
    print("Full pipeline complete.")
    print(f"Results: {os.path.abspath(args.results_dir)}")
    print(f"Adapters: {os.path.abspath(args.adapters_dir)}")
    print("="*65)


if __name__ == "__main__":
    main()
