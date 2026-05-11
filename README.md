# What Does Validation Actually Buy? A Systematic Ablation of the Neuro-Symbolic Gate

*Five adapters. One question. Does every gate component earn its keep?*

The original pipeline showed that a neuro-symbolic gate — JSON schema enforcement, SNOMED CT ontology grounding, AJCC clinical-logic checking — produces a synthetic corpus that trains a meaningfully better staging adapter than no gate at all. What it didn't isolate was which component does the work, and how much RAG grounding contributes beyond the gate itself.

This project answers that by making the ablation studies the primary experiment rather than a supporting analysis.

---

## The Design

Every condition generates synthetic lung cancer staging records through the same model, the same TNM grid, the same schema. The only variable is how much of the gate is applied and whether MedCPT retrieval grounds the generation.

**Five conditions. Five adapters. One TSTR evaluation table.**

| Adapter | Condition | Gate applied | RAG |
|---------|-----------|-------------|-----|
| A | `ungated` | None | No |
| B | `schema_only` | JSON schema completeness | No |
| C | `schema_onto` | Schema + SNOMED CT ontology | No |
| D | `full_norag` | Full G(x): schema + ontology + AJCC logic | No |
| E | `full_rag` | Full G(x) | Yes — MedCPT PubMed retrieval |

All five adapters are evaluated on the same three test sets under TSTR — trained on synthetic, tested on real MTSamples clinical notes.

---

## Pipeline

```
preprocessing/mtsamples_prep.py      extract TNM labels from MTSamples

Phase 1  (phases/phase1_generate.py)
  Generate records under a specific condition

Phase 2  (phases/phase2_audit.py)
  Schema compliance  ·  Gate decomposition  ·  Diversity audit
  SNOMED density  ·  AJCC violations  ·  Cross-condition summary table

Phase 4  (phases/phase4_finetune.py)
  QLoRA fine-tune one adapter per condition

Phase 3  (phases/phase3_benchmark.py)
  TSTR: all 5 adapters x 3 test sets (synthetic held-out, lung, all-cancer)
  Per-class T-stage accuracy with 95% bootstrap CI
```

---

## Repository

```
core/                    Shared utilities
  gate.py                G(x) gate components (schema, ontology, logic)
  generation.py          Model loading and generation (4-bit NF4, GPT-4o)
  schemas.py             rigid.v3 schema definition
  tnm_grid.py            32-cell TNM grid, diversity audit, entropy gates
  bioportal.py           SNOMED CT annotation
  medcpt.py              MedCPT RAG retrieval over FAISS PubMed index
  logging_utils.py       JSONL logging and checkpointing

phases/                  Phase 1–4 (each accepts a condition argument)

ablations/               Three independent ablation studies — each runs fully standalone
  1_gate_vs_nogate/
    run_ungated.py       Phase 1 + Phase 4 for Adapter A (no gate)
    run_gated.py         Phase 1 + Phase 4 for Adapter D (full G(x), no RAG)
    compare.py           Side-by-side quality table with per-model breakdown

  2_gate_decomposition/
    run_schema_only.py   Phase 1 + Phase 4 for Adapter B (schema only)
    run_schema_onto.py   Phase 1 + Phase 4 for Adapter C (schema + ontology)
    run_full_gate.py     Phase 1 + Phase 4 for Adapter D (full G(x))
    compare.py           Sequential gate decomposition table with per-model breakdown

  3_rag_vs_norag/
    run_norag.py         Phase 1 + Phase 4 for Adapter D (full gate, no retrieval)
    run_rag.py           Phase 1 + Phase 4 for Adapter E (full gate + MedCPT RAG)
    compare.py           SNOMED density comparison + Mann-Whitney U test

preprocessing/           MTSamples TNM extraction
config.py                All paths, model IDs, and hyperparameters
```

---

## Running the Studies

Each study folder runs independently. Run all three models per condition — ClinicalCamel's low compliance is where the gate decomposition story becomes meaningful.

```bash
# Set env vars first
export HF_TOKEN=your_token
export OPENAI_API_KEY=your_key
export BIOPORTAL_API_KEY=your_key

# Ablation 1 — Gate vs No-Gate
python ablations/1_gate_vs_nogate/run_ungated.py --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/1_gate_vs_nogate/run_ungated.py --model wanglab/ClinicalCamel-70B          --runs 128
python ablations/1_gate_vs_nogate/run_ungated.py --model gpt-4o                             --runs 128

python ablations/1_gate_vs_nogate/run_gated.py   --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/1_gate_vs_nogate/run_gated.py   --model wanglab/ClinicalCamel-70B          --runs 128
python ablations/1_gate_vs_nogate/run_gated.py   --model gpt-4o                             --runs 128

python ablations/1_gate_vs_nogate/compare.py

# Ablation 2 — Gate Decomposition
python ablations/2_gate_decomposition/run_schema_only.py --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/2_gate_decomposition/run_schema_onto.py --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/2_gate_decomposition/run_full_gate.py   --model meta-llama/Llama-3.3-70B-Instruct --runs 128
# (repeat for ClinicalCamel and GPT-4o)
python ablations/2_gate_decomposition/compare.py

# Ablation 3 — RAG vs No-RAG
python ablations/3_rag_vs_norag/run_norag.py --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/3_rag_vs_norag/run_rag.py   --model meta-llama/Llama-3.3-70B-Instruct --runs 128 \
  --faiss-index /path/to/medcpt_index.faiss
# (repeat for other models)
python ablations/3_rag_vs_norag/compare.py

# TSTR benchmark — after all adapters trained
python phases/phase3_benchmark.py
```

---

## Configuration

All paths, model IDs, and hyperparameters are in `config.py`.
Override via environment variable before running:

```bash
export GENERATOR_MODEL=gpt-4o
export BIOPORTAL_API_KEY=your_key
export HF_TOKEN=your_token
export FAISS_INDEX_PATH=/path/to/medcpt_index.faiss
export FAISS_TEXTS_PATH=/path/to/pubmed_abstracts.txt
```

---

## What Is Next

- Expand to colorectal and breast — same gate, new TNM grids
- Demographic stratification grid — address the sex/age bias in generated records
- MedCPT Cross-Encoder reranking — fix retrieval concentration to a small paper set
- Multi-institutional TSTR — MIMIC-III and eICU for validated ground truth

---

*Laxmigayathri Challa · PhD, Information Science (Data Science) · University of North Texas*
