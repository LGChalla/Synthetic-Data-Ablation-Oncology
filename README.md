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
preprocessing/mtsamples_prep.py     extract TNM labels from MTSamples

Phase 1  (phases/phase1_generate.py)
  Generate records under each condition (run per ablation)

Phase 2  (phases/phase2_audit.py)
  Schema compliance  ·  Gate decomposition  ·  Diversity audit
  SNOMED density  ·  AJCC violations  ·  Cross-condition summary table

Phase 4  (phases/phase4_finetune.py)
  QLoRA fine-tune one adapter per condition

Phase 3  (phases/phase3_benchmark.py)
  TSTR: all 5 adapters x 3 test sets (synthetic held-out, lung, all-cancer)
  Per-class T-stage accuracy with 95% bootstrap CI
```

**Ablation runners** (each orchestrates Phases 1 + 2 + 4 for their conditions):
```
ablations/ablation_gate_vs_nogate.py      A vs D
ablations/ablation_gate_decomposition.py  B vs C vs D
ablations/ablation_rag_vs_norag.py        D vs E
```

**Run everything:**
```bash
python run_all.py --model meta-llama/Llama-3.3-70B-Instruct --runs 64
```

**Run one ablation:**
```bash
python ablations/ablation_gate_vs_nogate.py --runs 64
python ablations/ablation_gate_decomposition.py --runs 64
python ablations/ablation_rag_vs_norag.py --runs 32 --faiss-index /path/to/index.faiss
```

**Skip adapter training** (use existing adapters for benchmark):
```bash
python run_all.py --skip-training
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

## Repository

```
core/          Gate, generation, schemas, TNM grid, BioPortal, MedCPT, logging
phases/        Phase 1-4 (each accepts a condition argument)
ablations/     Three ablation runners — each drives Phase 1+2+4
preprocessing/ MTSamples TNM extraction
analysis/      Figures and summary tables
data/          Ablation design CSVs
config.py      Central configuration
run_all.py     Full pipeline orchestrator
```

---

## What Is Next

- Expand to colorectal, breast — same gate, new TNM grids
- Demographic stratification grid — address the sex/age bias in generated records
- MedCPT Cross-Encoder reranking — fix retrieval concentration to a small paper set
- Multi-institutional TSTR — MIMIC-III and eICU for validated ground truth

---

*Laxmigayathri Challa · PhD, Information Science (Data Science) · University of North Texas*
