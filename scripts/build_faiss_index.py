"""
scripts/build_faiss_index.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build a FAISS flat-IP index over PubMed lung-cancer abstracts using
the MedCPT Article Encoder.  Saves two files:

    data/pubmed_index.faiss   — FAISS inner-product index
    data/pubmed_abstracts.txt — one abstract per line (parallel to index)

Usage (on HPC):
    pip install faiss-cpu biopython --break-system-packages
    export PYTHONPATH=/path/to/Synthetic-Data-Ablation-Oncology
    python3 scripts/build_faiss_index.py

    # With a specific email (required by NCBI):
    python3 scripts/build_faiss_index.py --email you@unt.edu

    # Larger corpus:
    python3 scripts/build_faiss_index.py --n-abstracts 5000

Then pass to run_rag.py:
    python3 ablations/3_rag_vs_norag/run_rag.py \
        --faiss-index data/pubmed_index.faiss \
        --faiss-texts data/pubmed_abstracts.txt
"""

import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--email",       default="laxmigayathrichalla@my.unt.edu")
parser.add_argument("--n-abstracts", type=int, default=2000)
parser.add_argument("--batch-size",  type=int, default=200)
parser.add_argument("--out-dir",     default="data")
parser.add_argument("--query",
    default=(
        "lung cancer TNM staging AJCC "
        "OR lung cancer treatment immunotherapy "
        "OR non-small cell lung cancer chemotherapy "
        "OR lung adenocarcinoma EGFR ALK "
        "OR lung cancer SNOMED clinical NLP"
    ))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
INDEX_PATH = os.path.join(args.out_dir, "pubmed_index.faiss")
TEXTS_PATH = os.path.join(args.out_dir, "pubmed_abstracts.txt")

# ── dependencies ──────────────────────────────────────────────────────────────
print("[1/4] Checking dependencies...")
missing = []
try:    import faiss
except: missing.append("faiss-cpu")
try:    from Bio import Entrez
except: missing.append("biopython")
if missing:
    print(f"  Missing: {missing}")
    print(f"  Run: pip install {' '.join(missing)} --break-system-packages")
    sys.exit(1)

import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

# ── fetch PubMed abstracts ────────────────────────────────────────────────────
print(f"\n[2/4] Fetching {args.n_abstracts} PubMed abstracts...")
Entrez.email = args.email

handle = Entrez.esearch(db="pubmed", term=args.query,
                        retmax=args.n_abstracts, sort="relevance")
pmids  = Entrez.read(handle)["IdList"]
handle.close()
print(f"  Found {len(pmids)} PMIDs.")

abstracts = []
for start in range(0, len(pmids), args.batch_size):
    batch = pmids[start:start + args.batch_size]
    try:
        fetch   = Entrez.efetch(db="pubmed", id=",".join(batch),
                                rettype="abstract", retmode="xml")
        records = Entrez.read(fetch)
        fetch.close()
        for art in records["PubmedArticle"]:
            try:
                title  = str(art["MedlineCitation"]["Article"]["ArticleTitle"])
                abtxt  = art["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [""])
                ab     = " ".join(str(s) for s in abtxt) if isinstance(abtxt, list) else str(abtxt)
                if ab.strip() and len(ab) > 50:
                    abstracts.append(f"{title}. {ab}")
            except Exception:
                pass
    except Exception as e:
        print(f"  [WARN] batch {start}: {e}")
    print(f"  {min(start+args.batch_size, len(pmids))}/{len(pmids)} fetched → {len(abstracts)} valid", end="\r")
    time.sleep(0.35)

print(f"\n  Total abstracts: {len(abstracts)}")
if not abstracts:
    print("  ERROR: No abstracts. Check network/query."); sys.exit(1)

# ── encode with MedCPT Article Encoder ───────────────────────────────────────
print("\n[3/4] Encoding with ncbi/MedCPT-Article-Encoder...")
device    = "cuda" if torch.cuda.is_available() else "cpu"
enc_id    = "ncbi/MedCPT-Article-Encoder"
tokenizer = AutoTokenizer.from_pretrained(enc_id)
model     = AutoModel.from_pretrained(enc_id).to(device).eval()
print(f"  Device: {device}")

embeddings = []
BATCH = 64
for i in range(0, len(abstracts), BATCH):
    batch = abstracts[i:i+BATCH]
    enc   = tokenizer(batch, truncation=True, padding=True,
                      max_length=512, return_tensors="pt").to(device)
    with torch.inference_mode():
        emb = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
    embeddings.append(emb)
    print(f"  {min(i+BATCH, len(abstracts))}/{len(abstracts)} encoded", end="\r")

print()
embeddings = np.vstack(embeddings)
faiss.normalize_L2(embeddings)
print(f"  Embeddings: {embeddings.shape}")

# ── build FAISS index ─────────────────────────────────────────────────────────
print("\n[4/4] Building FAISS IndexFlatIP...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
with open(TEXTS_PATH, "w", encoding="utf-8") as f:
    for line in abstracts:
        f.write(line.replace("\n", " ").strip() + "\n")

print(f"  Saved: {INDEX_PATH}  ({index.ntotal} vectors)")
print(f"  Saved: {TEXTS_PATH}  ({len(abstracts)} lines)")

# ── sanity check ──────────────────────────────────────────────────────────────
print("\nSanity check: querying 'lung cancer T3 N2 AJCC staging'...")
q_tok   = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
qenc    = q_tok("lung cancer T3 N2 AJCC staging treatment SNOMED",
                return_tensors="pt", truncation=True, max_length=64).to(device)
with torch.inference_mode():
    qemb = q_model(**qenc).last_hidden_state[:, 0, :].cpu().float().numpy()
faiss.normalize_L2(qemb)
_, ids = index.search(qemb, 3)
for rank, idx in enumerate(ids[0]):
    print(f"  [{rank+1}] {abstracts[idx][:120]}...")

print(f"""
Done. Run ablation 3 with real MedCPT retrieval:

  export PYTHONPATH=$(pwd)
  python3 ablations/3_rag_vs_norag/run_rag.py \\
      --faiss-index {INDEX_PATH} \\
      --faiss-texts {TEXTS_PATH} \\
      --runs 32
""")
