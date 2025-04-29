# Gen-AI-Merchant-Matching

**Turn raw card‑transaction descriptors into clean, structured merchant records using a retrieval‑augmented LLM pipeline.**

A few‑shot parser pulls out key fields, a hybrid Elasticsearch + FAISS search retrieves possible businesses, and a lightweight LLM ranks & explains the single best match—all in well under a second on a laptop.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `prompts/parsing.txt` | Few‑shot prompt that teaches the model to pull **brand, merchant_id, city, state** from raw descriptors |
| `scripts/elasticsearch_faiss_indexer.py` | One‑off utility that<br>1. ingests `Data/Business_Registry.csv`<br>2. builds an **Elasticsearch** index<br>3. builds a **FAISS** cosine‑sim index |
| `scripts/hybrid_retrieval.py` | Combines sparse (ES) and dense (FAISS) results, dedupes, and returns candidates for reranking |
| `scripts/merchant_matcher.py` | Orchestrates the full flow: parser → hybrid search → LLM rerank → optional judge → JSON |
| `config.py` | Central place to point to your local Ollama LLMs, prompts, and Elasticsearch creds |
| `Data/Business_Registry.csv` | supply a CSV with columns `merchant_id,brand,city,state,zip,mcc` |
| `models/` | Auto‑created directory that stores the FAISS index and the id⇆row mapping csv |

---

## Quick‑start (local laptop)

```bash
# 1. Clone & create env
git clone https://github.com/kaushik-holla/Gen-AI-Merchant-Matching.git
cd Gen-AI-Merchant-Matching
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # builds faiss-cpu, sentence-transformers, elasticsearch-py, etc.

# 2. Provide data
mkdir -p Data
cp /path/to/your/Business_Registry.csv Data/

# 3. Configure
cp .env.example .env          # edit ES_HOST/USER/PASS if needed
# or change them directly in config.py

# 4. Build search indexes (≈2‑3 min for 1 M rows)
python scripts/elasticsearch_faiss_indexer.py
```

---

## Matching a descriptor

```python
from scripts.merchant_matcher import match_descriptor

result = match_descriptor("VRIERTRS #727612092139916032 Monterey Park CA")
print(result["match"])
```

Typical output:

```json
{
  "merchant_id": 727612092139916032,
  "brand": "VRIERTRS",
  "city": "Monterey Park",
  "state": "CA",
  "score": 0.94,
  "why": "Brand and merchant-id exact match; city & state identical"
}
```

`score` and `why` come from the reranker prompt, giving both a confidence measure and a human‑readable rationale.  
If the optional **judge** LLM detects hallucinated evidence, the top result is rejected and the runner falls back to the next candidate.

---

## How it works

1. **Parsing** – Few‑shot prompt in `prompts/parsing.txt` extracts the four structured fields from raw text (helper in `scripts/parser.py`).  
2. **Hybrid retrieval** –  
   * **Sparse** query: exact/​fuzzy terms on `brand`, `state`, `city` in Elasticsearch.  
   * **Dense** query: Sentence‑Transformer *intfloat/e5-small-v2* embeddings over `brand city state` fed into FAISS.  
   * Results are merged, deduped, then trimmed to `top_k`.  
3. **LLM rerank** – A Gemma‑3‑12B (via Ollama) scores similarity and explains its score for each candidate.  
4. **LLM judge** (optional) – Second LLM guards against hallucinated evidence.  
5. **Return** – Fast, structured JSON + latency metrics.

---

## Configuration knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `ES_HOST / ES_USERNAME / ES_PASSWORD` | `http://localhost:9200` | Elasticsearch connection details |
| `ES_INDEX` | `merchants` | Name of the index to create/query |
| `parser_llm`, `rerank_llm`, `judge_llm` | see `config.py` | Point to any Ollama‑exposed model you prefer (Llama 3, Mistral, etc.) |
| `k_sparse`, `k_dense`, `top_k` | 50 / 50 / 20 | Tune recall vs speed trade‑off in `scripts/hybrid_retrieval.py` |

---

## Roadmap / TODO

* **Ship `scripts/parser.py`** – current import expects a custom parser; contribute yours!  
* **Docker compose** for ES + Ollama turnkey setup.  
* **Batch inference API** (FastAPI) + streaming mode.  
* **Unit tests** & CI with FAISS‑GPU.

---

## Contributing & license

PRs, issues, and feature requests are welcome—please open a discussion first so we can align on direction.  
No explicit license file yet; default copyright © 2025 Kaushik Holla. Until one is added, assume *All Rights Reserved*.

---

## Acknowledgements
Built on top of **LangChain**, **Sentence‑Transformers**, **FAISS**, **Elasticsearch**, and **Ollama**.
