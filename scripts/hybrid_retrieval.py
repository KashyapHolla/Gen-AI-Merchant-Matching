import numpy as np
import pandas as pd
import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants and paths
MODELS_DIR = Path("models")
FAISS_FILE = MODELS_DIR / "faiss_index"
ID_MAP_FILE = MODELS_DIR / "id_to_index_map.csv"
ES_INDEX = os.getenv("ES_INDEX", "merchants")

# Elasticsearch connection details
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")

def load_resources():
    """Load all necessary resources for hybrid search"""
    # Load the ID mapping
    if not ID_MAP_FILE.exists():
        raise FileNotFoundError(f"ID mapping file not found: {ID_MAP_FILE}")
    
    id_map = pd.read_csv(ID_MAP_FILE)
    
    # Load the original data
    registry = pd.read_csv('Data/Business_Registry.csv')
    registry = registry.dropna(subset=['brand', 'city', 'state'])
    
    # Load the FAISS index
    if not FAISS_FILE.exists():
        raise FileNotFoundError(f"FAISS index file not found: {FAISS_FILE}")
    
    faiss_idx = faiss.read_index(str(FAISS_FILE))
    
    # Load the sentence transformer model
    model = SentenceTransformer("intfloat/e5-small-v2")
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USERNAME, ES_PASSWORD)
    )
    
    return registry, id_map, faiss_idx, model, es

def hybrid_search(parsed: dict, k_sparse=50, k_dense=50, top_k=20, 
                 registry=None, id_map=None, faiss_idx=None, model=None, es=None):
    """
    Hybrid search using both Elasticsearch and FAISS
    
    Args:
        parsed: dict with keys "brand", "city", "state", "merchant_id"
        k_sparse: number of results to get from Elasticsearch
        k_dense: number of results to get from FAISS
        top_k: maximum number of final results to return
        
    Returns:
        list of dicts: Each dict is a row from the registry
    """
    # Load resources if not provided
    if any(x is None for x in [registry, id_map, faiss_idx, model, es]):
        registry, id_map, faiss_idx, model, es = load_resources()
    
    # ----- sparse ES query
    must = []
    if parsed.get("brand"):
        must.append({"term": {"brand": parsed["brand"]}})
    if parsed.get("state"):
        must.append({"term": {"state": parsed["state"]}})
    if parsed.get("city"):
        must.append({"match": {"city": parsed["city"]}})
    
    if not must:  # If no query conditions, add a match_all query
        es_body = {"query": {"match_all": {}}}
    else:
        es_body = {"query": {"bool": {"must": must}}}
    
    try:
        # Use proper format for ES 8.x
        res = es.search(index=ES_INDEX, query=es_body["query"], size=k_sparse)
        sparse_ids = [int(hit["_id"]) for hit in res["hits"]["hits"]]
    except Exception as e:
        print(f"Elasticsearch error: {e}")
        sparse_ids = []
    
    # ----- dense FAISS query
    query_parts = []
    if parsed.get("brand"):
        query_parts.append(parsed["brand"])
    if parsed.get("city"):
        query_parts.append(parsed["city"])
    if parsed.get("state"):
        query_parts.append(parsed["state"])
    
    text = " ".join(query_parts)
    if text:
        qvec = model.encode([text], normalize_embeddings=True).astype("float32")
        sim, idx = faiss_idx.search(qvec, k_dense)
        # Map FAISS indices to merchant_ids
        dense_ids = []
        for faiss_idx_pos in idx[0]:
            if faiss_idx_pos >= 0 and faiss_idx_pos < len(id_map):  # Valid index check
                # Use iloc for positional indexing
                merchant_id = id_map.iloc[faiss_idx_pos]['merchant_id']
                dense_ids.append(int(merchant_id))
    else:
        dense_ids = []
    
    # ----- merge & dedupe
    merged_ids = []
    for mid in sparse_ids + dense_ids:
        if mid not in merged_ids:
            merged_ids.append(mid)
        if len(merged_ids) >= top_k:
            break
    
    # ----- fetch full rows
    result = []
    for mid in merged_ids:
        row = registry[registry['merchant_id'] == mid]
        if not row.empty:
            result.append(row.iloc[0].to_dict())
    
    return result

def search_merchant(query_json):
    """
    Search for a merchant based on JSON query
    
    Args:
        query_json: JSON string {"brand": "...", "city": "...", "state": "...", "merchant_id": ...}
        
    Returns:
        list of dicts: Each dict is a row from the registry
    """
    # Parse the JSON query
    if isinstance(query_json, str):
        parsed = json.loads(query_json)
    else:
        parsed = query_json
    
    # Load all resources
    registry, id_map, faiss_idx, model, es = load_resources()
    
    # Perform the hybrid search
    return hybrid_search(parsed, registry=registry, id_map=id_map, 
                        faiss_idx=faiss_idx, model=model, es=es)

if __name__ == "__main__":
    # Example usage
    query = {
        "brand": "VRIERTRS",
        "city": "Monterey Park",
        "state": "CA"
    }
    
    print("Running hybrid search with query:", json.dumps(query, indent=2))
    results = search_merchant(query)
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Brand: {result['brand']}, City: {result['city']}, State: {result['state']}, ID: {result['merchant_id']}") 