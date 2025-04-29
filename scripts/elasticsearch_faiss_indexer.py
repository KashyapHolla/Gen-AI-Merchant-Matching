import os
import pandas as pd
import numpy as np
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Elasticsearch connection details
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_INDEX = os.getenv("ES_INDEX", "merchants")

# FAISS output path
FAISS_OUT = Path("models/faiss_index")
FAISS_OUT.parent.mkdir(exist_ok=True, parents=True)

def create_elasticsearch_index_with_mapping(es_client, index_name):
    """Create an Elasticsearch index with proper mapping"""
    mapping = {
        "mappings": {
            "properties": {
                "merchant_id": {"type": "long"},  # Use long for large numeric IDs
                "brand": {"type": "keyword"},     # Use keyword for exact matching
                "city": {"type": "text"},         # Use text for full-text search
                "state": {"type": "keyword"},     # Use keyword for states
                "zip": {"type": "keyword"},       # Use keyword for zip codes
                "mcc": {"type": "keyword"},       # Use keyword for MCC codes
                "full": {"type": "text"}          # Use text for full-text search
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # Create the index with mapping
    es_client.indices.create(index=index_name, body=mapping)
    print(f"Created Elasticsearch index '{index_name}' with mapping")

def generate_elasticsearch_actions(df, index_name):
    """Generate properly formatted documents for Elasticsearch bulk indexing"""
    for _, row in df.iterrows():
        doc = {
            "_index": index_name,
            "_id": str(row["merchant_id"]),  # Use merchant_id as document ID
            "_source": {
                "merchant_id": row["merchant_id"],
                "brand": row["brand"],
                "city": row["city"],
                "state": row["state"],
                "zip": row["zip"] if not pd.isna(row["zip"]) else None,
                "mcc": row["mcc"] if not pd.isna(row["mcc"]) else None,
                "full": row["full"]
            }
        }
        yield doc

def index_elasticsearch_data(df, es_client, index_name):
    """Index data to Elasticsearch with error handling"""
    # First check if index exists and delete if it does
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        print(f"Deleted existing Elasticsearch index '{index_name}'")
    
    # Create index with mapping
    create_elasticsearch_index_with_mapping(es_client, index_name)
    
    # Generate actions for bulk indexing
    actions = list(generate_elasticsearch_actions(df, index_name))
    
    # Use bulk helper with error handling
    success, failed = 0, 0
    errors = []
    
    # Process in smaller batches
    batch_size = 100
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i+batch_size]
        try:
            # Set error trace to get detailed error messages
            response = helpers.bulk(es_client, batch, stats_only=False, raise_on_error=False)
            success += response[0]
            if response[1]:
                failed += len(response[1])
                errors.extend(response[1])
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {str(e)}")
            failed += len(batch)
    
    # Print results
    print(f"Elasticsearch indexing complete: {success} succeeded, {failed} failed")
    
    # If there were failures, print the first few errors
    if errors:
        print(f"First {min(5, len(errors))} errors:")
        for i, error in enumerate(errors[:5]):
            print(f"Error {i+1}: {json.dumps(error, indent=2)}")
    
    return success, failed, errors

def create_faiss_index(df):
    """Create FAISS index from dataframe text"""
    try:
        # Load model
        print("Loading sentence transformer model...")
        model = SentenceTransformer("intfloat/e5-small-v2")
        
        # Convert to list and check for issues
        text_list = df["full"].tolist()
        print(f"Created text list with {len(text_list)} items")
        
        # Generate embeddings safely
        print("Generating embeddings...")
        emb = model.encode(text_list, normalize_embeddings=True)
        
        # Verify embedding shape
        print(f"Generated embeddings with shape: {emb.shape}")
        
        # Create and save FAISS index
        embedding_dim = emb.shape[1]  # Get embedding dimension
        print(f"Creating FAISS index with dimension: {embedding_dim}")
        
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(emb.astype("float32"))
        
        # Save the index
        faiss.write_index(index, str(FAISS_OUT))
        print(f"FAISS index saved to {FAISS_OUT}")
        
        # Save the mapping between IDs and indices
        id_map = pd.DataFrame({
            'merchant_id': df['merchant_id'],
            'index': range(len(df))
        })
        id_map.to_csv('models/id_to_index_map.csv', index=False)
        print(f"ID mapping saved to models/id_to_index_map.csv")
        
        return True
        
    except TypeError as e:
        print(f"TypeError in FAISS indexing: {e}")
        # Print diagnostic information
        print("\nDiagnostic information:")
        print(f"DataFrame shape: {df.shape}")
        print(f"Full column type: {type(df['full'])}")
        
        # Check for problematic values
        if 'full' in df.columns:
            print("\nChecking for problematic values in 'full' column:")
            for i, val in enumerate(df['full']):
                if not isinstance(val, str):
                    print(f"Row {i}: {val} (type: {type(val)})")
                    if i > 10:  # Only print first few problematic rows
                        print("...")
                        break
        return False
        
    except Exception as e:
        print(f"Unexpected error in FAISS indexing: {e}")
        return False

if __name__ == "__main__":
    # Connect to Elasticsearch
    try:
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USERNAME, ES_PASSWORD)
        )
        print(f"Connected to Elasticsearch at {ES_HOST}")
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        es = None
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv('Data/Business_Registry.csv')
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # Clean data and check for NaN values
    df_clean = df.dropna(subset=['brand', 'city', 'state'])  # Drop rows with NaN in key fields
    df_clean["full"] = df_clean["brand"] + " " + df_clean["city"] + " " + df_clean["state"]
    print(f"Processing {len(df_clean)} entries after removing NaN values")
    
    # Index to Elasticsearch if connection is available
    if es:
        print("\n=== ELASTICSEARCH INDEXING ===")
        index_elasticsearch_data(df_clean, es, ES_INDEX)
    
    # Create FAISS index
    print("\n=== FAISS INDEXING ===")
    create_faiss_index(df_clean)
    
    print("\nIndexing completed!") 