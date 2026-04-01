
import os
import sys
from pprint import pprint

# Add src to path
sys.path.append(os.path.abspath("."))

from src.kg.neo4j_client import Neo4jClient
from src.vectordb.chroma_client import ChromaClient
from src.cache.redis_client import RedisClient
from src.pipeline.inference import MedicalPipeline

def diagnose():
    print("--- DIAGNOSING RAG SYSTEM ---")
    
    # 1. Check Redis
    print("\n1. Checking Redis...")
    try:
        redis = RedisClient()
        ping = redis._client.ping()
        print(f"   - Redis Ping: {ping}")
        # Check some keys
        keys = redis._client.keys("*")
        print(f"   - Redis Keys Count: {len(keys)}")
    except Exception as e:
        print(f"   - Redis Error: {e}")

    # 2. Check Neo4j
    print("\n2. Checking Neo4j...")
    try:
        neo4j = Neo4jClient()
        if neo4j.ping():
            print("   - Neo4j Ping: Success")
            stats = neo4j.get_stats()
            print("   - Neo4j Stats:")
            pprint(stats)
        else:
            print("   - Neo4j Ping: Failed")
    except Exception as e:
        print(f"   - Neo4j Error: {e}")

    # 3. Check ChromaDB
    print("\n3. Checking ChromaDB...")
    try:
        chroma = ChromaClient()
        stats = chroma.get_stats()
        print(f"   - ChromaDB Stats: {stats}")
        
        # Test a search
        query = "ડાયાબિટીઝ"
        results = chroma.search(query, top_k=2)
        print(f"   - Search for '{query}': {len(results)} results found")
        for i, r in enumerate(results):
            print(f"     [{i}] Source: {r['source']}, Score: {r['score']}")
            print(f"         Text snippet: {r['text'][:100]}...")
    except Exception as e:
        print(f"   - ChromaDB Error: {e}")

    # 4. Check Ollama
    print("\n4. Checking Ollama API...")
    import requests
    try:
        res = requests.get("http://localhost:11434/api/tags")
        if res.status_code == 200:
            models = [m['name'] for m in res.json().get('models', [])]
            print(f"   - Ollama Status: Running")
            print(f"   - Available Models: {models}")
            if "gujarati_healthcare_ai:latest" in models:
                print("   - Target model 'gujarati_healthcare_ai' is available.")
            else:
                print("   - WARNING: Target model 'gujarati_healthcare_ai' NOT FOUND.")
        else:
            print(f"   - Ollama Status: Error {res.status_code}")
    except Exception as e:
        print(f"   - Ollama Connection Error: {e}")

    # 5. Pipeline Test
    print("\n5. Pipeline Step-by-Step Test...")
    try:
        pipeline = MedicalPipeline()
        query = "ડાયાબિટીઝના લક્ષણો શું છે?" # What are the symptoms of diabetes?
        print(f"   - Testing Query: {query}")
        
        # Manually run retriever
        print("   - Running Retriever...")
        retrieval = pipeline.retriever.retrieve(query)
        print(f"     * Extracted Entities: {retrieval['extracted_entities']}")
        print(f"     * KG Results: {list(retrieval['kg_results'].keys())}")
        print(f"     * Vector Results count: {len(retrieval['vector_results'])}")
        print(f"     * Combined Context length: {len(retrieval['combined_context'])}")
        print(f"     * Combined Context snippet: {retrieval['combined_context'][:200]}...")
        
        if len(retrieval['combined_context']) < 50:
            print("     * WARNING: Combined context is very short. Retrieval might be failing.")

    except Exception as e:
        print(f"   - Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
