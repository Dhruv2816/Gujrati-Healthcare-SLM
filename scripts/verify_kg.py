"""scripts/verify_kg.py — Verification script to validate Neo4j graph structure & density."""
import os
import sys
from dotenv import load_dotenv

# Ensure the root path is accessible to Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg.neo4j_client import Neo4jClient

load_dotenv()
neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")

def verify_kg():
    print("Verifying Knowledge Graph Schema & Density...")
    try:
        client = Neo4jClient(neo4j_uri, "neo4j", neo4j_pass)
        
        counts = client.query("""
            MATCH (n) 
            RETURN labels(n)[0] as Label, COUNT(n) as Count
        """)
        print("\n=== Node Counts ===")
        for c in counts:
            print(f"- {c['Label']}: {c['Count']}")
            
        rels = client.query("""
            MATCH ()-[r]->() 
            RETURN type(r) as Relationship, COUNT(r) as Count, AVG(r.weight) as AvgWeight
        """)
        print("\n=== Edge Counts & Weights ===")
        for r in rels:
            avg_w = r['AvgWeight'] if r['AvgWeight'] is not None else 0.0
            print(f"- {r['Relationship']}: {r['Count']} (Avg Weight: {avg_w:.2f})")
            
        print("\n=== Sample RAG Context Build ===")
        context = client.get_kg_context(['diabetes', 'insulin', 'kidney disease'])
        print(context if context else "(No edges found for these entities - run notebook 03!)")
        
        client.close()
        print("\n✅ Verification script complete.")
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")

if __name__ == "__main__":
    verify_kg()
