#!/usr/bin/env python3
"""
main.py — Gujarati Healthcare SLM — CLI Entry Point

Usage:
    python main.py --query "ડાયાબિટીઝ ના લક્ષણો?"
    python main.py --query "heart attack symptoms" --top-k 8
    python main.py --query "fever treatment" --no-cache
    python main.py --ingest-books        # Ingest all PDFs from data/books/
    python main.py --stats               # Show Neo4j + Redis + ChromaDB stats
"""
import argparse
import sys
import json


def cmd_query(args):
    """Run a single query through the full GraphRAG pipeline."""
    print(f"\n🔍 Query: {args.query}\n{'='*60}")
    
    from src.pipeline.inference import MedicalPipeline
    pipe = MedicalPipeline()
    result = pipe.answer(args.query, top_k=args.top_k)

    if result["is_emergency"]:
        print(f"\n⚠️  EMERGENCY RESPONSE:\n{result['answer']}")
    else:
        print(f"\n🏥 Answer:\n{result['answer']}")
        print(f"\n{'─'*60}")
        print(f"📦 Cache hit: {result['cache_hit']}")
        
        kg = result.get("kg_results", {})
        if kg.get("possible_diseases"):
            print(f"🕸️  KG Diseases:    {', '.join(kg['possible_diseases'][:5])}")
        if kg.get("suggested_treatments"):
            print(f"🕸️  KG Treatments:  {', '.join(kg['suggested_treatments'][:5])}")
        
        vr = result.get("vector_results", [])
        if vr:
            print(f"📚 Top source:     {vr[0]['source']} (score={vr[0]['score']:.2f})")

    if args.json:
        safe = {k: v for k, v in result.items() if k != "vector_results"}
        print(f"\n📄 JSON:\n{json.dumps(safe, ensure_ascii=False, indent=2)}")


def cmd_ingest_books(args):
    """Ingest all PDFs from data/books/ into ChromaDB + Neo4j."""
    from src.config import BOOKS_DIR
    from src.vectordb.chroma_client import ChromaClient
    from src.kg.neo4j_client import Neo4jClient
    from src.kg.entity_extractor import extract_entities
    import fitz, os

    print(f"\n📚 Ingesting books from {BOOKS_DIR}\n{'='*60}")
    
    chroma = ChromaClient()
    neo4j  = Neo4jClient()
    
    books_dir = str(BOOKS_DIR)
    summary = chroma.ingest_books_dir(books_dir)
    print(f"\n✅ ChromaDB: {sum(summary.values())} total chunks ingested")

    # Also build Neo4j KG from books
    for fname in os.listdir(books_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(books_dir, fname)
            doc = fitz.open(path)
            text = " ".join(page.get_text() for page in doc)
            entities = extract_entities(text)
            for disease in entities.diseases:
                neo4j.upsert_entity(disease, "Disease", source=fname)
            for symptom in entities.symptoms:
                neo4j.upsert_entity(symptom, "Symptom", source=fname)
            for drug in entities.drugs:
                neo4j.upsert_entity(drug, "Drug", source=fname)
                for disease in entities.diseases:
                    neo4j.upsert_relationship(drug, "Drug", "TREATS", disease, "Disease", fname)
            print(f"  🕸️  {fname}: KG nodes pushed to Neo4j")
    
    neo4j.close()
    print("\n✅ Ingestion complete!")


def cmd_stats(args):
    """Show database statistics."""
    print("\n📊 System Stats\n" + "="*60)
    
    try:
        from src.kg.neo4j_client import Neo4jClient
        neo4j = Neo4jClient()
        stats = neo4j.get_stats()
        print("🕸️  Neo4j:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
        neo4j.close()
    except Exception as e:
        print(f"❌ Neo4j: {e}")
    
    try:
        from src.cache.redis_client import RedisClient
        redis = RedisClient()
        print(f"\n⚡ Redis: {'connected ✅' if redis.ping() else 'offline ❌'}")
    except Exception as e:
        print(f"❌ Redis: {e}")

    try:
        from src.vectordb.chroma_client import ChromaClient
        chroma = ChromaClient()
        stats = chroma.get_stats()
        print(f"\n🔷 ChromaDB:")
        for k, v in stats.items():
            print(f"   {k}: {v}")
    except Exception as e:
        print(f"❌ ChromaDB: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🏥 Gujarati Healthcare SLM — GraphRAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # query
    q_parser = sub.add_parser("query", help="Run a healthcare query")
    q_parser.add_argument("--query", "-q", required=True, help="Healthcare question")
    q_parser.add_argument("--top-k", type=int, default=5)
    q_parser.add_argument("--no-cache", action="store_true")
    q_parser.add_argument("--json", action="store_true", help="Also print JSON output")

    # ingest
    sub.add_parser("ingest", help="Ingest PDFs from data/books/ into ChromaDB + Neo4j")

    # stats
    sub.add_parser("stats", help="Show Neo4j / Redis / ChromaDB stats")

    # Allow: python main.py --query "..." (no subcommand)
    parser.add_argument("--query", "-q", help="Quick query (no subcommand needed)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ingest-books", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    if args.stats or (hasattr(args, "command") and args.command == "stats"):
        cmd_stats(args)
    elif args.ingest_books or (hasattr(args, "command") and args.command == "ingest"):
        cmd_ingest_books(args)
    elif args.query or (hasattr(args, "command") and args.command == "query"):
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
