import os
import argparse
import warnings

# Ignore warnings for a clean output
warnings.filterwarnings("ignore")

from src.pipeline.inference import MedicalPipeline

def test_rag():
    print("="*60)
    print("🏥 GUJARATI HEALTHCARE GraphRAG - TERMINAL TEST 🏥")
    print("="*60)
    
    # Initialize Pipeline (Will load 3B model on CPU)
    pipeline = MedicalPipeline()
    print("\n✅ Pipeline Initialized Successfully!\n")
    
    test_queries = [
        "હાઈપરટેન્શન (Hypertension) ના મુખ્ય લક્ષણો અને કારણો શું છે?",
        "તાવ અને માથાના દુખાવા માટે ઘરેલૂ ઉપચાર કયા છે?",
        "જો મને અચાનક છાતીમાં દુખાવો થાય, તો મારે શું કરવું જોઈએ?"
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print("\n" + "~"*60)
        print(f"🔍 [TEST {idx}] USER QUERY: {query}")
        print("~"*60)
        
        # This triggers Graph Extraction, Vector Match, and LLM Inference
        result = pipeline.answer(query)
        answer = result["answer"]
        
        print("\n🤖 GraphRAG AI ANSWER:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        
    print("\n✅ RAG System Testing Complete!")

if __name__ == "__main__":
    test_rag()
