"""src/pipeline/inference.py — End-to-end Gujarati Healthcare QA pipeline."""
from __future__ import annotations
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.config import (
    HF_TOKEN,
    ADAPTER_PATH,
    BASE_MODEL_ID,
    SYSTEM_PROMPT,
    EMERGENCY_RESPONSE,
)
from src.retriever.graph_rag import GraphRAGRetriever, is_emergency
from src.cache.redis_client import RedisClient


import requests

def _load_model():
    """Bypassed: LLM backend is now handled independently via Ollama C++ Engine API."""
    print("⏳ Routing LLM logic to fast local Ollama C++ API server...")
    return None, None


class MedicalPipeline:
    """
    Full GraphRAG-powered Gujarati Healthcare QA pipeline.
    Caches both retrieval context and final answers in Redis.
    """

    def __init__(self):
        self.tokenizer, self.model = _load_model()
        self.retriever = GraphRAGRetriever()
        self._answer_cache = RedisClient()

    def answer(self, query: str, top_k: int = 5, max_new_tokens: int = 350) -> dict:
        """
        Full pipeline:
        1. Emergency check
        2. Redis answer cache check
        3. GraphRAG retrieval (Neo4j + ChromaDB)
        4. LLM generation
        5. Cache final answer
        """
        # ❶ Emergency
        if is_emergency(query):
            return {
                "query": query,
                "answer": EMERGENCY_RESPONSE,
                "is_emergency": True,
                "context": "",
                "kg_results": {},
                "cache_hit": False,
            }

        # ❷ Answer cache check
        cache_key = f"answer:{query}"
        cached_answer = self._answer_cache.get_cached(cache_key)
        if cached_answer:
            cached_answer["cache_hit"] = True
            return cached_answer

        # ❸ GraphRAG retrieval
        retrieval = self.retriever.retrieve(query, top_k=top_k)
        context = retrieval["combined_context"]

        # ❹ LLM generation via Ollama Chat Endpoint
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "gujarati_healthcare_ai",
                    "messages": [
                        {
                            "role": "system", 
                            "content": (
                                "તમે એક 'ગુજરાતી હેલ્થકેર એસિસ્ટન્ટ' છો. તમારું કામ નીચે આપેલ માહિતીના આધારે સવાલનો જવાબ આપવાનું છે.\n\n"
                                f"માહિતી:\n{context}\n\n"
                                "નિયમો: (1) માત્ર ગુજરાતીમાં જવાબ આપો, (2) સવાલ કે માહિતી ફરીથી ન લખો, (3) જો માહિતી ન હોય તો ડૉક્ટરની સલાહ લેવા માટે કહો."
                            )
                        },
                        {
                            "role": "user", 
                            "content": f"સવાલ: {query}\n\nજવાબ:"
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "stop": ["સવાલ:", "માહિતી:", "Question:", "Information:", "<|im_end|>"],
                        "num_predict": max_new_tokens
                    }
                }
            )
            answer_text = response.json().get("message", {}).get("content", "⚠️ Error generating response.").strip()
        except Exception as e:
            answer_text = f"Ollama API Error: {str(e)} - Make sure Ollama daemon is running locally."

        result = {
            "query": query,
            "answer": answer_text,
            "is_emergency": False,
            "context": context,
            "kg_results": retrieval["kg_results"],
            "vector_results": retrieval["vector_results"],
            "extracted_entities": retrieval["extracted_entities"],
            "cache_hit": False,
        }

        # ❺ Cache final answer
        self._answer_cache.set_cache(
            cache_key, {k: v for k, v in result.items() if k != "vector_results"}
        )

        return result
