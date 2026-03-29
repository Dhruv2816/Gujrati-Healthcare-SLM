"""src/pipeline/inference.py — End-to-end Gujarati Healthcare QA pipeline."""

from __future__ import annotations
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.retriever.graph_rag import GraphRAGRetriever, is_emergency
from src.cache.redis_client import RedisClient


from src.config import (
    Config,
    SYSTEM_PROMPT,
    EMERGENCY_RESPONSE,
)


def _load_model(config: Config):
    """Load Qwen base + LoRA adapter with 4-bit quantization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if device == "cuda"
        else None
    )
    tok_path = (
        config.ADAPTER_PATH
        if os.path.exists(config.ADAPTER_PATH)
        else config.BASE_MODEL_ID
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=True,
        token=config.HF_TOKEN,
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
        token=config.HF_TOKEN,
    )
    if os.path.exists(os.path.join(config.ADAPTER_PATH, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, config.ADAPTER_PATH)
        print(f"✅ Fine-tuned model loaded (LoRA from {config.ADAPTER_PATH})")
    else:
        model = base
        print("⚠️  LoRA adapter not found — using base model.")
    model.eval()
    return tokenizer, model


class MedicalPipeline:
    """
    Full GraphRAG-powered Gujarati Healthcare QA pipeline.
    Caches both retrieval context and final answers in Redis.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.tokenizer, self.model = _load_model(self.config)
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

        # ❷ Answer cache check (full answer, not just retrieval)
        cache_key = f"answer:{query}"
        cached_answer = self._answer_cache.get_cached(cache_key)
        if cached_answer:
            cached_answer["cache_hit"] = True
            return cached_answer

        # ❸ GraphRAG retrieval
        retrieval = self.retriever.retrieve(query, top_k=top_k)
        context = retrieval["combined_context"]

        # ❹ LLM generation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Medical Textbook Context:\n{context}\n\nQuery: {query}",
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

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
