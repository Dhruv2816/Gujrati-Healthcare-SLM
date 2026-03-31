"""src/retriever/graph_rag.py — GraphRAG: Neo4j KG + ChromaDB vector hybrid retrieval with Redis cache."""

from __future__ import annotations
from typing import Optional

from src.config import EMERGENCY_KEYWORDS
from src.kg.neo4j_client import Neo4jClient
from src.kg.entity_extractor import extract_entities
from src.vectordb.chroma_client import ChromaClient
from src.cache.redis_client import RedisClient


def is_emergency(query: str) -> bool:
    q = query.lower()
    for kw in EMERGENCY_KEYWORDS:
        # If it's a phrase, check if all words are present (more robust for Gujarati)
        words = kw.lower().split()
        if all(w in q for w in words):
            return True
    return False


class GraphRAGRetriever:
    """
    Hybrid retrieval combining:
    1. Redis  — cache check (fast path)
    2. Neo4j  — structured KG traversal (entities + relationships)
    3. ChromaDB — unstructured semantic search (medical book passages)
    """

    def __init__(
        self,
        neo4j: Optional[Neo4jClient] = None,
        chroma: Optional[ChromaClient] = None,
        redis: Optional[RedisClient] = None,
    ):
        # Allow lazy / offline init if services aren't running
        self._neo4j = neo4j or self._try_init(Neo4jClient)
        self._chroma = chroma or self._try_init(ChromaClient)
        self._redis = redis or self._try_init(RedisClient)

    @staticmethod
    def _try_init(cls):
        try:
            return cls()
        except Exception as e:
            print(f"⚠️  {cls.__name__} unavailable: {e}")
            return None

    # public API
    def retrieve(self, query: str, top_k: int = 5) -> dict:
        """
        Returns:
            {
                vector_results : [{"text", "source", "score"}, ...],
                kg_results     : {"matched_entities", "possible_diseases", ...},
                combined_context: str,
                cache_hit      : bool,
            }
        """
        # ❶ Cache check
        if self._redis:
            cached = self._redis.get_cached(query)
            if cached and "combined_context" in cached:
                cached["cache_hit"] = True
                return cached

        # ❷ Entity extraction
        entities = extract_entities(query)

        # ❸ Neo4j graph traversal (all extracted entities)
        kg_results = {"matched_entities": [], "possible_diseases": [], "suggested_drugs": [], "symptoms": [], "suggested_treatments": []}
        all_to_query = list(dict.fromkeys(entities.diseases + entities.symptoms + entities.drugs))
        
        if self._neo4j:
            for entity in all_to_query[:5]: # Query top 5 entities to avoid noise
                try:
                    res = self._neo4j.query_related(entity)
                    kg_results["matched_entities"].append(entity)
                    if res:
                        kg_results["possible_diseases"].extend(res.get("possible_diseases", []))
                        kg_results["suggested_drugs"].extend(res.get("suggested_drugs", []))
                        kg_results["symptoms"].extend(res.get("symptoms", []))
                        kg_results["suggested_treatments"].extend(res.get("suggested_treatments", []))
                except Exception as e:
                    print(f"⚠️  Neo4j query failed for {entity}: {e}")

        # Deduplicate KG results
        for k in kg_results:
            if isinstance(kg_results[k], list):
                kg_results[k] = list(dict.fromkeys(kg_results[k]))

        # ❹ ChromaDB semantic search
        vector_results: list[dict] = []
        if self._chroma:
            try:
                vector_results = self._chroma.search(query, top_k=top_k)
            except Exception as e:
                print(f"⚠️  ChromaDB search failed: {e}")

        # ❺ Build combined context
        combined = _build_context(vector_results, kg_results, entities)

        result = {
            "vector_results": vector_results,
            "kg_results": kg_results,
            "extracted_entities": entities.to_dict(),
            "combined_context": combined,
            "cache_hit": False,
        }

        # ❻ Cache the retrieval (not the LLM answer — that's done in pipeline)
        if self._redis:
            self._redis.set_cache(query, result)

        return result


def _build_context(vector_results: list[dict], kg_results: dict, entities) -> str:
    parts = []

    # Extract target english keywords (diseases, symptoms, drugs) from user Gujarati query
    target_keywords = set()
    for lst in [entities.diseases, entities.symptoms, entities.drugs]:
        if lst:
            target_keywords.update([kw.lower() for kw in lst])

    # Filter vector passages strictly so we DON'T feed garbage/unrelated context to the LLM
    valid_vectors = []
    if vector_results:
        for r in vector_results:
            text_lower = r['text'].lower()
            # Strict verification: chunk must contain the keyword, or if no keyword found, we pass them anyway
            if not target_keywords or any(kw in text_lower for kw in target_keywords):
                valid_vectors.append(r)

    # FALLBACK: If strict filter yields nothing, take the top 2 semantic results anyway
    # to avoid giving the LLM 'Empty Context' which leads to echoing.
    if not valid_vectors and vector_results:
        valid_vectors = vector_results[:2]

    if valid_vectors:
        parts.append("માહિતી (Medical Info):")
        valid_count = 0
        for r in valid_vectors:
            # Skip TOC/Index garbage with dots like (. . . .) or too short
            if ". . ." in r['text'] or r['text'].count('.') > 15:
                continue
            parts.append(f"- {r['text'][:400]}")
            valid_count += 1
            if valid_count >= 2: break

    # KG structured facts
    if kg_results:
        kg_data = []
        if kg_results.get("possible_diseases"):
            kg_data.append(f"Diseases: {', '.join(kg_results['possible_diseases'][:5])}")
        if kg_results.get("suggested_drugs"):
            kg_data.append(f"Drugs: {', '.join(kg_results['suggested_drugs'][:5])}")
        if kg_results.get("symptoms"):
            kg_data.append(f"Symptoms: {', '.join(kg_results['symptoms'][:5])}")
        
        if kg_data:
            parts.append("GRAPH KNOWLEDGE:")
            parts.append("\n".join(kg_data))

    return (
        "\n\n".join(parts) if parts else "No relevant context found."
    )
