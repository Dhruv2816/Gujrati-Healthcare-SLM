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
    return any(kw.lower() in q for kw in EMERGENCY_KEYWORDS)


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
        primary_entity = (
            entities.diseases or entities.symptoms or entities.drugs or [None]
        )[0]

        # ❸ Neo4j graph traversal
        kg_results: dict = {}
        if self._neo4j and primary_entity:
            try:
                kg_results = self._neo4j.query_related(primary_entity)
            except Exception as e:
                print(f"⚠️  Neo4j query failed: {e}")
                kg_results = {"matched_entities": [primary_entity]}

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

    # Vector (textbook passages)
    if vector_results:
        parts.append("## Relevant Medical Textbook Passages")
        for i, r in enumerate(vector_results[:5], 1):
            parts.append(
                f"[{i}] ({r['source']}, relevance={r['score']:.2f})\n{r['text'][:400]}"
            )

    # KG structured facts
    if kg_results:
        parts.append("\n## Medical Knowledge Graph Context")
        if kg_results.get("possible_diseases"):
            parts.append(
                f"Related Diseases: {', '.join(kg_results['possible_diseases'][:8])}"
            )
        if kg_results.get("suggested_drugs"):
            parts.append(
                f"Common Drugs: {', '.join(kg_results['suggested_drugs'][:8])}"
            )
        if kg_results.get("symptoms"):
            parts.append(
                f"Associated Symptoms: {', '.join(kg_results['symptoms'][:8])}"
            )
        if kg_results.get("suggested_treatments"):
            parts.append(
                f"Treatments: {', '.join(kg_results['suggested_treatments'][:8])}"
            )

    return (
        "\n\n".join(parts) if parts else "No relevant context found in medical books."
    )
