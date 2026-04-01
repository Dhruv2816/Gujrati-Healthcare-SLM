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

    # 1. Smarter keyword extraction for filtering
    # We want to match BOTH the canonical English name AND any Gujarati variant found in the maps
    from src.kg.entity_extractor import DISEASES_MAP, SYMPTOMS_MAP, DRUGS_MAP
    
    search_keywords = set()
    # Add canonical English names
    for lst in [entities.diseases, entities.symptoms, entities.drugs]:
        if lst: search_keywords.update([kw.lower() for kw in lst])
    
    # Add raw Gujarati names from the inverse maps for better coverage in Gujarati books
    all_maps = {**DISEASES_MAP, **SYMPTOMS_MAP, **DRUGS_MAP}
    for gu_kw, en_val in all_maps.items():
        if en_val in search_keywords:
            search_keywords.add(gu_kw.lower())

    # 2. Filter vector passages
    valid_vectors = []
    if vector_results:
        for r in vector_results:
            text_lower = r['text'].lower()
            # Verify: chunk must contain any of our keywords
            if not search_keywords or any(kw in text_lower for kw in search_keywords):
                valid_vectors.append(r)

    # Fallback if filter is too strict
    if not valid_vectors and vector_results:
        valid_vectors = vector_results[:2]

    # 3. Format Part 1: Medical Passages
    text_parts = []
    if valid_vectors:
        valid_count = 0
        for r in valid_vectors:
            text = r['text'].strip()
            # AGGRESSIVE GARBAGE FILTER:
            # Skip if it contains copyright notices, too many dots (index), or page numbers only
            garbage_keywords = ["copyright", "all rights reserved", "printed in the", "isbn", "editorial", "index", "table of contents"]
            if any(gk in text.lower() for gk in garbage_keywords):
                continue
            if ". . ." in text or text.count('.') > 15 or len(text) < 100:
                continue
            
            text_parts.append(text)
            valid_count += 1
            if valid_count >= 3: break
    
    if text_parts:
        parts.append("પુસ્તકની માહિતી (Medical Textbook Info):")
        for p in text_parts:
            parts.append(f"- {p}")

    # 4. Format Part 2: Graph Facts
    if kg_results:
        kg_data = []
        if kg_results.get("possible_diseases"):
            kg_data.append(f"સંભવિત બીમારીઓ: {', '.join(kg_results['possible_diseases'][:5])}")
        if kg_results.get("suggested_drugs"):
            kg_data.append(f"ભલામણ કરેલ દવાઓ: {', '.join(kg_results['suggested_drugs'][:5])}")
        if kg_results.get("symptoms"):
            kg_data.append(f"લક્ષણો: {', '.join(kg_results['symptoms'][:5])}")
        
        if kg_data:
            parts.append("ગ્રાફ માહિતી (Graph Knowledge):")
            parts.append("\n".join(kg_data))

    return "\n\n".join(parts) if parts else "કોઈ સચોટ માહિતી મળી નથી." # No relevant info found
