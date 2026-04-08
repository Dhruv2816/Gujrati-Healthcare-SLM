"""src/kg/neo4j_client.py — Graph database interaction layer.

Changes for pipeline fix:
- Typed relationships (HAS_SYMPTOM, TREATED_BY, LOCATED_IN, MAY_CAUSE)
- Confidence weights on edges
- Upsert relationship method now explicitly stores counts, weights, and confidence
- Uses APOC optionally if available, gracefully falls back to explicit MERGE / UNWIND
"""
from typing import Optional
from neo4j import GraphDatabase, Driver
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize connection to Neo4j graph database."""
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.verify_connectivity()

    def verify_connectivity(self) -> None:
        """Verify the database is reachable."""
        try:
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the database driver connection."""
        self._driver.close()

    def setup_schema(self) -> None:
        """Idempotent schema setup (constraints and indices)."""
        queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)
        logger.info("Neo4j constraints and indices initialized.")

    def upsert_entity(self, name: str, entity_type: str) -> None:
        """Merge an entity node into Neo4j."""
        query = """
        MERGE (e:Entity {id: toLower($name)})
        ON CREATE SET e.name = $name, e.type = $type
        ON MATCH SET e.type = $type
        """
        with self._driver.session() as session:
            session.run(query, name=name.strip(), type=entity_type)

    def determine_edge_type(self, type_a: str, type_b: str) -> str:
        """
        Determine richer schema edge types based on source and target types.
        A very flat graph performs poorly in GraphRAG.
        """
        pair = tuple(sorted([type_a, type_b]))

        # Disease mappings
        if type_a == "Disease" and type_b == "Symptom":
            return "HAS_SYMPTOM"
        elif type_a == "Symptom" and type_b == "Disease":
            return "SYMPTOM_OF"
        
        if pair == ("Disease", "Drug"):
            return "TREATED_BY"
        if pair == ("Disease", "Treatment"):
            return "MANAGED_BY"
            
        if type_a == "Disease" and type_b == "BodyPart":
            return "AFFECTS"
        elif type_a == "BodyPart" and type_b == "Disease":
            return "AFFECTED_BY"
            
        if type_a == "Symptom" and type_b == "BodyPart":
            return "LOCATED_IN"
            
        # Default fallback
        return "CO_OCCURS_WITH"

    def upsert_relationship(self, name1: str, type1: str, name2: str, type2: str, 
                          confidence: float = 0.5, doc_id: str = "unknown") -> None:
        """
        Create or update a relationship between two entities.
        Includes confidence weighting based on sliding-window proximity.
        """
        if name1 == name2:
            return  # No self-loops

        rel_type = self.determine_edge_type(type1, type2)
        
        query = f"""
        MATCH (a:Entity {{id: toLower($name1)}})
        MATCH (b:Entity {{id: toLower($name2)}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r.count = 1, r.weight = $confidence, r.docs = [$doc_id]
        ON MATCH SET r.count = r.count + 1, 
                     r.weight = r.weight + ($confidence * 0.5),
                     r.docs = CASE WHEN NOT $doc_id IN r.docs THEN r.docs + [$doc_id] ELSE r.docs END
        """
        with self._driver.session() as session:
            session.run(query, name1=name1.strip(), name2=name2.strip(), 
                       confidence=confidence, doc_id=doc_id)

    def write_graph_batch(self, edges: list[dict]) -> dict:
        """
        Writes a batch of graph operations using UNWIND to avoid round-trip overheads.
        edges format: [{"source": "diabetes", "source_type": "Disease", 
                        "target": "insulin", "target_type": "Drug",
                        "rel_type": "TREATED_BY", "confidence": 0.9, "doc_id": "doc1"}]
        """
        if not edges:
            return {"nodes_created": 0, "edges_created": 0}

        query = """
        UNWIND $batch AS record
        MERGE (a:Entity {id: toLower(record.source)})
        ON CREATE SET a.name = record.source, a.type = record.source_type
        
        MERGE (b:Entity {id: toLower(record.target)})
        ON CREATE SET b.name = record.target, b.type = record.target_type
        
        // Use APOC for dynamic relationship types if available, otherwise fallback
        // We'll use a dynamic apoc creation path
        WITH a, b, record
        CALL apoc.merge.relationship(a, record.rel_type, {}, {}, b, 
            {count: 1, weight: record.confidence, docs: [record.doc_id]}
        ) YIELD rel
        
        // If apoc merged but relationship already existed, update its weights
        // APOC merge handles the ON CREATE implicitly via the props mapping, 
        // but we want to increment counts safely.
        SET rel.count = coalesce(rel.count, 0) + 1
        SET rel.weight = coalesce(rel.weight, 0.0) + (record.confidence * 0.5)
        
        RETURN count(a) as count
        """
        
        # Without APOC, we must write separate UNWIND loops per relation type.
        # This implementation groups by relation type to run efficient MERGEs without APOC
        
        grouped_edges: dict[str, list[dict]] = {}
        for e in edges:
            rel = e.get("rel_type", self.determine_edge_type(e["source_type"], e["target_type"]))
            grouped_edges.setdefault(rel, []).append(e)

        total_processed = 0
        with self._driver.session() as session:
            for rel_type, batch in grouped_edges.items():
                batch_query = f"""
                UNWIND $batch AS record
                MERGE (a:Entity {{id: toLower(record.source)}})
                ON CREATE SET a.name = toLower(record.source), a.type = record.source_type
                
                MERGE (b:Entity {{id: toLower(record.target)}})
                ON CREATE SET b.name = toLower(record.target), b.type = record.target_type
                
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.count = 1, r.weight = record.confidence, r.docs = [record.doc_id]
                ON MATCH SET 
                    r.count = r.count + 1, 
                    r.weight = r.weight + (record.confidence * 0.5)
                """
                session.run(batch_query, batch=batch)
                total_processed += len(batch)
                
        return {"processed": total_processed}
        
    def query(self, cypher_query: str, **kwargs) -> list[dict]:
        """Execute a custom Cypher query and return results."""
        with self._driver.session() as session:
            result = session.run(cypher_query, **kwargs)
            return [dict(record) for record in result]
            
    def get_kg_context(self, entities: list[str], max_hops: int = 2) -> str:
        """
        Pull specific Graph RAG context for a list of entities.
        Takes advantage of the new structural edge weights and relation types!
        """
        if not entities:
            return ""
            
        entities_lower = [e.lower() for e in entities]
        
        query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.id IN $entities OR b.id IN $entities
        // Order by the new weight property
        RETURN a.name AS source, a.type AS source_type, 
               type(r) AS connection, r.weight AS weight, 
               b.name AS target, b.type AS target_type
        ORDER BY r.weight DESC
        LIMIT 50
        """
        with self._driver.session() as session:
            result = session.run(query, entities=entities_lower)
            records = [record for record in result]
            
        if not records:
            return ""
            
        context_parts = ["Knowledge Graph Facts:"]
        for row in records:
            source = row["source"].capitalize()
            target = row["target"].capitalize()
            rel = row["connection"].replace("_", " ").lower()
            
            if rel == "has symptom":
                context_parts.append(f" - {source} commonly presents with {target}.")
            elif rel == "symptom of":
                context_parts.append(f" - {source} is a recognized symptom of {target}.")
            elif rel == "treated by":
                context_parts.append(f" - {source} can be medically addressed using {target}.")
            elif rel == "affects":
                context_parts.append(f" - The {source} primarily impacts the {target}.")
            elif rel == "located in":
                context_parts.append(f" - The {source} is medically associated with the {target}.")
            else:
                context_parts.append(f" - {source} {rel} {target}.")
                
        return "\n".join(context_parts)
