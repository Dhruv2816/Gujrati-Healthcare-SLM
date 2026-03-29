"""src/kg/neo4j_client.py — Neo4j driver & CRUD helpers for the Medical Knowledge Graph."""

from __future__ import annotations
from neo4j import GraphDatabase, Driver
from src.config import Config


class Neo4jClient:
    """Thread-safe Neo4j client with medical KG helpers."""

    def __init__(
        self,
        config: Config | None = None,
    ):
        self.config = config or Config()
        self.uri = self.config.NEO4J_URI
        self.user = self.config.NEO4J_USER
        self.password = self.config.NEO4J_PASSWORD

        self._driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        self._create_constraints()

    #  lifecycle
    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def ping(self) -> bool:
        try:
            with self._driver.session() as s:
                s.run("RETURN 1")
            return True
        except Exception:
            return False

    #  schema setup
    def _create_constraints(self):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug)      REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BodyPart)  REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Treatment) REQUIRE n.name IS UNIQUE",
        ]
        with self._driver.session() as s:
            for q in queries:
                s.run(q)

    #  write
    def upsert_entity(self, name: str, label: str, source: str = "") -> None:
        """Merge (upsert) a medical entity node."""
        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n.source = $source, n.updated = timestamp()"
        )
        with self._driver.session() as s:
            s.run(query, name=name.lower().strip(), source=source)

    def upsert_relationship(
        self,
        from_name: str,
        from_label: str,
        rel_type: str,
        to_name: str,
        to_label: str,
        source: str = "",
    ) -> None:
        """Merge nodes and the relationship between them."""
        query = (
            f"MERGE (a:{from_label} {{name: $from_name}}) "
            f"MERGE (b:{to_label}  {{name: $to_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.source = $source"
        )
        with self._driver.session() as s:
            s.run(
                query,
                from_name=from_name.lower().strip(),
                to_name=to_name.lower().strip(),
                source=source,
            )

    #  read ─
    def query_related(self, entity_name: str, depth: int = 2) -> dict:
        """
        Return all nodes and relationships within `depth` hops of `entity_name`.
        Useful for building the graph context passed to the LLM.
        """
        query = (
            "MATCH (n {name: $name})-[r*1.." + str(depth) + "]-(related) "
            "RETURN n, r, related LIMIT 50"
        )
        diseases, drugs, symptoms, treatments = [], [], [], []
        relationships = []
        with self._driver.session() as s:
            result = s.run(query, name=entity_name.lower().strip())
            for record in result:
                related = record["related"]
                labels = list(related.labels)
                node_name = related.get("name", "")
                if "Disease" in labels:
                    diseases.append(node_name)
                elif "Drug" in labels:
                    drugs.append(node_name)
                elif "Symptom" in labels:
                    symptoms.append(node_name)
                elif "Treatment" in labels:
                    treatments.append(node_name)
                for rel in record["r"]:
                    relationships.append(rel.type)

        return {
            "matched_entities": [entity_name],
            "possible_diseases": list(set(diseases)),
            "suggested_drugs": list(set(drugs)),
            "symptoms": list(set(symptoms)),
            "suggested_treatments": list(set(treatments)),
            "relationship_types": list(set(relationships)),
        }

    def search_entities(self, keyword: str, limit: int = 10) -> list[dict]:
        """Full-text style search across all entity labels."""
        query = (
            "MATCH (n) WHERE n.name CONTAINS $kw "
            "RETURN n.name AS name, labels(n) AS labels LIMIT $limit"
        )
        results = []
        with self._driver.session() as s:
            for record in s.run(query, kw=keyword.lower(), limit=limit):
                results.append({"name": record["name"], "type": record["labels"][0]})
        return results

    def get_stats(self) -> dict:
        with self._driver.session() as s:
            counts = {}
            for label in ("Disease", "Drug", "Symptom", "BodyPart", "Treatment"):
                result = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                counts[label] = result.single()["c"]
            rel_result = s.run("MATCH ()-[r]->() RETURN count(r) AS c")
            counts["Relationships"] = rel_result.single()["c"]
        return counts
