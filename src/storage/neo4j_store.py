"""
Neo4j graph database connector for Evo-MedAgent.

Used for procedural memory (S): manages diagnostic rules as nodes
and their relationships (conflicts, evolutions, dependencies) as edges.

Schema:
  Nodes:
    (:Rule {rule_id, instruction, priority, success_rate, times_selected, enabled})
    (:Case  {case_id, category, question, correct_answer})
    (:Tool  {name, trust_label})

  Relationships:
    (:Rule)-[:DERIVED_FROM]->(:Case)
    (:Rule)-[:SUPERSEDES]->(:Rule)
    (:Rule)-[:CONFLICTS_WITH]->(:Rule)
    (:Rule)-[:RECOMMENDS]->(:Tool)

Usage:
    store = Neo4jStore(config)
    store.create_rule(...)
    store.find_related_rules(rule_id)
"""
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Neo4jStore:
    """
    Neo4j graph database connector for procedural memory.

    When enabled=False, acts as a no-op pass-through.
    When enabled=True, persists diagnostic rules + relationships in Neo4j.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: Optional[str] = None,
        database: str = "neo4j",
        enabled: bool = False,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.enabled = enabled

        self._driver = None

        if enabled:
            self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish Neo4j connection and verify."""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password or ""),
            )

            # Verify connection
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()

            # Ensure constraints
            self._init_schema()

            logger.info(f"Neo4j connected: {self.uri} (db={self.database})")

        except ImportError:
            logger.warning(
                "neo4j not installed. "
                "Install with: pip install neo4j"
            )
            self.enabled = False

        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Falling back to in-memory mode.")
            self.enabled = False

    def _init_schema(self) -> None:
        """Create constraints and indexes for the knowledge graph."""
        if not self._driver:
            return

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE",
        ]

        with self._driver.session(database=self.database) as session:
            for cypher in constraints:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.debug(f"Schema constraint (may already exist): {e}")

    # ------------------------------------------------------------------
    # Rule CRUD
    # ------------------------------------------------------------------

    def create_rule(
        self,
        rule_id: int,
        instruction: str,
        priority: int = 1,
        source_case: int = -1,
        times_selected: int = 0,
        success_rate: float = 0.0,
        enabled: bool = True,
    ) -> bool:
        """Create or update a diagnostic rule node in the graph."""
        if not self.enabled or not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    """
                    MERGE (r:Rule {rule_id: $rule_id})
                    SET r.instruction = $instruction,
                        r.priority = $priority,
                        r.source_case = $source_case,
                        r.times_selected = $times_selected,
                        r.success_rate = $success_rate,
                        r.enabled = $enabled
                    """,
                    rule_id=rule_id,
                    instruction=instruction,
                    priority=priority,
                    source_case=source_case,
                    times_selected=times_selected,
                    success_rate=success_rate,
                    enabled=enabled,
                )

                # Link rule to its source case
                if source_case >= 0:
                    session.run(
                        """
                        MERGE (c:Case {case_id: $case_id})
                        WITH c
                        MATCH (r:Rule {rule_id: $rule_id})
                        MERGE (r)-[:DERIVED_FROM]->(c)
                        """,
                        rule_id=rule_id,
                        case_id=f"case_{source_case}",
                    )

            logger.debug(f"Neo4j: Rule {rule_id} created/updated")
            return True

        except Exception as e:
            logger.error(f"Neo4j create_rule failed: {e}")
            return False

    def get_rule(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a rule by ID."""
        if not self.enabled or not self._driver:
            return None

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    "MATCH (r:Rule {rule_id: $rule_id}) RETURN r",
                    rule_id=rule_id,
                )
                record = result.single()
                if record:
                    return dict(record["r"])
            return None
        except Exception as e:
            logger.error(f"Neo4j get_rule failed: {e}")
            return None

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule and its relationships."""
        if not self.enabled or not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    "MATCH (r:Rule {rule_id: $rule_id}) DETACH DELETE r",
                    rule_id=rule_id,
                )
            return True
        except Exception as e:
            logger.error(f"Neo4j delete_rule failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_supersedes(self, old_rule_id: int, new_rule_id: int) -> bool:
        """Mark that new_rule supersedes old_rule."""
        if not self.enabled or not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    """
                    MATCH (old:Rule {rule_id: $old_id})
                    MATCH (new:Rule {rule_id: $new_id})
                    MERGE (new)-[:SUPERSEDES]->(old)
                    SET old.enabled = false
                    """,
                    old_id=old_rule_id,
                    new_id=new_rule_id,
                )
            return True
        except Exception as e:
            logger.error(f"Neo4j add_supersedes failed: {e}")
            return False

    def add_conflict(self, rule_id_1: int, rule_id_2: int) -> bool:
        """Mark two rules as conflicting."""
        if not self.enabled or not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    """
                    MATCH (r1:Rule {rule_id: $id1})
                    MATCH (r2:Rule {rule_id: $id2})
                    MERGE (r1)-[:CONFLICTS_WITH]->(r2)
                    """,
                    id1=rule_id_1,
                    id2=rule_id_2,
                )
            return True
        except Exception as e:
            logger.error(f"Neo4j add_conflict failed: {e}")
            return False

    def find_related_rules(self, rule_id: int, depth: int = 2) -> List[Dict[str, Any]]:
        """Find rules related to a given rule (graph traversal)."""
        if not self.enabled or not self._driver:
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (r:Rule {rule_id: $rule_id})-[rel*1..$depth]-(related:Rule)
                    WHERE related.rule_id <> $rule_id
                    RETURN DISTINCT related
                    LIMIT 20
                    """,
                    rule_id=rule_id,
                    depth=depth,
                )
                return [dict(record["related"]) for record in result]
        except Exception as e:
            logger.error(f"Neo4j find_related_rules failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Case tracking
    # ------------------------------------------------------------------

    def record_case(
        self,
        case_id: int,
        category: str = "",
        question: str = "",
        correct_answer: str = "",
    ) -> bool:
        """Record a benchmark case in the graph."""
        if not self.enabled or not self._driver:
            return False

        try:
            with self._driver.session(database=self.database) as session:
                session.run(
                    """
                    MERGE (c:Case {case_id: $case_id})
                    SET c.category = $category,
                        c.question = $question,
                        c.correct_answer = $correct_answer
                    """,
                    case_id=f"case_{case_id}",
                    category=category,
                    question=question[:200],
                    correct_answer=correct_answer,
                )
            return True
        except Exception as e:
            logger.error(f"Neo4j record_case failed: {e}")
            return False

    def find_cases_by_rule(self, rule_id: int) -> List[Dict[str, Any]]:
        """Find all cases that derived from or relate to a rule."""
        if not self.enabled or not self._driver:
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (r:Rule {rule_id: $rule_id})-[:DERIVED_FROM]->(c:Case)
                    RETURN c
                    """,
                    rule_id=rule_id,
                )
                return [dict(record["c"]) for record in result]
        except Exception as e:
            logger.error(f"Neo4j find_cases_by_rule failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_all_rules(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all enabled rules, sorted by priority."""
        if not self.enabled or not self._driver:
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (r:Rule)
                    WHERE r.enabled = true
                    RETURN r
                    ORDER BY r.priority ASC, r.success_rate DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                )
                return [dict(record["r"]) for record in result]
        except Exception as e:
            logger.error(f"Neo4j get_all_rules failed: {e}")
            return []

    def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        if not self._driver:
            return False
        try:
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None


def create_neo4j_store(config: dict) -> Neo4jStore:
    """Factory: create Neo4jStore from config dict."""
    neo4j_cfg = config.get("neo4j", {}) if config else {}
    password = os.getenv(neo4j_cfg.get("password_env", "")) or ""

    return Neo4jStore(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        username=neo4j_cfg.get("username", "neo4j"),
        password=password,
        database=neo4j_cfg.get("database", "neo4j"),
        enabled=neo4j_cfg.get("enabled", False),
    )
