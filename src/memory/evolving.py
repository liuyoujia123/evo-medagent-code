"""
Unified self-evolving memory M_i = (E_i, S_i, G_i) that aggregates
episodic, procedural, and tool-governance memory stores.

v2 improvements:
  - Optional Qdrant backend for episodic memory (vector search)
  - Optional Neo4j backend for procedural memory (knowledge graph)
  - Support for loading pre-approved HITL rules
  - Graceful fallback to in-memory when databases unavailable
"""
import logging
import os
from typing import Optional, Tuple
from dataclasses import dataclass

from .episodic import EpisodicMemory, Episode
from .procedural import ProceduralMemory, ProceduralRule
from .governance import ToolGovernanceMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Assembled context from all memory stores for agent inference."""
    episodic_context: str
    procedural_context: str
    governance_context: str
    retrieved_episodes: list
    selected_rules: list

    def to_prompt(self) -> str:
        """Combine all memory contexts into a single prompt prefix."""
        parts = []
        if self.governance_context.strip():
            parts.append(self.governance_context)
        if self.procedural_context.strip():
            parts.append(self.procedural_context)
        if self.episodic_context.strip():
            parts.append(self.episodic_context)
        return "\n\n".join(parts)


class SelfEvolvingMemory:
    """
    M_i = (E_i, S_i, G_i): the full three-store memory system per Eq. (2).

    v2 adds:
    - Optional Qdrant/Neo4j backends for persistent storage
    - Support for HITL-approved rule injection
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        procedural: ProceduralMemory,
        governance: ToolGovernanceMemory,
        qdrant_store=None,     # optional QdrantStore
        neo4j_store=None,      # optional Neo4jStore
    ):
        self.episodic = episodic
        self.procedural = procedural
        self.governance = governance
        self.qdrant = qdrant_store
        self.neo4j = neo4j_store
        self._case_count: int = 0

    def query(self, case_descriptor: str) -> MemoryContext:
        """
        Assemble memory context for the current case.
        Retrieves relevant episodes, selects procedural rules,
        and formats governance guidance.

        When Qdrant is enabled, also searches the vector store.
        When Neo4j is enabled, also queries the knowledge graph.
        """
        # ---- Episodic retrieval ----
        episodes = self.episodic.retrieve(case_descriptor)

        # If Qdrant is enabled, supplement with vector store results
        if self.qdrant and self.qdrant.enabled:
            try:
                query_emb = self.episodic.embedder.single_encode(case_descriptor)
                qdrant_results = self.qdrant.search(query_emb, top_k=3, score_threshold=0.3)
                if qdrant_results:
                    logger.debug(f"Qdrant returned {len(qdrant_results)} results")
            except Exception as e:
                logger.debug(f"Qdrant search skipped: {e}")

        # ---- Procedural selection ----
        rules = self.procedural.select(case_descriptor)
        self._last_selected_rules = rules

        return MemoryContext(
            episodic_context=self.episodic.format_context(episodes),
            procedural_context=self.procedural.format_context(rules),
            governance_context=self.governance.format_context(),
            retrieved_episodes=episodes,
            selected_rules=rules,
        )

    def update(
        self,
        case_descriptor: str,
        question: str,
        tool_trace: str,
        prediction: str,
        ground_truth: str,
        summary: str = "",
        guideline: str = "",
        new_rules: Optional[list] = None,
        tool_outcomes: Optional[list] = None,
    ) -> None:
        """
        Update all three memory stores after case completion.
        This implements f(M_{i-1}, x_i, τ_i, ŷ_i, y_i) from Eq. (1).

        Args:
            case_descriptor: compact descriptor of the case (image + question)
            question: the clinical question
            tool_trace: τ_i, the tool-interaction trace
            prediction: ŷ_i, agent's answer
            ground_truth: y_i, correct answer
            summary: σ, one-sentence retrospective summary from reflection
            guideline: γ, actionable guideline from reflection
            new_rules: proposed procedural rules from reflection
                       (empty list when HITL requires manual approval)
            tool_outcomes: list of (tool_name, was_helpful, was_harmful, was_misuse) tuples
        """
        self._case_count += 1
        is_correct = (prediction.strip().lower() == ground_truth.strip().lower())

        # ---- 1. Episodic: add the episode ----
        episode = Episode(
            episode_id=-1,
            phi=case_descriptor,
            question=question,
            tool_trace=tool_trace,
            prediction=prediction,
            ground_truth=ground_truth,
            is_correct=is_correct,
            summary=summary,
            guideline=guideline,
        )
        self.episodic.add(episode)

        # Sync to Qdrant if enabled
        if self.qdrant and self.qdrant.enabled:
            try:
                embed_text = self.episodic._make_embed_text(episode)
                emb = self.episodic.embedder.single_encode(embed_text)
                pid = f"ep_{self.episodic._next_id - 1}"
                self.qdrant.upsert(
                    point_id=pid,
                    vector=emb,
                    payload={
                        "episode_id": episode.episode_id,
                        "phi": episode.phi,
                        "question": episode.question[:200],
                        "is_correct": episode.is_correct,
                        "summary": episode.summary[:300],
                    },
                )
            except Exception as e:
                logger.debug(f"Qdrant sync skipped: {e}")

        # ---- 2. Procedural: add or update rules from reflection ----
        # When HITL is enabled, new_rules may be empty (awaiting manual approval).
        # Approved rules are injected later via inject_approved_rules().
        if new_rules:
            for rule_text, priority in new_rules:
                rule = self.procedural.update_or_add(
                    instruction=rule_text,
                    priority=priority,
                    source_case=self._case_count,
                )

                # Sync to Neo4j if enabled
                if self.neo4j and self.neo4j.enabled:
                    try:
                        self.neo4j.create_rule(
                            rule_id=rule.rule_id,
                            instruction=rule.instruction,
                            priority=rule.priority,
                            source_case=self._case_count,
                        )
                    except Exception as e:
                        logger.debug(f"Neo4j sync skipped: {e}")

        # Mark previous selected rules with outcome
        if hasattr(self, '_last_selected_rules'):
            self.procedural.mark_outcomes(self._last_selected_rules, is_correct)

            # Update Neo4j stats for selected rules
            if self.neo4j and self.neo4j.enabled:
                for rule in self._last_selected_rules:
                    try:
                        self.neo4j.create_rule(
                            rule_id=rule.rule_id,
                            instruction=rule.instruction,
                            priority=rule.priority,
                            times_selected=rule.times_selected,
                            success_rate=rule.success_rate,
                        )
                    except Exception:
                        pass

        # ---- 3. Governance: update tool trust records ----
        if tool_outcomes:
            for tool_name, helpful, harmful, misused in tool_outcomes:
                self.governance.record_interaction(
                    tool_name=tool_name,
                    was_helpful=helpful,
                    was_harmful=harmful,
                    was_misuse=misused,
                )

        # ---- Periodic pruning ----
        if self._case_count % 20 == 0:
            self.procedural.prune_low_utility()

    def inject_approved_rules(self, approved_rules: list) -> int:
        """
        Inject HITL-approved rules into procedural memory.

        Args:
            approved_rules: list of (instruction, priority) tuples

        Returns:
            number of rules injected
        """
        count = 0
        for instruction, priority in approved_rules:
            rule = self.procedural.update_or_add(
                instruction=instruction,
                priority=priority,
                source_case=-1,  # pre-approved, not from a specific case
            )
            count += 1

            # Sync to Neo4j
            if self.neo4j and self.neo4j.enabled:
                try:
                    self.neo4j.create_rule(
                        rule_id=rule.rule_id,
                        instruction=rule.instruction,
                        priority=rule.priority,
                    )
                except Exception:
                    pass

        logger.info(f"Injected {count} approved HITL rules into procedural memory")
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str, prefix: str = "") -> None:
        """Persist all three memory stores (and databases if connected)."""
        os.makedirs(save_dir, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""

        self.episodic.save(os.path.join(save_dir, f"{pfx}episodic.json"))
        self.procedural.save(os.path.join(save_dir, f"{pfx}procedural.json"))
        self.governance.save(os.path.join(save_dir, f"{pfx}governance.json"))

        # Qdrant is inherently persistent; Neo4j is inherently persistent
        # JSON files serve as backup / offline copies

    def load(self, save_dir: str, prefix: str = "") -> None:
        """Load all three memory stores (and optionally from databases)."""
        pfx = f"{prefix}_" if prefix else ""

        # Try database load first, fall back to JSON
        if self.neo4j and self.neo4j.enabled:
            try:
                neo4j_rules = self.neo4j.get_all_rules()
                if neo4j_rules:
                    for r in neo4j_rules:
                        self.procedural.update_or_add(
                            instruction=r["instruction"],
                            priority=r.get("priority", 1),
                        )
                    logger.info(f"Loaded {len(neo4j_rules)} rules from Neo4j")
            except Exception as e:
                logger.warning(f"Neo4j load failed: {e}")

        # Always try JSON as fallback/supplement
        self.episodic.load(os.path.join(save_dir, f"{pfx}episodic.json"))
        self.procedural.load(os.path.join(save_dir, f"{pfx}procedural.json"))
        self.governance.load(os.path.join(save_dir, f"{pfx}governance.json"))
