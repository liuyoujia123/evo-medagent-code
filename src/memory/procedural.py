"""
Procedural Memory (S): stores reusable diagnostic policies distilled across episodes.
Each policy pm = (um, ρm): natural-language instruction + priority level.
Tracks per-rule usage statistics for utility-driven selection.
"""
import json
import logging
import math
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from ..utils.embedding import Embedder, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ProceduralRule:
    """A single diagnostic policy / heuristic rule."""
    rule_id: int
    instruction: str               # u: natural-language diagnostic instruction
    priority: int = 1              # ρ ∈ {0, 1, 2}; 0 = highest urgency
    times_selected: int = 0        # how often this rule was selected
    times_helpful: int = 0         # how often agent answered correctly when rule active
    times_harmful: int = 0         # how often agent answered incorrectly when rule active
    source_case: int = -1          # case ID that created this rule
    enabled: bool = True

    @property
    def success_rate(self) -> float:
        """Rate of correct answers when this rule was active."""
        if self.times_selected == 0:
            return 0.0
        return self.times_helpful / self.times_selected

    def to_text(self) -> str:
        """Render rule as context string for agent prompt."""
        priority_label = {0: "CRITICAL", 1: "IMPORTANT", 2: "GUIDANCE"}
        return (f"[{priority_label.get(self.priority, 'GUIDANCE')}] "
                f"(used {self.times_selected}×, "
                f"helpful {self.times_helpful}/{self.times_selected}) {self.instruction}")

    def update_stats(self, was_helpful: bool) -> None:
        """Update usage statistics after a case."""
        self.times_selected += 1
        if was_helpful:
            self.times_helpful += 1
        else:
            self.times_harmful += 1


class ProceduralMemory:
    """
    Procedural memory that maintains reusable diagnostic rules with priority tags.
    Rules are selected by composite score: relevance × utility, with exploration bonus.
    """

    def __init__(self, embedder: Embedder, max_rules: int = 50,
                 retrieval_k: int = 5, exploration_weight: float = 0.15,
                 prune_threshold: float = 0.0):
        self.embedder = embedder
        self.max_rules = max_rules
        self.retrieval_k = retrieval_k
        self.exploration_weight = exploration_weight
        self.prune_threshold = prune_threshold

        self._rules: Dict[int, ProceduralRule] = {}
        self._embeddings: Dict[int, np.ndarray] = {}  # rule_id → embedding
        self._next_id: int = 0
        self._total_selections: int = 0  # across all rules, for UCB

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def rules(self) -> List[ProceduralRule]:
        return list(self._rules.values())

    @property
    def active_rules(self) -> List[ProceduralRule]:
        return [r for r in self._rules.values() if r.enabled]

    def add(self, instruction: str, priority: int = 1, source_case: int = -1) -> ProceduralRule:
        """Add a new procedural rule."""
        rule_id = self._next_id
        self._next_id += 1

        rule = ProceduralRule(
            rule_id=rule_id,
            instruction=instruction,
            priority=priority,
            source_case=source_case,
        )
        self._rules[rule_id] = rule

        # Compute embedding
        self._embeddings[rule_id] = self.embedder.single_encode(instruction)

        # Prune if over capacity
        self._prune_excess()
        logger.debug(f"Added rule #{rule_id}: {instruction[:80]}...")
        return rule

    def update_or_add(self, instruction: str, priority: int = 1,
                      source_case: int = -1) -> ProceduralRule:
        """
        Add a rule, or if a semantically similar rule exists, bump its priority.
        Returns the rule (existing or new).
        """
        # Check for near-duplicate
        existing = self._find_similar(instruction, threshold=0.85)
        if existing:
            # Boost priority of existing rule (lower = more urgent)
            existing.priority = max(0, existing.priority - 1)
            logger.debug(f"Merged to existing rule #{existing.rule_id}: {instruction[:80]}...")
            return existing

        return self.add(instruction, priority, source_case)

    def select(self, case_descriptor: str, k: Optional[int] = None) -> List[ProceduralRule]:
        """
        Select top-k procedural rules for the current case.
        Uses composite score: λ × relevance_score + (1-λ) × utility_score + exploration_bonus
        """
        k = k or self.retrieval_k
        active = self.active_rules
        if not active:
            return []

        # Compute relevance scores via embedding similarity
        query_emb = self.embedder.single_encode(case_descriptor)
        rule_embs = np.vstack([self._embeddings[r.rule_id] for r in active])
        relevance = cosine_similarity(query_emb.reshape(1, -1), rule_embs)[0]

        # Compute utility scores (UCB-inspired)
        scores = []
        for i, rule in enumerate(active):
            rel = relevance[i]

            # Utility: success rate with exploration bonus
            if rule.times_selected > 0:
                utility = rule.success_rate
                # UCB exploration bonus
                exploration = self.exploration_weight * math.sqrt(
                    math.log(max(1, self._total_selections) + 1) /
                    (rule.times_selected + 1)
                )
                utility = utility + exploration
            else:
                # Untested rules get exploration boost
                utility = 0.5 + self.exploration_weight * 2.0

            # Priority bonus (lower priority number = more urgent → higher score)
            priority_bonus = {0: 0.20, 1: 0.10, 2: 0.0}.get(rule.priority, 0.0)

            # Composite score
            composite = 0.4 * rel + 0.4 * utility + 0.2 * priority_bonus
            scores.append((composite, rule))

        # Sort descending, take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [rule for _, rule in scores[:k]]

        return selected

    def mark_outcomes(self, selected_rules: List[ProceduralRule],
                      was_correct: bool) -> None:
        """Update stats for rules that were active during a case."""
        for rule in selected_rules:
            rule.update_stats(was_correct)
        self._total_selections += len(selected_rules)

    def format_context(self, rules: List[ProceduralRule]) -> str:
        """Format selected rules as a context string."""
        if not rules:
            return "No procedural guidance available."

        lines = ["### Procedural Diagnostic Heuristics (evolved from prior experience):"]
        for i, rule in enumerate(rules):
            lines.append(f"{i+1}. {rule.to_text()}")
        return "\n".join(lines)

    def _find_similar(self, instruction: str, threshold: float = 0.85) -> Optional[ProceduralRule]:
        """Find a semantically similar existing rule."""
        if not self._rules:
            return None
        query_emb = self.embedder.single_encode(instruction)
        for rule in self._rules.values():
            emb = self._embeddings.get(rule.rule_id)
            if emb is None:
                continue
            sim = float(np.dot(query_emb, emb) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10))
            if sim >= threshold:
                return rule
        return None

    def _prune_excess(self) -> None:
        """Remove lowest-utility rules when over capacity."""
        while len(self._rules) > self.max_rules:
            # Find rule with lowest utility
            worst_id = min(
                self._rules.keys(),
                key=lambda rid: self._rules[rid].success_rate,
            )
            del self._rules[worst_id]
            self._embeddings.pop(worst_id, None)
            logger.debug(f"Pruned rule #{worst_id}")

    def prune_low_utility(self) -> int:
        """Remove rules with utility below threshold. Returns count removed."""
        to_remove = [
            rid for rid, rule in self._rules.items()
            if rule.times_selected >= 5 and rule.success_rate <= self.prune_threshold
        ]
        for rid in to_remove:
            del self._rules[rid]
            self._embeddings.pop(rid, None)
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} low-utility rules")
        return len(to_remove)

    def save(self, path: str) -> None:
        """Serialize to JSON."""
        data = {
            "next_id": self._next_id,
            "total_selections": self._total_selections,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "instruction": r.instruction,
                    "priority": r.priority,
                    "times_selected": r.times_selected,
                    "times_helpful": r.times_helpful,
                    "times_harmful": r.times_harmful,
                    "source_case": r.source_case,
                    "enabled": r.enabled,
                }
                for r in self._rules.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Procedural memory saved: {len(self._rules)} rules → {path}")

    def load(self, path: str) -> None:
        """Deserialize from JSON and rebuild embeddings."""
        import os
        if not os.path.exists(path):
            logger.warning(f"No procedural memory file at {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 0)
        self._total_selections = data.get("total_selections", 0)
        self._rules.clear()
        self._embeddings.clear()

        for rd in data.get("rules", []):
            rule = ProceduralRule(
                rule_id=rd["rule_id"],
                instruction=rd["instruction"],
                priority=rd["priority"],
                times_selected=rd["times_selected"],
                times_helpful=rd["times_helpful"],
                times_harmful=rd["times_harmful"],
                source_case=rd.get("source_case", -1),
                enabled=rd.get("enabled", True),
            )
            self._rules[rule.rule_id] = rule
            self._embeddings[rule.rule_id] = self.embedder.single_encode(rule.instruction)

        logger.info(f"Procedural memory loaded: {len(self._rules)} rules ← {path}")
