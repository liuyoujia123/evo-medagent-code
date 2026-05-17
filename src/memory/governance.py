"""
Tool-Governance Memory (G): tracks per-tool trustworthiness.
For each tool t, maintains gt = (ℓt, n⁺t, n⁻t, nᵐⁱˢt).
Trust labels: TRUSTED, CAUTION, AVOID.
"""
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrustLabel(str, Enum):
    TRUSTED = "TRUSTED"
    CAUTION = "CAUTION"
    AVOID = "AVOID"


@dataclass
class ToolGovernanceRecord:
    """Governance state for a single tool."""
    tool_name: str
    trust_label: TrustLabel = TrustLabel.CAUTION
    helpful_count: int = 0       # n⁺: times tool output helped
    harmful_count: int = 0       # n⁻: times tool output misled
    misuse_count: int = 0        # nᵐⁱˢ: times tool was used inappropriately
    total_interactions: int = 0
    enabled: bool = True

    @property
    def helpful_rate(self) -> float:
        """Proportion of interactions where tool was helpful."""
        if self.total_interactions == 0:
            return 0.0
        return self.helpful_count / self.total_interactions

    @property
    def effective_bad_rate(self) -> float:
        """Effective bad rate = (harmful + 0.5 × misuse) / total."""
        if self.total_interactions == 0:
            return 0.0
        return (self.harmful_count + 0.5 * self.misuse_count) / self.total_interactions

    def record_helpful(self) -> None:
        self.helpful_count += 1
        self.total_interactions += 1

    def record_harmful(self) -> None:
        self.harmful_count += 1
        self.total_interactions += 1

    def record_misuse(self) -> None:
        self.misuse_count += 1
        self.total_interactions += 1

    def to_text(self) -> str:
        """Render governance record as context string."""
        icon = {"TRUSTED": "[T]", "CAUTION": "[C]", "AVOID": "[X]"}
        return (f"{icon.get(self.trust_label.value, '❓')} [{self.tool_name}] "
                f"{self.trust_label.value} "
                f"(used {self.total_interactions}×, "
                f"helpful {self.helpful_count}, "
                f"harmful {self.harmful_count}, "
                f"misuse {self.misuse_count})")


class ToolGovernanceMemory:
    """
    Tool-governance memory that tracks per-tool reliability across interactions.
    Updates trust labels based on accumulated interaction statistics.
    """

    def __init__(self, trusted_threshold: float = 0.70,
                 trusted_min_interactions: int = 6,
                 avoid_threshold: float = 0.60,
                 avoid_min_interactions: int = 10):
        self.trusted_threshold = trusted_threshold
        self.trusted_min_interactions = trusted_min_interactions
        self.avoid_threshold = avoid_threshold
        self.avoid_min_interactions = avoid_min_interactions

        self._records: Dict[str, ToolGovernanceRecord] = {}

    def register_tool(self, tool_name: str) -> None:
        """Register a new tool in governance tracking."""
        if tool_name not in self._records:
            self._records[tool_name] = ToolGovernanceRecord(tool_name=tool_name)
            logger.debug(f"Registered tool: {tool_name}")

    def register_tools(self, tool_names: List[str]) -> None:
        for name in tool_names:
            self.register_tool(name)

    def get_record(self, tool_name: str) -> Optional[ToolGovernanceRecord]:
        return self._records.get(tool_name)

    def get_all_records(self) -> List[ToolGovernanceRecord]:
        return list(self._records.values())

    def record_interaction(self, tool_name: str,
                           was_helpful: bool = False,
                           was_harmful: bool = False,
                           was_misuse: bool = False) -> None:
        """Record a tool interaction outcome."""
        if tool_name not in self._records:
            self.register_tool(tool_name)

        record = self._records[tool_name]
        if was_helpful:
            record.record_helpful()
        if was_harmful:
            record.record_harmful()
        if was_misuse:
            record.record_misuse()

        # Re-evaluate trust label
        self._update_label(record)

    def _update_label(self, record: ToolGovernanceRecord) -> None:
        """Update trust label based on accumulated statistics."""
        n = record.total_interactions

        # Check TRUSTED condition
        if (n >= self.trusted_min_interactions and
            record.helpful_rate >= self.trusted_threshold and
            record.harmful_count == 0):
            record.trust_label = TrustLabel.TRUSTED

        # Check AVOID condition
        elif (n >= self.avoid_min_interactions and
              record.effective_bad_rate >= self.avoid_threshold):
            record.trust_label = TrustLabel.AVOID

        else:
            record.trust_label = TrustLabel.CAUTION

    def format_context(self) -> str:
        """Format all governance records as a context string for the agent."""
        if not self._records:
            return "No tool governance data available."

        lines = ["### Tool Governance Status:"]
        for record in sorted(self._records.values(),
                             key=lambda r: (r.trust_label.value, -r.total_interactions)):
            lines.append(f"- {record.to_text()}")
        return "\n".join(lines)

    def get_trusted_tools(self) -> List[str]:
        """Get list of currently TRUSTED tool names."""
        return [name for name, r in self._records.items()
                if r.trust_label == TrustLabel.TRUSTED]

    def get_avoid_tools(self) -> List[str]:
        """Get list of tools currently labeled AVOID."""
        return [name for name, r in self._records.items()
                if r.trust_label == TrustLabel.AVOID]

    def save(self, path: str) -> None:
        """Serialize to JSON."""
        data = {
            "records": [
                {
                    "tool_name": r.tool_name,
                    "trust_label": r.trust_label.value,
                    "helpful_count": r.helpful_count,
                    "harmful_count": r.harmful_count,
                    "misuse_count": r.misuse_count,
                    "total_interactions": r.total_interactions,
                    "enabled": r.enabled,
                }
                for r in self._records.values()
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Tool governance saved: {len(self._records)} tools → {path}")

    def load(self, path: str) -> None:
        """Deserialize from JSON."""
        import os
        if not os.path.exists(path):
            logger.warning(f"No governance file at {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._records.clear()
        for rd in data.get("records", []):
            record = ToolGovernanceRecord(
                tool_name=rd["tool_name"],
                trust_label=TrustLabel(rd["trust_label"]),
                helpful_count=rd["helpful_count"],
                harmful_count=rd["harmful_count"],
                misuse_count=rd["misuse_count"],
                total_interactions=rd["total_interactions"],
                enabled=rd.get("enabled", True),
            )
            self._records[record.tool_name] = record
        logger.info(f"Tool governance loaded: {len(self._records)} tools ← {path}")
