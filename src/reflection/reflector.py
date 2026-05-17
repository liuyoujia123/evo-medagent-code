"""
Reflection module: post-case analysis that distills actionable knowledge
from diagnostic successes and failures.

v2 — now supports:
  ✓ True multimodal reflection via VLMClient (real image analysis)
  ✓ Human-in-the-Loop: pending rules written to disk for manual review
  ✓ Fallback to text-only reflection when VLM unavailable
"""
import os
import json
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReflectionOutput:
    """Structured output from the reflection process."""
    summary: str                       # σ: one-sentence retrospective summary
    guideline: str                     # γ: actionable guideline distilled from the case
    new_rules: List[Tuple[str, int]]   # (instruction, priority) for procedural memory
    quality_score: float = 0.5         # self-assessed quality of the reflection


@dataclass
class PendingRule:
    """A rule awaiting manual approval (HITL)."""
    rule_id: str
    instruction: str
    priority: int
    source_case: int
    summary: str
    guideline: str
    quality_score: float
    created_at: str
    approved: bool = False


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT_REFLECTION = """You are a reflective medical AI assistant performing test-time learning.
Given a completed CXR diagnostic case with the actual image(s), your task is to:
1. Summarize the key lesson in one sentence
2. Extract an actionable diagnostic guideline
3. Propose new procedural rules for future cases

CRITICAL: You can SEE the CXR image(s). Use the visual information to
understand what the agent saw (or missed) and produce accurate, specific feedback.

Be specific and radiology-focused. Rules should be concrete heuristics."""


USER_PROMPT_REFLECTION = """Review the following CXR diagnostic case and provide reflections.

## Case Question
{question}

## Agent's Answer
{prediction}

## Ground Truth Answer
{ground_truth}

## Was the agent correct?
{was_correct}

## Tool Interaction Trace
{tool_trace}

---

Look carefully at the CXR image(s) provided. Consider what findings are visible and
what the agent may have missed or correctly identified.

Output a JSON object with these fields:
```json
{{
  "summary": "One-sentence lesson from this case",
  "guideline": "Actionable diagnostic guideline for future similar cases",
  "new_rules": [
    ["Rule instruction text 1", 0],
    ["Rule instruction text 2", 1]
  ],
  "quality_score": 0.8
}}
```

Rules should be concrete, actionable heuristics for CXR diagnosis.
Priority: 0=CRITICAL (life-threatening if missed), 1=IMPORTANT, 2=GUIDANCE.
For correct cases: extract rules that reinforce good practice.
For incorrect cases: extract corrective rules that prevent the specific mistake.
Only include rules that genuinely add value. Maximum 3 rules."""


# =============================================================================
# Reflector
# =============================================================================

class Reflector:
    """
    Post-feedback reflector that analyzes case outcomes and produces:
    - Episode summaries and guidelines (for episodic memory)
    - Procedural rule proposals (for procedural memory)

    v2 features:
    - Uses VLMClient for true multimodal reflection over CXR images
    - Supports Human-in-the-Loop: saves pending rules for manual review
    - Falls back to LLMClient (text-only) or simple rules when VLM unavailable
    """

    def __init__(
        self,
        text_llm=None,          # LLMClient (DeepSeek, text-only fallback)
        vlm_client=None,        # VLMClient (GPT-4o etc., real vision)
        enabled: bool = True,
        human_in_the_loop: bool = True,
        require_manual_approval: bool = True,
        pending_dir: str = "./pending_review",
    ):
        self._text_llm = text_llm
        self._vlm = vlm_client
        self.enabled = enabled
        self.human_in_the_loop = human_in_the_loop
        self.require_manual_approval = require_manual_approval
        self.pending_dir = pending_dir

        # In-memory queue for pending rules (per-case)
        self._pending_rules: List[PendingRule] = []
        self._case_counter: int = 0

        if self.human_in_the_loop:
            os.makedirs(self.pending_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main reflection entry point
    # ------------------------------------------------------------------

    def reflect(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        tool_trace: str = "",
        image_paths: Optional[List[str]] = None,
        case_index: int = -1,
    ) -> ReflectionOutput:
        """
        Perform post-case reflection. Uses VLM when available for true image analysis.

        When human_in_the_loop is enabled, proposed rules are staged for review
        rather than automatically committed to procedural memory.
        """
        was_correct = (prediction.strip().lower() == ground_truth.strip().lower())
        self._case_counter += 1

        if not self.enabled:
            return self._empty_reflection(was_correct)

        # Try VLM first (can actually see the images)
        if self._vlm is not None and image_paths:
            output = self._reflect_with_vlm(
                question, prediction, ground_truth, was_correct,
                tool_trace, image_paths
            )
        elif self._text_llm is not None:
            # Fallback: text-only LLM (no real image understanding)
            output = self._reflect_with_text_llm(
                question, prediction, ground_truth, was_correct, tool_trace
            )
        else:
            output = self._simple_reflection(question, prediction, ground_truth, was_correct)

        # ---- Human-in-the-Loop: stage rules for review ----
        if self.human_in_the_loop and output.new_rules:
            staged = self._stage_for_review(
                output=output,
                question=question,
                prediction=prediction,
                ground_truth=ground_truth,
                was_correct=was_correct,
                case_index=case_index,
            )
            logger.info(
                f"HITL: {len(staged)} rule(s) staged for manual approval "
                f"→ {self.pending_dir}"
            )

            if self.require_manual_approval:
                # Return rules as empty — they won't be auto-committed
                # The approved rules must be loaded separately via approve_pending()
                return ReflectionOutput(
                    summary=output.summary,
                    guideline=output.guideline,
                    new_rules=[],   # ★ NOT auto-committed
                    quality_score=output.quality_score,
                )

        return output

    # ------------------------------------------------------------------
    # VLM-based reflection (real image understanding)
    # ------------------------------------------------------------------

    def _reflect_with_vlm(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        was_correct: bool,
        tool_trace: str,
        image_paths: List[str],
    ) -> ReflectionOutput:
        """Use VLMClient to analyze the actual CXR images during reflection."""
        prompt = USER_PROMPT_REFLECTION.format(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            was_correct="YES" if was_correct else "NO",
            tool_trace=tool_trace or "No tools were used.",
        )

        try:
            response = self._vlm.chat_with_images(
                SYSTEM_PROMPT_REFLECTION,
                prompt,
                image_paths=image_paths,
                max_tokens=self._vlm.config.max_tokens,
            )
            return self._parse_response(response, was_correct)
        except Exception as e:
            logger.warning(f"VLM reflection failed: {e}. Falling back to text LLM.")
            if self._text_llm:
                return self._reflect_with_text_llm(
                    question, prediction, ground_truth, was_correct, tool_trace
                )
            return self._simple_reflection(question, prediction, ground_truth, was_correct)

    # ------------------------------------------------------------------
    # Text-only fallback reflection (no image analysis)
    # ------------------------------------------------------------------

    def _reflect_with_text_llm(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        was_correct: bool,
        tool_trace: str,
    ) -> ReflectionOutput:
        """Fallback: use text-only LLM for reflection (no image context)."""
        prompt = USER_PROMPT_REFLECTION.format(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            was_correct="YES" if was_correct else "NO",
            tool_trace=tool_trace or "No tools were used.",
        )

        try:
            response = self._text_llm.chat_with_images(
                SYSTEM_PROMPT_REFLECTION,
                prompt,
                max_tokens=600,
            )
            return self._parse_response(response, was_correct)
        except Exception as e:
            logger.warning(f"Text-LLM reflection failed: {e}.")
            return self._simple_reflection(question, prediction, ground_truth, was_correct)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Optional[str], was_correct: bool) -> ReflectionOutput:
        """Parse LLM/VLM reflection response (try JSON, fall back to text extraction)."""
        import re

        if not response:
            return self._simple_reflection("", "", "", was_correct)

        # Try JSON extraction
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReflectionOutput(
                    summary=data.get("summary", ""),
                    guideline=data.get("guideline", ""),
                    new_rules=[(r[0], int(r[1])) for r in data.get("new_rules", [])],
                    quality_score=float(data.get("quality_score", 0.5)),
                )
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                logger.warning(f"Failed to parse reflection JSON: {e}")

        # Text-based fallback
        summary = ""
        guideline = ""
        rules = []
        for line in response.strip().split("\n"):
            line = line.strip()
            lower = line.lower()
            if lower.startswith("summary:") or lower.startswith("lesson:"):
                summary = line.split(":", 1)[1].strip()
            elif lower.startswith("guideline:") or lower.startswith("actionable:"):
                guideline = line.split(":", 1)[1].strip()
            elif lower.startswith("rule") and ":" in line:
                rule_text = line.split(":", 1)[1].strip()
                rules.append((rule_text, 1))

        return ReflectionOutput(
            summary=summary or f"{'Correct' if was_correct else 'Incorrect'} diagnosis.",
            guideline=guideline or "",
            new_rules=rules,
            quality_score=0.3,
        )

    def _simple_reflection(
        self, question: str, prediction: str, ground_truth: str, was_correct: bool
    ) -> ReflectionOutput:
        """Minimal rule-based reflection when no LLM/VLM available."""
        if was_correct:
            return ReflectionOutput(
                summary=f"Correctly diagnosed: {ground_truth}.",
                guideline=(
                    f"For cases like '{question[:50]}...', "
                    f"continue the current reasoning approach."
                ),
                new_rules=[],
                quality_score=0.5,
            )
        else:
            rule = (
                f"When answering '{question[:80]}...', carefully consider that "
                f"the correct answer is '{ground_truth}' rather than '{prediction}'. "
                f"Review differential diagnoses before finalizing.", 0
            )
            return ReflectionOutput(
                summary=f"Mistakenly answered '{prediction}' instead of '{ground_truth}'.",
                guideline=(
                    f"Double-check differential diagnoses for cases similar to: "
                    f"{question[:80]}..."
                ),
                new_rules=[rule],
                quality_score=0.3,
            )

    def _empty_reflection(self, was_correct: bool) -> ReflectionOutput:
        return ReflectionOutput(
            summary="Reflection disabled.",
            guideline="",
            new_rules=[],
            quality_score=0.0,
        )

    # ------------------------------------------------------------------
    # Human-in-the-Loop: staging & approval
    # ------------------------------------------------------------------

    def _stage_for_review(
        self,
        output: ReflectionOutput,
        question: str,
        prediction: str,
        ground_truth: str,
        was_correct: bool,
        case_index: int,
    ) -> List[PendingRule]:
        """Save proposed rules + context to disk for manual review."""
        staged = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (instruction, priority) in enumerate(output.new_rules):
            rule_id = f"case{case_index}_rule{i}_{timestamp}"
            pending = PendingRule(
                rule_id=rule_id,
                instruction=instruction,
                priority=priority,
                source_case=case_index,
                summary=output.summary,
                guideline=output.guideline,
                quality_score=output.quality_score,
                created_at=timestamp,
                approved=False,
            )
            self._pending_rules.append(pending)
            staged.append(pending)

            # Save individual rule file for review
            review_file = os.path.join(self.pending_dir, f"{rule_id}.json")
            review_data = {
                "rule_id": rule_id,
                "instruction": instruction,
                "priority": priority,
                "priority_label": {0: "CRITICAL", 1: "IMPORTANT", 2: "GUIDANCE"}.get(priority, "?"),
                "source_case": case_index,
                "case_question": question,
                "agent_prediction": prediction,
                "ground_truth": ground_truth,
                "agent_was_correct": was_correct,
                "reflection_summary": output.summary,
                "reflection_guideline": output.guideline,
                "quality_score": output.quality_score,
                "created_at": timestamp,
                "status": "PENDING_REVIEW",
            }
            with open(review_file, "w", encoding="utf-8") as f:
                json.dump(review_data, f, ensure_ascii=False, indent=2)

        return staged

    def get_pending_rules(self) -> List[PendingRule]:
        """Get all rules awaiting manual approval."""
        return [r for r in self._pending_rules if not r.approved]

    def approve_rule(self, rule_id: str) -> Optional[PendingRule]:
        """Approve a specific pending rule by ID. Returns the rule if found."""
        for rule in self._pending_rules:
            if rule.rule_id == rule_id and not rule.approved:
                rule.approved = True
                # Update the pending file
                review_file = os.path.join(self.pending_dir, f"{rule_id}.json")
                if os.path.exists(review_file):
                    with open(review_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["status"] = "APPROVED"
                    with open(review_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"HITL: Rule '{rule_id}' APPROVED")
                return rule
        logger.warning(f"HITL: Rule '{rule_id}' not found or already approved.")
        return None

    def approve_all(self) -> int:
        """Approve all pending rules. Returns count of approved rules."""
        count = 0
        for rule in self._pending_rules:
            if not rule.approved:
                rule.approved = True
                count += 1
                review_file = os.path.join(self.pending_dir, f"{rule.rule_id}.json")
                if os.path.exists(review_file):
                    with open(review_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["status"] = "APPROVED"
                    with open(review_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"HITL: {count} rule(s) bulk-approved")
        return count

    def reject_rule(self, rule_id: str) -> bool:
        """Reject a pending rule. Returns True if found and rejected."""
        for rule in self._pending_rules:
            if rule.rule_id == rule_id and not rule.approved:
                rule.approved = True  # mark as processed (but rejected)
                review_file = os.path.join(self.pending_dir, f"{rule_id}.json")
                if os.path.exists(review_file):
                    with open(review_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["status"] = "REJECTED"
                    with open(review_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"HITL: Rule '{rule_id}' REJECTED")
                return True
        return False

    def get_approved_rules(self) -> List[Tuple[str, int]]:
        """Get all approved rules as (instruction, priority) tuples.
        Use this to feed approved rules back into procedural memory."""
        return [
            (r.instruction, r.priority)
            for r in self._pending_rules
            if r.approved
        ]

    def pending_count(self) -> int:
        """Number of rules awaiting review."""
        return len([r for r in self._pending_rules if not r.approved])
