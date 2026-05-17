"""
Evo-MedAgent: Core Reasoning Agent
===================================
Orchestrates memory-augmented inference, tool execution, and post-case reflection.
Implements the continuous cycle: Read → Reason → Feedback → Reflect → Write.
"""
import logging
import os
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .utils.llm_client import LLMClient
from .tools.base import ToolRegistry, ToolResult
from .memory.evolving import SelfEvolvingMemory
from .reflection.reflector import Reflector

logger = logging.getLogger(__name__)


BASE_SYSTEM_PROMPT = """You are an expert radiology AI assistant specializing in chest X-ray (CXR) interpretation.
Your task is to answer clinical questions about CXR images.

Guidelines:
1. Carefully analyze the provided CXR image(s)
2. Consider the clinical question and all relevant findings
3. Use available tools when helpful (classifier, segmenter, VQA, report_generator, grounding)
4. Apply any provided memory context (prior cases, procedural rules, tool governance) to improve your reasoning
5. Provide your final answer in the format: ANSWER: <your answer>

Be precise, evidence-based, and thorough."""


TOOL_USE_PROMPT = """
You may invoke tools using the following format:
<tool>tool_name: argument</tool>

Available tools:
{tool_descriptions}

Example: <tool>classifier: pneumothorax</tool>
After receiving tool output, continue your reasoning and then provide your final answer.
If no tools are needed, proceed directly to your answer."""


class EvoMedAgent:
    """
    Self-evolving medical agent with memory-augmented reasoning.

    Key features:
    - Memory context assembly before each case
    - Tool-augmented reasoning (optional)
    - Post-case reflection → memory update
    - Training-free test-time learning
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory: SelfEvolvingMemory,
        toolbox: Optional[ToolRegistry] = None,
        reflector: Optional[Reflector] = None,
        max_tool_calls: int = 5,
        use_tools: bool = True,
    ):
        self.llm = llm_client
        self.memory = memory
        self.toolbox = toolbox or ToolRegistry()
        self.reflector = reflector
        self.max_tool_calls = max_tool_calls
        self.use_tools = use_tools

        # Tracking
        self.stats: Dict[str, Any] = {
            "total_cases": 0,
            "correct": 0,
            "cumulative_accuracy": [],
            "tool_calls_per_case": [],
            "episodes_stored": 0,
            "rules_created": 0,
        }

    def diagnose(
        self,
        question: str,
        image_paths: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        case_descriptor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single CXR case with memory-augmented reasoning.
        This is the CORE LOOP of Evo-MedAgent.

        Args:
            question: clinical question to answer
            image_paths: list of image file paths
            ground_truth: (optional) correct answer, for evaluation/learning
            case_descriptor: textual description of the case for memory retrieval

        Returns:
            dict with prediction, trace, accuracy, and memory update info
        """
        case_desc = case_descriptor or question
        tool_trace_parts = []
        tool_governance_outcomes = []

        # ===== STEP 1: Assemble memory context =====
        memory_ctx = self.memory.query(case_desc)

        # ===== STEP 2: Build agent prompt with memory =====
        system_prompt = BASE_SYSTEM_PROMPT
        user_message = self._build_user_message(question, memory_ctx)

        # ===== STEP 3: Tool-augmented reasoning loop =====
        prediction, tool_trace_parts, tool_outcomes = self._reasoning_loop(
            system_prompt, user_message, image_paths
        )

        # ===== STEP 4: Evaluate =====
        is_correct = False
        if ground_truth is not None:
            is_correct = self._check_answer(prediction, ground_truth)
            self.stats["total_cases"] += 1
            if is_correct:
                self.stats["correct"] += 1
            self.stats["cumulative_accuracy"].append(
                self.stats["correct"] / self.stats["total_cases"]
            )

        tool_trace = "\n".join(tool_trace_parts) if tool_trace_parts else "No tools used."
        self.stats["tool_calls_per_case"].append(len(tool_trace_parts))

        # ===== STEP 5: Reflect and update memory =====
        reflection_info = {}
        if self.reflector and ground_truth is not None:
            reflection = self.reflector.reflect(
                question=question,
                prediction=prediction,
                ground_truth=ground_truth,
                tool_trace=tool_trace,
                image_paths=image_paths,
            )

            # Update all three memory stores
            self.memory.update(
                case_descriptor=case_desc,
                question=question,
                tool_trace=tool_trace,
                prediction=prediction,
                ground_truth=ground_truth,
                summary=reflection.summary,
                guideline=reflection.guideline,
                new_rules=reflection.new_rules,
                tool_outcomes=tool_outcomes,
            )

            self.stats["episodes_stored"] = len(self.memory.episodic)
            self.stats["rules_created"] = len(self.memory.procedural)

            reflection_info = {
                "summary": reflection.summary,
                "guideline": reflection.guideline,
                "new_rules": reflection.new_rules,
                "quality_score": reflection.quality_score,
            }

        return {
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "tool_trace": tool_trace,
            "memory_episodes_used": len(memory_ctx.retrieved_episodes),
            "memory_rules_used": len(memory_ctx.selected_rules),
            "reflection": reflection_info,
            "stats_snapshot": dict(self.stats),
        }

    def run_benchmark(
        self,
        cases: List[Dict[str, Any]],
        verbose: bool = True,
        save_every: int = 50,
        save_dir: str = "./checkpoints",
    ) -> List[Dict[str, Any]]:
        """
        Run Evo-MedAgent over a sequential benchmark stream.
        This is the main evaluation loop.

        Args:
            cases: list of case dicts with keys: question, image_paths, ground_truth, case_descriptor
            verbose: print progress
            save_every: save memory state every N cases
            save_dir: checkpoint directory

        Returns:
            list of per-case result dicts
        """
        results = []
        for i, case in enumerate(cases):
            question = case["question"]
            image_paths = case.get("image_paths", [])
            ground_truth = case.get("ground_truth")
            case_descriptor = case.get("case_descriptor", question)

            result = self.diagnose(
                question=question,
                image_paths=image_paths,
                ground_truth=ground_truth,
                case_descriptor=case_descriptor,
            )
            result["case_index"] = i
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                acc = self.stats["correct"] / self.stats["total_cases"]
                logger.info(
                    f"Case {i+1}/{len(cases)} | "
                    f"Cumulative acc: {acc:.3f} | "
                    f"Correct: {self.stats['correct']}/{self.stats['total_cases']} | "
                    f"Episodes: {self.stats['episodes_stored']} | "
                    f"Rules: {self.stats['rules_created']}"
                )

            if save_every and (i + 1) % save_every == 0:
                self.memory.save(save_dir, f"step_{i+1}")

        return results

    def _reasoning_loop(
        self,
        system_prompt: str,
        user_message: str,
        image_paths: Optional[List[str]],
    ) -> Tuple[str, List[str], List[Tuple[str, bool, bool, bool]]]:
        """
        Tool-augmented reasoning loop.
        Agent may call tools, receive outputs, and continue reasoning.
        Returns: (final_answer, trace_parts, governance_outcomes)
        """
        trace_parts = []
        governance_outcomes = []

        if not self.use_tools or not self.toolbox or len(self.toolbox.list_tools()) == 0:
            # Tool-free mode: single VLM call
            answer = self.llm.chat_with_images(
                system_prompt, user_message, image_paths
            )
            return self._extract_answer(answer), trace_parts, governance_outcomes

        # Tool-enabled mode: multi-turn reasoning with tool calls
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # DeepSeek is text-only: reference images by path, rely on case descriptor for findings
        if image_paths:
            img_names = [os.path.basename(p) for p in image_paths]
            user_message += f"\n[CXR image(s): {', '.join(img_names)}]"
        messages.append({"role": "user", "content": user_message})

        current_answer = ""
        for turn in range(self.max_tool_calls):
            response = self.llm._chat(messages)
            if response is None:
                break

            # Check for tool calls
            tool_calls = re.findall(r'<tool>(.*?)</tool>', response, re.DOTALL)
            if not tool_calls:
                # No tool call → final answer
                current_answer = response
                break

            # Execute tool calls
            for tool_str in tool_calls:
                parts = tool_str.strip().split(":", 1)
                tool_name = parts[0].strip()
                tool_arg = parts[1].strip() if len(parts) > 1 else ""

                result = self.toolbox.run_tool(tool_name, image_paths[0] if image_paths else "",
                                               **{"finding": tool_arg} if tool_arg else {})
                trace_parts.append(f"Tool: {tool_name}({tool_arg}) → {result.output}")

                # Governance: simplified — mark as helpful unless errored
                was_helpful = result.success and len(result.output) > 20
                governance_outcomes.append((tool_name, was_helpful, not result.success, False))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool_name} output:\n{result.output}\n\nContinue your reasoning or provide your final answer."
                })

        # Fallback: extract answer from last response
        if not current_answer:
            # Final call without tool format to get plain answer
            final = self.llm.chat_with_images(
                "Provide your final answer in format: ANSWER: <answer>",
                f"Based on all the information above, answer the original question.\n\n{user_message}",
                image_paths
            )
            current_answer = final or ""

        return self._extract_answer(current_answer), trace_parts, governance_outcomes

    def _build_user_message(self, question: str, memory_ctx) -> str:
        """Build the user message with memory context and tool instructions."""
        parts = []

        # Memory contexts
        memory_text = memory_ctx.to_prompt()
        if memory_text.strip():
            parts.append("## Accumulated Clinical Experience\n" + memory_text)

        # Tool descriptions (if tool-use enabled)
        if self.use_tools and self.toolbox and len(self.toolbox.list_tools()) > 0:
            parts.append(TOOL_USE_PROMPT.format(
                tool_descriptions=self.toolbox.get_schema_text()
            ))

        # The actual question
        parts.append("## Current Case\n" + question)
        parts.append("\nProvide your answer in format: ANSWER: <your answer>")

        return "\n\n".join(parts)

    def _extract_answer(self, text: Optional[str]) -> str:
        """Extract the final answer from agent output."""
        if not text:
            return ""

        # Try ANSWER: format
        m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try "Final Answer:" format
        m = re.search(r'Final\s+Answer:\s*(.+?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        # Fallback: take the last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()

    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth (with fuzzy matching)."""
        pred_clean = prediction.strip().lower().rstrip(".")
        gt_clean = ground_truth.strip().lower().rstrip(".")
        if pred_clean == gt_clean:
            return True

        # MCQ-style: match option letter (e.g. "A) ..." vs "A) ...")
        mcq_pred = re.match(r'^([a-d])\)', pred_clean)
        mcq_gt = re.match(r'^([a-d])\)', gt_clean)
        if mcq_pred and mcq_gt:
            return mcq_pred.group(1) == mcq_gt.group(1)

        # Yes/No polarity check: opposite polarity → not a match
        pred_yes = pred_clean.startswith("yes")
        pred_no = pred_clean.startswith("no")
        gt_yes = gt_clean.startswith("yes")
        gt_no = gt_clean.startswith("no")
        if (pred_yes and gt_no) or (pred_no and gt_yes):
            return False

        # Key content overlap: extract meaningful words and compare
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                     "in", "on", "at", "to", "of", "for", "with", "and", "or",
                     "no", "yes", "not", "left", "right", "this", "that", "it"}
        def _keywords(text: str) -> set:
            words = re.findall(r'[a-z]+', text)
            return {w for w in words if len(w) > 2 and w not in stopwords}

        pred_words = _keywords(pred_clean)
        gt_words = _keywords(gt_clean)
        if not pred_words or not gt_words:
            return False

        # Jaccard similarity on key content words
        overlap = pred_words & gt_words
        jaccard = len(overlap) / len(pred_words | gt_words)
        return jaccard >= 0.5
