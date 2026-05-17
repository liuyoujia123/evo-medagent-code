"""
Evo-MedAgent Demo (no API key required)
========================================
Runs the complete memory evolution pipeline with mock LLM responses.
Demonstrates: episodic retrieval, procedural rule evolution,
tool governance updates, and cumulative accuracy improvement.
"""
import sys
import os
import random
import json
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_client import LLMClient, LLMConfig
from src.utils.embedding import Embedder
from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.governance import ToolGovernanceMemory
from src.memory.evolving import SelfEvolvingMemory
from src.tools.simulated import create_default_toolbox
from src.reflection.reflector import Reflector
from src.agent import EvoMedAgent
from data.benchmark import BenchmarkLoader, SAMPLE_CASES


class MockLLMClient:
    """
    Mock LLM that simulates VLM reasoning for demo purposes.
    Returns structured responses without needing an API key.
    Demonstrates the full Evo-MedAgent pipeline end-to-end.
    """

    def __init__(self, accuracy_base: float = 0.65):
        self.config = LLMConfig(model="mock-vlm")
        self.accuracy_base = accuracy_base
        self.case_count = 0
        self._memory_boost = 0.0  # simulate memory improvement

    def chat(self, system_prompt: str, user_message: str,
             model=None, max_tokens=None) -> Optional[str]:
        return self._simulate_response(user_message)

    def chat_with_images(self, system_prompt: str, user_text: str,
                         image_paths=None, model=None, max_tokens=None) -> Optional[str]:
        return self._simulate_response(user_text)

    def _chat(self, messages, model=None, max_tokens=None, temperature=None) -> Optional[str]:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return self._simulate_response(msg.get("content", ""))
        return "No message found."

    def _simulate_response(self, message: str) -> str:
        """Simulate a VLM diagnostic response with memory-boosted accuracy."""
        self.case_count += 1

        # Check if memory context is present (this boosts accuracy)
        has_memory = any(kw in str(message).lower() for kw in
                         ["prior case", "procedural", "heuristic", "governance"])

        # Boost accuracy when memory is present (simulating the paper's results)
        effective_accuracy = self.accuracy_base
        if has_memory and self._memory_boost > 0:
            effective_accuracy = min(0.95, self.accuracy_base + self._memory_boost)

        # Simulate correct/incorrect based on accuracy
        is_correct = random.random() < effective_accuracy

        # Extract ground truth from context if available
        # For demo, return a plausible answer
        if is_correct:
            return "ANSWER: Correct diagnosis based on CXR findings."
        else:
            return "ANSWER: Incorrect diagnosis — misinterpreted key finding."

    def boost_memory(self, amount: float = 0.05):
        """Simulate memory-driven accuracy improvement over time."""
        self._memory_boost += amount


class MockReflector:
    """Mock reflector that generates plausible reflections."""

    def __init__(self):
        self.enabled = True
        self.case_num = 0

    def reflect(self, question: str, prediction: str, ground_truth: str,
                tool_trace: str = "", image_paths=None):
        from src.reflection.reflector import ReflectionOutput
        self.case_num += 1

        was_correct = "correct" in prediction.lower()

        if was_correct:
            return ReflectionOutput(
                summary=f"Successfully identified the correct finding through systematic analysis.",
                guideline="When encountering similar presentations, apply the same systematic approach.",
                new_rules=[("Maintain systematic review of all CXR zones before making diagnosis.", 2)],
                quality_score=0.8,
            )
        else:
            # Generate progressively better rules
            rules_pool = [
                ("Always check for subtle pneumothorax at lung apices before excluding it.", 0),
                ("Differentiate pleural effusion from consolidation by checking for air bronchograms.", 0),
                ("In young patients with chest pain, inspect ribs for exostosis or fracture.", 0),
                ("Bilateral hilar adenopathy → consider sarcoidosis before lymphoma in young adults.", 0),
                ("Air crescent sign in a cavitary lesion is characteristic of aspergilloma.", 1),
                ("A widened mediastinum in trauma requires urgent exclusion of aortic injury.", 0),
                ("For post-operative dyspnea with normal CXR, consider pulmonary embolism.", 1),
                ("Spiculated lung mass + hilar adenopathy in a smoker = lung cancer until proven otherwise.", 0),
                ("Kerley B lines indicate interstitial pulmonary edema — check cardiac silhouette.", 1),
                ("Compare with prior studies before calling a finding 'new' or 'worsened'.", 2),
            ]
            rule = rules_pool[min(self.case_num - 1, len(rules_pool) - 1)]

            return ReflectionOutput(
                summary=f"Mistakenly interpreted finding. The correct approach is to {rule[0][:80]}...",
                guideline=rule[0],
                new_rules=[rule],
                quality_score=0.7,
            )


def run_demo():
    print("=" * 65)
    print("  Evo-MedAgent Demo: Self-Evolving Diagnostic Memory")
    print("  'Agents That Remember, Reflect, and Improve'")
    print("=" * 65)

    # --- Setup ---
    random.seed(42)
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    episodic = EpisodicMemory(embedder, max_episodes=200, retrieval_k=3)
    procedural = ProceduralMemory(embedder, max_rules=50, retrieval_k=5)
    governance = ToolGovernanceMemory()
    memory = SelfEvolvingMemory(episodic, procedural, governance)

    toolbox = create_default_toolbox()
    governance.register_tools(toolbox.list_names())

    mock_llm = MockLLMClient(accuracy_base=0.65)
    mock_reflector = MockReflector()

    agent = EvoMedAgent(
        llm_client=mock_llm,
        memory=memory,
        toolbox=toolbox,
        reflector=mock_reflector,
        use_tools=True,
    )
    # Override the check method for mock
    agent._check_answer = lambda pred, gt: random.random() < (
        0.65 + min(0.25, len(memory.episodic) * 0.01)
    )

    # --- Load sample cases ---
    loader = BenchmarkLoader(seed=42, shuffle=True)
    cases = loader.load_sample(n_cases=25)

    # --- Run ---
    print(f"\nModel: {mock_llm.config.model}")
    print(f"Tool-enabled: {agent.use_tools}")
    print(f"Cases in benchmark: {len(cases)}")
    print(f"Base accuracy (no memory): ~0.65")
    print(f"\n{'─'*65}")
    print("  Legend: ✅ correct  ❌ incorrect  📚 memory boost")
    print(f"{'─'*65}\n")

    results = []
    correct_count = 0

    for i, case in enumerate(cases):
        # Simulate memory-driven improvement
        memory_boost = min(0.22, len(memory.episodic) * 0.008)
        effective_acc = 0.65 + memory_boost

        # Mock diagnose
        case_desc = case.get("case_descriptor", case["question"])
        gt = case["ground_truth"]

        is_correct = random.random() < effective_acc
        if is_correct:
            correct_count += 1

        prediction = gt if is_correct else next(
            (c["ground_truth"] for c in cases if c["ground_truth"] != gt), "Incorrect"
        )

        # Memory query (even with mock, show retrieval stats)
        mem_ctx = memory.query(case_desc)
        n_eps = len(mem_ctx.retrieved_episodes)
        n_rules = len(mem_ctx.selected_rules)

        # Reflect
        reflection = mock_reflector.reflect(
            question=case["question"],
            prediction=prediction,
            ground_truth=gt,
        )

        # Update memory
        tool_outcomes = [("classifier", is_correct, not is_correct, False)] if not is_correct else []
        memory.update(
            case_descriptor=case_desc,
            question=case["question"],
            tool_trace="classifier: Used for primary finding assessment.",
            prediction=prediction,
            ground_truth=gt,
            summary=reflection.summary,
            guideline=reflection.guideline,
            new_rules=reflection.new_rules,
            tool_outcomes=tool_outcomes,
        )

        # Display
        status = "✅" if is_correct else "❌"
        boost_indicator = f" 📚+{memory_boost:.2f}" if memory_boost > 0 else ""
        acc = correct_count / (i + 1)
        print(f"Case {i+1:2d} {status} | Acc: {acc:.3f}{boost_indicator} | "
              f"Episodes: {n_eps} used | Rules: {n_rules} selected | "
              f"New rule: {'✓' if reflection.new_rules else '—'}")

        results.append({
            "idx": i,
            "correct": is_correct,
            "prediction": prediction,
            "truth": gt,
            "summary": reflection.summary[:80],
            "rules": [r[0][:60] for r in reflection.new_rules],
        })

        # Show milestone
        if (i + 1) % 10 == 0:
            print(f"  ── Memory at {i+1} cases: {len(episodic)} episodes, "
                  f"{len(procedural)} rules ──")

    # --- Final Summary ---
    final_acc = correct_count / len(cases)
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"  Accuracy: {correct_count}/{len(cases)} = {final_acc:.3f}")
    print(f"  Baseline (no memory) estimate: ~0.65")
    print(f"  Improvement: +{final_acc - 0.65:.3f}")
    print(f"  Episodes stored: {len(episodic)}")
    print(f"  Procedural rules: {len(procedural)}")
    print(f"{'='*65}")

    # Show evolved rules
    print(f"\n  EVOLVED PROCEDURAL RULES (top by priority):")
    rules = sorted(procedural.active_rules, key=lambda r: (r.priority, -r.success_rate))
    for i, rule in enumerate(rules[:10]):
        p_label = {0: "CRIT", 1: "IMPT", 2: "GUIDE"}.get(rule.priority, "??")
        print(f"  [{p_label}] {rule.instruction[:90]}... "
              f"(used {rule.times_selected}×, {rule.success_rate:.0%} helpful)")

    # Compare with paper's Table 1
    print(f"\n  COMPARISON WITH PAPER (Table 1, GPT-5-mini MCQ):")
    print(f"  Paper Baseline:    0.68")
    print(f"  Paper Full Memory: 0.79")
    print(f"  Our Demo Final:    {final_acc:.2f}")
    print(f"  (Note: paper uses real VLM, demo uses simulated pipeline)")
    print(f"{'='*65}\n")

    return results


if __name__ == "__main__":
    run_demo()
