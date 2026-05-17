"""
Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve

Reference: Shen et al., arXiv:2604.14475 (2026)

v2 improvements:
  - DeepSeek-V4-Pro for text reasoning
  - VLM (GPT-4o / Gemini / Ollama) for true multimodal reflection
  - Human-in-the-Loop for rule quality assurance
  - Qdrant + Neo4j database connectors

Usage:
    python main.py                          # Run benchmark with sample cases
    python main.py --cases 50               # Run with 50 sample cases
    python main.py --config config.yaml     # Use custom config
    python main.py --tool-free              # Tool-free mode (as in Table 1)
    python main.py --permutations 3         # Run multiple order permutations
    python main.py --demo                   # Interactive demo mode
    python main.py --approve-all            # Bulk-approve all pending rules
    python main.py --approve <rule_id>      # Approve a specific pending rule
    python main.py --list-pending           # List rules awaiting review
    python main.py --save-dir ./results     # Save results and checkpoints
"""
import argparse
import logging
import os
import sys
import json
import yaml
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_client import LLMClient, LLMConfig, create_vlm_client
from src.utils.embedding import Embedder
from src.memory.episodic import EpisodicMemory
from src.memory.procedural import ProceduralMemory
from src.memory.governance import ToolGovernanceMemory
from src.memory.evolving import SelfEvolvingMemory
from src.tools.base import ToolRegistry
from src.tools.simulated import create_default_toolbox
from src.reflection.reflector import Reflector
from src.agent import EvoMedAgent
from data.benchmark import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evo-medagent")


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML file, with fallback defaults (v2)."""
    defaults = {
        "llm": {
            "provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro",
            "max_tokens": 2048, "temperature": 0.0,
            "vlm": {
                "provider": "openai", "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1", "model": "gpt-4o",
                "max_tokens": 600, "temperature": 0.0,
            },
        },
        "embedding": {"model": "all-MiniLM-L6-v2", "device": "cpu", "dimension": 384},
        "database": {
            "qdrant": {"enabled": False, "url": "http://localhost:6333",
                       "api_key_env": "QDRANT_API_KEY",
                       "collection_name": "evo_medagent_episodic", "vector_dim": 384},
            "neo4j": {"enabled": False, "uri": "bolt://localhost:7687",
                      "username": "neo4j", "password_env": "NEO4J_PASSWORD",
                      "database": "neo4j"},
        },
        "episodic": {"max_episodes": 200, "retrieval_k": 3, "relevance_threshold": 0.3},
        "procedural": {"max_rules": 50, "retrieval_k": 5, "exploration_weight": 0.15,
                       "prune_threshold": 0.0, "priority_default": 1},
        "governance": {"trusted_threshold": 0.70, "trusted_min_interactions": 6,
                       "avoid_threshold": 0.60, "avoid_min_interactions": 10},
        "reflection": {"enabled": True, "summarize_episode": True, "extract_rules": True,
                       "human_in_the_loop": True, "require_manual_approval": True,
                       "pending_dir": "./pending_review"},
        "benchmark": {"name": "ChestAgentBench", "random_seed": 42, "evaluate_every": 10},
        "persistence": {"save_dir": "./checkpoints", "save_every": 50, "resume": True},
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        for section in user_config:
            if section in defaults and isinstance(user_config[section], dict):
                defaults[section].update(user_config[section])
            else:
                defaults[section] = user_config[section]

    return defaults


# =============================================================================
# Agent construction
# =============================================================================

def build_agent(config: Dict[str, Any], use_tools: bool = True) -> EvoMedAgent:
    """Construct the Evo-MedAgent from config (v2)."""
    # ---- LLM client (text reasoning, DeepSeek-V4-Pro) ----
    llm_cfg_data = config["llm"]
    llm_cfg = LLMConfig(
        provider=llm_cfg_data.get("provider", "deepseek"),
        api_key_env=llm_cfg_data.get("api_key_env", "DEEPSEEK_API_KEY"),
        base_url=llm_cfg_data.get("base_url", "https://api.deepseek.com"),
        model=llm_cfg_data.get("model", "deepseek-v4-pro"),
        max_tokens=llm_cfg_data.get("max_tokens", 2048),
        temperature=llm_cfg_data.get("temperature", 0.0),
    )
    llm_client = LLMClient(llm_cfg)

    # ---- VLM client (multimodal reflection, GPT-4o etc.) ----
    vlm_config = llm_cfg_data.get("vlm", {})
    vlm_client = create_vlm_client(vlm_config)

    # ---- Embedder ----
    embedder = Embedder(
        model_name=config["embedding"]["model"],
        device=config["embedding"]["device"],
    )

    # ---- Memory stores ----
    episodic = EpisodicMemory(
        embedder=embedder,
        max_episodes=config["episodic"]["max_episodes"],
        retrieval_k=config["episodic"]["retrieval_k"],
        relevance_threshold=config["episodic"]["relevance_threshold"],
    )
    procedural = ProceduralMemory(
        embedder=embedder,
        max_rules=config["procedural"]["max_rules"],
        retrieval_k=config["procedural"]["retrieval_k"],
        exploration_weight=config["procedural"]["exploration_weight"],
        prune_threshold=config["procedural"]["prune_threshold"],
    )
    governance = ToolGovernanceMemory(
        trusted_threshold=config["governance"]["trusted_threshold"],
        trusted_min_interactions=config["governance"]["trusted_min_interactions"],
        avoid_threshold=config["governance"]["avoid_threshold"],
        avoid_min_interactions=config["governance"]["avoid_min_interactions"],
    )
    memory = SelfEvolvingMemory(episodic, procedural, governance)

    # ---- Toolbox ----
    toolbox = None
    if use_tools:
        toolbox = create_default_toolbox(llm_client)
        governance.register_tools(toolbox.list_names())

    # ---- Reflector (with VLM + HITL) ----
    reflection_cfg = config["reflection"]
    reflector = Reflector(
        text_llm=llm_client if reflection_cfg.get("enabled", True) else None,
        vlm_client=vlm_client,                                       # ★ KEY: VLM for real image analysis
        enabled=reflection_cfg.get("enabled", True),
        human_in_the_loop=reflection_cfg.get("human_in_the_loop", True),
        require_manual_approval=reflection_cfg.get("require_manual_approval", True),
        pending_dir=reflection_cfg.get("pending_dir", "./pending_review"),
    )

    if vlm_client:
        logger.info(f"[OK] Multimodal reflection enabled: {vlm_client.config.provider} @ {vlm_client.config.model}")
    else:
        logger.warning("[!] No VLM configured -- reflection will be text-only (no image analysis)")

    if reflector.human_in_the_loop:
        logger.info(f"[OK] Human-in-the-Loop enabled -- rules staged for review in {reflector.pending_dir}")

    # ---- Agent ----
    agent = EvoMedAgent(
        llm_client=llm_client,
        memory=memory,
        toolbox=toolbox,
        reflector=reflector,
        max_tool_calls=5,
        use_tools=use_tools,
    )

    return agent


# =============================================================================
# Benchmark runner
# =============================================================================

def run_benchmark(agent: EvoMedAgent, config: Dict[str, Any],
                  cases: list, save_dir: str) -> Dict[str, Any]:
    """Run benchmark evaluation."""
    save_every = config["persistence"]["save_every"]
    results = agent.run_benchmark(
        cases=cases,
        verbose=True,
        save_every=save_every,
        save_dir=save_dir,
    )

    final_acc = agent.stats["correct"] / max(1, agent.stats["total_cases"])
    final_rules = len(agent.memory.procedural)
    final_episodes = len(agent.memory.episodic)

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"  Total cases:      {agent.stats['total_cases']}")
    logger.info(f"  Correct:          {agent.stats['correct']}")
    logger.info(f"  Final accuracy:   {final_acc:.4f}")
    logger.info(f"  Episodes stored:  {final_episodes}")
    logger.info(f"  Rules created:    {final_rules}")

    # HITL summary
    if agent.reflector and agent.reflector.human_in_the_loop:
        pending = agent.reflector.pending_count()
        approved = len(agent.reflector.get_approved_rules())
        logger.info(f"  HITL pending:     {pending} rules awaiting review")
        logger.info(f"  HITL approved:    {approved} rules")
        if pending > 0:
            logger.info(f"  Review directory: {agent.reflector.pending_dir}")
            logger.info(f"  Run 'python main.py --list-pending' to see rules")
            logger.info(f"  Run 'python main.py --approve-all' to approve all")
    logger.info("=" * 60)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    output = {
        "config_summary": {
            "model": config["llm"]["model"],
            "vlm_model": config["llm"].get("vlm", {}).get("model", "none"),
            "use_tools": agent.use_tools,
            "hitl_enabled": config["reflection"].get("human_in_the_loop", False),
            "n_cases": agent.stats["total_cases"],
        },
        "final_accuracy": final_acc,
        "cumulative_accuracies": agent.stats["cumulative_accuracy"],
        "total_correct": agent.stats["correct"],
        "total_cases": agent.stats["total_cases"],
        "episodes_stored": final_episodes,
        "rules_created": final_rules,
        "hitl_pending": agent.reflector.pending_count() if agent.reflector else 0,
        "per_case": [
            {
                "idx": r["case_index"],
                "correct": r["is_correct"],
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
                "summary": r.get("reflection", {}).get("summary", ""),
            }
            for r in results
        ],
    }
    result_path = os.path.join(save_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {result_path}")

    agent.memory.save(save_dir, "final")
    return output


# =============================================================================
# Demo runner
# =============================================================================

def run_demo(agent: EvoMedAgent, cases: list):
    """Interactive demo mode."""
    print("\n" + "=" * 60)
    print("Evo-MedAgent Demo: Test-Time Learning in Action")
    print("=" * 60)
    print(f"Model: {agent.llm.config.model}")
    print(f"VLM:   {agent.reflector._vlm.config.model if agent.reflector._vlm else 'none (text-only reflection)'}")
    print(f"Tool-enabled: {agent.use_tools}")
    print(f"HITL: {agent.reflector.human_in_the_loop}")
    print(f"Cases: {len(cases)}")
    print("Each case: Read memory → Reason → Receive feedback → Reflect → Update memory")
    print("-" * 60)

    for i, case in enumerate(cases):
        print(f"\n{'─'*40}\nCase {i+1}/{len(cases)}: {case['category'].upper()}")
        print(f"Q: {case['question'][:100]}...")

        result = agent.diagnose(
            question=case["question"],
            image_paths=case.get("image_paths", []),
            ground_truth=case["ground_truth"],
            case_descriptor=case.get("case_descriptor", case["question"]),
        )

        status = "[OK] CORRECT" if result["is_correct"] else "[XX] INCORRECT"
        print(f"  Prediction: {result['prediction'][:80]}...")
        print(f"  Truth:      {result['ground_truth']}")
        print(f"  Result:     {status}")
        print(f"  Episodes used: {result['memory_episodes_used']} | Rules used: {result['memory_rules_used']}")

        reflection = result.get("reflection", {})
        if reflection.get("summary"):
            print(f"  Lesson: {reflection['summary'][:120]}...")
        if reflection.get("new_rules"):
            for rule_text, prio in reflection["new_rules"][:2]:
                print(f"  New rule [p={prio}]: {rule_text[:100]}...")

        acc = agent.stats["correct"] / agent.stats["total_cases"]
        print(f"  Cumulative accuracy: {acc:.3f} ({agent.stats['correct']}/{agent.stats['total_cases']})")

        if agent.reflector and agent.reflector.human_in_the_loop:
            pending = agent.reflector.pending_count()
            if pending > 0:
                print(f"  [!] {pending} rules awaiting manual approval in {agent.reflector.pending_dir}")

        if (i + 1) % 10 == 0:
            print(f"\n--- Memory state at case {i+1} ---")
            print(f"  Episodes: {len(agent.memory.episodic)} | Rules: {len(agent.memory.procedural)}")
            rules = agent.memory.procedural.active_rules
            if rules:
                for r in sorted(rules, key=lambda x: x.priority)[:3]:
                    print(f"  [p={r.priority}] {r.instruction[:80]}...")

    print(f"\n{'='*60}")
    print(f"FINAL: {agent.stats['correct']}/{agent.stats['total_cases']} "
          f"({agent.stats['correct']/agent.stats['total_cases']:.3f})")
    print(f"Memory: {len(agent.memory.episodic)} episodes, {len(agent.memory.procedural)} rules")

    if agent.reflector:
        pending = agent.reflector.pending_count()
        if pending > 0:
            print(f"[!] {pending} rules staged for review. Run --approve-all to commit them.")
    print(f"{'='*60}")


# =============================================================================
# HITL management commands
# =============================================================================

def handle_hitl_commands(args, config: Dict[str, Any]):
    """Handle human-in-the-loop management commands."""
    reflection_cfg = config["reflection"]
    pending_dir = reflection_cfg.get("pending_dir", "./pending_review")

    # Create a minimal reflector just for managing pending rules
    reflector = Reflector(
        text_llm=None,
        vlm_client=None,
        enabled=True,
        human_in_the_loop=True,
        require_manual_approval=True,
        pending_dir=pending_dir,
    )

    # Load existing pending files into memory
    if os.path.isdir(pending_dir):
        for fname in os.listdir(pending_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(pending_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "PENDING_REVIEW":
                    from src.reflection.reflector import PendingRule
                    rule = PendingRule(
                        rule_id=data["rule_id"],
                        instruction=data["instruction"],
                        priority=data["priority"],
                        source_case=data.get("source_case", -1),
                        summary=data.get("reflection_summary", ""),
                        guideline=data.get("reflection_guideline", ""),
                        quality_score=data.get("quality_score", 0.5),
                        created_at=data.get("created_at", ""),
                        approved=False,
                    )
                    reflector._pending_rules.append(rule)

    # ---- list-pending ----
    if args.list_pending:
        pending = reflector.get_pending_rules()
        if not pending:
            print("No pending rules awaiting review.")
        else:
            print(f"\n{'='*70}")
            print(f"  {len(pending)} RULE(S) AWAITING REVIEW")
            print(f"  Directory: {pending_dir}")
            print(f"{'='*70}")
            for rule in pending:
                p_label = {0: "CRITICAL", 1: "IMPORTANT", 2: "GUIDANCE"}.get(rule.priority, "?")
                print(f"\n  [{rule.rule_id}] {p_label}")
                print(f"  Instruction: {rule.instruction}")
                print(f"  From case #{rule.source_case}")
                print(f"  Summary: {rule.summary[:120]}...")
                print(f"  Quality: {rule.quality_score:.2f}")
                print(f"  Created: {rule.created_at}")
            print(f"\n  To approve all:  python main.py --approve-all")
            print(f"  To approve one:  python main.py --approve <rule_id>")
            print(f"  To reject one:   python main.py --reject <rule_id>")
            print(f"{'='*70}\n")
        return

    # ---- approve-all ----
    if args.approve_all:
        count = reflector.approve_all()
        print(f"[OK] {count} rule(s) approved. Ready to commit to procedural memory on next run.")

        # Also commit to a list for the build_agent to pick up
        approved_file = os.path.join(pending_dir, "_approved.json")
        approved_rules = reflector.get_approved_rules()
        with open(approved_file, "w", encoding="utf-8") as f:
            json.dump(
                [{"instruction": r[0], "priority": r[1]} for r in approved_rules],
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Approved rules saved to {approved_file}")
        return

    # ---- approve <id> ----
    if args.approve:
        rule = reflector.approve_rule(args.approve)
        if rule:
            print(f"[OK] Rule '{args.approve}' approved: {rule.instruction[:80]}...")
        else:
            print(f"[XX] Rule '{args.approve}' not found or already processed.")

        approved_file = os.path.join(pending_dir, "_approved.json")
        approved_rules = reflector.get_approved_rules()
        with open(approved_file, "w", encoding="utf-8") as f:
            json.dump(
                [{"instruction": r[0], "priority": r[1]} for r in approved_rules],
                f, ensure_ascii=False, indent=2,
            )
        return

    # ---- reject <id> ----
    if args.reject:
        ok = reflector.reject_rule(args.reject)
        if ok:
            print(f"[XX] Rule '{args.reject}' rejected.")
        else:
            print(f"[XX] Rule '{args.reject}' not found or already processed.")
        return


# =============================================================================
# Main entry
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evo-MedAgent v2: Self-Evolving Medical Diagnostic Agent "
                    "(VLM reflection + Human-in-the-Loop + Database connectors)"
    )
    # Standard args
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--cases", type=int, default=25, help="Number of sample cases")
    parser.add_argument("--categories", nargs="+", help="Filter by diagnostic categories")
    parser.add_argument("--case-file", help="Path to JSON case file")
    parser.add_argument("--tool-free", action="store_true",
                        help="Run in tool-free mode (VLM-only reasoning)")
    parser.add_argument("--permutations", type=int, default=1,
                        help="Number of order permutations")
    parser.add_argument("--save-dir", default="./results", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint directory")
    parser.add_argument("--demo", action="store_true", help="Interactive demo mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # HITL management args
    parser.add_argument("--list-pending", action="store_true",
                        help="List all rules awaiting manual approval")
    parser.add_argument("--approve-all", action="store_true",
                        help="Bulk-approve all pending rules")
    parser.add_argument("--approve", type=str, default=None,
                        help="Approve a specific pending rule by ID")
    parser.add_argument("--reject", type=str, default=None,
                        help="Reject a specific pending rule by ID")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # ---- HITL management mode ----
    if args.list_pending or args.approve_all or args.approve or args.reject:
        handle_hitl_commands(args, config)
        return

    # ---- Normal operation ----
    config["benchmark"]["random_seed"] = args.cases

    # Load cases
    loader = BenchmarkLoader(seed=config["benchmark"]["random_seed"])
    if args.case_file:
        cases = loader.load_json(args.case_file, n_cases=args.cases)
    else:
        cases = loader.load_sample(n_cases=args.cases, include_categories=args.categories)

    logger.info(f"Loaded {len(cases)} cases. Categories: {set(c['category'] for c in cases)}")

    # Build agent
    use_tools = not args.tool_free
    agent = build_agent(config, use_tools=use_tools)
    logger.info(f"Agent ready: tool-enabled={use_tools}, model={config['llm']['model']}")

    # Resume from checkpoint
    if args.resume:
        agent.memory.load(args.resume)
        logger.info(f"Memory resumed from {args.resume}")

    # Load previously approved HITL rules
    pending_dir = config["reflection"].get("pending_dir", "./pending_review")
    approved_file = os.path.join(pending_dir, "_approved.json")
    if os.path.exists(approved_file):
        with open(approved_file, "r", encoding="utf-8") as f:
            approved = json.load(f)
        for item in approved:
            agent.memory.procedural.update_or_add(
                instruction=item["instruction"],
                priority=item.get("priority", 1),
                source_case=-1,  # pre-approved
            )
        logger.info(f"Loaded {len(approved)} previously approved HITL rules")

    # ---- Demo mode ----
    if args.demo:
        run_demo(agent, cases)
        return

    # ---- Benchmark mode ----
    os.makedirs(args.save_dir, exist_ok=True)

    if args.permutations > 1:
        permutations = loader.generate_permutations(cases, args.permutations)
        all_results = []
        for p_idx, perm_cases in enumerate(permutations):
            logger.info(f"\n{'='*40}\nPermutation {p_idx+1}/{args.permutations}\n{'='*40}")
            perm_agent = build_agent(config, use_tools=use_tools)
            perm_save_dir = os.path.join(args.save_dir, f"perm_{p_idx+1}")
            result = run_benchmark(perm_agent, config, perm_cases, perm_save_dir)
            all_results.append(result)

        accs = [r["final_accuracy"] for r in all_results]
        print(f"\n{'='*60}")
        print(f"CROSS-PERMUTATION SUMMARY ({args.permutations} permutations)")
        print(f"  Mean accuracy: {sum(accs)/len(accs):.4f}")
        print(f"  Range: [{min(accs):.4f}, {max(accs):.4f}]")
        print(f"  Std: {(sum((a - sum(accs)/len(accs))**2 for a in accs)/len(accs))**0.5:.4f}")
        print(f"{'='*60}")
    else:
        run_benchmark(agent, config, cases, args.save_dir)


if __name__ == "__main__":
    main()
