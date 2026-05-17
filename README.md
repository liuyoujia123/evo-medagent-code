# Evo-MedAgent

**Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve**

[![arXiv](https://img.shields.io/badge/arXiv-2604.14475-b31b1b.svg)](https://arxiv.org/abs/2604.14475)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An independent implementation of the Evo-MedAgent architecture proposed in Shen et al. (2026). Evo-MedAgent introduces a **self-evolving memory framework** for medical LLM agents — instead of solving each case from scratch, the agent accumulates experience across cases, much like a radiologist improving with every patient.

## Overview

Medical diagnosis is inherently iterative: clinicians learn from past cases, internalize diagnostic patterns, and calibrate their trust in different tools over time. Standard LLM agents treat every case independently. Evo-MedAgent changes this by maintaining three persistent memory stores:

| Store | Role | Real-world analogue |
|---|---|---|
| **Episodic Memory (E)** | Records of past cases and their outcomes | "I've seen this presentation before" |
| **Procedural Memory (S)** | Distilled diagnostic rules with priority levels | "Always rule out tension pneumothorax first" |
| **Tool-Governance Memory (G)** | Per-tool reliability tracking | "This segmenter is unreliable for small nodules" |

All three evolve continuously at test time — no fine-tuning, no RLHF, no parameter updates.

## Architecture

```
For each incoming case:
  1. Query memory → retrieve relevant episodes, rules, and tool guidance
  2. Assemble a context prefix that guides the agent's reasoning
  3. Agent diagnoses → produces answer and tool-interaction trace
  4. Receive ground-truth feedback
  5. Reflect (optionally with multimodal VLM) → extract lessons
  6. Update all three memory stores
  7. Next case benefits from accumulated experience
```

## Quick Start

### Installation

```bash
git clone https://github.com/liuyoujia123/evo-medagent-code.git
cd Evo-MedAgent
pip install -r requirements.txt
```

### API Keys

Set the following environment variables:

```bash
export DEEPSEEK_API_KEY="sk-your-key"     # LLM for text reasoning
export DASHSCOPE_API_KEY="sk-your-key"    # VLM for multimodal reflection
```

Or edit `config.yaml` directly (see `config.yaml.example` for a template). The `api_key_env` field accepts either an environment variable name or a literal key.

### Demo (no API key needed)

```bash
python demo.py
```

Runs the full pipeline with mock LLM responses across 25 cases, demonstrating memory evolution and cumulative accuracy improvement.

### Benchmark Evaluation

```bash
# Tool-free mode (VLM-only reasoning with memory)
python main.py --cases 25 --tool-free

# Tool-enabled mode (VLM orchestrates CXR analysis tools)
python main.py --cases 25

# Order-sensitivity analysis (multiple permutations)
python main.py --cases 25 --permutations 3

# Filter by diagnostic categories
python main.py --cases 10 --categories diagnosis detection

# Resume from a saved checkpoint
python main.py --resume ./checkpoints --cases 25
```

### Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Dataset

This implementation uses **ChestAgentBench** (Wang et al.), a benchmark of 2,500 CXR MCQs across 675 cases spanning 7 diagnostic categories: detection, classification, localization, diagnosis, reasoning, segmentation, and report generation.

- `chestagentbench/metadata.jsonl` — Raw benchmark metadata (included)
- `data/real_cases.json` — Preprocessed cases in Evo-MedAgent format (included)
- CXR images — Download separately from [wanglab/chestagentbench](https://huggingface.co/datasets/wanglab/chestagentbench) on HuggingFace

To regenerate `real_cases.json` from the raw metadata after downloading images:

```bash
python data/chestagentbench_loader.py --data-dir ./chestagentbench --output ./data/real_cases.json
```

## Project Structure

```
Evo-MedAgent/
├── main.py                     # Entry point: CLI + benchmark runner
├── demo.py                     # Self-contained demo (mock LLM, no API key)
├── config.yaml                 # Runtime configuration
├── config.yaml.example         # Configuration template
├── requirements.txt
│
├── src/
│   ├── agent.py                # Core reasoning loop with memory augmentation
│   ├── memory/
│   │   ├── episodic.py         # Episodic memory (case records + embeddings)
│   │   ├── procedural.py       # Procedural memory (diagnostic rules + utility)
│   │   ├── governance.py       # Tool-governance memory (per-tool reliability)
│   │   └── evolving.py         # Unified M_i = (E_i, S_i, G_i) coordinator
│   ├── tools/
│   │   ├── base.py             # Tool interface and registry
│   │   └── simulated.py        # Simulated CXR tools (classifier, segmenter, VQA)
│   ├── reflection/
│   │   └── reflector.py        # Post-case multimodal reflection (VLM-based)
│   ├── storage/
│   │   ├── qdrant_store.py     # Qdrant vector DB connector (optional)
│   │   └── neo4j_store.py      # Neo4j graph DB connector (optional)
│   ├── retrieval/              # Unified semantic retrieval over memory stores
│   └── utils/
│       ├── llm_client.py       # Unified LLM/VLM client (DeepSeek, OpenAI, Gemini, Ollama)
│       └── embedding.py        # Sentence-transformer embeddings
│
├── data/
│   ├── benchmark.py            # Benchmark loader and sample case generator
│   ├── chestagentbench_loader.py  # ChestAgentBench JSONL parser
│   └── real_cases.json         # Preprocessed benchmark cases
│
├── chestagentbench/
│   ├── metadata.jsonl          # Raw benchmark metadata (2,500 MCQs)
│   └── README.md               # Upstream dataset documentation
│
└── tests/
    └── test_memory.py          # Unit tests for memory modules
```

## Configuration

Key settings in `config.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `llm` | `provider` | `deepseek` | LLM backend for text reasoning |
| `llm` | `model` | `deepseek-v4-pro` | Model name |
| `llm.vlm` | `provider` | `openai` | VLM backend for image reflection |
| `llm.vlm` | `model` | `qwen-vl-max` | Vision model (via DashScope) |
| `episodic` | `max_episodes` | `200` | Max stored case records |
| `episodic` | `retrieval_k` | `3` | Top-K similar episodes retrieved |
| `procedural` | `max_rules` | `50` | Max diagnostic rules |
| `procedural` | `retrieval_k` | `5` | Top-K relevant rules retrieved |
| `governance` | `trusted_threshold` | `0.70` | Helpful rate to mark a tool TRUSTED |
| `reflection` | `human_in_the_loop` | `true` | Require manual rule approval |

For persistent storage, enable Qdrant (vector search) and/or Neo4j (graph) under `database`.

## Design Principles

- **Training-free** — No gradient updates. Memory evolves purely through retrieval, reflection, and structured updates at inference time.
- **Plug-and-play** — Works on top of any frozen VLM. Add the memory module to your existing medical agent.
- **Human-in-the-loop** — New procedural rules are held in `pending_review/` until manually approved (`python main.py --approve <rule_id>`), preventing hallucinated rules from contaminating the knowledge base.
- **Complementary memories** — Episodic memory excels at recurring patterns; procedural memory generalizes to novel combinations. Together they outperform either alone.

## Extending

### Adding a new tool

```python
from src.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_tool", description="Custom CXR analysis tool")

    def run(self, image_path: str, **kwargs) -> ToolResult:
        # Your logic here
        return ToolResult(self.name, success=True, output="Finding: ...")

toolbox.register(MyTool())
memory.governance.register_tools(["my_tool"])
```

### Adding custom cases

```json
[
  {
    "question": "Is there a pneumothorax in this CXR?",
    "ground_truth": "Yes, left-sided",
    "category": "detection",
    "case_descriptor": "CXR detection: pneumothorax, unilateral left",
    "image_paths": ["/path/to/cxr.png"]
  }
]
```

Then run: `python main.py --case-file my_cases.json --cases 50`

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{shen2026evomedagent,
  title   = {Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That
             Remember, Reflect, and Improve},
  author  = {Shen, Weixiang and Jian, Bailiang and Li, Jun and Liu, Che
             and Moll, Johannes and Hu, Xiaobin and Rueckert, Daniel
             and Li, Hongwei Bran and Pan, Jiazhen},
  journal = {arXiv preprint arXiv:2604.14475},
  year    = {2026}
}
```

Also cite the ChestAgentBench dataset if you use the benchmark:

```bibtex
@misc{wang2025chestagentbench,
  title   = {ChestAgentBench: A Benchmark for Chest X-ray Agent},
  author  = {Wang, ...},
  year    = {2025}
}
```

## License

This implementation is released under the [MIT License](LICENSE). The original paper is under CC BY-NC-SA 4.0.
