# Evo-MedAgent

**Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve**

An independent implementation of the Evo-MedAgent architecture from Shen et al. (arXiv:2604.14475, 2026).

Evo-MedAgent introduces a **self-evolving memory framework** for medical LLM agents. Instead of solving each case from scratch, the agent accumulates experience across cases — like a radiologist improving with every patient they read.

## Key Idea

| Without Memory | With Evo-MedAgent |
|---|---|
| Each case = fresh start | Learns from prior cases |
| Same mistake repeated 50x | Reflects → corrects → avoids repetition |
| Static tool trust | Tracks which tools actually help |
| No procedural learning | Distills diagnostic heuristics over time |

## Three Memory Stores

### 1. Episodic Memory (E)
Stores compressed records of prior diagnostic episodes: case descriptors, tool-interaction traces, predicted vs. ground-truth answers, retrospective summaries, and actionable guidelines for future cases.

### 2. Procedural Memory (S)
Distilled diagnostic rules with priority tags:
- `ρ = 0`: CRITICAL (life-threatening misses)
- `ρ = 1`: IMPORTANT (best practices)
- `ρ = 2`: GUIDANCE (helpful reminders)

Rules evolve via **utility-driven selection** — proven rules persist, harmful ones are pruned.

### 3. Tool-Governance Memory (G)
Tracks per-tool reliability:
- TRUSTED: helpful rate > 0.70, zero harm
- CAUTION: insufficient data or mixed results
- AVOID: effective-bad rate exceeds threshold

## Workflow

```
For each case:
  1. Query memory → retrieve relevant episodes, rules, tool guidance
  2. Assemble context prefix → guide agent reasoning
  3. Agent diagnoses → produces answer + tool trace
  4. Receive ground-truth feedback
  5. Reflect → summarize lesson, extract new rules
  6. Update all three memory stores
  7. Next case benefits from accumulated experience
```

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/Evo-MedAgent.git
cd Evo-MedAgent
pip install -r requirements.txt
```

### Set API Keys

```bash
# Required: LLM backend for text reasoning
export DEEPSEEK_API_KEY="sk-your-deepseek-key"

# Required: VLM for multimodal reflection
export DASHSCOPE_API_KEY="sk-your-dashscope-key"
```

Alternatively, edit `config.yaml` (see `config.yaml.example` for a template). The `api_key_env` field accepts either an environment variable name or a literal key string.

### Quick Demo (no API key needed)

```bash
python demo.py
```
Runs the full pipeline with simulated responses, showing memory evolution across 25 cases.

### Run on Sample Benchmark

```bash
# Tool-free mode (VLM-only reasoning with memory)
python main.py --cases 25 --tool-free

# Tool-enabled mode (VLM orchestrates CXR tools)
python main.py --cases 25

# Multi-permutation evaluation (order sensitivity)
python main.py --cases 25 --permutations 3

# With specific categories only
python main.py --cases 10 --categories diagnosis detection

# Resume from checkpoint
python main.py --resume ./checkpoints --cases 25
```

### Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
Evo-MedAgent/
├── main.py              # Entry point: CLI + benchmark runner
├── demo.py              # Self-contained demo (no API key needed)
├── config.yaml          # Configuration (LLM, memory, benchmark)
├── config.yaml.example  # Configuration template for reference
├── requirements.txt
├── src/
│   ├── agent.py         # Core EvoMedAgent: reasoning loop
│   ├── memory/
│   │   ├── episodic.py  # Episodic memory store
│   │   ├── procedural.py# Procedural memory (rules + utility)
│   │   ├── governance.py# Tool-governance memory
│   │   └── evolving.py  # Unified M_i = (E_i, S_i, G_i)
│   ├── tools/
│   │   ├── base.py      # Tool interface + registry
│   │   └── simulated.py # CXR tools (classifier, segmenter, VQA, etc.)
│   ├── reflection/
│   │   └── reflector.py # Post-case multimodal reflection
│   ├── storage/
│   │   ├── qdrant_store.py  # Qdrant vector DB connector
│   │   └── neo4j_store.py   # Neo4j graph DB connector
│   ├── retrieval/        # Unified retrieval over three stores
│   └── utils/
│       ├── llm_client.py # Unified LLM client (DeepSeek, OpenAI, Gemini, Ollama)
│       └── embedding.py  # Sentence-transformers embeddings
├── data/
│   ├── benchmark.py      # ChestAgentBench loader + sample cases
│   └── chestagentbench_loader.py
├── tests/
│   └── test_memory.py    # Unit tests for all memory modules
└── chestagentbench/      # Benchmark metadata
```

## Configuration

Copy `config.yaml.example` to `config.yaml` and customize:

```yaml
llm:
  provider: "deepseek"
  api_key_env: "DEEPSEEK_API_KEY"    # env var or literal key
  model: "deepseek-v4-pro"

# Optional: persistent databases
database:
  qdrant:
    enabled: false    # set true to enable Qdrant vector search
  neo4j:
    enabled: false    # set true to enable Neo4j graph storage
```

## Design Highlights

- **Training-Free Test-Time Learning** — No fine-tuning, no RLHF, no parameter updates. Memory evolves purely through retrieval + reflection at test time.
- **Plug-and-Play** — Works on top of any frozen VLM. Just add this memory module to an existing medical agent.
- **Human-in-the-Loop** — New procedural rules require manual approval before entering the procedural memory, preventing hallucinated rules from contaminating the knowledge base.
- **Low Overhead** — Per case: one additional embedding retrieval pass + one reflection LLM call.

## Extending

### Adding a New Tool

```python
from src.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(name="my_tool", description="Custom CXR tool")

    def run(self, image_path: str, **kwargs) -> ToolResult:
        return ToolResult(self.name, success=True, output="Result")

toolbox.register(MyTool())
memory.governance.register_tools(["my_tool"])
```

### Adding Custom Cases

```json
[
  {
    "question": "Is there a pneumothorax?",
    "ground_truth": "Yes, right-sided",
    "category": "detection",
    "case_descriptor": "CXR detection: pneumothorax right",
    "image_paths": ["/path/to/cxr.png"]
  }
]
```

Then run: `python main.py --case-file my_cases.json --cases 50`

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{shen2026evomedagent,
  title={Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve},
  author={Shen, Weixiang and Jian, Bailiang and Li, Jun and Liu, Che and Moll, Johannes
          and Hu, Xiaobin and Rueckert, Daniel and Li, Hongwei Bran and Pan, Jiazhen},
  journal={arXiv preprint arXiv:2604.14475},
  year={2026}
}
```

## License

This implementation is released under the MIT License. See [LICENSE](LICENSE) for details.

The original paper is under CC BY-NC-SA 4.0.
