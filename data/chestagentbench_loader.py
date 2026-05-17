"""
ChestAgentBench loader v2 — handles the actual dataset format
from wanglab/chestagentbench on HuggingFace.

Dataset structure:
  - metadata.jsonl  → 2500 MCQ questions across 675 cases
  - figures.zip      → CXR images (extracts to figures/figures/<case_id>/)
  
Each JSONL line:
  {
    "images": ["figures/11583/figure_1.jpg"],
    "question": "MCQ text with A)... B)... options inline",
    "answer": "B",            ← letter only
    "case_id": "11583",
    "categories": "localization,diagnosis,reasoning",
    "explanation": "..."
  }

Usage:
    python data/chestagentbench_loader.py --data-dir ./chestagentbench --output ./data/real_cases.json
"""
import json
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def extract_options_from_question(question: str) -> List[str]:
    """
    Extract MCQ options from question text.
    Pattern: "A) some text B) more text C) ..." or "A. text B. text"
    """
    # Try ") " pattern: "A) text B) text"
    options = re.findall(r'([A-F])\)\s*(.+?)(?=\s*[A-F]\)|\s*$)', question)
    if options and len(options) >= 3:
        return [f"{letter}) {text.strip()}" for letter, text in options]

    # Try ". " pattern
    options = re.findall(r'([A-F])\.\s*(.+?)(?=\s*[A-F]\.|\s*$)', question)
    if options and len(options) >= 3:
        return [f"{letter}. {text.strip()}" for letter, text in options]

    return []


def expand_answer(raw: dict) -> str:
    """
    Convert letter answer (e.g. "B") to full answer text using question options.
    Returns "B) actual answer text" or just the letter if options can't be parsed.
    """
    letter = raw.get("answer", "").strip().upper()
    question = raw.get("question", "")

    options = extract_options_from_question(question)
    if options and letter in "ABCDEF":
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]

    return letter


def resolve_image_paths(raw: dict, data_dir: str) -> List[str]:
    """Resolve image paths from metadata to absolute filesystem paths."""
    images = raw.get("images", [])
    resolved = []

    for img_rel in images:
        # img_rel like "figures/11583/figure_1.jpg"
        # After extracting figures.zip to <data_dir>/figures/,
        # the actual path is <data_dir>/figures/figures/11583/figure_1.jpg
        # (because the zip's internal root is "figures/")
        candidate1 = os.path.join(data_dir, img_rel)          # figures/figures/11583/...
        candidate2 = os.path.join(data_dir, "figures", img_rel)  # figures/figures/figures/...

        if os.path.exists(candidate1):
            resolved.append(os.path.abspath(candidate1))
        elif os.path.exists(candidate2):
            resolved.append(os.path.abspath(candidate2))
        else:
            # Try searching
            basename = os.path.basename(img_rel)
            case_id = raw.get("case_id", "")
            for root, _, files in os.walk(data_dir):
                if basename in files:
                    resolved.append(os.path.abspath(os.path.join(root, basename)))
                    break

    return resolved


def parse_categories(raw: dict) -> str:
    """Parse primary category from comma-separated categories field."""
    cats = raw.get("categories", "")
    if not cats:
        return "diagnosis"
    return cats.split(",")[0].strip().lower()


def normalize_case(raw: dict, data_dir: str) -> Optional[Dict[str, Any]]:
    """
    Convert a ChestAgentBench JSONL record into Evo-MedAgent format.
    Returns None if the case can't be normalized.
    """
    question = raw.get("question", "")
    if not question:
        return None

    # Expand letter answer → full answer text
    ground_truth = expand_answer(raw)
    if not ground_truth:
        return None

    category = parse_categories(raw)

    # Case descriptor for embedding-based retrieval
    case_id = raw.get("case_id", "")
    explanation = raw.get("explanation", "")
    descriptor = f"CXR {category}: case {case_id} — {explanation[:100]}" if explanation else f"CXR {category}: case {case_id}"

    # Image paths
    image_paths = resolve_image_paths(raw, data_dir)

    return {
        "question": question.strip(),
        "ground_truth": ground_truth.strip(),
        "category": category,
        "case_descriptor": descriptor,
        "image_paths": image_paths,
        "case_id": case_id,
        "type": raw.get("type", ""),
    }


def load_chestagentbench(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load ChestAgentBench from local directory.

    Args:
        data_dir: path to the downloaded dataset dir (contains metadata.jsonl + figures/)

    Returns:
        list of case dicts in Evo-MedAgent format
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Find metadata file
    candidates = ["metadata.jsonl", "data.jsonl", "cases.jsonl", "data.json", "cases.json"]
    data_path = ""
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data_path = path
            break

    if not data_path:
        # List what's there to help debug
        contents = os.listdir(data_dir)
        raise FileNotFoundError(
            f"No metadata file found in {data_dir}. Contents: {contents}"
        )

    # Load
    if data_path.endswith(".jsonl"):
        raw_cases = load_jsonl(data_path)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_cases = data if isinstance(data, list) else list(data.values())

    logger.info(f"Raw records: {len(raw_cases)}")

    # Normalize
    cases = []
    skipped = 0
    for raw in raw_cases:
        normalized = normalize_case(raw, data_dir)
        if normalized:
            cases.append(normalized)
        else:
            skipped += 1

    logger.info(f"Normalized: {len(cases)} cases (skipped {skipped})")

    # Stats
    cats = {}
    with_img = 0
    for c in cases:
        cat = c["category"]
        cats[cat] = cats.get(cat, 0) + 1
        if c["image_paths"]:
            with_img += 1

    logger.info(f"Categories: {cats}")
    logger.info(f"Cases with resolved images: {with_img}/{len(cases)}")

    # Show sample
    if cases:
        sample = cases[0]
        logger.info(f"Sample case: [{sample['category']}] {sample['question'][:80]}...")
        logger.info(f"  Answer: {sample['ground_truth']}")
        logger.info(f"  Images: {sample['image_paths']}")

    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Load ChestAgentBench → Evo-MedAgent format"
    )
    parser.add_argument("--data-dir", default="./chestagentbench",
                        help="Path to downloaded ChestAgentBench dataset")
    parser.add_argument("--output", default="./data/real_cases.json",
                        help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of cases (0 = all)")
    args = parser.parse_args()

    cases = load_chestagentbench(args.data_dir)

    if args.limit and args.limit < len(cases):
        cases = cases[:args.limit]

    # Save
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(cases)} cases saved to {args.output}")
    print(f"  Run: python main.py --case-file {args.output} --cases {len(cases)}")


if __name__ == "__main__":
    main()
