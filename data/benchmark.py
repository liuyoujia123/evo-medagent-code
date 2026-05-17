"""
ChestAgentBench data loader and sample case generator.

Provides a structured benchmark stream for Evo-MedAgent evaluation.
When real data is unavailable, generates realistic CXR MCQ cases.
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

# =============================================================================
# Sample CXR cases — representative of ChestAgentBench's 7 diagnostic categories
# =============================================================================

SAMPLE_CASES = [
    # ---- DETECTION ----
    {
        "question": "Is there a pneumothorax visible in this chest X-ray?",
        "ground_truth": "Yes, left-sided pneumothorax",
        "category": "detection",
        "case_descriptor": "CXR detection: pneumothorax, unilateral left",
    },
    {
        "question": "Does this CXR show any evidence of pleural effusion?",
        "ground_truth": "Yes, bilateral pleural effusions",
        "category": "detection",
        "case_descriptor": "CXR detection: pleural effusion, bilateral",
    },
    {
        "question": "Is there cardiomegaly on this chest radiograph?",
        "ground_truth": "No cardiomegaly",
        "category": "detection",
        "case_descriptor": "CXR detection: cardiomegaly, negative",
    },
    {
        "question": "Are there any pulmonary nodules visible?",
        "ground_truth": "Yes, solitary pulmonary nodule in right upper lobe",
        "category": "detection",
        "case_descriptor": "CXR detection: solitary pulmonary nodule, right upper lobe",
    },
    # ---- CLASSIFICATION ----
    {
        "question": "Classify the pattern of interstitial lung disease: "
                     "A) UIP B) NSIP C) COP D) LIP",
        "ground_truth": "A) UIP",
        "category": "classification",
        "case_descriptor": "CXR classification: interstitial lung disease pattern, UIP vs NSIP differentiation",
    },
    {
        "question": "What is the most likely diagnosis? "
                     "A) Bacterial pneumonia B) Viral pneumonia C) Pulmonary edema D) ARDS",
        "ground_truth": "C) Pulmonary edema",
        "category": "classification",
        "case_descriptor": "CXR classification: diffuse bilateral opacities, pneumonia vs edema",
    },
    # ---- LOCALIZATION ----
    {
        "question": "Where is the mass located? "
                     "A) Right upper lobe B) Right lower lobe C) Left upper lobe D) Left lower lobe",
        "ground_truth": "B) Right lower lobe",
        "category": "localization",
        "case_descriptor": "CXR localization: mass, right lower lobe",
    },
    {
        "question": "Which lobe contains the consolidation?",
        "ground_truth": "Left upper lobe",
        "category": "localization",
        "case_descriptor": "CXR localization: consolidation, left upper lobe",
    },
    # ---- COMPARISON ----
    {
        "question": "Compared to the prior study, has the pleural effusion "
                     "A) Increased B) Decreased C) Stable D) Resolved",
        "ground_truth": "A) Increased",
        "category": "comparison",
        "case_descriptor": "CXR comparison: pleural effusion change, increased",
    },
    {
        "question": "How has the cardiomediastinal silhouette changed since the previous exam? "
                     "A) Enlarged B) Decreased C) Unchanged D) New mass",
        "ground_truth": "C) Unchanged",
        "category": "comparison",
        "case_descriptor": "CXR comparison: cardiomediastinal silhouette stability",
    },
    # ---- RELATIONSHIP ----
    {
        "question": "How does the mass relate to the hilum? "
                     "A) Overlaps B) Separate C) Obscures D) Cannot determine",
        "ground_truth": "A) Overlaps",
        "category": "relationship",
        "case_descriptor": "CXR relationship: mass-hilum overlap",
    },
    # ---- CHARACTERIZATION ----
    {
        "question": "Characterize the opacity pattern: "
                     "A) Airspace B) Interstitial C) Mixed D) Nodular",
        "ground_truth": "B) Interstitial",
        "category": "characterization",
        "case_descriptor": "CXR characterization: interstitial opacity pattern",
    },
    # ---- DIAGNOSIS ----
    {
        "question": "A 14-year-old male presents with acute left chest pain and fever. "
                     "Left pleural effusion is seen. What is the most likely diagnosis? "
                     "A) Pneumonia with parapneumonic effusion "
                     "B) Spontaneous hemothorax from rib exostosis "
                     "C) Tuberculous pleurisy "
                     "D) Pulmonary embolism with infarction",
        "ground_truth": "B) Spontaneous hemothorax from rib exostosis",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: adolescent, chest pain, fever, left effusion, rib exostosis vs pneumonia",
    },
    {
        "question": "A 65-year-old smoker presents with cough and weight loss. "
                     "CXR shows a spiculated right upper lobe mass with hilar adenopathy. "
                     "Most likely diagnosis? "
                     "A) Tuberculosis B) Lung cancer C) Fungal infection D) Sarcoidosis",
        "ground_truth": "B) Lung cancer",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: elderly smoker, spiculated mass, hilar adenopathy — lung cancer",
    },
    # ---- More detection cases for longer benchmark streams ----
    {
        "question": "Is there evidence of rib fracture on this CXR?",
        "ground_truth": "Yes, fracture of the left 7th rib",
        "category": "detection",
        "case_descriptor": "CXR detection: rib fracture, left 7th",
    },
    {
        "question": "Does this chest X-ray show signs of hyperinflation?",
        "ground_truth": "Yes, consistent with COPD",
        "category": "detection",
        "case_descriptor": "CXR detection: hyperinflation, COPD pattern",
    },
    {
        "question": "Is there a mediastinal mass present?",
        "ground_truth": "Yes, anterior mediastinal mass",
        "category": "detection",
        "case_descriptor": "CXR detection: anterior mediastinal mass",
    },
    {
        "question": "Are there signs of pulmonary edema?",
        "ground_truth": "Yes, interstitial pulmonary edema with Kerley B lines",
        "category": "detection",
        "case_descriptor": "CXR detection: pulmonary edema, interstitial with Kerley B lines",
    },
    {
        "question": "Is there atelectasis present? "
                     "A) Yes, right lower lobe B) Yes, left lower lobe C) No D) Cannot determine",
        "ground_truth": "A) Yes, right lower lobe",
        "category": "detection",
        "case_descriptor": "CXR detection: atelectasis, right lower lobe",
    },
    {
        "question": "Does this CXR show a cavitary lesion?",
        "ground_truth": "Yes, cavitary lesion in left upper lobe",
        "category": "detection",
        "case_descriptor": "CXR detection: cavitary lesion, left upper lobe",
    },
    # ---- More diagnosis cases ----
    {
        "question": "A 45-year-old female with dyspnea and bilateral hilar lymphadenopathy. "
                     "Most likely diagnosis? "
                     "A) Sarcoidosis B) Lymphoma C) Tuberculosis D) Metastatic disease",
        "ground_truth": "A) Sarcoidosis",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: bilateral hilar lymphadenopathy, sarcoidosis",
    },
    {
        "question": "A 72-year-old post-operative patient develops acute dyspnea. "
                     "CXR is normal. Most likely? "
                     "A) Pulmonary embolism B) Atelectasis C) Pneumonia D) Heart failure",
        "ground_truth": "A) Pulmonary embolism",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: post-op dyspnea, normal CXR, pulmonary embolism",
    },
    {
        "question": "This CXR shows widening of the mediastinum in a trauma patient. "
                     "Most likely finding? "
                     "A) Aortic injury B) Thymic hyperplasia C) Lymphoma D) Goiter",
        "ground_truth": "A) Aortic injury",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: trauma, widened mediastinum, aortic injury",
    },
    {
        "question": "A 30-year-old with HIV presents with bilateral interstitial infiltrates. "
                     "Most likely? "
                     "A) PCP pneumonia B) Bacterial pneumonia C) TB D) CMV pneumonitis",
        "ground_truth": "A) PCP pneumonia",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: HIV, bilateral interstitial infiltrates, PCP pneumonia",
    },
    {
        "question": "A 55-year-old with hemoptysis. CXR shows a mass with an air crescent sign. "
                     "Most likely? "
                     "A) Aspergilloma B) Lung abscess C) Cavitary tumor D) Hydatid cyst",
        "ground_truth": "A) Aspergilloma",
        "category": "diagnosis",
        "case_descriptor": "CXR diagnosis: hemoptysis, air crescent sign, aspergilloma",
    },
]


class BenchmarkLoader:
    """
    Loads and manages ChestAgentBench-compatible case streams.
    Supports sample data, custom JSON, and permutation-based ordering.
    """

    def __init__(self, seed: int = 42, shuffle: bool = True):
        self.seed = seed
        self.shuffle = shuffle

    def load_sample(self, n_cases: int = 25,
                    include_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load sample CXR cases (ideal for quick evaluation and demo).
        Cycles through available cases if n_cases > len(SAMPLE_CASES).
        """
        available = SAMPLE_CASES
        if include_categories:
            available = [c for c in available if c["category"] in include_categories]

        if n_cases <= len(available):
            cases = available[:n_cases]
        else:
            # Cycle through available cases
            cases = []
            for i in range(n_cases):
                cases.append(dict(available[i % len(available)]))
                cases[-1]["_cycle_index"] = i

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(cases)

        return cases

    def load_json(self, path: str, n_cases: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load cases from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data if isinstance(data, list) else data.get("cases", [])
        if n_cases:
            cases = cases[:n_cases]
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(cases)
        return cases

    def generate_permutations(self, cases: List[Dict[str, Any]],
                              n_permutations: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Generate multiple randomized permutations of the case stream
        (for order-sensitivity evaluation, as done in the paper).
        """
        rng = random.Random(self.seed)
        permutations = []
        for i in range(n_permutations):
            perm = list(cases)
            rng.shuffle(perm)
            permutations.append(perm)
        return permutations

    @staticmethod
    def get_categories() -> List[str]:
        """Return all diagnostic categories in ChestAgentBench."""
        return ["detection", "classification", "localization", "comparison",
                "relationship", "characterization", "diagnosis"]
