"""
Simulated CXR diagnostic tools for Evo-MedAgent.

In a production setting, these would be real ML models.
For research/reproducibility, we provide:
1. LLM-Tool mode: uses the agent's own VLM to simulate tool outputs
2. Rule-based mode: heuristics for testing the memory framework
"""
import logging
from typing import Optional, Dict, Any
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SimulatedClassifier(BaseTool):
    """Pathology classifier — detects presence/absence of CXR findings."""

    def __init__(self, llm_client=None):
        super().__init__(
            name="classifier",
            description="Classifies CXR for common pathologies "
                        "(pneumothorax, pleural effusion, consolidation, "
                        "cardiomegaly, edema, atelectasis, mass, nodule, fracture). "
                        "Returns finding name + confidence."
        )
        self._llm = llm_client

    def run(self, image_path: str, finding: Optional[str] = None, **kwargs) -> ToolResult:
        if self._llm:
            prompt = (f"You are a CXR pathology classifier. Analyze the image for "
                      f"{finding or 'all common pathologies'}. "
                      f"Output: '<finding>: PRESENT/ABSENT/<confidence>'. Be precise.")
            output = self._llm.chat_with_images(
                "You are a radiology classifier. Be brief and precise.",
                prompt, [image_path], max_tokens=200
            )
            return ToolResult(self.name, success=True, output=output or "No finding detected.")
        else:
            # Rule-based fallback
            return ToolResult(
                self.name, success=True,
                output=f"classifier({finding or 'all'}): no significant abnormality detected."
            )


class SimulatedSegmenter(BaseTool):
    """Anatomical/pathological segmentation tool."""

    def __init__(self, llm_client=None):
        super().__init__(
            name="segmenter",
            description="Segments anatomical structures or pathological regions in CXR. "
                        "Returns mask boundaries and measurements."
        )
        self._llm = llm_client

    def run(self, image_path: str, target: Optional[str] = None, **kwargs) -> ToolResult:
        if self._llm:
            prompt = (f"Segment{' the ' + target if target else ' key regions'} in this CXR. "
                      f"Describe location, size, and shape. Be precise with coordinates.")
            output = self._llm.chat_with_images(
                "You are a CXR segmentation model. Describe findings with spatial precision.",
                prompt, [image_path], max_tokens=200
            )
            return ToolResult(self.name, success=True, output=output or "No segmentation output.")
        else:
            return ToolResult(
                self.name, success=True,
                output=f"segmentation({target or 'full'}): normal anatomy, no focal lesions."
            )


class SimulatedVQA(BaseTool):
    """Visual Question Answering for CXR."""

    def __init__(self, llm_client=None):
        super().__init__(
            name="vqa",
            description="Answers focused visual questions about CXR (e.g., "
                        "'Is there a pneumothorax?'). Returns yes/no/detail."
        )
        self._llm = llm_client

    def run(self, image_path: str, question: str = "", **kwargs) -> ToolResult:
        if self._llm:
            prompt = f"Question about this CXR: {question}\nAnswer concisely."
            output = self._llm.chat_with_images(
                "You are a CXR visual QA system. Answer briefly and precisely.",
                prompt, [image_path], max_tokens=150
            )
            return ToolResult(self.name, success=True, output=output or "Unable to answer.")
        else:
            return ToolResult(self.name, success=True, output=f"VQA({question}): No findings.")


class SimulatedReportGenerator(BaseTool):
    """CXR report generator."""

    def __init__(self, llm_client=None):
        super().__init__(
            name="report_generator",
            description="Generates structured radiology report from CXR "
                        "(findings, impression, recommendations)."
        )
        self._llm = llm_client

    def run(self, image_path: str, **kwargs) -> ToolResult:
        if self._llm:
            prompt = ("Generate a structured radiology report for this CXR with: "
                      "FINDINGS, IMPRESSION. Be thorough and precise.")
            output = self._llm.chat_with_images(
                "You are an expert radiologist. Write a structured CXR report.",
                prompt, [image_path], max_tokens=400
            )
            return ToolResult(self.name, success=True, output=output or "Normal CXR.")
        else:
            return ToolResult(
                self.name, success=True,
                output="FINDINGS: Clear lungs. Normal cardiac silhouette. "
                       "No pleural effusion or pneumothorax.\nIMPRESSION: Normal CXR."
            )


class SimulatedPhraseGrounding(BaseTool):
    """Phrase grounding — localizes textual findings in the image."""

    def __init__(self, llm_client=None):
        super().__init__(
            name="grounding",
            description="Locates regions in CXR corresponding to a textual description. "
                        "Returns bounding box or spatial region."
        )
        self._llm = llm_client

    def run(self, image_path: str, phrase: str = "", **kwargs) -> ToolResult:
        if self._llm:
            prompt = f"Locate '{phrase}' in this CXR. Describe the spatial position precisely."
            output = self._llm.chat_with_images(
                "You localize radiographic findings. Give spatial coordinates.",
                prompt, [image_path], max_tokens=150
            )
            return ToolResult(self.name, success=True, output=output or "Not localized.")
        else:
            return ToolResult(
                self.name, success=True,
                output=f"grounding('{phrase}'): no matching region found."
            )


def create_default_toolbox(llm_client=None):
    """Create the standard CXR toolbox (like MedRAX's 7-tool suite)."""
    from .base import ToolRegistry

    registry = ToolRegistry()
    registry.register_all([
        SimulatedClassifier(llm_client),
        SimulatedSegmenter(llm_client),
        SimulatedVQA(llm_client),
        SimulatedReportGenerator(llm_client),
        SimulatedPhraseGrounding(llm_client),
    ])
    return registry
