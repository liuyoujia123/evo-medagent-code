# ChestAgentBench

Existing medical VQA benchmarks typically focus on simple, single-step reasoning tasks. In contrast, ChestAgentBench offers several distinctive advantages:
- It represents one of the largest medical VQA benchmarks, with 2,500 questions derived from expert-validated clinical cases, each with comprehensive radiological findings, detailed discussions, and multi-modal imaging data.
- The benchmark combines complex multi-step reasoning assessment with a structured six-choice format, enabling both rigorous evaluation of advanced reasoning capabilities and straightforward, reproducible evaluation.
- The benchmark features diverse questions across seven core competencies in CXR interpretation, requiring integration of multiple visual findings and reasoning to mirror the complexity of real-world clinical decision-making.


We utilize [Eurorad](https://www.eurorad.org/), the largest peer-reviewed radiological case report database maintained by the European Society of Radiology (ESR). This database contains detailed clinical cases consisting of patient histories, clinical presentations, and multi-modal imaging findings. Each case includes detailed radiological interpretations across different modalities, complemented by in-depth discussions that connect findings with clinical context, and concludes with reasoned interpretations, differential diagnosis list and a final diagnoses.

From its chest imaging section, we curated 675 patient cases with associated chest X-rays and complete clinical documentation. These cases covered 53 unique areas of interest including lung, thorax, and mediastinum. \autoref{fig:benchmark} provides an overview of the benchmark, showing (a) the creation pipeline, (b) patient gender distribution, (c) age distribution, and (d) most frequent anatomical areas of interest.

ChestAgentBench comprises six-choice questions, each designed to evaluate complex CXR interpretation capabilities.

We first established seven core competencies alongside reasoning that are essential for CXR interpretation:

- **Detection** Identifying specific findings. (e.g., ``Is there a nodule present in the right upper lobe?")
- **Classification** Classifying specific findings. (e.g., ``Is this mass benign or malignant in appearance?")   
- **Localization** Precise positioning of findings. (e.g., ``In which bronchopulmonary segment is the mass located?")
- **Comparison** Analyzing relative sizes and positions. (e.g., ``How has the pleural effusion volume changed compared to prior imaging?")
- **Relationship** Understanding relationship of findings. (e.g., ``Does the mediastinal lymphadenopathy correlate with the lung mass?")
- **Diagnosis** Interpreting findings for clinical decisions. (e.g., ``Given the CXR, what is the likely diagnosis?")
- **Characterization** Describing specific finding attributes. (e.g., ``What are the margins of the nodule - smooth, spiculated, or irregular?")
- **Reasoning** Explaining medical rationale and thought. (e.g., ``Why do these findings suggest infectious rather than malignant etiology?")

# How to Use
Download the benchmark:
```
huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench
```
The metadata contains `question`, the question to ask the agent, and `images`, a list of paths to all the images necessary for answering each question.
For more details check out https://github.com/bowang-lab/MedRAX .