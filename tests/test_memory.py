"""
Unit tests for Evo-MedAgent memory modules.
Run with: python -m pytest tests/ -v
"""
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.embedding import Embedder, cosine_similarity
from src.memory.episodic import EpisodicMemory, Episode
from src.memory.procedural import ProceduralMemory, ProceduralRule
from src.memory.governance import ToolGovernanceMemory, TrustLabel
from src.memory.evolving import SelfEvolvingMemory


def get_embedder():
    """Get embedder with fallback support."""
    return Embedder(model_name="all-MiniLM-L6-v2")


class TestEmbedder:
    """Test embedding utilities."""

    def test_single_encode(self):
        emb = get_embedder()
        vec = emb.single_encode("pneumothorax left chest X-ray")
        assert len(vec.shape) == 1
        assert vec.shape[0] > 0
        assert abs(float(sum(vec * vec)) - 1.0) < 0.2  # roughly normalized

    def test_batch_encode(self):
        emb = get_embedder()
        texts = ["normal CXR", "pneumonia right lower lobe", "cardiomegaly moderate"]
        vecs = emb.encode(texts)
        assert vecs.shape == (3, emb.dimension)

    def test_cosine_similarity(self):
        import numpy as np
        a = np.array([[1, 0, 0]], dtype=np.float32)  # unit vector
        b = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim[0, 0] - 1.0) < 0.01
        assert abs(sim[0, 1] - 0.0) < 0.01


class TestEpisodicMemory:
    """Test episodic memory store."""

    def test_add_and_retrieve(self):
        emb = get_embedder()
        store = EpisodicMemory(emb, max_episodes=100, retrieval_k=3)

        # Add episodes
        for i in range(5):
            ep = Episode(
                episode_id=-1,
                phi=f"CXR case {i}: pneumonia",
                question=f"Is there pneumonia in case {i}?",
                tool_trace="classifier: pneumonia",
                prediction="Yes" if i % 2 == 0 else "No",
                ground_truth="Yes",
                is_correct=(i % 2 == 0),
                summary="Good" if i % 2 == 0 else "Missed",
            )
            store.add(ep)

        assert len(store) == 5

        # Retrieve
        results = store.retrieve("CXR with possible pneumonia")
        assert len(results) > 0

    def test_max_episodes(self):
        emb = get_embedder()
        store = EpisodicMemory(emb, max_episodes=5, retrieval_k=3)

        for i in range(10):
            store.add(Episode(
                episode_id=-1, phi=f"Case {i}", question="Q",
                tool_trace="", prediction="A", ground_truth="A",
                is_correct=True,
            ))

        assert len(store) == 5  # should evict oldest

    def test_format_context(self):
        emb = get_embedder()
        store = EpisodicMemory(emb)

        ep = Episode(
            episode_id=0, phi="Pneumothorax case",
            question="Is there pneumothorax?",
            tool_trace="classifier: negative",
            prediction="No pneumothorax",
            ground_truth="Left pneumothorax",
            is_correct=False,
            summary="Missed subtle apical pneumothorax.",
            guideline="Always inspect lung apices carefully.",
        )
        store.add(ep)

        ctx = store.format_context([ep])
        assert "INCORRECT" in ctx
        assert "pneumothorax" in ctx.lower()

    def test_save_load(self):
        emb = get_embedder()
        store = EpisodicMemory(emb, max_episodes=100)

        store.add(Episode(
            episode_id=-1, phi="Test", question="Test Q",
            tool_trace="none", prediction="A", ground_truth="A",
            is_correct=True, summary="OK", guideline="Keep doing this.",
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            store.save(path)
            store2 = EpisodicMemory(emb)
            store2.load(path)
            assert len(store2) == 1
            assert store2._episodes[0].question == "Test Q"
        finally:
            os.unlink(path)


class TestProceduralMemory:
    """Test procedural memory store."""

    def test_add_and_select(self):
        emb = get_embedder()
        store = ProceduralMemory(emb, max_rules=50, retrieval_k=5)

        store.add("Always check lung apices for pneumothorax.", priority=0)
        store.add("Consider sarcoidosis for bilateral hilar adenopathy.", priority=1)

        assert len(store) == 2

        selected = store.select("CXR showing possible pneumothorax")
        assert len(selected) > 0
        # Top rule should be more relevant to pneumothorax
        assert "pneumothorax" in selected[0].instruction.lower()

    def test_update_or_add_merges(self):
        emb = get_embedder()
        store = ProceduralMemory(emb)

        r1 = store.update_or_add("Never miss apical pneumothorax.", priority=1)
        r2 = store.update_or_add("Never miss apical pneumothorax in CXR.", priority=1)

        # Should merge (same or similar rule), not duplicate
        assert r1.rule_id == r2.rule_id

    def test_mark_outcomes(self):
        emb = get_embedder()
        store = ProceduralMemory(emb)

        store.add("Check for rib fracture in trauma.", priority=0)
        selected = store.select("Chest trauma with pain")
        store.mark_outcomes(selected, was_correct=True)

        for rule in selected:
            assert rule.times_selected == 1
            assert rule.times_helpful == 1

    def test_max_rules(self):
        emb = get_embedder()
        store = ProceduralMemory(emb, max_rules=5)

        for i in range(10):
            store.add(f"Test rule number {i} for radiology diagnosis.", priority=2)

        assert len(store) <= 5

    def test_priority_bonus_in_scoring(self):
        emb = get_embedder()
        store = ProceduralMemory(emb, retrieval_k=10)

        # Add a critical rule
        store.add("CRITICAL: Check mediastinum in trauma. This is urgent.", priority=0)
        # Add less important rules
        for i in range(10):
            store.add(f"General guidance rule {i}: review all films carefully.", priority=2)

        # Select for trauma case — critical rule should appear
        selected = store.select("Trauma patient with CXR showing widened mediastinum")
        priorities = [r.priority for r in selected]
        assert 0 in priorities  # at least one critical rule

    def test_save_load(self):
        emb = get_embedder()
        store = ProceduralMemory(emb)

        store.add("Check for Kerley B lines in suspected edema.", priority=0)
        store.add("Compare with prior studies.", priority=2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            store.save(path)
            store2 = ProceduralMemory(emb)
            store2.load(path)
            assert len(store2) == 2
        finally:
            os.unlink(path)


class TestGovernanceMemory:
    """Test tool-governance memory."""

    def test_register_and_track(self):
        store = ToolGovernanceMemory()
        store.register_tools(["classifier", "segmenter", "vqa"])

        # Simulate interactions
        for _ in range(8):
            store.record_interaction("classifier", was_helpful=True)
        # classifier should now be TRUSTED (8 helpful, 0 harmful, rate=1.0 > 0.70)

        assert store.get_record("classifier").trust_label == TrustLabel.TRUSTED
        assert store.get_record("segmenter").trust_label == TrustLabel.CAUTION  # no data yet

    def test_avoid_threshold(self):
        store = ToolGovernanceMemory()

        # Simulate a bad tool
        for _ in range(12):
            store.record_interaction("noisy_segmenter", was_harmful=True)

        record = store.get_record("noisy_segmenter")
        assert record.trust_label == TrustLabel.AVOID
        assert record.effective_bad_rate >= 0.60

    def test_format_context(self):
        store = ToolGovernanceMemory()
        store.record_interaction("classifier", was_helpful=True)
        ctx = store.format_context()
        assert "classifier" in ctx

    def test_save_load(self):
        store = ToolGovernanceMemory()
        store.record_interaction("classifier", was_helpful=True, was_harmful=False)
        store.record_interaction("segmenter", was_helpful=False, was_harmful=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            store.save(path)
            store2 = ToolGovernanceMemory()
            store2.load(path)
            assert len(store2.get_all_records()) == 2
        finally:
            os.unlink(path)


class TestEvolvingMemory:
    """Test the unified three-store memory system."""

    def test_query_and_update(self):
        emb = get_embedder()
        episodic = EpisodicMemory(emb)
        procedural = ProceduralMemory(emb)
        governance = ToolGovernanceMemory()
        memory = SelfEvolvingMemory(episodic, procedural, governance)

        # Initial query (empty memory)
        ctx = memory.query("CXR with pneumothorax")
        assert len(ctx.retrieved_episodes) == 0
        assert len(ctx.selected_rules) == 0

        # Add some rules first
        procedural.add("Always check apices for pneumothorax.", priority=0)

        # Update with a case
        memory.update(
            case_descriptor="CXR: pneumothorax left apex",
            question="Is there a pneumothorax?",
            tool_trace="classifier: negative",
            prediction="No",
            ground_truth="Yes, left pneumothorax",
            summary="Missed subtle pneumothorax.",
            guideline="Always examine apices carefully.",
            new_rules=[
                ("Inspect lung apices on every CXR for subtle pneumothorax.", 0),
            ],
            tool_outcomes=[
                ("classifier", False, True, False),  # classifier was harmful
            ],
        )

        assert len(episodic) == 1
        assert len(procedural) >= 1

        # Second query should retrieve the stored episode
        ctx2 = memory.query("CXR showing apical lucency, possible pneumothorax")
        assert len(ctx2.retrieved_episodes) >= 1

    def test_save_load_all(self):
        emb = get_embedder()
        episodic = EpisodicMemory(emb)
        procedural = ProceduralMemory(emb)
        governance = ToolGovernanceMemory()
        governance.register_tool("classifier")

        memory = SelfEvolvingMemory(episodic, procedural, governance)

        memory.update(
            case_descriptor="Test case", question="Q?",
            tool_trace="", prediction="A", ground_truth="A",
            summary="OK", new_rules=[("Test rule.", 1)],
            tool_outcomes=[("classifier", True, False, False)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            memory.save(tmpdir)
            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "episodic.json"))
            assert os.path.exists(os.path.join(tmpdir, "procedural.json"))
            assert os.path.exists(os.path.join(tmpdir, "governance.json"))

            # Load into new memory
            episodic2 = EpisodicMemory(emb)
            procedural2 = ProceduralMemory(emb)
            governance2 = ToolGovernanceMemory()
            memory2 = SelfEvolvingMemory(episodic2, procedural2, governance2)
            memory2.load(tmpdir)

            assert len(episodic2) == 1
            assert len(procedural2) == 1
            assert len(governance2.get_all_records()) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
