import numpy as np
import pytest
import torch

from idea.config import IDEAConfig
from idea.dataset_loader import DatasetSplit, Sample
from idea.experiment import IDEAExperiment
from idea.feature_calibration import CalibrationConfig, FeatureCalibrator, l2_normalize
from idea.lsconv import (
    PatchTokenLSConv,
    build_patch_token_lsconv,
    load_lsnet_weights,
)
from idea.priors import estimate_priors
from idea.probabilistic_alignment import PCMAConfig, ProbabilisticAlignmentModule
from idea.selection import (
    AdaptiveDiagnostics,
    SelectionWeights,
    adapt_selection_weights,
    compute_adaptive_diagnostics,
)


def test_patch_token_lsconv_uses_fixed_orthogonal_projection_and_structure_pooling():
    model = build_patch_token_lsconv(device="cpu")
    assert isinstance(model, PatchTokenLSConv)
    assert "R" in dict(model.named_buffers())
    assert "R" not in dict(model.named_parameters())
    identity = model.R.T @ model.R
    torch.testing.assert_close(identity, torch.eye(128), atol=1e-5, rtol=1e-5)

    tokens = torch.randn(1, 1024, 1024)
    structural = model(tokens)
    assert structural.shape == (1, 1024)
    torch.testing.assert_close(structural.norm(dim=1), torch.ones(1))


def test_lsnet_checkpoint_covers_and_replaces_all_block_parameters(tmp_path):
    pretrained = PatchTokenLSConv(seed=91)
    checkpoint = {
        "state_dict": {
            f"module.{key}": value.clone()
            for key, value in pretrained.state_dict().items()
            if key.startswith("blocks.")
        }
    }
    path = tmp_path / "lsnet_t.pth"
    torch.save(checkpoint, path)

    model = PatchTokenLSConv(seed=42)
    before = next(model.blocks.parameters()).detach().clone()
    missing, unexpected = load_lsnet_weights(model, path)
    assert not unexpected
    assert all(not key.startswith("blocks.") for key in missing)
    assert not torch.equal(before, next(model.blocks.parameters()))
    for key, value in pretrained.state_dict().items():
        if key.startswith("blocks."):
            torch.testing.assert_close(model.state_dict()[key], value)


def test_official_lsnet_t_c128_mixer_is_copied_to_both_blocks(tmp_path):
    pretrained = PatchTokenLSConv(seed=73)
    source_state = pretrained.state_dict()
    translations = (
        ("lkp.reduce.0.", "lkp.cv1.c."),
        ("lkp.reduce.1.", "lkp.cv1.bn."),
        ("lkp.depthwise.0.", "lkp.cv2.c."),
        ("lkp.depthwise.1.", "lkp.cv2.bn."),
        ("lkp.project.0.", "lkp.cv3.c."),
        ("lkp.project.1.", "lkp.cv3.bn."),
        ("lkp.to_kernel.", "lkp.cv4."),
        ("lkp.norm.", "lkp.norm."),
        ("bn.", "bn."),
    )
    official = {}
    expected = {}
    for target_key, value in source_state.items():
        if not target_key.startswith("blocks.0."):
            continue
        suffix = target_key[len("blocks.0.") :]
        for target_prefix, source_prefix in translations:
            if suffix.startswith(target_prefix):
                source_key = (
                    "blocks2.3.mixer."
                    + source_prefix
                    + suffix[len(target_prefix) :]
                )
                official[source_key] = value.clone()
                expected[suffix] = value
                break

    path = tmp_path / "lsnet_t.pth"
    torch.save({"model": official}, path)
    model = PatchTokenLSConv(seed=42)
    load_lsnet_weights(model, path)
    for block_index in (0, 1):
        for suffix, value in expected.items():
            torch.testing.assert_close(
                model.state_dict()[f"blocks.{block_index}.{suffix}"], value
            )


def test_feature_fusion_is_unweighted_5120_dimensional_concat():
    rng = np.random.default_rng(4)
    semantic = rng.standard_normal((3, 4096))
    structural = rng.standard_normal((3, 1024))
    calibrator = FeatureCalibrator(CalibrationConfig(use_structural=True))
    calibrator.fit(semantic, structural)
    fused = calibrator.transform(semantic, structural)

    expected = l2_normalize(
        np.concatenate([l2_normalize(semantic), l2_normalize(structural)], axis=1)
    )
    assert fused.shape == (3, 5120)
    np.testing.assert_allclose(fused, expected)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1.0)


def test_pcma_stores_scalar_trace_and_uses_paper_bandwidth():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    module = ProbabilisticAlignmentModule().fit(features, ["a", "a", "b"])
    assert PCMAConfig().sigma_sq == 1.0
    assert isinstance(module["a"].trace, float)
    centered = features[:2] - features[:2].mean(axis=0)
    expected_trace = np.mean(np.sum(centered**2, axis=1))
    assert module["a"].trace == expected_trace


def test_adaptive_weights_preserve_global_and_intrinsic_sums():
    base = SelectionWeights()
    diagnostics = AdaptiveDiagnostics(0.4, 0.6, 0.8, 0.2, 0.9)
    adapted = adapt_selection_weights(base, diagnostics)
    assert np.isclose(
        adapted.alpha + adapted.beta + adapted.gamma + adapted.delta,
        base.alpha + base.beta + base.gamma + base.delta,
    )
    assert np.isclose(
        adapted.omega1 + adapted.omega2, base.omega1 + base.omega2
    )
    assert adapted.delta > base.delta
    assert adapted.omega2 > base.omega2


def test_adaptive_diagnostics_are_bounded():
    diagnostics = compute_adaptive_diagnostics(
        context_labels=[["a", "a"], ["a", "b"]],
        context_embeddings=[np.eye(2), np.array([[1.0, 0.0], [0.9, 0.1]])],
        train_embeddings=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        train_labels=["a", "a", "b"],
        val_true=["a", "b"],
        val_predictions=[None, "b"],
        hard_classes={"a"},
        n_classes=2,
    )
    assert all(0.0 <= value <= 1.0 for value in diagnostics.__dict__.values())
    assert diagnostics.r_hard == 1.0
    assert diagnostics.r_conf == 0.0
    # Class a has per-dimension variances [0.0025, 0.0025], class b has zero.
    assert np.isclose(diagnostics.r_var, 0.00125)


def test_unparseable_validation_prediction_counts_as_error():
    priors = estimate_priors(
        ["a", "a", "b"], ["a", None, "b"], ["a", "b"]
    )
    assert priors.difficulty["a"].total == 2
    assert priors.difficulty["a"].accuracy == 0.5
    assert priors.confusion.get("a", {}) == {}


def test_baseline_candidates_use_semantic_cache_only():
    cfg = IDEAConfig()
    experiment = IDEAExperiment(cfg)
    sample = Sample("image.tif", "a", "train")
    experiment._semantic_embeddings[sample.image_path] = np.ones(4096)
    experiment._idea_embeddings[sample.image_path] = np.ones(5120)
    candidates = experiment._build_baseline_candidates([sample])
    assert candidates[0].embedding.shape == (4096,)


def test_baseline_evaluation_always_disables_token_budget(monkeypatch):
    cfg = IDEAConfig()
    cfg.experiment.token_budget = 1
    experiment = IDEAExperiment(cfg)
    query = Sample("query.tif", "a", "test")
    dataset = DatasetSplit("mock", ["a"], [], [], [query], seed=42)
    experiment._dataset = dataset
    experiment._calibrator = object()
    experiment._semantic_embeddings[query.image_path] = np.ones(4096)
    experiment._idea_embeddings[query.image_path] = np.ones(5120)

    class Backend:
        def generate(self, prompt, image_paths, max_new_tokens=100):
            return "a"

    experiment._backend = Backend()
    captured = {}

    def fake_random(query_embedding, candidates, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("idea.baselines.random_select", fake_random)
    experiment.evaluate_baseline("random", dataset)
    assert captured["token_budget"] is None


def test_paper_defaults_disable_ambiguous_token_cap():
    cfg = IDEAConfig()
    assert cfg.experiment.token_budget is None
    assert cfg.experiment.max_new_tokens == 100
    assert cfg.pcma.sigma_sq == 1.0
    assert cfg.selection.adaptive is True
    assert cfg.lsconv.checkpoint_path == "lsnet_t.pth"


def test_structural_pipeline_rejects_missing_lsnet_checkpoint():
    cfg = IDEAConfig()
    cfg.lsconv.checkpoint_path = None
    experiment = IDEAExperiment(cfg)
    dataset = DatasetSplit("mock", ["a"], [], [], [], seed=42)
    with pytest.raises(ValueError, match="lsconv.checkpoint_path is required"):
        experiment.setup(dataset)
