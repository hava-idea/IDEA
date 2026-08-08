"""Main IDEA experiment pipeline.

Pipeline (no test-set leakage)
-------------------------------
1. Load dataset (train / val / test split).
2. Extract semantic features (InternVL2.5-8B mean-pooled visual tokens).
3. Extract structural features (LSConv, pretrained LSNet-T backbone, frozen).
4. Fit FeatureCalibrator on train features; transform all features.
5. Fit ProbabilisticAlignmentModule (PCMA) on calibrated train features.
6. Run zero-shot inference on the **validation** split; estimate retrieval
   priors (C_hard, H) from validation predictions -- never from test.
7. Fit SACScorer from train prototypes and priors.hard_negatives.
8. Build AdaptiveSelectionEngine from calibrated train candidates.
9. For each test query: select exemplars -> build prompt -> generate -> parse.
10. Report per-class and overall accuracy.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import IDEAConfig
from .dataset_loader import DatasetSplit, Sample, load_dataset
from .feature_calibration import CalibrationConfig, FeatureCalibrator
from .lsconv import build_lsconv_backbone_pretrained, extract_structural_features_from_paths
from .mllm_backend import MLLMBackend, load_backend
from .probabilistic_alignment import PCMAConfig, ProbabilisticAlignmentModule
from .priors import Priors, estimate_priors, empty_priors
from .prompt_builder import build_classification_prompt, parse_label, text_token_cost_for_exemplar
from .self_supervised_alignment import ClassPrototypes, HardNegativeSet, SACConfig, SACScorer
from .selection import AdaptiveSelectionEngine, Candidate, SelectionWeights

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Inference result for one test query."""

    query_path: str
    true_label: str
    predicted_label: Optional[str]
    n_exemplars: int
    correct: bool = field(init=False)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.correct = self.predicted_label == self.true_label


@dataclass
class EvalResult:
    """Aggregated evaluation results for one method on one dataset."""

    dataset_name: str
    method: str
    n_queries: int
    n_correct: int
    accuracy: float
    per_class_accuracy: Dict[str, float]
    per_class_n: Dict[str, int]
    results: List[QueryResult] = field(repr=False, default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"EvalResult({self.dataset_name}, method={self.method})",
            f"  accuracy : {self.accuracy:.4f}  ({self.n_correct}/{self.n_queries})",
            f"  elapsed  : {self.elapsed_s:.1f}s",
            "  per-class accuracy:",
        ]
        for lbl in sorted(self.per_class_accuracy):
            lines.append(
                f"    {lbl:<28s} {self.per_class_accuracy[lbl]:.4f}"
                f"  (n={self.per_class_n[lbl]})"
            )
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_name": self.dataset_name,
            "method": self.method,
            "n_queries": self.n_queries,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "elapsed_s": self.elapsed_s,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_n": self.per_class_n,
            "per_query": [
                {"query": r.query_path, "true": r.true_label,
                 "pred": r.predicted_label, "correct": r.correct,
                 "n_exemplars": r.n_exemplars, "elapsed_s": r.elapsed_s}
                for r in self.results
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("results written to %s", path)


def _make_eval_result(
    dataset_name: str, method: str, results: List[QueryResult], elapsed_s: float
) -> EvalResult:
    per_class_correct: Dict[str, int] = defaultdict(int)
    per_class_n: Dict[str, int] = defaultdict(int)
    for r in results:
        per_class_n[r.true_label] += 1
        if r.correct:
            per_class_correct[r.true_label] += 1
    n_correct = sum(1 for r in results if r.correct)
    n = len(results)
    return EvalResult(
        dataset_name=dataset_name,
        method=method,
        n_queries=n,
        n_correct=n_correct,
        accuracy=n_correct / n if n else 0.0,
        per_class_accuracy={lbl: per_class_correct[lbl] / cnt for lbl, cnt in per_class_n.items()},
        per_class_n=dict(per_class_n),
        results=results,
        elapsed_s=elapsed_s,
    )


class IDEAExperiment:
    """Full IDEA pipeline: feature extraction -> prior estimation -> selection -> eval.

    Usage::

        cfg = IDEAConfig.from_yaml("configs/ucmerced_20shot.yaml")
        exp = IDEAExperiment(cfg)
        dataset = load_dataset(cfg.dataset.name, cfg.dataset.root, seed=cfg.dataset.seed)
        exp.setup(dataset)
        result = exp.evaluate(dataset)
        print(result.summary())
    """

    def __init__(self, cfg: IDEAConfig) -> None:
        self.cfg = cfg
        self._backend: Optional[MLLMBackend] = None
        self._lsconv = None
        self._calibrator: Optional[FeatureCalibrator] = None
        self._pcma: Optional[ProbabilisticAlignmentModule] = None
        self._sac_scorer: Optional[SACScorer] = None
        self._priors: Optional[Priors] = None
        self._engine: Optional[AdaptiveSelectionEngine] = None
        self._dataset: Optional[DatasetSplit] = None
        self._embeddings: Dict[str, np.ndarray] = {}  # calibrated, keyed by image_path
        self._train_candidates: List[Candidate] = []

    # Feature extraction

    def _extract_features(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and return (semantic, structural) feature arrays.

        Returns
        -------
        sem : (N, 4096) float32 -- raw InternVL visual embeddings
        str : (N, 64)   float32 -- LSConv structural embeddings
        """
        assert self._backend is not None
        assert self._lsconv is not None

        paths = [s.image_path for s in samples]
        logger.info("extracting semantic features for %d images ...", len(paths))
        sem = self._backend.extract_features(paths).astype(np.float32)  # (N, 4096)

        logger.info("extracting structural features ...")
        str_feats = extract_structural_features_from_paths(self._lsconv, paths)  # (N, 64) ndarray
        return sem, str_feats

    # Zero-shot validation inference (for prior estimation)

    def _run_zero_shot(self, samples: List[Sample]) -> List[Optional[str]]:
        """Run zero-shot classification on *samples* without any exemplars.

        Returns a list of predicted labels (same length as *samples*).
        Unparseable responses become ``None``.
        """
        assert self._backend is not None
        assert self._dataset is not None
        classes = self._dataset.classes

        predictions: List[Optional[str]] = []
        for i, sample in enumerate(samples):
            prompt, img_paths = build_classification_prompt(
                query_image_path=sample.image_path,
                exemplars=[],
                classes=classes,
                dataset_name=self._dataset.name,
            )
            response = self._backend.generate(prompt, img_paths)
            pred = parse_label(response, classes)
            predictions.append(pred)
            if (i + 1) % 50 == 0:
                n_parsed = sum(1 for p in predictions if p is not None)
                logger.info("zero-shot val: %d/%d done, %d parsed", i + 1, len(samples), n_parsed)
        return predictions

    # Candidate construction

    def _build_candidates(
        self,
        samples: List[Sample],
        pcma: ProbabilisticAlignmentModule,
        sac_scorer: SACScorer,
    ) -> List[Candidate]:
        """Build Candidate objects from training samples with calibrated embeddings."""
        candidates = []
        for sample in samples:
            emb = self._embeddings.get(sample.image_path)
            if emb is None:
                logger.warning("no embedding for %s -- skipping", sample.image_path)
                continue
            cand = Candidate(
                id=sample.image_path,
                image_path=sample.image_path,
                label=sample.label,
                embedding=emb,
                text_token_cost=text_token_cost_for_exemplar(sample.label),
            )
            candidates.append(cand)

        # Pre-compute intrinsic scores keyed by candidate id
        embs = np.stack([c.embedding for c in candidates])
        labels = [c.label for c in candidates]

        typ_scores = pcma.typicality_batch(embs, labels)
        disc_scores = sac_scorer.score_batch(embs, labels)

        typ_dict = {c.id: float(t) for c, t in zip(candidates, typ_scores)}
        disc_dict = {c.id: float(d) for c, d in zip(candidates, disc_scores)}

        # Store on engine (returned for caller to pass to AdaptiveSelectionEngine)
        self._typ_dict = typ_dict
        self._disc_dict = disc_dict
        return candidates

    # Setup (call once per dataset before evaluate)

    def setup(self, dataset: DatasetSplit) -> None:
        """Fit all modules. Must be called before evaluate().

        Parameters
        ----------
        dataset:
            Three-way split produced by :func:`~idea.dataset_loader.load_dataset`.
        """
        t0 = time.perf_counter()
        self._dataset = dataset
        cfg = self.cfg

        # 1. Backend
        self._backend = load_backend(
            cfg.model.model_path,
            device=cfg.experiment.device,
            torch_dtype=cfg.model.torch_dtype,
            load_in_8bit=cfg.model.load_in_8bit,
            mock_classes=dataset.classes,
        )

        # 2. LSConv backbone (pretrained LSNet-T, frozen)
        self._lsconv = build_lsconv_backbone_pretrained(
            device=cfg.experiment.device,
            dim=cfg.lsconv.dim,
        )

        # 3. Extract raw features for train + val
        all_samples = dataset.train + dataset.val
        sem, str_feats = self._extract_features(all_samples)

        # 4. Fit calibrator on train, transform all
        n_train = len(dataset.train)
        cal_cfg = CalibrationConfig(
            w_sem=cfg.calibration.w_sem,
            w_str=cfg.calibration.w_str,
            weight_mode=cfg.calibration.weight_mode,
            use_structural=cfg.calibration.use_structural,
            standardise=cfg.calibration.standardise,
        )
        self._calibrator = FeatureCalibrator(config=cal_cfg)
        self._calibrator.fit(sem[:n_train], str_feats[:n_train])
        calibrated = self._calibrator.transform(sem, str_feats)  # (N_train+N_val, D)

        # Store calibrated embeddings indexed by path (train + val)
        for sample, emb in zip(all_samples, calibrated):
            self._embeddings[sample.image_path] = emb.astype(np.float32)

        # 5. Fit PCMA on calibrated train features
        train_embs = calibrated[:n_train]
        train_labels = [s.label for s in dataset.train]
        pcma_cfg = PCMAConfig(
            sigma_sq=cfg.pcma.sigma_sq,
            min_variance=cfg.pcma.min_variance,
            shrinkage=cfg.pcma.shrinkage,
        )
        self._pcma = ProbabilisticAlignmentModule(config=pcma_cfg)
        self._pcma.fit(train_embs, train_labels)
        logger.info(self._pcma.summary())

        # 6. Zero-shot inference on VAL; estimate priors
        logger.info("running zero-shot inference on %d val samples ...", len(dataset.val))
        val_preds = self._run_zero_shot(dataset.val)
        val_true = [s.label for s in dataset.val]
        self._priors = estimate_priors(
            val_true, val_preds, dataset.classes,
            hard_class_threshold=cfg.priors.hard_class_threshold,
            hard_negative_k=cfg.priors.hard_negative_k,
            source_split="val",
        )
        self._priors.assert_no_test_contamination()
        logger.info(self._priors.summary())

        # 7. Fit SAC
        prototypes = ClassPrototypes.fit(train_embs, train_labels)
        hn_set = HardNegativeSet(hard_negatives=self._priors.hard_negatives)
        hn_set.fill_missing_from_prototypes(prototypes)
        sac_cfg = SACConfig(
            temperature=cfg.sac.temperature,
            hard_negative_k=cfg.sac.hard_negative_k,
            use_hard_negatives=cfg.sac.use_hard_negatives,
        )
        self._sac_scorer = SACScorer(
            prototypes=prototypes,
            hard_negative_set=hn_set,
            config=sac_cfg,
        )

        # 8. Build candidates and engine
        self._train_candidates = self._build_candidates(dataset.train, self._pcma, self._sac_scorer)
        weights = SelectionWeights(
            alpha=cfg.selection.alpha,
            beta=cfg.selection.beta,
            gamma=cfg.selection.gamma,
            delta=cfg.selection.delta,
            omega1=cfg.selection.omega1,
            omega2=cfg.selection.omega2,
        )
        self._engine = AdaptiveSelectionEngine(
            candidates=self._train_candidates,
            typicality=self._typ_dict,
            discriminativeness=self._disc_dict,
            hard_classes=set(self._priors.hard_classes),
            weights=weights,
        )
        logger.info("setup complete in %.1fs", time.perf_counter() - t0)

    # Evaluation

    def evaluate(self, dataset: Optional[DatasetSplit] = None) -> EvalResult:
        """Run IDEA selection + inference on test queries and return accuracy."""
        assert self._engine is not None, "call setup() first"
        ds = dataset or self._dataset
        assert ds is not None
        cfg = self.cfg

        t0 = time.perf_counter()
        results: List[QueryResult] = []

        for i, sample in enumerate(ds.test):
            t_q = time.perf_counter()
            # Get calibrated query embedding
            if sample.image_path not in self._embeddings:
                # Test images not yet extracted -- do it now
                sem, str_f = self._extract_features([sample])
                emb = self._calibrator.transform(sem, str_f)[0]
                self._embeddings[sample.image_path] = emb.astype(np.float32)
            q_emb = self._embeddings[sample.image_path]

            exemplars, _ = self._engine.select(
                query_embedding=q_emb,
                max_shots=cfg.experiment.shots,
                token_budget=cfg.experiment.token_budget,
            )
            prompt, img_paths = build_classification_prompt(
                query_image_path=sample.image_path,
                exemplars=exemplars,
                classes=ds.classes,
                dataset_name=ds.name,
            )
            response = self._backend.generate(prompt, img_paths)
            pred = parse_label(response, ds.classes)
            results.append(QueryResult(
                query_path=sample.image_path,
                true_label=sample.label,
                predicted_label=pred,
                n_exemplars=len(exemplars),
                elapsed_s=time.perf_counter() - t_q,
            ))
            if cfg.experiment.log_every > 0 and (i + 1) % cfg.experiment.log_every == 0:
                acc_so_far = sum(r.correct for r in results) / len(results)
                logger.info("eval IDEA: %d/%d  acc=%.4f", i + 1, len(ds.test), acc_so_far)

        return _make_eval_result(ds.name, "idea", results, time.perf_counter() - t0)

    def evaluate_baseline(
        self, method: str, dataset: Optional[DatasetSplit] = None
    ) -> EvalResult:
        """Run a named baseline against test queries.

        Parameters
        ----------
        method:
            ``"random"``, ``"cosine"``, ``"rices"``, or ``"diverse"``.
        """
        from .baselines import cosine_select, diverse_select, random_select, rices_select

        assert self._calibrator is not None, "call setup() first"
        ds = dataset or self._dataset
        assert ds is not None

        _selector = {
            "random": random_select,
            "cosine": cosine_select,
            "rices": rices_select,
            "diverse": diverse_select,
        }.get(method)
        if _selector is None:
            raise ValueError(f"unknown baseline method {method!r}")

        # Baselines use raw calibrated embeddings with no PCMA/SAC priors.
        t0 = time.perf_counter()
        results: List[QueryResult] = []
        for i, sample in enumerate(ds.test):
            t_q = time.perf_counter()
            if sample.image_path not in self._embeddings:
                sem, str_f = self._extract_features([sample])
                self._embeddings[sample.image_path] = self._calibrator.transform(sem, str_f)[0]
            q_emb = self._embeddings[sample.image_path]
            exemplars = _selector(
                q_emb, self._train_candidates,
                k=self.cfg.experiment.shots,
                token_budget=self.cfg.experiment.token_budget,
            )
            prompt, img_paths = build_classification_prompt(
                query_image_path=sample.image_path,
                exemplars=exemplars,
                classes=ds.classes,
                dataset_name=ds.name,
            )
            response = self._backend.generate(prompt, img_paths)
            pred = parse_label(response, ds.classes)
            results.append(QueryResult(
                query_path=sample.image_path,
                true_label=sample.label,
                predicted_label=pred,
                n_exemplars=len(exemplars),
                elapsed_s=time.perf_counter() - t_q,
            ))
        return _make_eval_result(ds.name, method, results, time.perf_counter() - t0)

    def run_all_methods(self, dataset: Optional[DatasetSplit] = None) -> Dict[str, EvalResult]:
        """Evaluate IDEA and all four baselines; return results keyed by method name."""
        idea = self.evaluate(dataset)
        baselines = {m: self.evaluate_baseline(m, dataset) for m in ("random", "cosine", "rices", "diverse")}
        return {"idea": idea, **baselines}
