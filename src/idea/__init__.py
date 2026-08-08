"""IDEA package public API."""

from .baselines import cosine_select, diverse_select, random_select, rices_select
from .config import IDEAConfig
from .dataset_loader import DatasetSplit, Sample, load_dataset
from .experiment import EvalResult, IDEAExperiment, QueryResult
from .feature_calibration import FeatureCalibrator
from .mllm_backend import InternVLBackend, MockBackend, load_backend
from .probabilistic_alignment import ProbabilisticAlignmentModule
from .priors import Priors, estimate_priors
from .prompt_builder import build_classification_prompt, parse_label
from .self_supervised_alignment import SACScorer
from .selection import AdaptiveSelectionEngine, Candidate

__all__ = [
    "IDEAConfig",
    "IDEAExperiment",
    "EvalResult",
    "QueryResult",
    "DatasetSplit",
    "Sample",
    "load_dataset",
    "FeatureCalibrator",
    "ProbabilisticAlignmentModule",
    "SACScorer",
    "AdaptiveSelectionEngine",
    "Candidate",
    "Priors",
    "estimate_priors",
    "build_classification_prompt",
    "parse_label",
    "load_backend",
    "InternVLBackend",
    "MockBackend",
    "random_select",
    "cosine_select",
    "rices_select",
    "diverse_select",
]
