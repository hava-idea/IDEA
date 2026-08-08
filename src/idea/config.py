"""Unified IDEA configuration.

All hyper-parameters are specified here, grouped by module. The dataclass
hierarchy mirrors the module layout, so any single default can be overridden
from a YAML file or a CLI flag without touching module code.

Usage example::

    from idea.config import IDEAConfig
    cfg = IDEAConfig()                   # paper defaults
    cfg = IDEAConfig.from_yaml("configs/ucmerced_20shot.yaml")

YAML schema (all fields optional; absent fields keep defaults)::

    dataset:
      name: ucmerced
      root: /data/UCMerced_LandUse
      seed: 42

    experiment:
      shots: 20
      token_budget: 4096
      device: cuda:0
      results_dir: ./results

    model:
      model_path: /weights/InternVL2_5-8B
      torch_dtype: bfloat16
      load_in_8bit: false

    pcma:
      sigma_sq: 100.0
      min_variance: 1.0e-6
      shrinkage: 0.0

    sac:
      temperature: 0.07
      hard_negative_k: 5
      use_hard_negatives: true

    calibration:
      w_sem: 0.7
      w_str: 0.3
      weight_mode: amplitude
      use_structural: true
      standardise: true

    lsconv:
      dim: 64

    selection:
      alpha: 0.4
      beta: 0.3
      gamma: 0.2
      delta: 0.3
      omega1: 0.2
      omega2: 0.15

    priors:
      hard_class_threshold: 0.5
      hard_negative_k: 5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str = "ucmerced"          # "ucmerced" or "rsscn7"
    root: str = ""                  # path to dataset root directory
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42


@dataclass
class ExperimentConfig:
    shots: int = 20                 # k (number of in-context examples per query)
    token_budget: int = 4096        # max total tokens per prompt (images + text)
    device: str = "cuda:0"
    results_dir: str = "./results"
    log_every: int = 50             # log accuracy every N queries


@dataclass
class ModelConfig:
    model_path: Optional[str] = None  # None => MockBackend (testing only)
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False


@dataclass
class PCMAConfig:
    sigma_sq: float = 100.0
    min_variance: float = 1e-6
    shrinkage: float = 0.0


@dataclass
class SACConfig:
    temperature: float = 0.07
    hard_negative_k: int = 5
    use_hard_negatives: bool = True


@dataclass
class CalibrationConfig:
    w_sem: float = 0.7
    w_str: float = 0.3
    weight_mode: str = "amplitude"  # "amplitude" or "similarity"
    use_structural: bool = True
    standardise: bool = True


@dataclass
class LSConvConfig:
    dim: int = 64                   # paper: Phi_str in R^64 (Eq. 6)


@dataclass
class SelectionConfig:
    alpha: float = 0.4              # S_sim weight
    beta: float = 0.3               # S_bal weight
    gamma: float = 0.2              # S_div weight
    delta: float = 0.3              # S_hard weight
    omega1: float = 0.2             # typicality sub-weight in S_sim
    omega2: float = 0.15            # discriminativeness sub-weight in S_sim


@dataclass
class PriorsConfig:
    hard_class_threshold: float = 0.5
    hard_negative_k: int = 5


@dataclass
class IDEAConfig:
    """Root configuration object.  All sub-configs use paper defaults."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pcma: PCMAConfig = field(default_factory=PCMAConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    lsconv: LSConvConfig = field(default_factory=LSConvConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "IDEAConfig":
        """Load configuration from a YAML file, overriding only supplied keys."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "IDEAConfig.from_yaml requires PyYAML: pip install pyyaml"
            ) from exc

        data: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        def _update(obj, updates: Dict[str, Any]) -> None:
            for key, value in updates.items():
                if hasattr(obj, key):
                    if isinstance(value, dict):
                        _update(getattr(obj, key), value)
                    else:
                        setattr(obj, key, value)
                else:
                    logger.warning("unknown config key %r in %s", key, path)

        cfg = cls()
        _update(cfg, data)
        logger.info("loaded config from %s", path)
        return cfg

    def to_yaml(self, path: str) -> None:
        """Persist the configuration as YAML for reproducibility."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "IDEAConfig.to_yaml requires PyYAML: pip install pyyaml"
            ) from exc
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.dump(_to_dict(self), default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("config written to %s", path)

    def summary(self) -> str:
        import dataclasses
        lines = ["IDEAConfig:"]
        for f in dataclasses.fields(self):
            sub = getattr(self, f.name)
            lines.append(f"  [{f.name}]")
            for sf in dataclasses.fields(sub):
                lines.append(f"    {sf.name} = {getattr(sub, sf.name)!r}")
        return "\n".join(lines)
