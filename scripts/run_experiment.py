"""Command-line runner for IDEA experiments.

Usage
-----
Evaluate IDEA on UCMerced (20 shots, default config)::

    python scripts/run_experiment.py \\
        --dataset ucmerced \\
        --data-root /data/UCMerced_LandUse \\
        --model-path /weights/InternVL2_5-8B \\
        --shots 20 \\
        --output-dir ./results

Evaluate all methods (IDEA + 4 baselines)::

    python scripts/run_experiment.py ... --all-methods

Use a custom config YAML (all CLI flags override the YAML)::

    python scripts/run_experiment.py --config configs/ucmerced_20shot.yaml --shots 31
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from the repo root without `pip install`.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from idea.config import IDEAConfig
from idea.dataset_loader import load_dataset
from idea.experiment import IDEAExperiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run IDEA exemplar-selection experiments on remote sensing datasets."
    )
    p.add_argument("--config", default=None, help="Path to a YAML config file (optional).")
    p.add_argument("--dataset", default=None, choices=["ucmerced", "rsscn7"])
    p.add_argument("--data-root", default=None, help="Dataset root directory.")
    p.add_argument("--model-path", default=None,
                   help="InternVL2_5-8B checkpoint directory (omit to use MockBackend).")
    p.add_argument("--shots", type=int, default=None, help="Number of in-context examples.")
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Dataset split seed.")
    p.add_argument("--device", default=None, help="e.g. cuda:0")
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--all-methods", action="store_true",
                   help="Evaluate IDEA + all four baselines.")
    p.add_argument("--save-config", action="store_true",
                   help="Write resolved config YAML to output-dir/config.yaml.")
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    # Load base config from YAML (if provided) then apply CLI overrides.
    cfg = IDEAConfig.from_yaml(args.config) if args.config else IDEAConfig()
    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.data_root:
        cfg.dataset.root = args.data_root
    if args.model_path:
        cfg.model.model_path = args.model_path
    if args.shots is not None:
        cfg.experiment.shots = args.shots
    if args.token_budget is not None:
        cfg.experiment.token_budget = args.token_budget
    if args.seed is not None:
        cfg.dataset.seed = args.seed
    if args.device:
        cfg.experiment.device = args.device
    cfg.experiment.results_dir = args.output_dir

    if not cfg.dataset.root:
        print("error: --data-root is required", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        cfg.to_yaml(str(out_dir / "config.yaml"))

    print(cfg.summary())

    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.root,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.dataset.seed,
    )

    exp = IDEAExperiment(cfg)
    exp.setup(dataset)

    if args.all_methods:
        all_results = exp.run_all_methods(dataset)
        for method, result in all_results.items():
            print(result.summary())
            result.to_json(str(out_dir / f"{dataset.name}_{method}_results.json"))
        # Print comparison table
        print("\n=== Summary ===")
        for method, r in sorted(all_results.items(), key=lambda kv: -kv[1].accuracy):
            print(f"  {method:<10s}  {r.accuracy:.4f}")
    else:
        result = exp.evaluate(dataset)
        print(result.summary())
        result.to_json(str(out_dir / f"{dataset.name}_idea_results.json"))


if __name__ == "__main__":
    main()
