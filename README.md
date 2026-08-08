# IDEA: Integrated Distribution-aware Exemplar Adaptation

Official implementation of **IDEA** — a training-free many-shot in-context
learning framework for remote sensing image classification.

## Method overview

IDEA selects an ordered sequence of labelled exemplars for each query image
without any fine-tuning. It wraps InternVL2.5-8B and combines four components:

| Component | Role |
|-----------|------|
| **PCMA** | Probabilistic Calibration and Matching Alignment — typicality score via diagonal Gaussian manifold (Eq. 7-8) |
| **SAC** | Self-Supervised Alignment Contrast — discriminativeness score via hard-negative contrastive scoring (Eq. 9-10) |
| **LSConv** | Large-Small Convolution backbone — ImageNet-pretrained LSNet-T structural features Φ_str ∈ R^64 (Eq. 4-5) |
| **Adaptive Selection** | Iterative MMR balancing similarity, diversity, label balance, and hard-class coverage (Eq. 11-13) |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- InternVL2.5-8B checkpoint (download from HuggingFace: `OpenGVLab/InternVL2_5-8B`)
- LSNet-T checkpoint

See `requirements.txt` for the full dependency list.

## Installation

```bash
git clone https://github.com/hava-idea/IDEA.git
cd IDEA
pip install -e .
```

## Datasets

Download instructions are in [`docs/DATASETS.md`](docs/DATASETS.md).

Expected layout after download:

```
/data/
  UCMerced_LandUse/
    Images/
      agricultural/  airplane/  baseballdiamond/  ...
  RSSCN7/
    aGricultural/  fForest/  gGolf/  mMeadow/  pParking/  rResidential/  sSea/
```

## Quick start

```bash
# Evaluate IDEA on UCMerced (20 shots)
python scripts/run_experiment.py \
    --dataset ucmerced \
    --data-root /data/UCMerced_LandUse \
    --model-path /weights/InternVL2_5-8B \
    --shots 20 \
    --output-dir ./results

# Evaluate all methods
python scripts/run_experiment.py ... --all-methods
```

Or use the Python API:

```python
from idea.config import IDEAConfig
from idea.dataset_loader import load_dataset
from idea.experiment import IDEAExperiment

cfg = IDEAConfig.from_yaml("configs/ucmerced_20shot.yaml")
dataset = load_dataset("ucmerced", "/data/UCMerced_LandUse")
exp = IDEAExperiment(cfg)
exp.setup(dataset)
result = exp.evaluate(dataset)
print(result.summary())
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use synthetic random features and `MockBackend` — no GPU or dataset
downloads are needed.

## Citation

```bibtex
@article{idea,
  title   = {IDEA: Integrated Distribution-aware Exemplar Adaptation for
             Multimodal Many-Shot ICL in Remote Sensing},
  author  = {Feng, Yi and Gao, Silu and Yang, Aiping},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
