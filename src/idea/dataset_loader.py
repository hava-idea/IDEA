"""Remote sensing dataset loaders for UCMerced and RSSCN7.

Supported datasets
------------------
* **UCMerced_LandUse** -- 21 land-use classes, 100 images per class (2100 total).
  Expected layout: ``root/Images/<class>/<image>.tif``
* **RSSCN7** -- 7 scene categories, 400 images per class (2800 total).
  Expected layout: ``root/<class>/<image>.jpg``

Split protocol
--------------
All splits are created with a **seeded stratified shuffle**: the random seed is
fixed, a per-class shuffle is applied, then the first ``train_ratio`` fraction
becomes the candidate pool, the next ``val_ratio`` becomes validation, and the
rest is the test set. This ensures reproducible, class-balanced splits and must
never be changed once the paper's Table 2 numbers are committed.

Default ratios: 70 % train / 15 % val / 15 % test, matching the paper (§4.2).

Only the test split is used as queries; train is the candidate pool for FAISS
indexing and prior estimation; val is used for zero-shot prior estimation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class DatasetName(str, Enum):
    UCMERCED = "ucmerced"
    RSSCN7 = "rsscn7"


@dataclass
class Sample:
    """One image sample.

    Attributes
    ----------
    image_path: Absolute path to the image file.
    label:      Human-readable class name (lower-case, spaces replaced by ``_``).
    split:      One of ``"train"``, ``"val"``, ``"test"``.
    """

    image_path: str
    label: str
    split: str

    def __repr__(self) -> str:  # pragma: no cover
        name = Path(self.image_path).name
        return f"Sample({name!r}, label={self.label!r}, split={self.split!r})"


@dataclass
class DatasetSplit:
    """Three-way split of a remote sensing dataset.

    Attributes
    ----------
    name:    Dataset identifier.
    classes: Sorted list of class names.
    train:   Candidate pool (feature extraction, FAISS index, prior fitting).
    val:     Validation set (zero-shot prior estimation *only*).
    test:    Query set (final evaluation).
    seed:    The RNG seed used, recorded for reproducibility.
    """

    name: str
    classes: List[str]
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    seed: int

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def class_to_train(self) -> Dict[str, List[Sample]]:
        out: Dict[str, List[Sample]] = {c: [] for c in self.classes}
        for s in self.train:
            out[s.label].append(s)
        return out

    def summary(self) -> str:
        return (
            f"DatasetSplit({self.name}): "
            f"{len(self.train)} train / {len(self.val)} val / {len(self.test)} test "
            f"| {self.n_classes} classes | seed={self.seed}"
        )


def _normalise_label(raw: str) -> str:
    """Lower-case, collapse spaces/hyphens to underscores."""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _stratified_split(
    class_images: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """Produce reproducible per-class splits.

    Returns three lists of (path, label) tuples for train/val/test.
    """
    train_items: List[Tuple[Path, str]] = []
    val_items: List[Tuple[Path, str]] = []
    test_items: List[Tuple[Path, str]] = []

    rng = random.Random(seed)

    for label, paths in sorted(class_images.items()):
        paths_sorted = sorted(paths)  # deterministic before shuffle
        shuffled = paths_sorted[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Give any rounding remainder to test.
        n_test = n - n_train - n_val
        if n_test < 1:
            # Reduce val by one to guarantee at least one test sample.
            n_val -= 1
            n_test = 1

        for p in shuffled[:n_train]:
            train_items.append((p, label))
        for p in shuffled[n_train : n_train + n_val]:
            val_items.append((p, label))
        for p in shuffled[n_train + n_val :]:
            test_items.append((p, label))

    return train_items, val_items, test_items


def _to_samples(items: List[Tuple[Path, str]], split: str) -> List[Sample]:
    return [Sample(image_path=str(p), label=label, split=split) for p, label in items]


def load_ucmerced(
    root: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Load UCMerced LandUse dataset from ``root/Images/<class>/<file>.tif``.

    Parameters
    ----------
    root:
        Path to the ``UCMerced_LandUse`` directory.
    train_ratio, val_ratio:
        Fraction of each class assigned to train / val. The remainder is test.
    seed:
        RNG seed for the stratified shuffle.
    """
    images_dir = Path(root) / "Images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"UCMerced Images directory not found: {images_dir}\n"
            "Expected layout: <root>/Images/<class>/<image>.tif"
        )

    class_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _normalise_label(class_dir.name)
        files = sorted(class_dir.glob("*.tif"))
        if files:
            class_images[label] = files

    if not class_images:
        raise ValueError(f"No .tif images found under {images_dir}")

    logger.info(
        "UCMerced: found %d classes, %d total images",
        len(class_images),
        sum(len(v) for v in class_images.values()),
    )

    train_items, val_items, test_items = _stratified_split(
        class_images, train_ratio, val_ratio, seed
    )

    classes = sorted(class_images.keys())
    ds = DatasetSplit(
        name=DatasetName.UCMERCED.value,
        classes=classes,
        train=_to_samples(train_items, "train"),
        val=_to_samples(val_items, "val"),
        test=_to_samples(test_items, "test"),
        seed=seed,
    )
    logger.info(ds.summary())
    return ds


def load_rsscn7(
    root: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Load RSSCN7 dataset from ``root/<class>/<file>.jpg``.

    Parameters
    ----------
    root:
        Path to the ``RSSCN7`` directory.
    train_ratio, val_ratio:
        Fraction of each class assigned to train / val.
    seed:
        RNG seed for the stratified shuffle.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"RSSCN7 root directory not found: {root_path}")

    class_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _normalise_label(class_dir.name)
        files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if files:
            class_images[label] = files

    if not class_images:
        raise ValueError(f"No images found under {root_path}")

    logger.info(
        "RSSCN7: found %d classes, %d total images",
        len(class_images),
        sum(len(v) for v in class_images.values()),
    )

    train_items, val_items, test_items = _stratified_split(
        class_images, train_ratio, val_ratio, seed
    )

    classes = sorted(class_images.keys())
    ds = DatasetSplit(
        name=DatasetName.RSSCN7.value,
        classes=classes,
        train=_to_samples(train_items, "train"),
        val=_to_samples(val_items, "val"),
        test=_to_samples(test_items, "test"),
        seed=seed,
    )
    logger.info(ds.summary())
    return ds


def load_dataset(
    name: str,
    root: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Dispatch to the right loader by dataset name.

    Parameters
    ----------
    name:
        ``"ucmerced"`` or ``"rsscn7"``.
    root:
        Dataset root directory.
    train_ratio, val_ratio, seed:
        Passed through to the underlying loader.
    """
    name = name.lower().strip()
    if name == DatasetName.UCMERCED.value:
        return load_ucmerced(root, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    if name == DatasetName.RSSCN7.value:
        return load_rsscn7(root, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    raise ValueError(
        f"Unknown dataset {name!r}. Supported: "
        + ", ".join(e.value for e in DatasetName)
    )
