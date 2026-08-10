"""MLLM backend abstraction (InternVL2.5-8B and a mock for testing).

This module isolates all model-weight dependencies behind a single interface:
:class:`MLLMBackend`. Tests and CI that cannot load the 8-B parameter checkpoint
use :class:`MockBackend` instead; the rest of the pipeline is identical.

Feature extraction
------------------
InternVL2.5-8B processes a 448×448 image as follows:

1. InternViT-300M-448px encodes the image as 32×32 = 1024 patch tokens.
2. Pixel shuffle with factor 0.5 reduces to 16×16 = 256 visual tokens per tile.
3. The visual tokens are projected into the language model's embedding space
   (hidden size 4096 for InternLM2_5-7B-chat).
4. We mean-pool over the 256 token positions → one R^4096 vector per image.

The resulting ``Phi_sem`` is fed to :class:`~idea.feature_calibration.FeatureCalibrator`.

Prompt and inference
--------------------
Context construction uses ``<image>`` as the per-image placeholder, which is the
token InternVL2.5's tokeniser replaces with the visual token sequence. Using a
literal text string such as ``[Image 1]`` or ``[图像i]`` bypasses the visual
tokeniser and sends only text tokens, so the model never sees the image. The
prompt builder in :mod:`idea.prompt_builder` uses ``<image>`` exclusively.

The classification response is a free-text label from the class list. No special
tokens or post-processing are needed; the model is prompted in zero/few-shot
classification format and the label is extracted from its first line.
"""

from __future__ import annotations

import abc
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: InternVL2.5-8B semantic embedding dimension (InternLM2_5-7B hidden size).
SEMANTIC_DIM = 4096

#: Visual tokens per 448×448 tile after pixel-shuffle (32×32 -> 0.5 -> 16×16).
VISUAL_TOKENS_PER_IMAGE = 256
PATCH_TOKENS_PER_IMAGE = 1024
PATCH_TOKEN_DIM = 1024


@dataclass(frozen=True)
class VisualFeatures:
    """Both branches produced by one frozen InternVL visual forward pass."""

    patch_tokens: np.ndarray
    """Pre-projector ViT tokens, shape ``(B, 1024, 1024)``."""
    visual_tokens: np.ndarray
    """Post-pixel-shuffle/projector tokens, shape ``(B, 256, 4096)``."""
    semantic: np.ndarray
    """Mean-pooled visual tokens, shape ``(B, 4096)``."""


class MLLMBackend(abc.ABC):
    """Abstract interface for the MLLM.

    Implementors must provide:
    * ``extract_features`` -- image path(s) -> ``Phi_sem`` vectors;
    * ``generate`` -- prompt + image paths -> free-text response.
    """

    @abc.abstractmethod
    def extract_visual_features(
        self, image_paths: Sequence[str]
    ) -> VisualFeatures:
        """Extract patch and semantic streams with one visual forward per image."""

    def extract_features(self, image_paths: Sequence[str]) -> np.ndarray:
        """Extract semantic embeddings for a batch of images.

        Parameters
        ----------
        image_paths:
            Local paths to 448×448-compatible images. Preprocessing (resize,
            normalise) is handled internally by the backend.

        Returns
        -------
        ``(N, SEMANTIC_DIM)`` float32 array -- ``Phi_sem`` for each image, NOT
        yet L2-normalised (normalisation happens inside the calibrator).
        """
        return self.extract_visual_features(image_paths).semantic

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str],
        max_new_tokens: int = 100,
    ) -> str:
        """Run the MLLM on a multi-image prompt.

        Parameters
        ----------
        prompt:
            Text prompt containing one ``<image>`` placeholder per image in
            ``image_paths``.
        image_paths:
            Paths in the same order as the ``<image>`` placeholders.
        max_new_tokens:
            Generation budget in tokens. Keep this small for classification
            (the answer is a short label).

        Returns
        -------
        The model's decoded output text.
        """

    @property
    @abc.abstractmethod
    def semantic_dim(self) -> int:
        """Dimensionality of the output of ``extract_features``."""


class InternVLBackend(MLLMBackend):
    """InternVL2.5-8B backend using the HuggingFace InternVL-main repo.

    Parameters
    ----------
    model_path:
        Local path to the InternVL2_5-8B checkpoint directory
        (contains ``config.json``, ``tokenizer.model``, and weight shards).
    device:
        PyTorch device string, e.g. ``"cuda:0"``.
    torch_dtype:
        Weight dtype. ``"bfloat16"`` is recommended; ``"float16"`` also works.
    load_in_8bit:
        Whether to use LLM.int8() quantisation. Reduces VRAM by ~4 GB at a
        small accuracy cost.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "InternVLBackend requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:
            raise ImportError(
                "InternVLBackend requires torchvision. "
                "Install with: pip install torchvision"
            ) from exc

        self._torch = torch
        self._T = T
        self._InterpolationMode = InterpolationMode

        dtype = getattr(torch, torch_dtype)
        logger.info(
            "loading InternVL2.5-8B from %s (device=%s, dtype=%s, int8=%s)",
            model_path,
            device,
            torch_dtype,
            load_in_8bit,
        )
        self._model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self._device = device
        self._dtype = dtype
        logger.info("InternVL2.5-8B loaded successfully")

    def _preprocess(self, image_path: str) -> "torch.Tensor":
        """Load and normalise one image to (1, 3, 448, 448) float."""
        from PIL import Image

        transform = self._T.Compose(
            [
                self._T.Resize(
                    (448, 448), interpolation=self._InterpolationMode.BICUBIC
                ),
                self._T.ToTensor(),
                self._T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0).to(self._device, dtype=self._dtype)

    def extract_visual_features(
        self, image_paths: Sequence[str]
    ) -> VisualFeatures:
        import torch

        patch_batches = []
        visual_batches = []
        semantic_batches = []
        with torch.no_grad():
            for path in image_paths:
                pixel_values = self._preprocess(path)
                output_hidden_states = getattr(self._model, "select_layer", -1) != -1
                vision_output = self._model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                if output_hidden_states:
                    vit_tokens = vision_output.hidden_states[self._model.select_layer]
                else:
                    vit_tokens = vision_output.last_hidden_state

                # InternVL's first vision token is CLS. The remaining 32x32
                # patch tokens feed both branches below.
                patch_tokens = vit_tokens[:, 1:, :]
                batch, count, dim = patch_tokens.shape
                if (count, dim) != (PATCH_TOKENS_PER_IMAGE, PATCH_TOKEN_DIM):
                    raise ValueError(
                        "InternVL checkpoint is incompatible with the paper path: "
                        f"expected patch tokens (*, 1024, 1024), got {tuple(patch_tokens.shape)}"
                    )

                side = int(count**0.5)
                shuffled = patch_tokens.reshape(batch, side, side, dim)
                shuffled = self._model.pixel_shuffle(
                    shuffled, scale_factor=self._model.downsample_ratio
                )
                shuffled = shuffled.reshape(batch, -1, shuffled.shape[-1])
                visual_tokens = self._model.mlp1(shuffled)
                if tuple(visual_tokens.shape[1:]) != (
                    VISUAL_TOKENS_PER_IMAGE,
                    SEMANTIC_DIM,
                ):
                    raise ValueError(
                        "InternVL checkpoint is incompatible with the paper path: "
                        f"expected visual tokens (*, 256, 4096), got {tuple(visual_tokens.shape)}"
                    )

                patch_batches.append(patch_tokens.float().cpu().numpy())
                visual_batches.append(visual_tokens.float().cpu().numpy())
                semantic_batches.append(
                    visual_tokens.mean(dim=1).float().cpu().numpy()
                )

        return VisualFeatures(
            patch_tokens=np.concatenate(patch_batches, axis=0),
            visual_tokens=np.concatenate(visual_batches, axis=0),
            semantic=np.concatenate(semantic_batches, axis=0),
        )

    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str],
        max_new_tokens: int = 100,
    ) -> str:
        import torch
        from PIL import Image

        pixel_values_list = [self._preprocess(p) for p in image_paths]
        pixel_values = torch.cat(pixel_values_list, dim=0)  # (N, 3, 448, 448)

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        with torch.no_grad():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        return response if isinstance(response, str) else str(response)

    @property
    def semantic_dim(self) -> int:
        return SEMANTIC_DIM


class MockBackend(MLLMBackend):
    """Deterministic fake backend for unit tests and CI without GPU/weights.

    Features are generated by hashing (image_path + random seed) so they are
    consistent across calls: the same path always yields the same vector.
    Classification always returns a random class from the provided list.

    The mock is *not* supposed to produce meaningful results -- it exists so
    the rest of the pipeline can be tested for correctness (data splits,
    calibration arithmetic, selection logic, no-leakage guard) without a GPU.
    """

    def __init__(
        self,
        dim: int = SEMANTIC_DIM,
        seed: int = 0,
        classes: Optional[Sequence[str]] = None,
    ) -> None:
        self._dim = dim
        self._seed = seed
        self._cache: Dict[str, np.ndarray] = {}
        self._classes = list(classes) if classes is not None else []
        logger.info(
            "MockBackend initialised (dim=%d, seed=%d); "
            "suitable for testing only",
            dim,
            seed,
        )

    def _get_or_make(self, path: str) -> np.ndarray:
        if path not in self._cache:
            # Seed derived from the path string so the result is reproducible
            # across calls even when paths arrive in different order.
            digest = hashlib.sha256(f"{self._seed}:{path}".encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(h)
            vec = rng.standard_normal(self._dim).astype(np.float32)
            self._cache[path] = vec
        return self._cache[path]

    def extract_visual_features(
        self, image_paths: Sequence[str]
    ) -> VisualFeatures:
        semantics = np.stack([self._get_or_make(p) for p in image_paths], axis=0)
        patches = []
        for path in image_paths:
            digest = hashlib.sha256(
                f"{self._seed}:patch:{path}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            patches.append(
                rng.standard_normal(
                    (PATCH_TOKENS_PER_IMAGE, PATCH_TOKEN_DIM), dtype=np.float32
                )
            )
        patch_tokens = np.stack(patches, axis=0)
        # The mock does not emulate InternVL's learned projector. It preserves
        # the public shapes and deterministic behavior needed by pipeline tests.
        visual_tokens = np.repeat(
            semantics[:, None, :], VISUAL_TOKENS_PER_IMAGE, axis=1
        )
        return VisualFeatures(patch_tokens, visual_tokens, semantics)

    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str],
        max_new_tokens: int = 100,
    ) -> str:
        if not self._classes:
            return "unknown"
        key = image_paths[-1] if image_paths else prompt
        digest = hashlib.sha256(f"{self._seed}:{key}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "little") % len(self._classes)
        return self._classes[idx]

    @property
    def semantic_dim(self) -> int:
        return self._dim


def load_backend(
    model_path: Optional[str],
    *,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    load_in_8bit: bool = False,
    mock_classes: Optional[Sequence[str]] = None,
    mock_seed: int = 0,
) -> MLLMBackend:
    """Factory: return :class:`InternVLBackend` if ``model_path`` is given, else mock.

    Parameters
    ----------
    model_path:
        Path to an InternVL2_5-8B checkpoint directory. ``None`` returns the
        mock backend instead, which is appropriate for testing.
    device, torch_dtype, load_in_8bit:
        Forwarded to :class:`InternVLBackend`.
    mock_classes:
        Class list for the mock backend's ``generate()``.
    mock_seed:
        RNG seed for the mock backend.
    """
    if model_path is not None:
        return InternVLBackend(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
        )
    logger.warning(
        "model_path is None; using MockBackend (testing only, not for reported results)"
    )
    return MockBackend(seed=mock_seed, classes=mock_classes)
