"""Structural calibration over frozen InternVL patch tokens.

Input
-----
P: pre-projector ViT patch tokens, shape ``(B, 1024, 1024)``.

Pipeline
--------
P
 -> fixed orthogonal projection 1024 -> 128
 -> reshape to (B, 128, 32, 32)
 -> two LSConv blocks initialized from the ImageNet-1K LSNet-T C=128 mixer
    and then frozen
 -> channel-average structural saliency
 -> softmax weighting with tau_s = 0.07
 -> weighted aggregation of the original ViT patch tokens
 -> L2-normalized f_str in R^1024

LSConv determines where to aggregate, while the original ViT tokens provide
the content that is aggregated.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VIT_DIM = 1024
PATCH_COUNT = 1024
STRUCTURAL_DIM = VIT_DIM

logger = logging.getLogger(__name__)


class Conv2dBN(nn.Sequential):
    """Convolution followed by batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class LargeKernelPerception(nn.Module):
    """Generate a local dynamic kernel from a large receptive field."""

    def __init__(
        self,
        dim: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if dim % groups:
            raise ValueError(f"dim ({dim}) must be divisible by groups ({groups})")
        hidden = dim // 2
        self.reduce = Conv2dBN(dim, hidden)
        self.depthwise = Conv2dBN(
            hidden,
            hidden,
            kernel_size=large_kernel,
            padding=large_kernel // 2,
            groups=hidden,
        )
        self.project = Conv2dBN(hidden, hidden)
        kernel_channels = small_kernel**2 * dim // groups
        self.to_kernel = nn.Conv2d(hidden, kernel_channels, kernel_size=1)
        self.norm = nn.GroupNorm(dim // groups, kernel_channels)
        self.dim = dim
        self.groups = groups
        self.small_kernel = small_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.reduce(x), inplace=True)
        hidden = F.relu(self.project(self.depthwise(hidden)), inplace=True)
        weights = self.norm(self.to_kernel(hidden))
        batch, _, height, width = weights.shape
        return weights.reshape(
            batch,
            self.dim // self.groups,
            self.small_kernel**2,
            height,
            width,
        )


class SmallKernelAggregation(nn.Module):
    """Apply per-location dynamic kernels to unfolded neighborhoods."""

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        num_groups = weights.shape[1]
        kernel = math.isqrt(weights.shape[2])
        if kernel * kernel != weights.shape[2]:
            raise ValueError("dynamic weight axis must describe a square kernel")
        unfolded = F.unfold(x, kernel_size=kernel, padding=kernel // 2)
        unfolded = unfolded.reshape(
            batch,
            num_groups,
            channels // num_groups,
            kernel * kernel,
            height,
            width,
        )
        weights = weights.reshape(
            batch, num_groups, 1, kernel * kernel, height, width
        )
        return (unfolded * weights).sum(dim=3).reshape(
            batch, channels, height, width
        )


class LSConvBlock(nn.Module):
    """One residual large-small convolution block."""

    def __init__(self, dim: int, groups: int = 8) -> None:
        super().__init__()
        self.lkp = LargeKernelPerception(dim, groups=groups)
        self.ska = SmallKernelAggregation()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn(self.ska(x, self.lkp(x)))


class PatchTokenLSConv(nn.Module):
    """Produce a structural representation by weighting original ViT tokens."""

    def __init__(
        self,
        vit_dim: int = VIT_DIM,
        proj_dim: int = 128,
        num_blocks: int = 2,
        tau_s: float = 0.07,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if vit_dim < proj_dim:
            raise ValueError("vit_dim must be at least proj_dim")
        if proj_dim % 8:
            raise ValueError("proj_dim must be divisible by 8")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_matrix = torch.randn(vit_dim, proj_dim, generator=generator)
        projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("R", projection)

        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            self.blocks = nn.Sequential(
                *[LSConvBlock(proj_dim) for _ in range(num_blocks)]
            )
        finally:
            torch.random.set_rng_state(rng_state)

        self.vit_dim = vit_dim
        self.proj_dim = proj_dim
        self.tau_s = tau_s

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must have shape (B, N, D), got {tuple(patch_tokens.shape)}"
            )
        batch, count, dim = patch_tokens.shape
        side = math.isqrt(count)
        if count != PATCH_COUNT or side * side != count or dim != self.vit_dim:
            raise ValueError(
                f"expected patch tokens (B, {PATCH_COUNT}, {self.vit_dim}), "
                f"got {tuple(patch_tokens.shape)}"
            )

        projected = patch_tokens @ self.R.to(dtype=patch_tokens.dtype)
        feature_map = projected.transpose(1, 2).reshape(
            batch, self.proj_dim, side, side
        )
        feature_map = self.blocks(feature_map)
        saliency = feature_map.mean(dim=1).flatten(1)
        alpha = torch.softmax(saliency / self.tau_s, dim=-1)
        structural = torch.sum(alpha.unsqueeze(-1) * patch_tokens, dim=1)
        return F.normalize(structural, dim=-1)


def build_patch_token_lsconv(
    *,
    device: Union[str, torch.device] = "cpu",
    vit_dim: int = VIT_DIM,
    proj_dim: int = 128,
    num_blocks: int = 2,
    tau_s: float = 0.07,
    seed: int = 42,
    checkpoint_path: Union[str, Path, None] = None,
) -> PatchTokenLSConv:
    """Build and freeze the structural module.

    ``checkpoint_path`` is required by the paper experiment path. Leaving it
    unset is supported only for isolated architecture tests.
    """
    model = PatchTokenLSConv(
        vit_dim=vit_dim,
        proj_dim=proj_dim,
        num_blocks=num_blocks,
        tau_s=tau_s,
        seed=seed,
    )
    if checkpoint_path is not None:
        load_lsnet_weights(model, checkpoint_path)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _unwrap_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("LSNet checkpoint must contain a state-dict mapping")
    for container_key in ("state_dict", "model", "model_state_dict"):
        nested = checkpoint.get(container_key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    return {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _canonical_block_key(key: str) -> str | None:
    """Map common checkpoint prefixes to this module's ``blocks.*`` keys."""
    while key.startswith(("module.", "model.", "backbone.")):
        key = key.split(".", 1)[1]
    if key.startswith("blocks."):
        return key
    marker = ".blocks."
    if marker in key:
        return "blocks." + key.split(marker, 1)[1]
    return None


def _official_lsnet_t_c128_key(key: str) -> str | None:
    """Translate the official LSNet-T C=128 mixer key to an LSConv suffix."""
    prefix = "blocks2.3.mixer."
    if not key.startswith(prefix):
        return None
    suffix = key[len(prefix) :]
    translations = (
        ("lkp.cv1.c.", "lkp.reduce.0."),
        ("lkp.cv1.bn.", "lkp.reduce.1."),
        ("lkp.cv2.c.", "lkp.depthwise.0."),
        ("lkp.cv2.bn.", "lkp.depthwise.1."),
        ("lkp.cv3.c.", "lkp.project.0."),
        ("lkp.cv3.bn.", "lkp.project.1."),
        ("lkp.cv4.", "lkp.to_kernel."),
        ("lkp.norm.", "lkp.norm."),
        ("bn.", "bn."),
    )
    for source_prefix, target_prefix in translations:
        if suffix.startswith(source_prefix):
            return target_prefix + suffix[len(source_prefix) :]
    return None


def load_lsnet_weights(
    model: PatchTokenLSConv,
    checkpoint_path: Union[str, Path],
) -> tuple[list[str], list[str]]:
    """Transfer the C=128 LSConv blocks from an ImageNet-1K LSNet-T checkpoint.

    Direct ``blocks.*`` checkpoints and the official
    ``blocks2.3.mixer.*`` layout are supported. Every transferred tensor must
    match by role and shape. The function fails if any LSConv parameter or
    normalization state remains unmatched, preventing an invalid
    random-initialized frozen model from running.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"LSNet-T checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only support
        checkpoint = torch.load(path, map_location="cpu")
    source = _unwrap_state_dict(checkpoint)
    target = model.state_dict()

    matched: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for source_key, tensor in source.items():
        target_key = _canonical_block_key(source_key)
        if target_key is None or target_key not in target:
            continue
        if target[target_key].shape != tensor.shape:
            shape_mismatches.append(
                f"{source_key}: checkpoint {tuple(tensor.shape)} != "
                f"model {tuple(target[target_key].shape)}"
            )
            continue
        matched[target_key] = tensor

    # Official LSNet-T stores the C=128 LSConv mixer at blocks2.3. The paper
    # uses two consecutive blocks at that matched width, so both inherit this
    # pretrained mixer state.
    for source_key, tensor in source.items():
        target_suffix = _official_lsnet_t_c128_key(source_key)
        if target_suffix is None:
            continue
        for block_index in range(len(model.blocks)):
            target_key = f"blocks.{block_index}.{target_suffix}"
            if target_key not in target:
                continue
            if target[target_key].shape != tensor.shape:
                shape_mismatches.append(
                    f"{source_key} -> {target_key}: checkpoint {tuple(tensor.shape)} "
                    f"!= model {tuple(target[target_key].shape)}"
                )
                continue
            matched[target_key] = tensor

    required_state = {
        name
        for name in target
        if name.startswith("blocks.") and not name.endswith("num_batches_tracked")
    }
    missing_state = sorted(required_state - matched.keys())
    if missing_state:
        details = ""
        if shape_mismatches:
            details = " Shape mismatches: " + "; ".join(shape_mismatches[:5])
        raise RuntimeError(
            "LSNet-T checkpoint did not cover the complete LSConv block state. "
            f"Missing: {', '.join(missing_state)}.{details}"
        )

    incompatible = model.load_state_dict(matched, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected LSNet-T block keys: {unexpected}")
    logger.info(
        "loaded %d C=%d LSConv tensors from ImageNet-1K LSNet-T checkpoint %s",
        len(matched),
        model.proj_dim,
        path,
    )
    return list(incompatible.missing_keys), unexpected


@torch.no_grad()
def extract_structural_features(
    model: PatchTokenLSConv,
    patch_tokens: torch.Tensor,
) -> torch.Tensor:
    """Run Structure Pooling on pre-projector InternVL patch tokens."""
    return model(patch_tokens.to(device=model.R.device, dtype=model.R.dtype)).float()
