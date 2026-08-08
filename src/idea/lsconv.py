"""LSConv structural stream (Section 3.1, Eq. 3-5).

The structural stream complements the frozen MLLM's semantic embedding with
local texture and geometric structure. The backbone is an LSNet-T encoder
(Wang et al., 2025) pretrained on ImageNet-1K and frozen during all IDEA
inference — no gradients are ever computed on task data.

Architecture (feature extraction path)::

    x (B, 3, 224, 224)
      -> LSNet-T stage-1 feature extractor   (pretrained, frozen)
      -> global average pool -> flatten      (B, C_stage1)
      -> linear projection                   (B, 64)
      -> Phi_str (B, 64)

The output width is 64, matching ``Phi_str in R^64`` in Eq. 6.

Training-free operation
-----------------------
The backbone is initialised from the official ``lsnet_t`` ImageNet-1K
checkpoint (``jameslahm/lsnet`` on HuggingFace) and kept frozen. It
contributes genuine low-level texture and geometric priors — edge detectors,
spatial organisation, locality — that complement the MLLM's pooled semantic
embedding. The downstream
:class:`~idea.feature_calibration.FeatureCalibrator` standardises its output
using candidate-pool statistics so that uninformative channels are
down-weighted automatically.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Width of the structural embedding ``Phi_str`` (Eq. 6).
STRUCTURAL_DIM = 64


class Conv2dBN(nn.Sequential):
    """Conv2d followed by BatchNorm2d, with BN initialised to identity scaling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0.0)


class LargeKernelPerception(nn.Module):
    """LKP: generate per-location dynamic kernels from a large receptive field.

    Implements Eq. 4, ``W_l = GroupNorm(H_dw(F_{l-1}))``. The depthwise 7x7
    convolution supplies the wide receptive field that captures object
    skeletons and spatial organisation -- road grids, building clusters -- which
    tend to be blurred by the MLLM's pooled semantic embedding.

    Returns
    -------
    Tensor of shape ``(B, dim // groups, sks * sks, H, W)``: one ``sks x sks``
    kernel per channel-group per spatial location.
    """

    def __init__(self, dim: int, large_kernel: int = 7, small_kernel: int = 3, groups: int = 8) -> None:
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by groups ({groups})")

        hidden = dim // 2
        self.reduce = Conv2dBN(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.depthwise = Conv2dBN(
            hidden,
            hidden,
            kernel_size=large_kernel,
            padding=(large_kernel - 1) // 2,
            groups=hidden,
        )
        self.project = Conv2dBN(hidden, hidden)

        num_channels = small_kernel**2 * dim // groups
        self.to_kernel = nn.Conv2d(hidden, num_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=num_channels)

        self.dim = dim
        self.groups = groups
        self.small_kernel = small_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.reduce(x))
        h = self.act(self.project(self.depthwise(h)))
        weights = self.norm(self.to_kernel(h))
        batch, _, height, width = weights.shape
        return weights.view(
            batch,
            self.dim // self.groups,
            self.small_kernel**2,
            height,
            width,
        )


class SmallKernelAggregation(nn.Module):
    """SKA: modulate unfolded local neighbourhoods with the dynamic kernels.

    Implements the summation inside Eq. 5. ``unfold`` rearranges the input into
    sliding ``K x K`` blocks so each local region can be weighted by the kernel
    LKP produced for that location. Channels within a group share a kernel,
    which is what keeps the operation cheap.
    """

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        num_groups = weights.shape[1]
        kernel = int(round(weights.shape[2] ** 0.5))
        if kernel * kernel != weights.shape[2]:
            raise ValueError(f"dynamic weight axis {weights.shape[2]} is not a square kernel")
        if channels % num_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by weight groups ({num_groups})"
            )

        unfolded = F.unfold(x, kernel_size=kernel, padding=kernel // 2)
        unfolded = unfolded.view(batch, num_groups, channels // num_groups, kernel * kernel, height, width)
        weights = weights.view(batch, num_groups, 1, kernel * kernel, height, width)
        aggregated = (unfolded * weights).sum(dim=3)
        return aggregated.view(batch, channels, height, width)


class LSConvBlock(nn.Module):
    """One LSConv block: ``F_l = F_{l-1} + BN(SKA(F_{l-1}, LKP(F_{l-1})))``.

    Normalisation is applied only to the modulated branch before the residual
    addition, so the identity path stays linear and the original local response
    remains available alongside the texture-aware modulation (Eq. 5).
    """

    def __init__(
        self,
        dim: int,
        large_kernel: int = 7,
        small_kernel: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.lkp = LargeKernelPerception(dim, large_kernel, small_kernel, groups)
        self.ska = SmallKernelAggregation()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn(self.ska(x, self.lkp(x)))


class LSConvBackbone(nn.Module):
    """Structural encoder mapping ``(B, 3, H, W)`` to ``Phi_str`` in ``R^dim``."""

    def __init__(
        self,
        dim: int = STRUCTURAL_DIM,
        num_blocks: int = 2,
        large_kernel: int = 7,
        small_kernel: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.stem = nn.Sequential(
            Conv2dBN(3, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[LSConvBlock(dim, large_kernel, small_kernel, groups) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class PretrainedLSConvBackbone(nn.Module):
    """LSNet-T stage-1 feature extractor + linear projection to ``dim``.

    The LSNet-T backbone is loaded from the official ImageNet-1K checkpoint
    and kept entirely frozen. A single linear projection maps stage-1 output
    channels to ``dim`` (default 64). Use
    :func:`build_lsconv_backbone_pretrained` to construct this module.
    """

    def __init__(self, backbone: nn.Module, in_channels: int, dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, dim, bias=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[0]           # (B, C, H, W) — stage-1 maps
        pooled = self.pool(feats).flatten(1)  # (B, C)
        return self.proj(pooled)              # (B, dim)


def build_lsconv_backbone_pretrained(
    device: Union[str, torch.device] = "cpu",
    dim: int = STRUCTURAL_DIM,
) -> PretrainedLSConvBackbone:
    """Build the structural encoder from the official LSNet-T pretrained weights.

    Downloads (or uses the local cache of) the ``lsnet_t`` checkpoint from
    ``hf_hub:jameslahm/lsnet`` via timm. The backbone is frozen; no gradients
    are ever computed on task data.

    Parameters
    ----------
    device:
        Device to place the module on.
    dim:
        Width of ``Phi_str``. Defaults to 64 to match Eq. 6.

    Returns
    -------
    A :class:`PretrainedLSConvBackbone` in ``eval()`` mode with
    ``requires_grad=False`` on all parameters.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "build_lsconv_backbone_pretrained requires timm. "
            "Install with: pip install timm>=1.0.0"
        ) from exc

    backbone = timm.create_model(
        "hf_hub:jameslahm/lsnet_t",
        pretrained=True,
        features_only=True,
        out_indices=(0,),  # stage-1 only (stride 4)
    )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        stage1_channels = backbone(dummy)[0].shape[1]

    model = PretrainedLSConvBackbone(backbone, stage1_channels, dim)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LSConv backbone built: LSNet-T pretrained (ImageNet-1K), "
        "stage1_channels=%d, proj->%d, params=%d, device=%s (frozen)",
        stage1_channels,
        dim,
        num_params,
        device,
    )
    return model


def build_lsconv_backbone(
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    dim: int = STRUCTURAL_DIM,
    num_blocks: int = 2,
) -> LSConvBackbone:
    """Build a randomly initialised structural encoder (reference / ablation).

    .. note::
        The default pipeline uses :func:`build_lsconv_backbone_pretrained`.
        This function is retained for ablation studies.

    Parameters
    ----------
    seed:
        Seeds the random initialisation for reproducibility.
    device:
        Device to place the module on.
    dim:
        Width of ``Phi_str``. Defaults to 64 to match Eq. 6.
    num_blocks:
        Number of LSConv blocks.

    Returns
    -------
    A module in ``eval()`` mode with ``requires_grad=False`` on all parameters.
    """
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = LSConvBackbone(dim=dim, num_blocks=num_blocks)
    finally:
        torch.random.set_rng_state(generator_state)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LSConv backbone built: dim=%d, blocks=%d, params=%d, seed=%d, device=%s",
        dim, num_blocks, num_params, seed, device,
    )
    return model


@torch.no_grad()
def extract_structural_features(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen structural encoder on a batch of normalised images.

    Parameters
    ----------
    model:
        A backbone from :func:`build_lsconv_backbone_pretrained` (or
        :func:`build_lsconv_backbone` for ablation).
    images:
        ``(B, 3, H, W)`` float tensor, already resized and normalised.

    Returns
    -------
    ``(B, dim)`` float32 tensor.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B, 3, H, W) input, got shape {tuple(images.shape)}")

    device = next(model.parameters()).device
    target_dtype = next(model.parameters()).dtype
    return model(images.to(device=device, dtype=target_dtype)).float()


def extract_structural_features_from_paths(
    model: nn.Module,
    image_paths: Sequence[str],
) -> np.ndarray:
    """Load images from disk, preprocess, and extract structural features.

    This is the convenience entry-point used by the experiment pipeline.
    It handles PIL loading, resizing to 224×224, and ImageNet normalisation
    before calling :func:`extract_structural_features`.

    Parameters
    ----------
    model:
        A backbone produced by :func:`build_lsconv_backbone_pretrained`.
    image_paths:
        Sequence of local file paths.

    Returns
    -------
    ``(N, STRUCTURAL_DIM)`` float32 numpy array.
    """
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "extract_structural_features_from_paths requires torchvision and Pillow. "
            "Install with: pip install torchvision pillow"
        ) from exc

    transform = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    batch = torch.stack(tensors)  # (N, 3, 224, 224)
    feats = extract_structural_features(model, batch)  # (N, STRUCTURAL_DIM)
    return feats.cpu().numpy().astype(np.float32)
