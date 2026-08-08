"""Prompt construction for InternVL2.5-8B few-shot classification.

InternVL2.5 image token
-----------------------
Each image in the prompt must be represented by the literal string ``<image>``
in the text. This is the visual-token placeholder that InternVL2.5's tokeniser
replaces with the 256 projected visual tokens. Using any other placeholder
(e.g. ``[Image 1]``, ``[图像i]``, ``(image)``) causes the tokeniser to pass
those characters as plain text tokens, so the model never receives the visual
embedding for that example.

The number of ``<image>`` tokens in the prompt must exactly match the number of
paths passed to the backend's ``generate()`` call, and the order must match.

Token cost estimation
---------------------
The selection engine enforces a token budget over both image tokens (256 each)
and text tokens. ``text_token_cost`` below estimates the text portion for one
exemplar line using the 4-chars-per-token approximation; this is stored on each
``Candidate`` so the engine can make incremental budget checks without
constructing the full prompt.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Literal placeholder used by InternVL2.5's tokeniser.
IMAGE_PLACEHOLDER = "<image>"

#: Conservative chars-per-token approximation for English text.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count via character count (no tokeniser needed at build time)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def build_classification_prompt(
    query_image_path: str,
    exemplars: Sequence,            # Sequence[Candidate]
    classes: Sequence[str],
    dataset_name: str = "remote sensing",
) -> Tuple[str, List[str]]:
    """Build a multi-image few-shot classification prompt.

    Parameters
    ----------
    query_image_path:
        Path to the query image. Its ``<image>`` token appears last.
    exemplars:
        Ordered list of :class:`~idea.selection.Candidate` objects to use as
        in-context examples. Empty list produces a zero-shot prompt.
    classes:
        Full class list for the dataset, used to build the label enumeration.
    dataset_name:
        Short description of the domain, used in the task description line.

    Returns
    -------
    prompt:
        Text string containing exactly ``len(exemplars) + 1`` occurrences of
        ``<image>`` (examples first, query last).
    image_paths:
        Ordered list of image paths in the same order as the ``<image>``
        placeholders. Pass this to :meth:`~idea.mllm_backend.MLLMBackend.generate`.
    """
    image_paths: List[str] = []
    lines: List[str] = []

    # --- Preamble -------------------------------------------------------
    class_list_str = ", ".join(classes)
    lines.append(
        f"You are an expert in {dataset_name} image classification. "
        "I will show you some labelled examples followed by a query image. "
        "Answer with exactly one label from the list below and nothing else."
    )
    lines.append(f"Categories: {class_list_str}")
    lines.append("")

    # --- In-context examples --------------------------------------------
    if exemplars:
        lines.append("Examples:")
        for i, ex in enumerate(exemplars, start=1):
            lines.append(f"{IMAGE_PLACEHOLDER}")
            lines.append(f"Label: {ex.label}")
            image_paths.append(ex.image_path)
        lines.append("")

    # --- Query ----------------------------------------------------------
    lines.append("Now classify the following image.")
    lines.append(IMAGE_PLACEHOLDER)
    lines.append("Label:")
    image_paths.append(query_image_path)

    prompt = "\n".join(lines)
    assert prompt.count(IMAGE_PLACEHOLDER) == len(image_paths), (
        "IMAGE_PLACEHOLDER count does not match image_paths length — "
        "this would cause an InternVL tokenisation error."
    )
    return prompt, image_paths


def text_token_cost_for_exemplar(label: str) -> int:
    """Estimate text token cost for a single in-context example.

    This is stored on each ``Candidate`` at index-build time so the selection
    engine can perform incremental budget checks without constructing the full
    prompt string.

    Accounts for the label line (``"Label: <label>\\n"``) and a blank line.
    The ``<image>`` placeholder itself is replaced by 256 visual tokens (tracked
    separately as ``IMAGE_TOKEN_COST``).
    """
    label_line = f"Label: {label}\n\n"
    return _estimate_tokens(label_line)


def parse_label(response: str, classes: Sequence[str]) -> Optional[str]:
    """Extract the predicted class label from the model's free-text response.

    Strategy (in order of preference):
    1. Exact match against the class list (case-insensitive, strip punctuation).
    2. Prefix match -- response starts with a class name.
    3. Substring match -- response contains a class name.
    4. Return ``None`` if no class can be identified.

    Parameters
    ----------
    response:
        Raw text returned by :meth:`~idea.mllm_backend.MLLMBackend.generate`.
    classes:
        Full class list.

    Returns
    -------
    The matched class label (in original casing from *classes*), or ``None``.
    """
    if not response:
        return None

    first_line = response.strip().split("\n")[0].strip()
    cleaned = first_line.lower().strip(".,:;!?\"' ")

    # Build a lookup: normalised form -> original label
    norm: Dict[str, str] = {}
    for label in classes:
        key = label.lower().strip().replace("-", "_").replace(" ", "_")
        norm[key] = label
        # Also map without underscores for fuzzy matching
        norm[key.replace("_", "")] = label

    # 1. Exact match on cleaned first line
    cleaned_key = cleaned.replace(" ", "_").replace("-", "_")
    if cleaned_key in norm:
        return norm[cleaned_key]
    if cleaned.replace(" ", "") in norm:
        return norm[cleaned.replace(" ", "")]

    # 2. Prefix match
    for key, label in sorted(norm.items(), key=lambda kv: -len(kv[0])):
        if cleaned_key.startswith(key) or cleaned.replace(" ", "").startswith(key):
            return label

    # 3. Substring match (longest match wins to avoid partial overlaps)
    matches = [
        (key, label) for key, label in norm.items() if key in cleaned_key
    ]
    if matches:
        best_key, best_label = max(matches, key=lambda kv: len(kv[0]))
        return best_label

    logger.debug("could not parse label from response: %r", response[:100])
    return None
