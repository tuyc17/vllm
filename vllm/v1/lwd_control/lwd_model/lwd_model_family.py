# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model-family request metadata the edge extracts at admission.

Only the edge holds the real prompt. Model families whose cloud-side forward
reads more than the prompt embeddings get that extra input derived here, once
per request, in a form that exposes no text token ids: whole-prompt positions
for M-RoPE models; image placeholder runs (positions and public image ids)
for DeepSeek V4.1. The edge scheduler and engine use the family selected for
the model.

Families represent metadata contracts, not individual architectures. Any
`model_config.uses_mrope` model automatically uses `LwdMropeModelFamily`; no
family-specific registration is needed when its native model class exposes a
compatible `_get_mrope_input_positions(input_tokens, mm_features, config)`
and LWD supports its input modalities.

Models needing additional metadata should subclass the closest family, such
as `LwdMropeModelFamily`, and override only the differing hooks. Their
selector case goes before the generic M-RoPE case. New wire fields also
require matching notify, cloud validation, and consumer changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.lwd_logger import lwd_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.request import Request

    LwdMropePositionsFn = Callable[..., tuple[torch.Tensor, Any]]

_LWD_DEEPSEEK_V41_ARCHITECTURE = "DeepseekV41ForCausalLM"

# Width of the id range, starting at `image_token_id`, that the DeepSeek MoE
# router sends through the vision routing bias: the `+ 5` of the vLLM
# `_dsv4_topk_kernel` sentinel test and vllm-ascend's
# DEEPSEEK_V4_IMAGE_SENTINEL_COUNT; neither is importable from here.
_LWD_DEEPSEEK_IMAGE_ROUTING_IDS = 5


@dataclass(frozen=True)
class LwdRequestMetadata:
    """Whole-request values `LwdRequestNotify` carries for the model family;
    an empty field is not used by the family."""

    image_placeholder_spans: list[int] = field(default_factory=list)


class LwdModelFamily:
    """Default model family, which derives no additional request metadata."""

    def admit(self, request: Request) -> None:
        """Hook for retaining per-request metadata derived at admission."""

    def request_metadata(self, request: Request) -> LwdRequestMetadata:
        """Whole-request values announced with the request."""
        return LwdRequestMetadata()

    def has_mrope_positions(self, request_id: str) -> bool:
        """Whether the request carries M-RoPE positions. Read per range
        item, so it must not serialize them."""
        return False

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        """Raw int64 `[3, num_prompt_tokens]` M-RoPE positions of the whole
        prompt; None when the request carries none."""
        return None

    def release(self, request_id: str) -> None:
        """Release per-request metadata retained by `admit()`."""

    def held_request_ids(self) -> list[str]:
        """Request IDs whose admission metadata is still retained."""
        return []


class LwdMropeModelFamily(LwdModelFamily):
    """M-RoPE model family: whole-prompt positions for multimodal prompts.

    Positions need the real prompt and the image grids, so the edge computes
    them for the whole prompt at admission and the request notify carries
    them whole.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config
        # Resolved on the first multimodal request; model classes lacking it
        # fail fast there.
        self._positions_fn: LwdMropePositionsFn | None = None
        # Whole-prompt [3, N] CPU positions per multimodal request.
        self._positions: dict[str, torch.Tensor] = {}

    def admit(self, request: Request) -> None:
        if not _lwd_request_has_mm(request):
            return
        assert request.prompt_token_ids is not None
        mm_features = [f for f in request.mm_features if f.modality != "prompt_embeds"]
        positions, _delta = self._resolve_positions_fn(request)(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mm_features,
            config=self._vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._positions[request.request_id] = positions

    def _resolve_positions_fn(self, request: Request) -> LwdMropePositionsFn:
        if self._positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_config = self._vllm_config.model_config
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architectures, model_config
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions(input_tokens, mm_features, "
                    f"config); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._positions_fn = fn
        return self._positions_fn

    def has_mrope_positions(self, request_id: str) -> bool:
        return request_id in self._positions

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        positions = self._positions.get(request_id)
        if positions is None:
            return None
        return positions.contiguous().numpy().tobytes()

    def release(self, request_id: str) -> None:
        self._positions.pop(request_id, None)

    def held_request_ids(self) -> list[str]:
        return list(self._positions)


class LwdDeepseekV41ModelFamily(LwdModelFamily):
    """DeepSeek V4.1: image placeholder runs.

    The cloud's MoE vision routing and Engram image mask read image
    placeholder ids, so the edge sends their runs (public ids and positions,
    no text), once with the request notify.

    The first decode positions hash the prompt tail, whose ids the cloud also
    lacks, but the tail needs nothing on the wire: its positions are prompt
    positions, so their embeddings ride the UP payload like every other prompt
    row, and the cloud looks the ids back up from its own embedding table
    (`LwdEmbedLookup`).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._image_token_id: int = model_config.hf_config.image_token_id

    def request_metadata(self, request: Request) -> LwdRequestMetadata:
        assert request.prompt_token_ids is not None
        prompt = torch.tensor(request.prompt_token_ids, dtype=torch.int64)
        spans = self._image_placeholder_spans(prompt)
        return LwdRequestMetadata(image_placeholder_spans=spans)

    def _image_placeholder_spans(self, prompt: torch.Tensor) -> list[int]:
        """Flattened `(start, length, token_id)` of the maximal runs of equal
        ids in the image routing range.

        Scans the ids rather than the multimodal features: placeholder text
        written out in the prompt routes as image too.
        """
        is_image = (prompt >= self._image_token_id) & (
            prompt < self._image_token_id + _LWD_DEEPSEEK_IMAGE_ROUTING_IDS
        )
        if not bool(is_image.any()):
            return []
        values = torch.where(is_image, prompt, -1)
        run_begins = torch.ones_like(is_image)
        run_begins[1:] = values[1:] != values[:-1]
        run_starts = torch.nonzero(run_begins).flatten()
        run_ends = torch.cat((run_starts[1:], torch.tensor([prompt.numel()])))
        image_runs = is_image[run_starts]
        starts = run_starts[image_runs]
        spans = torch.stack(
            (starts, run_ends[image_runs] - starts, prompt[starts]), dim=1
        )
        return spans.flatten().tolist()


def _lwd_request_has_mm(request: Request) -> bool:
    """Whether the request carries real multimodal data (prompt_embeds
    passthrough does not count -- it has no grid metadata, mrope is
    handled as text positions, consistent with the native definition)."""
    return any(f.modality != "prompt_embeds" for f in request.mm_features)


def lwd_select_model_family(vllm_config: VllmConfig) -> LwdModelFamily:
    """The request metadata family of the served model."""
    model_config = vllm_config.model_config
    family: LwdModelFamily
    if model_config.architecture == _LWD_DEEPSEEK_V41_ARCHITECTURE:
        family = LwdDeepseekV41ModelFamily(model_config)
    elif model_config.uses_mrope:
        family = LwdMropeModelFamily(vllm_config)
    else:
        family = LwdModelFamily()
    lwd_logger.info("[LWD] request metadata family: %s", type(family).__name__)
    return family
