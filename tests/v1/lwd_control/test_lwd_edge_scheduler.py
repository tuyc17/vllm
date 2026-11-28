# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LwdEdgeScheduler 覆写契约测试: 覆写方法必须与原生 Scheduler 调用点的
返回元数一致, 锁死上游 sync 后覆写漏改的回归形态 (5 元组解包失败)。"""

from __future__ import annotations

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.request import Request


def _mm_request() -> Request:
    return Request(
        "mm-req",
        prompt_token_ids=[1] * 600,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-hash",
                mm_position=PlaceholderRange(offset=10, length=576),
            )
        ],
    )


def test_encoder_passthrough_matches_native_unpacking() -> None:
    """Regression: upstream #59942 extended the native call site to unpack
    five values (duplicate_encoder_inputs); the edge passthrough kept
    returning a 4-tuple, so every multimodal prefill crashed the engine
    core with "not enough values to unpack (expected 5, got 4)"."""
    request = _mm_request()
    assert request.has_encoder_inputs

    scheduler = object.__new__(LwdEdgeScheduler)
    (
        encoder_inputs_to_schedule,
        num_new_tokens,
        new_encoder_compute_budget,
        external_load_encoder_input,
        duplicate_encoder_inputs,
    ) = scheduler._try_schedule_encoder_inputs(
        request, 0, 600, 1024, shift_computed_tokens=1
    )

    assert encoder_inputs_to_schedule == []
    assert num_new_tokens == 600
    assert new_encoder_compute_budget == 1024
    assert external_load_encoder_input == []
    assert duplicate_encoder_inputs == []
