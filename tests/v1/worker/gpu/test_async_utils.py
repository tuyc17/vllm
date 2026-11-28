# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from unittest.mock import Mock

import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu_model_runner import (
    AsyncGPUModelRunnerOutput,
    AsyncGPUPoolingModelRunnerOutput,
)


def _make_async_output(monkeypatch, check_ep_fault: bool) -> AsyncOutput:
    class _Stream:
        def wait_stream(self, _stream) -> None:
            pass

    class _Event:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "Stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: _Event())
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    inner = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.empty((0, 1), dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.empty(0, dtype=torch.int32),
        num_rejected=torch.empty(0, dtype=torch.int32),
    )
    return AsyncOutput(
        model_runner_output=inner,
        sampler_output=sampler_output,
        num_sampled_tokens=torch.empty(0, dtype=torch.int32),
        main_stream=torch.cuda.Stream(),
        copy_stream=torch.cuda.Stream(),
        check_ep_fault=check_ep_fault,
        pending_aux_output=None,
    )


def test_async_output_exposes_inner_output_through_public_api(monkeypatch) -> None:
    output = _make_async_output(monkeypatch, check_ep_fault=False)

    assert output.get_model_runner_output() is output.model_runner_output


def test_async_output_queries_ep_fault_when_enabled(monkeypatch) -> None:
    manager = Mock()
    manager.query_fault.return_value = torch.tensor([True])
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.async_utils.get_ep_all2all_manager",
        lambda: manager,
    )

    output = _make_async_output(monkeypatch, check_ep_fault=True)

    manager.query_fault.assert_called_once_with()
    assert output._has_fault is not None
    assert output._has_fault.item() is True


class _Stream:
    def wait_stream(self, _stream) -> None:
        pass


class _Event:
    def record(self, _stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _patch_cuda_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: _Event())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a, **_k: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())


def _make_v1_gpu_async_output(monkeypatch) -> AsyncGPUModelRunnerOutput:
    _patch_cuda_runtime(monkeypatch)
    return AsyncGPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        sampled_token_ids=torch.empty((0, 1), dtype=torch.int64),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=_Stream(),
        vocab_size=1,
    )


def test_v1_gpu_async_output_exposes_inner_output_through_public_api(
    monkeypatch,
) -> None:
    """Regression: the V1 wrapper stored only ``_model_runner_output`` while
    the shared accessor reads the public name, so unwrapping an async output
    raised AttributeError on the first sampled step."""
    output = _make_v1_gpu_async_output(monkeypatch)

    assert output.model_runner_output is output._model_runner_output
    assert output.get_model_runner_output() is output._model_runner_output


def test_v1_pooling_async_output_exposes_inner_output_through_public_api(
    monkeypatch,
) -> None:
    _patch_cuda_runtime(monkeypatch)
    output = AsyncGPUPoolingModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(req_ids=[], req_id_to_index={}),
        raw_pooler_output=[],
        finished_mask=[],
        async_output_copy_stream=_Stream(),
    )

    assert output.model_runner_output is output._model_runner_output
    assert output.get_model_runner_output() is output._model_runner_output
