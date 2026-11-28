# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import PreTrainedConfig

from vllm.config import (
    CUDAGraphMode,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.v1.worker.gpu.lwd.model_runner import (
    LwdCloudModelRunnerV2,
    _validate_lwd_v2_config,
)
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelState
from vllm.v1.worker.gpu.lwd.mtp_speculator import LwdCloudMTPSpeculator
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.worker_base import WorkerBase


class _EmbeddingModel(torch.nn.Module):
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.bfloat16)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


class _CustomStateModel(_EmbeddingModel):
    def get_model_state_cls(self) -> type[DefaultModelState]:
        return DefaultModelState


def _model_config(tmp_path, model_type: str = "qwen3") -> ModelConfig:
    config_cls = type("_TestConfig", (PreTrainedConfig,), {"model_type": model_type})
    hf_config = config_cls(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=2,
        vocab_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    hf_config.save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), enable_prompt_embeds=True, max_model_len=16)


def _dspark_config(tmp_path) -> VllmConfig:
    speculative_config = SpeculativeConfig(
        method="draft_model",
        target_model_config=_model_config(tmp_path),
        target_parallel_config=ParallelConfig(),
        model=str(tmp_path),
        num_speculative_tokens=2,
    )
    speculative_config.method = "dspark"
    return VllmConfig(
        model_config=_model_config(tmp_path),
        speculative_config=speculative_config,
    )


@pytest.fixture
def distributed_runtime(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29519")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(backend="gloo")
        initialize_model_parallel()
        yield
        destroy_model_parallel()


def test_lwd_v2_rejects_deferred_context_parallelism() -> None:
    config = VllmConfig()
    config.parallel_config.prefill_context_parallel_size = 2

    with pytest.raises(ValueError, match="PCP support is deferred"):
        _validate_lwd_v2_config(config)


def test_lwd_v2_rejects_batch_sharded_sampling() -> None:
    config = VllmConfig()
    config.parallel_config.enable_batch_sharded_sampling = True

    with pytest.raises(ValueError, match="batch-sharded sampling"):
        _validate_lwd_v2_config(config)


def test_lwd_v2_engram_allows_eager(tmp_path) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    config.model_config.hf_text_config.engram_layer_ids = [1]
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    _validate_lwd_v2_config(config)


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_lwd_v2_engram_allows_cudagraphs(tmp_path, mode: CUDAGraphMode) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    config.model_config.hf_text_config.engram_layer_ids = [1]
    config.compilation_config.cudagraph_mode = mode

    _validate_lwd_v2_config(config)


def test_lwd_v2_allows_eager_dspark(tmp_path) -> None:
    config = _dspark_config(tmp_path)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    _validate_lwd_v2_config(config)


def test_lwd_v2_dspark_allows_cudagraphs(tmp_path) -> None:
    config = _dspark_config(tmp_path)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    _validate_lwd_v2_config(config)


def test_lwd_v2_rejects_dspark_adaptive_verification(tmp_path) -> None:
    config = _dspark_config(tmp_path)
    config.speculative_config.enable_adaptive_verification = True

    with pytest.raises(ValueError, match="adaptive verification is deferred"):
        _validate_lwd_v2_config(config)


def test_lwd_runner_selects_default_lwd_state(tmp_path, distributed_runtime) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    runner = LwdCloudModelRunnerV2(config, torch.device("cpu"))
    runner.model = _EmbeddingModel()

    state = runner.create_model_state()

    assert type(state) is LwdCloudModelState


def test_lwd_runner_maps_custom_default_state(tmp_path, distributed_runtime) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    runner = LwdCloudModelRunnerV2(config, torch.device("cpu"))
    runner.model = _CustomStateModel()

    state = runner.create_model_state()

    assert type(state) is LwdCloudModelState


def test_lwd_runner_selects_lwd_mtp_speculator(tmp_path, distributed_runtime) -> None:
    speculative_config = SpeculativeConfig(
        method="draft_model",
        target_model_config=_model_config(tmp_path),
        target_parallel_config=ParallelConfig(),
        model=str(tmp_path),
        num_speculative_tokens=1,
    )
    speculative_config.method = "mtp"
    config = VllmConfig(
        model_config=_model_config(tmp_path),
        speculative_config=speculative_config,
    )

    runner = LwdCloudModelRunnerV2(config, torch.device("cpu"))

    assert isinstance(runner.speculator, LwdCloudMTPSpeculator)


def test_lwd_runner_binds_prompt_stage_with_torch_global_rank(
    tmp_path, distributed_runtime
) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    runner = LwdCloudModelRunnerV2(config, torch.device("cpu"))
    runner.model = _EmbeddingModel()
    runner.worker = WorkerBase(
        vllm_config=config,
        local_rank=0,
        rank=7,
        distributed_init_method="",
    )

    runner.load_model(model=runner.model)

    assert runner.model_state.lwd_input_assembler._local_rank == 0
