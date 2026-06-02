# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CosyVoice3 pipeline topology (frozen).

Stage 0: Talker   — text prompt → speech tokens (LLM autoregressive).
Stage 1: Code2Wav — flow-matching decoder → acoustic features → waveform.
  * ``sync_process_input_func`` runs when ``deploy.async_chunk=false``:
    stage 1 builds full-sequence flow input via ``text2flow``.
  * ``async_chunk_process_next_stage_input_func`` runs when
    ``deploy.async_chunk=true``: stage 0 streams codec chunks to stage 1
    through the shared-memory connector.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.cosyvoice3.hf_talker_env import (
    hf_metadata,
    hf_talker_enabled,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.cosyvoice3"

# HF-Qwen2 talker (COSYVOICE3_HF_TALKER=1) emits a single eos token; the custom
# fp32 talker merges 200 stop logits into 6562 via logsumexp in compute_logits.
_TALKER_STOP_TOKEN_IDS = [hf_metadata()["eos_token_id"]] if hf_talker_enabled() else [6562]

COSYVOICE3_PIPELINE = PipelineConfig(
    model_type="cosyvoice3",
    model_arch="CosyVoice3Model",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="cosyvoice3_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "stop_token_ids": _TALKER_STOP_TOKEN_IDS,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="cosyvoice3_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.text2flow",
        ),
    ),
)
