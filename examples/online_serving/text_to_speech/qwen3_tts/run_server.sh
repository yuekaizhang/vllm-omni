#!/bin/bash
# Launch vLLM-Omni server for Qwen3-TTS models
#
# Usage:
#   ./run_server.sh                           # Default: CustomVoice model
#   ./run_server.sh CustomVoice               # CustomVoice model
#   ./run_server.sh VoiceDesign               # VoiceDesign model
#   ./run_server.sh Base                      # Base (voice clone) model
#   PORT=8095 ./run_server.sh Base            # override port
#   GPU_UTIL=0.45 ./run_server.sh Base        # override gpu-mem (see note below)
#
# NOTE: gpu-memory-utilization is NOT passed by default. This is a two-stage
# deploy (talker + code2wav); qwen3_tts.yaml sets a per-stage budget
# (0.3 + 0.3, verified on 1x H100). A command-line --gpu-memory-utilization
# overrides the yaml GLOBALLY for every stage, so a single value like 0.9
# makes stage 0 consume the GPU and stage 1 OOM on a single card. Only set
# GPU_UTIL when you deliberately want to override the yaml for all stages.

set -e

TASK_TYPE="${1:-CustomVoice}"
PORT="${PORT:-8091}"
GPU_UTIL="${GPU_UTIL:-}"

case "$TASK_TYPE" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    VoiceDesign)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: CustomVoice, VoiceDesign, Base"
        exit 1
        ;;
esac

echo "Starting Qwen3-TTS server with model: $MODEL (port $PORT)"

EXTRA_ARGS=()
if [[ -n "$GPU_UTIL" ]]; then
    EXTRA_ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
fi

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni \
    "${EXTRA_ARGS[@]}"
