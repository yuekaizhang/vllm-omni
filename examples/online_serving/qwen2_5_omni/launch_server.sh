
config_file=/workspace_yuekai/asr/vllm-omni/vllm_omni/model_executor/stage_configs/qwen2_5_omni.h20.yaml
model_path=/workspace_yuekai/Qwen2.5-Omni-3B
port=8091
vllm serve $model_path --omni --port $port --stage-configs-path $config_file