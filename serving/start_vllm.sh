#!/bin/bash
# Launch a vLLM OpenAI-compatible server on Intel Gaudi 2 HPUs.
#
# Usage:
#   start_vllm.sh <model_id_or_path> <tensor_parallel> <port> [<max_model_len>]
#
# Default path: uses Sol's pre-built /packages/envs/gaudi-pytorch-vllm (Phase-1
# proven). To run inside an Apptainer container instead, set APPTAINER_IMAGE
# to a .sif path.
#
# Env vars (all set in serving/sol_env.sh):
#   VLLM_ENGINE_ITERATION_TIMEOUT_S, VLLM_RPC_TIMEOUT, VLLM_PROMPT_USE_FUSEDSDPA,
#   PT_HPU_LAZY_MODE, PT_HPU_ENABLE_LAZY_COLLECTIVES, HABANA_VISIBLE_DEVICES,
#   GC_KERNEL_PATH, VLLM_GRAPH_RESERVED_MEM, HF_HOME.

set -euo pipefail

MODEL="${1:?model id or path required}"
TP="${2:?tensor parallel size required}"
PORT="${3:?port required}"
MAX_LEN="${4:-4096}"

# Source Sol env if not already sourced by caller SLURM script
if [[ -z "${VLLM_ENGINE_ITERATION_TIMEOUT_S:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/sol_env.sh"
fi

ARGS=(
    --model "$MODEL"
    --tensor-parallel-size "$TP"
    --port "$PORT"
    --host 0.0.0.0
    --dtype bfloat16
    --block-size 128
    --max-model-len "$MAX_LEN"
    --max-num-seqs "${MAX_NUM_SEQS:-16}"
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}"
    --trust-remote-code
)

# Set served-model-name so clients can use the HF ID instead of the local path
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
    ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
elif [[ "$MODEL" == *"Llama-3.3-70B-Instruct"* ]]; then
    ARGS+=(--served-model-name "meta-llama/Llama-3.3-70B-Instruct")
elif [[ "$MODEL" == *"Qwen3-32B"* ]]; then
    ARGS+=(--served-model-name "Qwen/Qwen3-32B")
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2086
    ARGS+=($VLLM_EXTRA_ARGS)
fi

if [[ -n "${APPTAINER_IMAGE:-}" && -f "$APPTAINER_IMAGE" ]]; then
    echo "[start_vllm] launching vLLM inside Apptainer: $APPTAINER_IMAGE"
    exec apptainer exec --nv "$APPTAINER_IMAGE" \
        python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
else
    echo "[start_vllm] launching vLLM via Sol host Python (/packages/envs/gaudi-pytorch-vllm)"
    echo "[start_vllm] which python: $(which python)"
    exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
fi
