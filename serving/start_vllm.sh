#!/bin/bash
# Launch a vLLM OpenAI-compatible server on Intel Gaudi 2 HPUs.
#
# Usage:
#   start_vllm.sh <model_id> <tensor_parallel> <port> [<max_model_len>]
#
# Env:
#   APPTAINER_IMAGE   path to container .sif (required if running in container)
#   VLLM_EXTRA_ARGS   optional extra args forwarded to vllm serve
#
# Designed to be `srun`ed from a SLURM script (see scripts/*.slurm).

set -euo pipefail

MODEL="${1:?model id required}"
TP="${2:?tensor parallel size required}"
PORT="${3:?port required}"
MAX_LEN="${4:-2048}"

export HABANA_VISIBLE_DEVICES="${HABANA_VISIBLE_DEVICES:-all}"
export PT_HPU_LAZY_MODE=1
export VLLM_GRAPH_RESERVED_MEM="${VLLM_GRAPH_RESERVED_MEM:-0.1}"
# Helps avoid idle timeout on long runs
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"

ARGS=(
    --model "$MODEL"
    --tensor-parallel-size "$TP"
    --port "$PORT"
    --host 0.0.0.0
    --dtype bfloat16
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}"
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --enable-prefix-caching
    --trust-remote-code
)

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2086
    ARGS+=($VLLM_EXTRA_ARGS)
fi

if [[ -n "${APPTAINER_IMAGE:-}" && -f "$APPTAINER_IMAGE" ]]; then
    echo "launching vLLM inside Apptainer: $APPTAINER_IMAGE"
    exec apptainer exec --nv "$APPTAINER_IMAGE" python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
else
    echo "launching vLLM on host Python (no container)"
    exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
fi
