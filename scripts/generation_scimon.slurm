#!/bin/bash
#SBATCH --job-name=phase2-scimon-gen
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scimon_gen_%j.out
#SBATCH --error=logs/scimon_gen_%j.err
#SBATCH --signal=B:SIGTERM@60

# -------- SCIMON generation (Llama-3.3-70B, TP=8) --------
# Sharding: pass --shard i/N or --start/--end via $SHARD_ARGS
# Example:
#   sbatch --export=ALL,SHARD_ARGS='--shard 1/4',SHARD_TAG=shard_1 \
#       scripts/generation_scimon.slurm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs outputs

# Phase-1 proven env (Sol pre-built vLLM + cached Llama-3.3-70B weights)
# shellcheck disable=SC1091
source serving/sol_env.sh

export VLLM_LLAMA_URL="http://localhost:8000/v1"
PORT=8000
MODEL="${LLAMA_MODEL_PATH}"   # local filesystem path, avoids HF download
TP=8
MAX_LEN=4096

# Launch vLLM in background
bash serving/start_vllm.sh "$MODEL" "$TP" "$PORT" "$MAX_LEN" \
    > logs/vllm_scimon_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[slurm] cleanup: killing vLLM pid=$VLLM_PID"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

# Wait for /health — up to 15 min (Llama-3.3-70B TP=8 warmup can take 5-10 min)
echo "[slurm] waiting for vLLM readiness..."
for i in $(seq 1 180); do
    if curl -fsS "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[slurm] vLLM ready after $((i*5))s"
        break
    fi
    sleep 5
done

# Launch orchestration pipeline
python run.py \
    --benchmark scimon \
    --config configs/scimon.yaml \
    ${SHARD_ARGS:-} \
    ${SHARD_TAG:+--shard-tag $SHARD_TAG}

echo "[slurm] generation done"
