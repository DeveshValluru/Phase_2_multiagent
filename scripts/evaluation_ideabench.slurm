#!/bin/bash
#SBATCH --job-name=phase2-ideabench-judge
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --nodes=1
#SBATCH --gres=gpu:hl225:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ideabench_judge_%j.out
#SBATCH --error=logs/ideabench_judge_%j.err
#SBATCH --signal=B:SIGTERM@60

# -------- IdeaBench judge pass (Qwen3-32B, TP=8) --------
# Runs AFTER generation jobs finish and results are merged.
# Usage:
#   sbatch --export=ALL,PREDICTIONS=outputs/ideabench_final.jsonl,OUT=outputs/ideabench_judge.jsonl \
#       scripts/evaluation_ideabench.slurm

set -euo pipefail

# SLURM copies the script to its spool dir; use the submission dir as repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[slurm] cwd=$PWD host=$(hostname) user=$(whoami)"
mkdir -p logs outputs

# Phase-1 proven env
# shellcheck disable=SC1091
source serving/sol_env.sh

export VLLM_QWEN_URL="http://localhost:8001/v1"
PORT=8001
MODEL="${QWEN_MODEL_ID}"
TP=8
MAX_LEN=16384

PREDICTIONS="${PREDICTIONS:-outputs/ideabench_final.jsonl}"
OUT="${OUT:-outputs/ideabench_judge.jsonl}"

bash serving/start_vllm.sh "$MODEL" "$TP" "$PORT" "$MAX_LEN" \
    > logs/vllm_judge_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[slurm] cleanup: killing vLLM pid=$VLLM_PID"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "[slurm] waiting for Qwen vLLM readiness..."
for i in $(seq 1 180); do
    if curl -fsS "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[slurm] Qwen ready after $((i*5))s"
        break
    fi
    sleep 5
done

python eval/ideabench_eval.py \
    --config configs/ideabench.yaml \
    --predictions "$PREDICTIONS" \
    --out "$OUT"

echo "[slurm] judge done"
