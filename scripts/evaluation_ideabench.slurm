#!/bin/bash
#SBATCH --job-name=phase2-ideabench-judge
#SBATCH --partition=gaudi
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ideabench_judge_%j.out
#SBATCH --error=logs/ideabench_judge_%j.err
#SBATCH --signal=B:SIGTERM@60

# -------- IdeaBench judge pass (Qwen3-32B, TP=4) --------
# Runs AFTER generation jobs finish and results are merged.
# Usage:
#   sbatch --export=PREDICTIONS=outputs/ideabench_final.jsonl,OUT=outputs/ideabench_judge.jsonl \
#     scripts/evaluation_ideabench.slurm

set -euo pipefail

module purge || true
module load cuda || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs outputs

export VLLM_QWEN_URL="http://localhost:8001/v1"
PORT=8001
MODEL="Qwen/Qwen3-32B"
TP=4
MAX_LEN=16384

PREDICTIONS="${PREDICTIONS:-outputs/ideabench_final.jsonl}"
OUT="${OUT:-outputs/ideabench_judge.jsonl}"

bash serving/start_vllm.sh "$MODEL" "$TP" "$PORT" "$MAX_LEN" > logs/vllm_judge_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[slurm] cleanup: killing vLLM pid=$VLLM_PID"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "[slurm] waiting for Qwen vLLM readiness..."
for i in $(seq 1 240); do
    if curl -fsS "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[slurm] Qwen ready after ${i}s"
        break
    fi
    sleep 5
done

python eval/ideabench_eval.py \
    --config configs/ideabench.yaml \
    --predictions "$PREDICTIONS" \
    --out "$OUT"

echo "[slurm] judge done"
