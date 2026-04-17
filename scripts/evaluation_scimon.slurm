#!/bin/bash
#SBATCH --job-name=phase2-scimon-eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scimon_eval_%j.out
#SBATCH --error=logs/scimon_eval_%j.err

# -------- SCIMON evaluation (pure Python, no LLM) --------
# Runs after all SCIMON shards are merged.
# Usage:
#   sbatch --export=PREDICTIONS=outputs/scimon_final.jsonl,OUT=outputs/scimon_scores.json \
#     scripts/evaluation_scimon.slurm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs outputs

PREDICTIONS="${PREDICTIONS:-outputs/scimon_final.jsonl}"
OUT="${OUT:-outputs/scimon_scores.json}"

python eval/scimon_eval.py \
    --predictions "$PREDICTIONS" \
    --out "$OUT"
