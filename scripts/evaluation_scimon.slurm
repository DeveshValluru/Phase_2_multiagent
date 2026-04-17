#!/bin/bash
#SBATCH --job-name=phase2-scimon-eval
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scimon_eval_%j.out
#SBATCH --error=logs/scimon_eval_%j.err

# -------- SCIMON evaluation (pure Python, no LLM) --------
# Runs after all SCIMON shards are merged. CPU-only — no Gaudi needed.
# Usage:
#   sbatch --export=ALL,PREDICTIONS=outputs/scimon_final.jsonl,OUT=outputs/scimon_scores.json \
#       scripts/evaluation_scimon.slurm

set -euo pipefail

# SLURM copies the script to its spool dir; use the submission dir as repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
echo "[slurm] cwd=$PWD host=$(hostname) user=$(whoami)"
mkdir -p logs outputs

# bert-score / rouge-score live in the Sol vLLM env too
# shellcheck disable=SC1091
source serving/sol_env.sh

PREDICTIONS="${PREDICTIONS:-outputs/scimon_final.jsonl}"
OUT="${OUT:-outputs/scimon_scores.json}"

python eval/scimon_eval.py \
    --predictions "$PREDICTIONS" \
    --out "$OUT"
