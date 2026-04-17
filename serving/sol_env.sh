#!/bin/bash
# Sol Gaudi-2 runtime environment — sourced by every SLURM script.
#
# Uses the pre-built vLLM env that Phase-1 used successfully (98.6% coverage
# on IdeaBench). No Apptainer container needed.
#
# Source this from SLURM scripts:
#     source serving/sol_env.sh

# Pre-built Gaudi vLLM env shipped on Sol
export PATH=/packages/envs/gaudi-pytorch-vllm/bin:$PATH
export PYTHONNOUSERSITE=0

# HF caches → user scratch (community cache is read-only)
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/.cache/huggingface/hub
mkdir -p "$HF_HOME/hub"

# Gaudi / TPC kernel path
TPC_LIB=$(find /usr/lib /opt /packages/envs/gaudi-pytorch-vllm -name libtpc_kernels.so 2>/dev/null | head -1)
[ -n "$TPC_LIB" ] && export GC_KERNEL_PATH="$TPC_LIB"

# Gaudi / vLLM runtime tuning — these are the Phase-1 values that fixed the
# vLLM idle-crash. DO NOT LOWER VLLM_ENGINE_ITERATION_TIMEOUT_S.
export HABANA_VISIBLE_DEVICES=all
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_PROMPT_USE_FUSEDSDPA=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=86400
export VLLM_RPC_TIMEOUT=100000

# Sanity — clear old graph dumps so compilation doesn't reuse stale kernels
rm -rf .graph_dumps/ 2>/dev/null || true

# Cached model paths on Sol (Phase-1 verified)
export LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-/data/datasets/community/huggingface/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}
# Qwen may not be locally cached; vLLM will download to HF_HOME on first use.
export QWEN_MODEL_ID=${QWEN_MODEL_ID:-Qwen/Qwen3-32B}
