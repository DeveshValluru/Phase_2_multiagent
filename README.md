# Phase 2 — Multi-Agent Scientific Ideation (CSE 598 Spring 2026)

A unified multi-agent pipeline that beats the Phase-1 Llama-3.3-70B baselines on
both the **SCIMON** and **IdeaBench** benchmarks.

## Baselines to Beat (Phase-1 Llama-3.3-70B)

| Benchmark | Metric | Phase-1 | Phase-2 target |
|---|---|---|---|
| SCIMON | ROUGE-L | 0.1132 | > 0.1132 |
| SCIMON | BERTScore (SciBERT) | 0.5547 | > 0.5547 |
| IdeaBench | BERTScore F1 | 0.5252 | > 0.5252 |
| IdeaBench | LLM Rating (1–10) | 5.83 | > 5.83 |
| IdeaBench | Insight-Novelty | 0.0863 | > 0.0863 |
| IdeaBench | Insight-Feasibility | 0.0253 | > 0.0253 |

## Architecture

One shared agent graph runs either benchmark by switching a dataset adapter:

```
Retriever → Planner → Generator (N drafts) → Critic → Refiner (≤2 iters) → Selector
```

- **Generator / Critic / Refiner / Selector**: Llama-3.3-70B-Instruct (all roles)
- **Judge (IdeaBench only)**: Qwen3-32B (runs in a *separate* SLURM job after generation)
- **Retriever**: SentenceBERT top-k neighbors (SCIMON) / re-rank of 3 refs (IdeaBench)

Key design choices:
- **Gold-style Critic rubric** (not dissimilarity-to-training) fixes Phase-1's novelty-boost regression.
- **Retrieval-mix on SCIMON** (2 drafts with retrieved neighbors, 2 without) avoids the ROUGE-L regression that pure retrieval caused in Phase 1.
- **Submodular top-3 selection** on IdeaBench boosts Insight-Novelty via diversity.
- **Robust `<think>`-stripping + JSON-with-regex-fallback parser** addresses Phase-1's 40–55% malformed-ranking issue.
- **Per-instance JSONL checkpointing** — every member can claim/resume any shard, idempotent merge at the end.
- **Separate generation and judge SLURM jobs** — avoids the Phase-1 vLLM idle-server failure mode.

## Repository Layout

```
phase2-multiagent/
  README.md                    ← you are here
  requirements.txt
  apptainer.def                ← container def for Gaudi-2 + vLLM
  run.py                       ← main entry point
  configs/
    scimon.yaml
    ideabench.yaml
  agents/                      ← retriever, planner, generator, critic, refiner, selector
  adapters/                    ← scimon.py, ideabench.py (only place the two benchmarks differ)
  orchestrator/                ← pipeline.py, checkpoint.py, sharding.py
  serving/
    vllm_client.py             ← OpenAI-compatible client w/ health check + retry + keepalive
    start_vllm.sh              ← launches vLLM on Gaudi
  eval/
    scimon_eval.py             ← ROUGE-L + BERTScore
    ideabench_eval.py          ← BERTScore + LLM Rating + Insight Score
  scripts/
    generation_scimon.slurm
    generation_ideabench.slurm
    evaluation_scimon.slurm
    evaluation_ideabench.slurm
    merge.py                   ← merges shard JSONLs
    calibrate.py               ← throughput calibration on first N instances
  utils/                       ← parsing.py (<think>-aware), logging.py
  external/                    ← put clones of SCIMON and IdeaBench repos here (gitignored)
```

---

## Upload to GitHub

```bash
# 1) Create a new empty repo on GitHub (e.g. phase2-multiagent) via the UI.
# 2) From the project directory:
cd phase2-multiagent
git init
git add .
git commit -m "Initial commit: Phase 2 multi-agent system"
git branch -M main
git remote add origin https://github.com/<your-user-or-org>/phase2-multiagent.git
git push -u origin main
```

The `.gitignore` already excludes `external/SCIMON/`, `external/IdeaBench/`,
`logs/`, `outputs/`, `checkpoints/`, `*.sif`, and `*.npz`, so nothing bulky or
license-entangled will be pushed.

---

## Running on ASU Sol (Gaudi-2)

### 1. Clone this repo + external deps

```bash
ssh sol
cd $SCRATCH
git clone https://github.com/<your-user-or-org>/phase2-multiagent.git
cd phase2-multiagent

# Vendor the two original paper repos under external/
git clone https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty external/SCIMON
git clone https://github.com/amir-hassan25/IdeaBench external/IdeaBench
```

### 2. Build (or reuse) the Apptainer image

```bash
# If Phase-1 left an image behind, point to it:
export APPTAINER_IMAGE=$SCRATCH/images/phase2.sif

# Otherwise build from the definition file (~20 min):
mkdir -p $SCRATCH/images
apptainer build $SCRATCH/images/phase2.sif apptainer.def
export APPTAINER_IMAGE=$SCRATCH/images/phase2.sif
```

### 3. Calibrate throughput on a small batch (optional but recommended)

```bash
# Requests a small allocation, runs the pipeline on 50 instances, prints per-instance time
sbatch --export=ALL --gres=gpu:8 --time=02:00:00 \
    scripts/generation_scimon.slurm
# or:
python scripts/calibrate.py --benchmark ideabench --n 50
```

### 4. Submit generation jobs

For SCIMON (194 instances total, one job usually enough):

```bash
sbatch scripts/generation_scimon.slurm                           # full run
# or split across teammates:
sbatch --export=ALL,SHARD_ARGS="--shard 1/4",SHARD_TAG=shard_1 scripts/generation_scimon.slurm
sbatch --export=ALL,SHARD_ARGS="--shard 2/4",SHARD_TAG=shard_2 scripts/generation_scimon.slurm
sbatch --export=ALL,SHARD_ARGS="--shard 3/4",SHARD_TAG=shard_3 scripts/generation_scimon.slurm
sbatch --export=ALL,SHARD_ARGS="--shard 4/4",SHARD_TAG=shard_4 scripts/generation_scimon.slurm
```

For IdeaBench (2,374 papers, shard into 16 × ~150):

```bash
for i in $(seq 1 16); do
    sbatch --export=ALL,SHARD_ARGS="--shard ${i}/16",SHARD_TAG=shard_${i} \
        scripts/generation_ideabench.slurm
done
```

Sharding is completely flexible — use any of:
- `--shard i/N` (deterministic split)
- `--start S --end E` (explicit range)
- `--ids file.txt` (one instance_id per line)

All members can claim shards independently. Each writes to
`checkpoints/<bench>/<shard_tag>/generation.jsonl`. **Running the same shard
twice is harmless** — JSONL is idempotent, merge dedupes.

### 5. Merge shards

```bash
# SCIMON
python scripts/merge.py --benchmark scimon \
    --shards checkpoints/scimon/shard_{1,2,3,4} \
    --out outputs/scimon_final.jsonl

# IdeaBench
python scripts/merge.py --benchmark ideabench \
    --shards checkpoints/ideabench/shard_{1..16} \
    --out outputs/ideabench_final.jsonl
```

### 6. Run evaluation

SCIMON (no LLM, runs on any node):

```bash
sbatch --export=ALL,PREDICTIONS=outputs/scimon_final.jsonl,OUT=outputs/scimon_scores.json \
    scripts/evaluation_scimon.slurm
```

IdeaBench (Qwen-32B judge — separate vLLM server, separate job):

```bash
sbatch --export=ALL,PREDICTIONS=outputs/ideabench_final.jsonl,OUT=outputs/ideabench_judge.jsonl \
    scripts/evaluation_ideabench.slurm
```

Results print to stdout in the SLURM log and land in:
- `outputs/scimon_scores.json`
- `outputs/ideabench_judge_summary.json`

Each summary includes the baseline comparison and a `beat_baseline` flag per metric.

---

## Local (non-Sol) Dev Loop

You can run the orchestration layer anywhere — point it at *any* OpenAI-compatible
vLLM endpoint via `VLLM_LLAMA_URL` and `VLLM_QWEN_URL`:

```bash
pip install -r requirements.txt
export VLLM_LLAMA_URL=http://gpu-host:8000/v1
python run.py --benchmark scimon --limit 10     # smoke test
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not locate SCIMON gold test set" | Set `data.gold_test_path` in `configs/scimon.yaml` to your clone's actual path. |
| vLLM `/health` never returns 200 | Check `logs/vllm_*.log` — usually model download, OOM, or HPU driver issue. Bump `--time` if first load is slow. |
| Judge output has 40% parse failures | Already handled: `utils/parsing.py` strips `<think>` and falls back to regex. Check `logs/` for the few that still fail. |
| SLURM job preempted mid-run | Just resubmit the same command — checkpoints pick up where it left off. |
| "ROUGE-L below baseline" after eval | Inspect `outputs/scimon_final.jsonl`, confirm Critic rubric weights in `configs/scimon.yaml`. You can re-run with tuned weights without re-running generation — only re-evaluation. |

---

## Citation

Original benchmarks:
- SCIMON: Wang et al., ACL 2024 — <https://aclanthology.org/2024.acl-long.18/>
- IdeaBench: <https://arxiv.org/abs/2411.02429>
