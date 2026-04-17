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
  apptainer.def                ← legacy (not used; retained for reference)
  run.py                       ← main entry point
  configs/
    scimon.yaml
    ideabench.yaml
  agents/                      ← retriever, planner, generator, critic, refiner, selector
  adapters/                    ← scimon.py, ideabench.py (only place the two benchmarks differ)
  orchestrator/                ← pipeline.py, checkpoint.py, sharding.py
  serving/
    sol_env.sh                 ← Sol Gaudi-2 runtime env (sourced by every SLURM script)
    start_vllm.sh              ← launches vLLM on Gaudi (uses --served-model-name)
    vllm_client.py             ← OpenAI-compatible client w/ health check + retry + keepalive
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

## Running on ASU Sol (Gaudi-2)

**This pipeline runs directly against Sol's pre-built vLLM env via `serving/sol_env.sh`.
No Apptainer container is needed.** The `apptainer.def` file is kept for historical
reference only.

### 1. Clone this repo + external deps

```bash
ssh sol
git clone https://github.com/DeveshValluru/Phase_2_multiagent.git
cd Phase_2_multiagent

# Vendor the two original paper repos under external/
git clone https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty external/SCIMON
git clone https://github.com/amir-hassan25/IdeaBench external/IdeaBench
```

If you already have one of those repos cloned elsewhere, you can symlink instead:
```bash
mkdir -p external
ln -s /path/to/existing/IdeaBench external/IdeaBench
```

### 2. Install missing Python packages (one-time, per user)

Sol's pre-built env at `/packages/envs/gaudi-pytorch-vllm/` provides most deps
(`torch`, `transformers`, `vllm`, `openai`, `pandas`, `tenacity`, etc.) but is
**missing three packages** needed by the pipeline's retriever and evaluation code:

```bash
pip install --user sentence-transformers rouge-score bert-score
```

The install goes to `~/.local/` so it's available on every compute node
automatically (`sol_env.sh` sets `PYTHONNOUSERSITE=0`).

Verify on a Gaudi compute node:
```bash
srun --partition=gaudi --qos=class_gaudi --account=class_cse59827694spring2026 \
    --gres=gpu:hl225:1 --cpus-per-task=2 --mem=24G --time=0-1:00 --pty bash
# On the compute node:
source serving/sol_env.sh
python -c "import torch, transformers, sentence_transformers, rouge_score, bert_score; print('all OK')"
```

### 3. Calibrate throughput on a small batch (optional)

```bash
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

**Splitting work between teammates:** each teammate's outputs land in their
own repo clone. Before merging, collect all shards to one location:

```bash
# From the merging teammate's side:
scp -r <other_user>@sol:/path/to/Phase_2_multiagent/checkpoints/ideabench/shard_{9..16} \
    checkpoints/ideabench/
```

**First-time vLLM warmup:** expect 15–30 minutes on first model load (HPU
graph compilation). Watch `logs/vllm_<bench>_<jobid>.log` for
`Application startup complete` and the main SLURM log for
`[slurm] vLLM ready after XXXs`.

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
| `ModuleNotFoundError: sentence_transformers` / `rouge_score` / `bert_score` | Run the `pip install --user` step in section 2. |
| vLLM `/health` returns 200 but chat completions return 404 `model does not exist` | `start_vllm.sh` now passes `--served-model-name` for Llama/Qwen; make sure your clone is up-to-date (`git pull`). |
| vLLM `/health` never returns 200 | Check `logs/vllm_*.log` — usually model download, OOM, or HPU driver issue. Bump `--time` if first load is slow. |
| Judge output has 40% parse failures | Already handled: `utils/parsing.py` strips `<think>` and falls back to regex. Check `logs/` for the few that still fail. |
| SLURM job preempted mid-run | Just resubmit the same command — checkpoints pick up where it left off. |
| "ROUGE-L below baseline" after eval | Inspect `outputs/scimon_final.jsonl`, confirm Critic rubric weights in `configs/scimon.yaml`. You can re-run with tuned weights without re-running generation — only re-evaluation. |

---

## Citation

Original benchmarks:
- SCIMON: Wang et al., ACL 2024 — <https://aclanthology.org/2024.acl-long.18/>
- IdeaBench: <https://arxiv.org/abs/2411.02429>
