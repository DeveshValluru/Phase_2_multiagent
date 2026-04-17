"""IdeaBench evaluation: BERTScore + LLM Rating + Insight Score (Novelty & Feasibility).

The Qwen3-32B judge runs in a SEPARATE SLURM job (its own vLLM server) — not in
the same process as generation. This matches Phase 1's separation and avoids
the idle-server issue.

Target baselines (Phase-1 Llama-3.3-70B generator, Qwen-32B judge):
  BERTScore F1     > 0.5252
  LLM Rating       > 5.83
  Insight-Novelty  > 0.0863
  Insight-Feasibil > 0.0253
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.prompts import (
    JUDGE_SYSTEM_RATING, JUDGE_USER_RATING,
    JUDGE_SYSTEM_RANK, JUDGE_USER_RANK,
)
from orchestrator.checkpoint import JsonlCheckpoint
from serving.vllm_client import client_from_config
from utils.logging import setup_logging
from utils.parsing import extract_json, parse_ranking

log = setup_logging()


# --------------------------- BERTScore (no LLM) ---------------------------

def compute_bertscore(pairs: list[tuple[str, str]], model: str = "roberta-large") -> list[float]:
    """Return per-pair F1 scores. pairs = [(prediction, target_abstract), ...]"""
    from bert_score import score as bscore
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    P, R, F = bscore(preds, refs, model_type=model, verbose=False, batch_size=32,
                     rescale_with_baseline=False)
    return [float(x) for x in F.tolist()]


# --------------------------- Judge passes ---------------------------------

def load_config(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(os.path.expandvars(raw))


def load_generation_outputs(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "done":
            out.append(r)
    return out


def build_hypotheses_block(hyps: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hyps):
        letter = chr(ord("A") + i + 1)   # A is reserved for target
        if isinstance(h, dict):
            text = h.get("hypothesis") or h.get("title") or json.dumps(h, ensure_ascii=False)
        else:
            text = str(h)
        parts.append(f"[{letter}] {text}")
    return "\n".join(parts)


def build_items_block(target_abstract: str, hyps: list[dict]) -> str:
    lines = [f"[A] {target_abstract}"]
    for i, h in enumerate(hyps):
        letter = chr(ord("B") + i)
        if isinstance(h, dict):
            text = h.get("hypothesis") or h.get("title") or json.dumps(h, ensure_ascii=False)
        else:
            text = str(h)
        lines.append(f"[{letter}] {text}")
    return "\n\n".join(lines)


def run_llm_rating(client, record: dict, max_tokens: int) -> float | None:
    hyps = record.get("hypotheses", [])
    if not hyps:
        return None
    block = build_hypotheses_block(hyps)
    msg = [
        {"role": "system", "content": JUDGE_SYSTEM_RATING},
        {"role": "user", "content": JUDGE_USER_RATING.format(
            target_abstract=record.get("target_abstract", ""),
            hypotheses_block=block,
        )},
    ]
    try:
        text = client.chat(msg, temperature=0.0, max_tokens=max_tokens)
    except Exception as e:
        log.warning("rating call failed for %s: %s", record.get("instance_id"), e)
        return None
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("ratings"), list):
        try:
            nums = [float(x) for x in obj["ratings"]]
            if nums:
                return sum(nums) / len(nums)
        except (ValueError, TypeError):
            pass
    return None


def run_insight_ranking(client, record: dict, criterion: str, max_tokens: int) -> float | None:
    """Compute Insight Score = (rank_of_target - 1) / n, higher = model beat GT."""
    hyps = record.get("hypotheses", [])
    if not hyps:
        return None
    n_items = len(hyps) + 1
    items_block = build_items_block(record.get("target_abstract", ""), hyps)
    msg = [
        {"role": "system", "content": JUDGE_SYSTEM_RANK},
        {"role": "user", "content": JUDGE_USER_RANK.format(
            n=n_items, criterion=criterion, items_block=items_block,
        )},
    ]
    try:
        text = client.chat(msg, temperature=0.0, max_tokens=max_tokens)
    except Exception as e:
        log.warning("ranking (%s) call failed for %s: %s", criterion, record.get("instance_id"), e)
        return None

    order = parse_ranking(text, n_items=n_items)
    if order is None:
        return None
    # Find where target (position 0 = "A") landed
    try:
        rank_of_target = order.index(0) + 1    # 1-indexed
    except ValueError:
        return None
    n_generated = len(hyps)
    return (rank_of_target - 1) / max(1, n_generated)


# --------------------------- main -----------------------------------------

def main():
    p = argparse.ArgumentParser(description="IdeaBench eval: BERTScore + LLM Rating + Insight Score")
    p.add_argument("--config", default="configs/ideabench.yaml")
    p.add_argument("--predictions", required=True, help="merged generation JSONL")
    p.add_argument("--out", required=True, help="judge output JSONL")
    p.add_argument("--bertscore-model", default="roberta-large")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-rating", action="store_true")
    p.add_argument("--skip-insight", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    records = load_generation_outputs(Path(args.predictions))
    log.info("evaluating %d records", len(records))

    # 1. BERTScore (no LLM; runs fast)
    bs_by_id: dict[str, float] = {}
    if not args.skip_bertscore:
        pairs = []
        ids = []
        for r in records:
            hyps_text = " ".join(
                (h.get("hypothesis", "") if isinstance(h, dict) else str(h))
                for h in r.get("hypotheses", [])
            )
            pairs.append((hyps_text, r.get("target_abstract", "")))
            ids.append(r["instance_id"])
        if pairs:
            fs = compute_bertscore(pairs, model=args.bertscore_model)
            bs_by_id = dict(zip(ids, fs))
        log.info("BERTScore done")

    # 2 & 3. Judge passes (need vLLM)
    need_llm = not (args.skip_rating and args.skip_insight)
    client = None
    if need_llm:
        client = client_from_config(cfg["judge"])
        client.wait_ready(timeout_s=900.0)
        client.start_keepalive(60.0)

    ckpt = JsonlCheckpoint(args.out)
    done = ckpt.completed_ids()
    log.info("resuming judge: %d already scored", len(done))

    try:
        from tqdm import tqdm
        for r in tqdm(records, desc="judge", unit="paper"):
            iid = str(r["instance_id"])
            if iid in done:
                continue
            out = {
                "instance_id": iid,
                "bertscore_f1": bs_by_id.get(iid),
                "llm_rating": None,
                "insight_novelty": None,
                "insight_feasibility": None,
                "status": "done",
            }
            if client is not None and client.term_requested:
                log.warning("SIGTERM received — exiting judge loop cleanly")
                break
            if client is not None and not args.skip_rating:
                out["llm_rating"] = run_llm_rating(client, r, cfg["judge"]["max_tokens"])
            if client is not None and not args.skip_insight:
                out["insight_novelty"] = run_insight_ranking(client, r, "novel", cfg["judge"]["max_tokens"])
                out["insight_feasibility"] = run_insight_ranking(client, r, "feasible", cfg["judge"]["max_tokens"])
            ckpt.append(out)
    finally:
        if client is not None:
            client.stop_keepalive()

    # 4. Aggregate + print
    aggregate(Path(args.out))


def aggregate(judge_path: Path) -> None:
    recs = list(JsonlCheckpoint(judge_path).all_records())
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    bs = mean([r.get("bertscore_f1") for r in recs])
    rating = mean([r.get("llm_rating") for r in recs])
    nov = mean([r.get("insight_novelty") for r in recs])
    feas = mean([r.get("insight_feasibility") for r in recs])

    summary = {
        "benchmark": "ideabench",
        "n_records": len(recs),
        "bertscore_f1": bs,
        "llm_rating": rating,
        "insight_novelty": nov,
        "insight_feasibility": feas,
        "coverage": {
            "bertscore": sum(1 for r in recs if r.get("bertscore_f1") is not None),
            "rating": sum(1 for r in recs if r.get("llm_rating") is not None),
            "novelty": sum(1 for r in recs if r.get("insight_novelty") is not None),
            "feasibility": sum(1 for r in recs if r.get("insight_feasibility") is not None),
        },
        "baselines": {
            "bertscore_f1": 0.5252,
            "llm_rating": 5.83,
            "insight_novelty": 0.0863,
            "insight_feasibility": 0.0253,
        },
        "beat_baseline": {
            "bertscore_f1": bs is not None and bs > 0.5252,
            "llm_rating": rating is not None and rating > 5.83,
            "insight_novelty": nov is not None and nov > 0.0863,
            "insight_feasibility": feas is not None and feas > 0.0253,
        },
    }
    print(json.dumps(summary, indent=2))
    out_summary = judge_path.with_name(judge_path.stem + "_summary.json")
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", out_summary)


if __name__ == "__main__":
    main()
