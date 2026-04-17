"""Main entry point.

Usage examples:
    python run.py --benchmark scimon
    python run.py --benchmark scimon --shard 1/4
    python run.py --benchmark ideabench --start 0 --end 594
    python run.py --benchmark ideabench --ids my_papers.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.retriever import CorpusIndex, InlineReRanker
from orchestrator.pipeline import Pipeline
from orchestrator.sharding import select_shard, summarize_shard
from utils.logging import setup_logging


def load_config(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(os.path.expandvars(raw))


def main():
    log = setup_logging()

    p = argparse.ArgumentParser(description="Phase 2 multi-agent runner")
    p.add_argument("--benchmark", choices=["scimon", "ideabench"], required=True)
    p.add_argument("--config", default=None, help="YAML config (default: configs/<benchmark>.yaml)")
    # Sharding (mutually exclusive)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--shard", default=None, help='"i/N", e.g. "1/4"')
    p.add_argument("--ids", default=None, help="file with one instance_id per line")
    # Override output dir (useful to separate shards on disk)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--shard-tag", default=None, help="appended to output dir, e.g. 'shard_1'")
    # Limits (for debugging / calibration)
    p.add_argument("--limit", type=int, default=None, help="cap processed instances (after shard)")
    args = p.parse_args()

    cfg_path = args.config or f"configs/{args.benchmark}.yaml"
    cfg = load_config(cfg_path)
    if args.output_dir:
        cfg["run"]["output_dir"] = args.output_dir
    if args.shard_tag:
        cfg["run"]["output_dir"] = str(Path(cfg["run"]["output_dir"]) / args.shard_tag)

    # Build adapter + load instances
    if args.benchmark == "scimon":
        from adapters.scimon import ScimonAdapter
        adapter = ScimonAdapter(cfg)
        all_instances = adapter.load_test_instances()
        # Build / load SBERT index
        corpus_texts = adapter.load_training_corpus()
        idx = CorpusIndex(cfg["data"]["sbert_index_path"], cfg["data"]["sbert_model"])
        idx.build_or_load(corpus_texts)
        inline = None
    else:
        from adapters.ideabench import IdeaBenchAdapter
        adapter = IdeaBenchAdapter(cfg)
        all_instances = adapter.load_test_instances()
        idx = None
        inline = InlineReRanker()

    all_instances.sort(key=lambda x: x["instance_id"])
    all_ids = [x["instance_id"] for x in all_instances]
    selected_ids = select_shard(all_ids, start=args.start, end=args.end,
                                shard=args.shard, ids_file=args.ids)
    id_set = set(selected_ids)
    instances = [x for x in all_instances if x["instance_id"] in id_set]
    if args.limit is not None:
        instances = instances[: args.limit]

    log.info("shard: %s", summarize_shard(instances, len(all_instances)))

    pipeline = Pipeline(cfg)
    pipeline.run(instances, adapter=adapter, corpus_index=idx, inline_reranker=inline)
    log.info("done. output: %s",
             Path(cfg["run"]["output_dir"]) / "generation.jsonl")


if __name__ == "__main__":
    main()
