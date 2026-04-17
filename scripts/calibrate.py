"""Calibration: run the pipeline on the first N instances and report throughput.

Use this once per benchmark to size SLURM wall-clock accurately.

Usage:
    python scripts/calibrate.py --benchmark ideabench --n 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_config
from agents.retriever import CorpusIndex, InlineReRanker
from orchestrator.pipeline import Pipeline
from utils.logging import setup_logging

log = setup_logging()


def main():
    p = argparse.ArgumentParser(description="Throughput calibration")
    p.add_argument("--benchmark", choices=["scimon", "ideabench"], required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg_path = args.config or f"configs/{args.benchmark}.yaml"
    cfg = load_config(cfg_path)
    # Route calibration output to its own dir to avoid polluting checkpoints
    cfg["run"]["output_dir"] = str(Path(cfg["run"]["output_dir"]) / "calibration")

    if args.benchmark == "scimon":
        from adapters.scimon import ScimonAdapter
        adapter = ScimonAdapter(cfg)
        instances = adapter.load_test_instances()[: args.n]
        corpus_texts = adapter.load_training_corpus()
        idx = CorpusIndex(cfg["data"]["sbert_index_path"], cfg["data"]["sbert_model"])
        idx.build_or_load(corpus_texts)
        inline = None
    else:
        from adapters.ideabench import IdeaBenchAdapter
        adapter = IdeaBenchAdapter(cfg)
        instances = adapter.load_test_instances()[: args.n]
        idx = None
        inline = InlineReRanker()

    pipeline = Pipeline(cfg)
    t0 = time.monotonic()
    pipeline.run(instances, adapter=adapter, corpus_index=idx, inline_reranker=inline)
    dt = time.monotonic() - t0

    per = dt / max(1, args.n)
    log.info("calibration: %d instances in %.1fs (%.2fs/inst)", args.n, dt, per)

    # Sol projection
    if args.benchmark == "scimon":
        remaining = 194 - args.n
    else:
        remaining = 2374 - args.n
    projected_h = remaining * per / 3600
    log.info("projected full run (single shard): %.1f hours for remaining %d",
             projected_h, remaining)


if __name__ == "__main__":
    main()
