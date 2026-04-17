"""Merge per-shard JSONL checkpoints into one final JSONL.

Usage:
    python scripts/merge.py \
        --benchmark scimon \
        --shards checkpoints/scimon/shard_1 checkpoints/scimon/shard_2 ... \
        --out outputs/scimon_final.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.checkpoint import merge_jsonl
from utils.logging import setup_logging

log = setup_logging()


def main():
    p = argparse.ArgumentParser(description="Merge shard JSONLs")
    p.add_argument("--benchmark", choices=["scimon", "ideabench"], required=True)
    p.add_argument("--shards", nargs="+", required=True,
                   help="shard directories (each should contain generation.jsonl)")
    p.add_argument("--out", required=True, help="output merged JSONL path")
    args = p.parse_args()

    paths = []
    for s in args.shards:
        sp = Path(s)
        if sp.is_dir():
            sp = sp / "generation.jsonl"
        if not sp.exists():
            log.warning("shard file missing: %s", sp)
            continue
        paths.append(sp)

    if not paths:
        raise SystemExit("no shard files found")

    n = merge_jsonl(paths, args.out)
    log.info("merged %d shards -> %s (%d unique instances)", len(paths), args.out, n)


if __name__ == "__main__":
    main()
