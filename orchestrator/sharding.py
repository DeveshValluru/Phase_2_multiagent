"""Shard selection: --start/--end, --shard i/N, --ids file.txt"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def select_shard(
    ids: list[str],
    start: int | None = None,
    end: int | None = None,
    shard: str | None = None,
    ids_file: str | None = None,
) -> list[str]:
    """Return the subset of `ids` to process based on shard args.

    Args:
      ids:       full list of instance IDs (already sorted, deterministic)
      start:     zero-indexed start (inclusive)
      end:       zero-indexed end (exclusive)
      shard:     "i/N" form, e.g. "1/4" for the first quarter
      ids_file:  path to a text file with one instance_id per line

    Exactly one of (start/end), shard, ids_file may be specified.
    If none are given, returns the full list.
    """
    specifiers = sum(x is not None for x in (start, shard, ids_file))
    if end is not None and start is None:
        specifiers += 1
    if specifiers > 1:
        raise ValueError(
            "Specify only one of: --start/--end, --shard i/N, --ids file.txt"
        )

    if ids_file:
        wanted = {line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
        return [x for x in ids if x in wanted]

    if shard:
        i, n = shard.split("/")
        i, n = int(i), int(n)
        if not (1 <= i <= n):
            raise ValueError(f"shard index {i} out of range 1..{n}")
        total = len(ids)
        size = total // n
        rem = total % n
        # First `rem` shards get one extra instance
        s = (i - 1) * size + min(i - 1, rem)
        e = s + size + (1 if i - 1 < rem else 0)
        return ids[s:e]

    if start is not None or end is not None:
        s = start or 0
        e = end if end is not None else len(ids)
        return ids[s:e]

    return list(ids)


def summarize_shard(selected: Iterable[str], total: int) -> str:
    selected = list(selected)
    return f"{len(selected)} / {total} instances"
