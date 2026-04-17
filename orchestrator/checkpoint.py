"""Per-instance JSONL checkpointing with resumable reads.

Each line is a full JSON record: {"instance_id": ..., "status": ..., ...}
Append-only; resume by reading completed instance_ids before processing.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


class JsonlCheckpoint:
    """Append-only JSONL checkpoint store, one record per instance."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed_ids(self) -> set[str]:
        """Return set of instance_ids already written with status == 'done'."""
        done: set[str] = set()
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "done" and "instance_id" in rec:
                    done.add(str(rec["instance_id"]))
        return done

    def all_records(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def append(self, record: dict[str, Any]) -> None:
        """Append a record. Flushed + fsynced so a crash doesn't lose it."""
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def merge_jsonl(shard_paths: list[str | Path], output_path: str | Path) -> int:
    """Concatenate shard JSONL files, deduping by instance_id.

    Later occurrences overwrite earlier ones. Returns # records written.
    """
    latest: dict[str, dict] = {}
    for sp in shard_paths:
        sp = Path(sp)
        if not sp.exists():
            continue
        with sp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = rec.get("instance_id")
                if iid is None:
                    continue
                latest[str(iid)] = rec

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via tempfile-rename
    fd, tmp = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for iid in sorted(latest.keys()):
                f.write(json.dumps(latest[iid], ensure_ascii=False) + "\n")
        os.replace(tmp, output_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(latest)
