"""SCIMON dataset adapter.

Loads the 194-instance gold test set and the 114K training corpus from the
EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty repo.

The SCIMON repo's dataset lives in `data/local_context_dataset.zip`. We try a
few standard filenames and fall back to any JSON/JSONL we find with the right
schema. If layout differs in your clone, set paths explicitly in scimon.yaml.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# Plausible layouts observed across SCIMON forks
_GOLD_CANDIDATES = [
    "data/gold/test.json",
    "data/gold/gold_test.json",
    "data/local_context_dataset/test_gold.json",
    "data/local_context_dataset/test.json",
    "data/test_gold.json",
    "data/test.json",
]
_TRAIN_CANDIDATES = [
    "data/local_context_dataset/train.json",
    "data/local_context_dataset/train.jsonl",
    "data/train.json",
    "data/train.jsonl",
]


def _load_json_any(path: Path) -> list[dict]:
    """Load JSON or JSONL into a list of dicts."""
    text = path.read_text(encoding="utf-8")
    # JSONL
    if path.suffix == ".jsonl" or text.lstrip().startswith("{") and "\n{" in text:
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if out:
            return out
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # common: {"instances": [...]}
        for k in ("instances", "data", "examples"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def _extract_zip_if_needed(repo_root: Path) -> None:
    z = repo_root / "data" / "local_context_dataset.zip"
    target = repo_root / "data" / "local_context_dataset"
    if z.exists() and not target.exists():
        log.info("extracting %s", z)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(repo_root / "data")


class ScimonAdapter:
    """SCIMON I/O: load test instances, load training corpus, format outputs."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        data_cfg = cfg["data"]
        self.repo_root = Path(os.path.expandvars(data_cfg["repo_root"]))
        self.gold_path = Path(os.path.expandvars(data_cfg["gold_test_path"])) if data_cfg.get("gold_test_path") else None
        self.train_path = Path(os.path.expandvars(data_cfg["train_corpus_path"])) if data_cfg.get("train_corpus_path") else None
        _extract_zip_if_needed(self.repo_root)

    # ---- data loading ---------------------------------------------------

    def load_test_instances(self) -> list[dict]:
        path = self._resolve(self.gold_path, _GOLD_CANDIDATES, "gold test set")
        raw = _load_json_any(path)
        instances = [self._normalize_instance(x, i) for i, x in enumerate(raw)]
        log.info("loaded %d SCIMON test instances from %s", len(instances), path)
        return instances

    def load_training_corpus(self) -> list[str]:
        path = self._resolve(self.train_path, _TRAIN_CANDIDATES, "training corpus")
        raw = _load_json_any(path)
        texts = []
        for x in raw:
            t = self._extract_finding_text(x)
            if t:
                texts.append(t)
        log.info("loaded %d training corpus texts from %s", len(texts), path)
        return texts

    def _resolve(self, explicit: Path | None, candidates: list[str], what: str) -> Path:
        if explicit and explicit.exists():
            return explicit
        for c in candidates:
            p = self.repo_root / c
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not locate SCIMON {what}. Tried: "
            + ", ".join(str(self.repo_root / c) for c in candidates)
            + ". Set data.gold_test_path / data.train_corpus_path in scimon.yaml."
        )

    # ---- per-instance normalization -----------------------------------

    def _normalize_instance(self, raw: dict, idx: int) -> dict:
        """Map a raw SCIMON instance to {instance_id, context, seeds, relation, gold}."""
        iid = str(raw.get("id") or raw.get("instance_id") or idx)
        context = (
            raw.get("context")
            or raw.get("local_context")
            or raw.get("background")
            or ""
        )
        if isinstance(context, list):
            context = " ".join(str(c) for c in context)
        seeds = (
            raw.get("seed_terms")
            or raw.get("seeds")
            or raw.get("entities")
            or []
        )
        if isinstance(seeds, str):
            seeds = [s.strip() for s in seeds.split(",") if s.strip()]
        relation = str(raw.get("relation") or raw.get("relation_type") or "used-for")
        gold = raw.get("gold") or raw.get("target") or raw.get("output") or raw.get("idea") or ""
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        return {
            "instance_id": iid,
            "context": str(context),
            "seeds": list(seeds),
            "relation": relation,
            "gold": str(gold),
        }

    def _extract_finding_text(self, raw: dict) -> str:
        """Pull a finding-sentence-like string from a training record."""
        for k in ("finding", "idea", "target", "output", "key_finding", "gold"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v and isinstance(v[0], str):
                return v[0].strip()
        return ""

    # ---- output formatting --------------------------------------------

    def format_output(self, instance: dict, final: dict) -> dict:
        """Shape a final agent output into the per-instance record."""
        return {
            "instance_id": instance["instance_id"],
            "benchmark": "scimon",
            "gold": instance["gold"],
            "prediction": final.get("text", ""),
            "composite_score": final.get("composite", 0.0),
            "status": "done",
        }
