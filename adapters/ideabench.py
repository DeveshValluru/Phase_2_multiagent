"""IdeaBench dataset adapter.

Loads the 2,374 target papers and their filtered reference abstracts from the
amir-hassan25/IdeaBench repo. Low-resource setting (num_ref=3, num_hyp=3).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


_PAPER_CANDIDATES = [
    "data/papers.csv",
    "data/target_papers.csv",
    "data/papers.jsonl",
    "data/papers.json",
]
_REFERENCE_CANDIDATES = [
    "data/references.csv",
    "data/filtered_references.csv",
    "data/references.jsonl",
    "data/references.json",
]


def _load_tabular(path: Path) -> list[dict]:
    if path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        return df.to_dict("records")
    if path.suffix == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    raise ValueError(f"unsupported file type: {path}")


class IdeaBenchAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        data_cfg = cfg["data"]
        self.repo_root = Path(os.path.expandvars(data_cfg["repo_root"]))
        self.papers_path = Path(os.path.expandvars(data_cfg["papers_path"])) if data_cfg.get("papers_path") else None
        self.references_path = Path(os.path.expandvars(data_cfg["references_path"])) if data_cfg.get("references_path") else None
        self.num_ref = int(data_cfg.get("num_ref", 3))
        self.num_hyp = int(data_cfg.get("num_hyp", 3))
        self.filtered_ref = bool(data_cfg.get("filtered_ref", True))
        self.all_ref = bool(data_cfg.get("all_ref", False))

    def _resolve(self, explicit: Path | None, candidates: list[str], what: str) -> Path:
        if explicit and explicit.exists():
            return explicit
        for c in candidates:
            p = self.repo_root / c
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not locate IdeaBench {what}. Tried: "
            + ", ".join(str(self.repo_root / c) for c in candidates)
            + ". Set data.papers_path / data.references_path in ideabench.yaml."
        )

    def load_papers(self) -> list[dict]:
        path = self._resolve(self.papers_path, _PAPER_CANDIDATES, "papers")
        rows = _load_tabular(path)
        log.info("loaded %d papers from %s", len(rows), path)
        return rows

    def load_references(self) -> dict[str, list[dict]]:
        path = self._resolve(self.references_path, _REFERENCE_CANDIDATES, "references")
        rows = _load_tabular(path)
        by_paper: dict[str, list[dict]] = {}
        for r in rows:
            pid = str(r.get("paper_id") or r.get("target_paper_id") or r.get("parent_id") or "")
            if not pid:
                continue
            by_paper.setdefault(pid, []).append(r)
        log.info("loaded references for %d papers from %s", len(by_paper), path)
        return by_paper

    # ---- per-instance normalization -----------------------------------

    def load_test_instances(self) -> list[dict]:
        papers = self.load_papers()
        refs = self.load_references()
        instances = []
        skipped = 0
        for p in papers:
            iid = str(p.get("paper_id") or p.get("id") or "")
            if not iid:
                skipped += 1
                continue
            target_abstract = str(p.get("abstract") or p.get("target_abstract") or "")
            title = str(p.get("title") or "")
            refs_for_paper = refs.get(iid, [])

            if self.filtered_ref:
                refs_for_paper = [r for r in refs_for_paper if self._is_filtered(r)]

            ref_abstracts = [str(r.get("abstract") or r.get("text") or "") for r in refs_for_paper]
            ref_abstracts = [a for a in ref_abstracts if a.strip()]

            if not self.all_ref:
                ref_abstracts = ref_abstracts[: self.num_ref]

            if not ref_abstracts:
                skipped += 1
                continue

            instances.append({
                "instance_id": iid,
                "title": title,
                "target_abstract": target_abstract,
                "references": ref_abstracts,
                "num_hyp": self.num_hyp,
            })
        if skipped:
            log.warning("skipped %d papers (missing id or references)", skipped)
        log.info("prepared %d IdeaBench instances", len(instances))
        return instances

    @staticmethod
    def _is_filtered(r: dict) -> bool:
        # If the dataset tags filtered refs, honor it; otherwise accept.
        for k in ("filtered", "is_filtered", "keep"):
            if k in r:
                return bool(r[k])
        return True

    # ---- output formatting --------------------------------------------

    def format_output(self, instance: dict, final_list: list[dict]) -> dict:
        """Shape 3 final hypotheses into the per-instance record."""
        hyps = []
        for f in final_list[: instance["num_hyp"]]:
            h = f.get("hypothesis") if isinstance(f.get("hypothesis"), dict) else {"hypothesis": f.get("text", "")}
            hyps.append(h)
        return {
            "instance_id": instance["instance_id"],
            "benchmark": "ideabench",
            "title": instance["title"],
            "target_abstract": instance["target_abstract"],
            "hypotheses": hyps,
            "composite_scores": [f.get("composite", 0.0) for f in final_list[: instance["num_hyp"]]],
            "status": "done",
        }
