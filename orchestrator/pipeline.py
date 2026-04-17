"""Multi-agent pipeline orchestrator.

Shared agent graph; dataset adapter is the only thing that differs between
SCIMON and IdeaBench. Per-instance JSONL checkpointing makes the run resumable
from any point, and sharding lets multiple team members work in parallel.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agents.critic import Critic
from agents.generator import Generator
from agents.planner import Planner
from agents.refiner import Refiner
from agents.retriever import CorpusIndex, InlineReRanker
from agents.selector import select_best_scimon, select_submodular_top_k
from orchestrator.checkpoint import JsonlCheckpoint
from serving.vllm_client import VLLMClient, client_from_config

log = logging.getLogger(__name__)


class Pipeline:
    """Entry-point object; one instance per process."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.benchmark = cfg["benchmark"]
        self.client: VLLMClient | None = None
        self._built = False

    # ---- setup ---------------------------------------------------------

    def build(self) -> None:
        if self._built:
            return
        self.client = client_from_config(self.cfg["llm"])
        log.info("waiting for vLLM readiness at %s", self.client.cfg.base_url)
        self.client.wait_ready(timeout_s=900.0)
        self.client.start_keepalive(60.0)

        # Shared agents
        self.critic = Critic(self.client, self.cfg["critic"]["rubric_weights"])
        self.generator = Generator(self.client)
        self.refiner = Refiner(
            self.client, self.critic,
            max_iters=int(self.cfg["refiner"]["max_iters"]),
            plateau_delta=float(self.cfg["refiner"]["plateau_delta"]),
        )
        self.planner = Planner(
            self.client,
            num_angles=int(self.cfg.get("planner", {}).get("num_angles", 1)),
        )
        self._built = True

    def teardown(self) -> None:
        if self.client is not None:
            self.client.stop_keepalive()

    # ---- public entry --------------------------------------------------

    def run(
        self,
        instances: list[dict],
        adapter,
        corpus_index: CorpusIndex | None = None,
        inline_reranker: InlineReRanker | None = None,
    ) -> None:
        self.build()
        try:
            ckpt = JsonlCheckpoint(self._output_path())
            done = ckpt.completed_ids()
            log.info("resuming: %d already-completed instances skipped", len(done))

            todo = [x for x in instances if x["instance_id"] not in done]
            log.info("processing %d / %d instances", len(todo), len(instances))

            for inst in tqdm(todo, desc=f"{self.benchmark}", unit="inst"):
                if self.client is not None and self.client.term_requested:
                    log.warning("SIGTERM received — flushing and exiting cleanly")
                    break
                try:
                    if self.benchmark == "scimon":
                        rec = self._run_scimon(inst, corpus_index)
                    elif self.benchmark == "ideabench":
                        rec = self._run_ideabench(inst, inline_reranker)
                    else:
                        raise ValueError(f"unknown benchmark: {self.benchmark}")
                    out = adapter.format_output(inst, rec)
                except Exception as e:
                    log.exception("instance %s failed: %s", inst.get("instance_id"), e)
                    out = {
                        "instance_id": inst["instance_id"],
                        "benchmark": self.benchmark,
                        "status": "error",
                        "error": str(e),
                    }
                ckpt.append(out)
        finally:
            self.teardown()

    # ---- benchmark-specific flows -------------------------------------

    def _run_scimon(self, inst: dict, corpus_index: CorpusIndex | None) -> dict:
        # 1. Retrieval
        neighbors: list[str] = []
        if corpus_index is not None:
            query = f"{' '.join(inst['seeds'])} {inst['relation']} {inst['context']}"
            top = corpus_index.top_k(query, k=int(self.cfg["retrieval"]["top_k"]))
            neighbors = [t for _, _, t in top]

        # 2. Generate N drafts with retrieval-mix
        gcfg = self.cfg["generator"]
        drafts = self.generator.generate_scimon(
            context=inst["context"], seeds=inst["seeds"], relation=inst["relation"],
            neighbors=neighbors,
            retrieval_mix=gcfg["retrieval_mix"],
            temperatures=[float(t) for t in gcfg["temperatures"]],
        )

        # 3. Critique each draft
        for d in drafts:
            scored = self.critic.score_scimon(
                candidate=d["text"], context=inst["context"],
                seeds=inst["seeds"], relation=inst["relation"], neighbors=neighbors,
            )
            d.update({"composite": scored["composite"], "scores": scored["scores"],
                      "critique": scored["critique"]})

        # 4. Refine top-1
        best = max(drafts, key=lambda d: d["composite"])
        refined = self.refiner.refine_scimon(
            candidate=best["text"], critique=best["critique"],
            context=inst["context"], seeds=inst["seeds"], relation=inst["relation"],
            neighbors=neighbors, starting_composite=best["composite"],
        )
        best["text"] = refined["text"]
        best["composite"] = refined["composite"]
        best["refine_history"] = refined["history"]

        # 5. Select (redundant here but keeps API consistent)
        final = select_best_scimon(drafts)
        return final

    def _run_ideabench(self, inst: dict, inline_reranker: InlineReRanker | None) -> list[dict]:
        refs = inst["references"]
        # 1. Re-rank references by relevance to target
        if inline_reranker is not None and inst.get("title"):
            reranked = inline_reranker.rerank(
                query=f"{inst['title']} {inst['target_abstract'][:500]}",
                candidates=refs,
            )
            refs = [t for _, _, t in reranked]

        # 2. Plan angles
        angles = self.planner.propose_angles(refs)

        # 3. Generate one draft per angle
        gcfg = self.cfg["generator"]
        temps = [float(t) for t in gcfg["temperatures"]]
        if len(temps) < len(angles):
            temps = (temps * ((len(angles) // len(temps)) + 1))[: len(angles)]
        drafts = self.generator.generate_ideabench(
            references=refs, angles=angles, temperatures=temps[: len(angles)],
        )

        # 4. Score each draft (parallel — drafts are independent)
        def _score(d: dict) -> dict:
            return self.critic.score_ideabench(candidate=d["hypothesis"], references=refs)

        with ThreadPoolExecutor(max_workers=max(1, len(drafts))) as ex:
            scored_list = list(ex.map(_score, drafts))

        for d, scored in zip(drafts, scored_list):
            d.update({"composite": scored["composite"], "scores": scored["scores"],
                      "critique": scored["critique"]})

        # 5. Refine top-k
        refine_top_k = int(self.cfg["refiner"].get("refine_top_k", inst["num_hyp"]))
        drafts_sorted = sorted(drafts, key=lambda d: d["composite"], reverse=True)
        for d in drafts_sorted[:refine_top_k]:
            refined = self.refiner.refine_ideabench(
                candidate=d["hypothesis"], critique=d["critique"],
                references=refs, starting_composite=d["composite"],
            )
            d["hypothesis"] = refined["hypothesis"]
            d["composite"] = refined["composite"]
            d["refine_history"] = refined["history"]

        # 6. Submodular top-3
        scfg = self.cfg["selector"]
        k = int(inst["num_hyp"])
        final = select_submodular_top_k(
            drafts_sorted, k=k,
            diversity_lambda=float(scfg.get("diversity_lambda", 0.3)),
        )
        return final

    # ---- helpers -------------------------------------------------------

    def _output_path(self) -> Path:
        return Path(self.cfg["run"]["output_dir"]).expanduser() / "generation.jsonl"
