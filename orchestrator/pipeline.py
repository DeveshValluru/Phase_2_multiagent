"""Multi-agent pipeline orchestrator.

Shared agent graph; dataset adapter is the only thing that differs between
SCIMON and IdeaBench. Per-instance JSONL checkpointing makes the run resumable
from any point, and sharding lets multiple team members work in parallel.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

            # Instance-level concurrency: multiple papers processed in parallel.
            # Default 1 keeps original sequential behavior. Raising this feeds
            # vLLM more concurrent requests -> higher HPU utilization.
            instance_concurrency = int(
                self.cfg.get("run", {}).get("instance_concurrency", 1)
            )
            ckpt_lock = threading.Lock()
            # InlineReRanker wraps a SentenceTransformer with a lazy-loaded
            # model; serialize calls to avoid init races and concurrent encode
            # contention. The call is short (CPU-only, <1s) so no throughput loss.
            reranker_lock = threading.Lock()

            def _process_one(inst: dict) -> None:
                if self.client is not None and self.client.term_requested:
                    return
                try:
                    if self.benchmark == "scimon":
                        rec = self._run_scimon(inst, corpus_index)
                    elif self.benchmark == "ideabench":
                        wrapped = _LockedReRanker(inline_reranker, reranker_lock) \
                            if inline_reranker is not None else None
                        rec = self._run_ideabench(inst, wrapped)
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
                with ckpt_lock:
                    ckpt.append(out)

            if instance_concurrency <= 1:
                # Sequential (original behavior)
                for inst in tqdm(todo, desc=f"{self.benchmark}", unit="inst"):
                    if self.client is not None and self.client.term_requested:
                        log.warning("SIGTERM received — flushing and exiting cleanly")
                        break
                    _process_one(inst)
            else:
                log.info("outer-loop concurrency: %d papers in flight", instance_concurrency)
                with ThreadPoolExecutor(max_workers=instance_concurrency) as ex:
                    futures = {ex.submit(_process_one, inst): inst for inst in todo}
                    try:
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"{self.benchmark}",
                            unit="inst",
                        ):
                            # Force any exceptions inside the future to surface
                            # (they're already logged inside _process_one, but
                            # re-raise so debuggers see them).
                            try:
                                fut.result()
                            except Exception as e:
                                log.exception("worker future raised: %s", e)

                            if self.client is not None and self.client.term_requested:
                                log.warning("SIGTERM received — cancelling remaining futures")
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break
                    except KeyboardInterrupt:
                        log.warning("KeyboardInterrupt — cancelling futures")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        raise
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

        # 5. Refine top-k (parallel across drafts — each refinement is independent;
        # the inner iteration loop within one refinement stays sequential because
        # iter N+1 depends on iter N's critique.)
        refine_top_k = int(self.cfg["refiner"].get("refine_top_k", inst["num_hyp"]))
        drafts_sorted = sorted(drafts, key=lambda d: d["composite"], reverse=True)
        to_refine = drafts_sorted[:refine_top_k]

        def _refine(d: dict) -> dict:
            return self.refiner.refine_ideabench(
                candidate=d["hypothesis"], critique=d["critique"],
                references=refs, starting_composite=d["composite"],
            )

        with ThreadPoolExecutor(max_workers=max(1, len(to_refine))) as ex:
            refined_list = list(ex.map(_refine, to_refine))

        for d, refined in zip(to_refine, refined_list):
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


class _LockedReRanker:
    """Thin wrapper that serializes InlineReRanker.rerank() calls.

    SentenceTransformer lazy-init + encode() isn't guaranteed thread-safe; since
    a reranker call is short (CPU-only) we just serialize it rather than try
    to make it concurrent.
    """

    def __init__(self, inner, lock: threading.Lock):
        self._inner = inner
        self._lock = lock

    def rerank(self, *args, **kwargs):
        with self._lock:
            return self._inner.rerank(*args, **kwargs)
