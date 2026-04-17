"""Critic agent: rubric scoring of candidates.

SCIMON rubric (fixes Phase-1 novelty-boost regression by optimizing for
gold-likeness, not dissimilarity-to-training):
  grounding, specificity, coherence, local_novelty, gold_likeness

IdeaBench rubric:
  grounding, specificity, coherence, novelty, feasibility
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agents.prompts import CRITIC_SYSTEM, CRITIC_USER_SCIMON, CRITIC_USER_IDEABENCH
from serving.vllm_client import VLLMClient
from utils.parsing import parse_rubric_scores

log = logging.getLogger(__name__)


SCIMON_AXES = ["grounding", "specificity", "coherence", "local_novelty", "gold_likeness"]
IDEABENCH_AXES = ["grounding", "specificity", "coherence", "novelty", "feasibility"]


class Critic:
    def __init__(self, client: VLLMClient, rubric_weights: dict[str, float]):
        self.client = client
        self.weights = rubric_weights

    # ----- SCIMON -------------------------------------------------------

    def score_scimon(
        self,
        *,
        candidate: str,
        context: str,
        seeds: list[str],
        relation: str,
        neighbors: list[str],
    ) -> dict[str, Any]:
        seeds_str = ", ".join(seeds) if isinstance(seeds, list) else str(seeds)
        neighbor_block = "\n".join(f"- {n}" for n in neighbors[:5]) or "(none)"
        user = CRITIC_USER_SCIMON.format(
            context=context, seeds=seeds_str, relation=relation,
            neighbors=neighbor_block, candidate=candidate,
        )
        return self._score(user, SCIMON_AXES)

    # ----- IdeaBench -----------------------------------------------------

    def score_ideabench(
        self,
        *,
        candidate: dict | str,
        references: list[str],
    ) -> dict[str, Any]:
        cand_text = candidate if isinstance(candidate, str) else json.dumps(candidate, ensure_ascii=False)
        ref_block = "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(references))
        user = CRITIC_USER_IDEABENCH.format(references=ref_block, candidate=cand_text)
        return self._score(user, IDEABENCH_AXES)

    # ----- shared -------------------------------------------------------

    def _score(self, user: str, axes: list[str]) -> dict[str, Any]:
        msg = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            text = self.client.chat(msg, temperature=0.2, max_tokens=400)
        except Exception as e:
            log.warning("critic LLM call failed: %s", e)
            text = ""

        scores = parse_rubric_scores(text, axes)
        if scores is None:
            log.warning("critic parse failed; assigning neutral 5.0 to all axes")
            scores = {ax: 5.0 for ax in axes}

        # Compute weighted composite
        composite = 0.0
        for ax, v in scores.items():
            w = self.weights.get(ax, 1.0 / len(axes))
            composite += w * float(v)

        # Extract critique text if present
        critique = ""
        try:
            from utils.parsing import extract_json
            obj = extract_json(text)
            if isinstance(obj, dict):
                critique = str(obj.get("critique", "")).strip()
        except Exception:
            pass

        return {
            "scores": scores,
            "composite": composite,
            "critique": critique,
            "raw": text,
        }
