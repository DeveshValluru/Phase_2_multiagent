"""Refiner agent: bounded rewrite loop with plateau early-stop."""
from __future__ import annotations

import json
import logging
from typing import Any

from agents.critic import Critic
from agents.prompts import REFINER_SYSTEM, REFINER_USER_SCIMON, REFINER_USER_IDEABENCH
from serving.vllm_client import VLLMClient
from utils.parsing import extract_json

log = logging.getLogger(__name__)


class Refiner:
    def __init__(
        self,
        client: VLLMClient,
        critic: Critic,
        max_iters: int = 2,
        plateau_delta: float = 0.5,
    ):
        self.client = client
        self.critic = critic
        self.max_iters = max_iters
        self.plateau_delta = plateau_delta

    # ----- SCIMON -------------------------------------------------------

    def refine_scimon(
        self,
        *,
        candidate: str,
        critique: str,
        context: str,
        seeds: list[str],
        relation: str,
        neighbors: list[str],
        starting_composite: float,
    ) -> dict[str, Any]:
        seeds_str = ", ".join(seeds) if isinstance(seeds, list) else str(seeds)
        best_text = candidate
        best_composite = starting_composite
        history = []
        current = candidate
        current_critique = critique

        for it in range(self.max_iters):
            user = REFINER_USER_SCIMON.format(
                context=context, seeds=seeds_str, relation=relation,
                candidate=current, critique=current_critique or "Make it more specific and better grounded.",
            )
            msg = [
                {"role": "system", "content": REFINER_SYSTEM},
                {"role": "user", "content": user},
            ]
            try:
                revised = self.client.chat(msg, temperature=0.4, max_tokens=256).strip()
                revised = revised.strip('"').strip("'").strip()
            except Exception as e:
                log.warning("SCIMON refiner call failed at iter %d: %s", it, e)
                break
            if not revised:
                break

            rescored = self.critic.score_scimon(
                candidate=revised, context=context, seeds=seeds, relation=relation, neighbors=neighbors,
            )
            delta = rescored["composite"] - best_composite
            history.append({"iter": it, "text": revised, "composite": rescored["composite"], "delta": delta})

            if rescored["composite"] > best_composite:
                best_text = revised
                best_composite = rescored["composite"]
                current = revised
                current_critique = rescored["critique"]

            # Early stop on plateau or regression
            if delta < self.plateau_delta:
                break

        return {"text": best_text, "composite": best_composite, "history": history}

    # ----- IdeaBench -----------------------------------------------------

    def refine_ideabench(
        self,
        *,
        candidate: dict,
        critique: str,
        references: list[str],
        starting_composite: float,
    ) -> dict[str, Any]:
        best = dict(candidate)
        best_composite = starting_composite
        history = []
        current = dict(candidate)
        current_critique = critique
        ref_block = "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(references))

        for it in range(self.max_iters):
            user = REFINER_USER_IDEABENCH.format(
                references=ref_block,
                candidate=json.dumps(current, ensure_ascii=False),
                critique=current_critique or "Make it more specific and novel.",
            )
            msg = [
                {"role": "system", "content": REFINER_SYSTEM},
                {"role": "user", "content": user},
            ]
            try:
                revised_text = self.client.chat(msg, temperature=0.5, max_tokens=768)
            except Exception as e:
                log.warning("IdeaBench refiner call failed at iter %d: %s", it, e)
                break
            revised = extract_json(revised_text)
            if not isinstance(revised, dict):
                log.warning("IdeaBench refiner JSON parse failed at iter %d", it)
                break

            rescored = self.critic.score_ideabench(candidate=revised, references=references)
            delta = rescored["composite"] - best_composite
            history.append({"iter": it, "hypothesis": revised, "composite": rescored["composite"], "delta": delta})

            if rescored["composite"] > best_composite:
                best = revised
                best_composite = rescored["composite"]
                current = revised
                current_critique = rescored["critique"]

            if delta < self.plateau_delta:
                break

        return {"hypothesis": best, "composite": best_composite, "history": history}
