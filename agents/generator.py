"""Generator agent: produces N candidate drafts.

- SCIMON: retrieval-mix (some drafts use retrieved neighbors as few-shot, some don't).
  This prevents the ROUGE-L regression seen in Phase 1 where retrieval-augmented
  generation dropped ROUGE-L from 0.1132 -> 0.1120.
- IdeaBench: one draft per Planner angle, all conditioned on re-ranked references.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.prompts import (
    GEN_SYSTEM_SCIMON, GEN_USER_SCIMON_NO_RETRIEVAL, GEN_USER_SCIMON_WITH_NEIGHBORS,
    GEN_SYSTEM_IDEABENCH, GEN_USER_IDEABENCH,
)
from serving.vllm_client import VLLMClient
from utils.parsing import extract_json

log = logging.getLogger(__name__)


class Generator:
    def __init__(self, client: VLLMClient):
        self.client = client

    # ----- SCIMON -------------------------------------------------------

    def generate_scimon(
        self,
        *,
        context: str,
        seeds: list[str],
        relation: str,
        neighbors: list[str],
        retrieval_mix: list[str],
        temperatures: list[float],
    ) -> list[dict[str, Any]]:
        """Produce `len(retrieval_mix)` drafts. Returns list of {text, retrieval, temp}."""
        assert len(retrieval_mix) == len(temperatures), "retrieval_mix and temperatures must align"
        seeds_str = ", ".join(seeds) if isinstance(seeds, list) else str(seeds)
        neighbor_block = "\n".join(f"- {n}" for n in neighbors[:5]) if neighbors else "(no neighbors)"

        drafts = []
        for mode, temp in zip(retrieval_mix, temperatures):
            if mode == "with_neighbors":
                user = GEN_USER_SCIMON_WITH_NEIGHBORS.format(
                    context=context, seeds=seeds_str, relation=relation,
                    neighbors=neighbor_block,
                )
            else:
                user = GEN_USER_SCIMON_NO_RETRIEVAL.format(
                    context=context, seeds=seeds_str, relation=relation,
                )
            msg = [
                {"role": "system", "content": GEN_SYSTEM_SCIMON},
                {"role": "user", "content": user},
            ]
            try:
                text = self.client.chat(msg, temperature=temp, max_tokens=256).strip()
            except Exception as e:
                log.warning("SCIMON generator call failed: %s", e)
                text = ""
            # strip any quotes / trailing period artifacts
            text = text.strip().strip('"').strip("'").strip()
            drafts.append({"text": text, "retrieval": mode, "temperature": temp})
        return drafts

    # ----- IdeaBench -----------------------------------------------------

    def generate_ideabench(
        self,
        *,
        references: list[str],
        angles: list[str],
        temperatures: list[float],
    ) -> list[dict[str, Any]]:
        """One draft per angle. Returns list of {hypothesis_json, angle, temp}."""
        assert len(angles) == len(temperatures), "angles and temperatures must align"
        ref_block = "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(references))

        drafts = []
        for angle, temp in zip(angles, temperatures):
            user = GEN_USER_IDEABENCH.format(references=ref_block, angle=angle)
            msg = [
                {"role": "system", "content": GEN_SYSTEM_IDEABENCH},
                {"role": "user", "content": user},
            ]
            try:
                text = self.client.chat(msg, temperature=temp, max_tokens=768)
            except Exception as e:
                log.warning("IdeaBench generator call failed: %s", e)
                text = ""
            obj = extract_json(text)
            if not isinstance(obj, dict):
                # Fallback: wrap raw text
                obj = {"title": "", "hypothesis": text.strip(), "method": "", "expected_outcome": ""}
            drafts.append({"hypothesis": obj, "angle": angle, "temperature": temp, "raw": text})
        return drafts
