"""Planner agent: decomposes an IdeaBench task into N orthogonal angles.

For SCIMON this is a no-op (single-sentence output, no decomposition).
"""
from __future__ import annotations

import logging
from typing import Any

from agents.prompts import PLANNER_SYSTEM_IDEABENCH, PLANNER_USER_IDEABENCH
from serving.vllm_client import VLLMClient
from utils.parsing import extract_json

log = logging.getLogger(__name__)


class Planner:
    def __init__(self, client: VLLMClient, num_angles: int):
        self.client = client
        self.num_angles = num_angles

    def propose_angles(self, references: list[str]) -> list[str]:
        """Return `num_angles` angle strings. Falls back to generic angles on parse fail."""
        ref_block = "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(references))
        msg = [
            {"role": "system", "content": PLANNER_SYSTEM_IDEABENCH},
            {"role": "user", "content": PLANNER_USER_IDEABENCH.format(
                references=ref_block, n=self.num_angles)},
        ]
        try:
            text = self.client.chat(msg, temperature=0.7, max_tokens=768)
        except Exception as e:
            log.warning("planner LLM call failed: %s; using fallback angles", e)
            return self._fallback_angles()

        obj = extract_json(text)
        if isinstance(obj, dict) and isinstance(obj.get("angles"), list):
            angles = [str(a).strip() for a in obj["angles"] if str(a).strip()]
            if len(angles) >= self.num_angles:
                return angles[: self.num_angles]
            if angles:
                # Pad with fallback angles if short
                return angles + self._fallback_angles()[: self.num_angles - len(angles)]

        log.warning("planner JSON parse failed; using fallback angles")
        return self._fallback_angles()

    def _fallback_angles(self) -> list[str]:
        base = [
            "Propose a new method that improves efficiency or accuracy.",
            "Propose a cross-domain application or transfer scenario.",
            "Propose a theoretical framing or unification of existing results.",
            "Propose a novel evaluation protocol or benchmark.",
            "Propose addressing a limitation or failure mode.",
            "Propose a scaling or regularization study.",
        ]
        return base[: self.num_angles] if self.num_angles <= len(base) else base + base[: self.num_angles - len(base)]
