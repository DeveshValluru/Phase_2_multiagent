"""Selector: pick final output from scored drafts.

- SCIMON: best-1 by composite score.
- IdeaBench: submodular top-3 balancing quality with mutual diversity.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def select_best_scimon(drafts: list[dict]) -> dict:
    """Pick the single highest-composite draft."""
    if not drafts:
        return {"text": "", "composite": 0.0}
    scored = [d for d in drafts if "composite" in d]
    if not scored:
        return drafts[0]
    return max(scored, key=lambda d: d["composite"])


def select_submodular_top_k(
    drafts: list[dict],
    k: int = 3,
    diversity_lambda: float = 0.3,
    embedder=None,
) -> list[dict]:
    """Greedy submodular selection balancing composite score with diversity.

    score(draft) + lambda * min_pairwise_dissimilarity_to_already_selected

    `embedder`: callable(list[str]) -> np.ndarray (L2-normalized); if None, a
    cheap fallback uses token-set Jaccard dissimilarity.
    """
    if not drafts:
        return []
    if len(drafts) <= k:
        return list(drafts)

    # Pull representation text
    def to_text(d: dict) -> str:
        h = d.get("hypothesis")
        if isinstance(h, dict):
            return " ".join(str(v) for v in h.values())
        if isinstance(h, str):
            return h
        return d.get("text", "") or ""

    texts = [to_text(d) for d in drafts]

    if embedder is not None:
        embs = embedder(texts)
    else:
        embs = None

    scores = np.array([float(d.get("composite", 0.0)) for d in drafts])

    selected: list[int] = []
    remaining = list(range(len(drafts)))

    while len(selected) < k and remaining:
        if not selected:
            # First pick is pure score
            best = max(remaining, key=lambda i: scores[i])
        else:
            def marginal(i: int) -> float:
                if embs is not None:
                    sims = embs[i] @ embs[selected].T
                    diss = 1.0 - float(np.max(sims)) if len(sims) else 1.0
                else:
                    # Jaccard dissimilarity fallback
                    ti = set(texts[i].lower().split())
                    diss = 1.0
                    for j in selected:
                        tj = set(texts[j].lower().split())
                        if not ti or not tj:
                            continue
                        jac = len(ti & tj) / max(1, len(ti | tj))
                        diss = min(diss, 1.0 - jac)
                return float(scores[i]) + diversity_lambda * diss
            best = max(remaining, key=marginal)
        selected.append(best)
        remaining.remove(best)

    return [drafts[i] for i in selected]
