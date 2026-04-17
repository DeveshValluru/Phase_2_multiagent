"""Robust parsing for LLM outputs.

Handles Qwen3-32B <think>...</think> traces + malformed JSON with regex fallbacks.
Phase 1 had 40-55% parse failures on IdeaBench rankings — this module fixes that.
"""
from __future__ import annotations

import json
import re
from typing import Any


_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_RANK_LINE_RE = re.compile(r"\b([A-Z])\b\s*(?:[:.)\-]|$)")


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict | list | None:
    """Extract a JSON object/array from arbitrary model output.

    Tries in order:
      1. Direct json.loads on stripped text
      2. Content inside ```json ... ``` fence
      3. First {...} block by brace matching
    Returns None if everything fails.
    """
    if not text:
        return None

    text = strip_think(text)

    # Try direct parse
    for candidate in (text, text.strip()):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try fenced code block
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try first balanced {...}
    obj = _balanced_brace_extract(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try first [...]
    arr = _balanced_bracket_extract(text)
    if arr is not None:
        try:
            return json.loads(arr)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _balanced_brace_extract(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _balanced_bracket_extract(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_rubric_scores(text: str, axes: list[str]) -> dict[str, float] | None:
    """Parse a rubric-score JSON. Falls back to regex if JSON parse fails.

    Expects keys matching `axes` with numeric values (0-10).
    """
    obj = extract_json(text)
    if isinstance(obj, dict):
        try:
            out = {}
            for ax in axes:
                val = obj.get(ax)
                if val is None:
                    # Case-insensitive fallback
                    for k, v in obj.items():
                        if k.lower() == ax.lower():
                            val = v
                            break
                if val is None:
                    return None
                out[ax] = float(val)
            return out
        except (ValueError, TypeError):
            pass

    # Regex fallback: look for "axis: 7.5" or "axis = 7"
    text = strip_think(text)
    out = {}
    for ax in axes:
        pat = re.compile(
            rf"\"?{re.escape(ax)}\"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        )
        m = pat.search(text)
        if not m:
            return None
        out[ax] = float(m.group(1))
    return out


def parse_ranking(text: str, n_items: int) -> list[int] | None:
    """Parse an LLM ranking output into a list of 0-indexed positions.

    Expected formats:
      - JSON array: [1, 3, 0, 2] or ["B", "D", "A", "C"]
      - Text list: "1. B\n2. D\n3. A\n4. C"
      - Inline: "Ranking: B, D, A, C"
    Returns None if parsing fails completely.
    """
    text = strip_think(text)

    obj = extract_json(text)
    if isinstance(obj, list) and len(obj) == n_items:
        try:
            return [_normalize_rank_item(x, n_items) for x in obj]
        except (ValueError, TypeError):
            pass

    # Regex fallback: find letters A, B, C, ... in order
    letters = re.findall(r"\b([A-" + chr(ord("A") + n_items - 1) + r"])\b", text)
    if len(letters) >= n_items:
        seen = []
        for L in letters:
            idx = ord(L) - ord("A")
            if 0 <= idx < n_items and idx not in seen:
                seen.append(idx)
            if len(seen) == n_items:
                return seen

    # Regex fallback: find numbers
    nums = re.findall(r"\b([0-9]+)\b", text)
    if len(nums) >= n_items:
        seen = []
        for n in nums:
            idx = int(n)
            if 0 < idx <= n_items and (idx - 1) not in seen:
                seen.append(idx - 1)
            if len(seen) == n_items:
                return seen

    return None


def _normalize_rank_item(x: Any, n_items: int) -> int:
    if isinstance(x, int):
        if 0 <= x < n_items:
            return x
        if 1 <= x <= n_items:
            return x - 1
        raise ValueError(f"rank index out of range: {x}")
    if isinstance(x, str):
        x = x.strip().strip(".:,;)(")
        if len(x) == 1 and x.isalpha():
            return ord(x.upper()) - ord("A")
        try:
            i = int(x)
            if 1 <= i <= n_items:
                return i - 1
            return i
        except ValueError:
            raise ValueError(f"unparseable rank item: {x!r}")
    raise ValueError(f"unexpected rank item type: {type(x)}")
