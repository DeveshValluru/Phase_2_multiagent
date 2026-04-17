"""SCIMON evaluation: ROUGE-L + BERTScore (SciBERT).

Matches Phase 1 metric setup. Operates on the merged final JSONL.
Target baselines (Phase-1 Llama-3.3-70B):
  ROUGE-L    > 0.1132
  BERTScore  > 0.5547
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from utils.logging import setup_logging

log = setup_logging()


SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"


def load_predictions(path: Path) -> tuple[list[str], list[str], list[str]]:
    ids, golds, preds = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") != "done":
            continue
        ids.append(str(r["instance_id"]))
        golds.append(str(r.get("gold", "")))
        preds.append(str(r.get("prediction", "")))
    return ids, golds, preds


def compute_rouge_l(preds: list[str], golds: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fs = []
    for p, g in zip(preds, golds):
        fs.append(scorer.score(g, p)["rougeL"].fmeasure)
    return sum(fs) / max(1, len(fs))


def compute_bertscore(
    preds: list[str], golds: list[str], model: str = SCIBERT_MODEL
) -> float:
    from bert_score import score as bscore
    P, R, F = bscore(
        preds, golds, model_type=model, num_layers=9,
        verbose=False, device=None, batch_size=64, rescale_with_baseline=False,
    )
    return float(F.mean().item())


def main():
    p = argparse.ArgumentParser(description="SCIMON eval")
    p.add_argument("--predictions", required=True, help="merged final JSONL")
    p.add_argument("--model", default=SCIBERT_MODEL, help="BERTScore model")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    ids, golds, preds = load_predictions(Path(args.predictions))
    log.info("evaluating %d predictions", len(ids))
    if not ids:
        raise SystemExit("no completed predictions found")

    rl = compute_rouge_l(preds, golds)
    bs = compute_bertscore(preds, golds, model=args.model)

    result = {
        "benchmark": "scimon",
        "n_instances": len(ids),
        "rouge_l": rl,
        "bertscore_f1": bs,
        "model": args.model,
        "baselines": {"rouge_l": 0.1132, "bertscore_f1": 0.5547},
        "beat_baseline": {
            "rouge_l": rl > 0.1132,
            "bertscore_f1": bs > 0.5547,
        },
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
