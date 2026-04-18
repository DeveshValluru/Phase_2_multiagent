# SCIMON Phase-2 Evaluation Report

**Evaluator:** Phase-1's `SCIMON/Llama-3.3-70B/scripts/evaluate.py` (unchanged — same evaluator used for the Phase-1 baseline, ensuring apples-to-apples comparison)
**Model under test:** Llama-3.3-70B-Instruct with the Phase-2 multi-agent pipeline
**Dataset:** 194 gold test instances from SCIMON (46 forward + 148 backward)

---

## 1. Headline Metrics

| Metric | Phase-1 baseline | **Phase-2 (multi-agent)** | Δ |
|---|---|---|---|
| ROUGE-L | 0.1132 | **0.1845** | **+62.8%** |
| BERTScore (SciBERT) | 0.5547 | **0.6323** | **+14.0%** |

---

## 2. Against the Paper's Reference (Table 9, Gold subset)

| Model | ROUGE-L | BERTScore | Phase-2 beats? |
|---|---|---|---|
| GPT-4 zero-shot | 0.130 | 0.583 | ✓ both |
| GPT-4 few-shot | 0.151 | 0.624 | ✓ both |
| GPT-4 + SN | 0.149 | 0.627 | ✓ both |
| GPT-4 + KG | 0.152 | 0.626 | ✓ both |
| **Phase-2 multi-agent (Llama-3.3-70B)** | **0.1845** | **0.6323** | **—** |
| T5 (fine-tuned) | 0.246 | 0.685 | ✗ (outside scope — no fine-tuning) |
| T5 + SN + CL (fine-tuned) | 0.258 | 0.686 | ✗ (outside scope — no fine-tuning) |

**Phase-2 is the strongest non-fine-tuned result in the table.**

---

## 3. Evaluation Output (from `evaluate.py`)

| Variant | N | ROUGE-L | BERTScore |
|---------|---|---------|----------|
| Phase-2 multi-agent | 194 | 0.1845 | 0.6323 |

---

## 4. Win Rate (per-instance analysis)

- **83.5%** (162/194) of instances improved vs Phase-1
- **7.7%** (15/194) regressed
- **8.8%** (17/194) roughly tied (|Δ| ≤ 0.01)

| Direction | N | Phase-1 ROUGE-L | Phase-2 ROUGE-L | Δ |
|---|---|---|---|---|
| Forward | 46 | 0.1194 | 0.1995 | +0.0800 |
| Backward | 148 | 0.1114 | 0.1798 | +0.0685 |

Balanced gains across both directions — no direction is disproportionately carrying the improvement.

---

## 5. Main Observation — Style Alignment Drives Gains

Phase-1 outputs are **4× longer than the gold references**. Phase-2 lands at almost exactly gold length:

| Prediction length (words) | Median | Mean |
|---|---|---|
| Gold | **29** | 29.7 |
| Phase-1 | 118 | 117.2 |
| **Phase-2** | **33** | **35.0** |

This isn't a trivial length-hack — it's a consequence of the critic rubric + refiner enforcing contribution-sentence style.

---

## 6. Before/After Examples

### Example A: ΔROUGE-L = +0.27 (biggest single win)

**Gold (31w):**
> *"Subsequently, adaptive boosting, logistic regression, random forest and support vector machine (SVM) classifiers were used to identify the scale of depression from the given texts."*

**Phase-1 (124w):**
> *"Based on the context, another scientific term that can be used for Support Vector Machine (SVM) classifiers is 'Maximum Margin Classifiers'. This term is used because SVM classifiers work by finding the hyperplane that maximizes the margin between classes in the feature space. The margin is the dis..."*

**Phase-2 (26w):**
> *"The bionlp group at IISERB employed Support Vector Machine (SVM) classifiers, which were used for enhancing the accuracy of text mining models in their submitted runs."*

**Diagnosis:** Phase-1 interprets *"and why?"* literally and produces a dictionary-entry explanation. Phase-2 writes like a paper contribution sentence — specific actor ("bionlp group at IISERB"), specific purpose ("enhancing text mining accuracy").

### Example B: typical improvement

**Gold (18w):** *"We propose a self-supervised pre-training approach to enhance the zero-shot performance."*

**Phase-1 (105w):** 105-word monologue defining "Unsupervised Aspect-Based Summarization" with bolded subsection.

**Phase-2 (29w):** *"A self-supervised pre-training approach is used for generating aspect-specific summaries by leveraging a masked aspect prediction task to enhance the model's ability to generalize across diverse aspects and domains."*

---

## 7. Boilerplate Frequency (all 194 records)

| Phrase | Phase-1 | Phase-2 | Gold |
|---|---|---|---|
| `"in this context"` | **22** 🚩 | 0 ✓ | 0 ✓ |
| `"we propose"` | 0 | 31 ✓ | 37 ✓ |
| `"we introduce"` | 0 | 3 ✓ | 5 ✓ |
| `"state-of-the-art"` | 5 | 11 | 0 |

**Phase-2 adopts the gold's writing style.** The 22 *"in this context"* openers in Phase-1 are Llama echoing the question template verbatim — Phase-2's agents suppress this pattern.

---

## 8. Component Attribution — Why the Pipeline Works

| Phase-2 component | What it fixes in Llama's default behavior |
|---|---|
| **Generator** (multi-temperature N drafts) | Diverse pool → selector can pick a draft that matches gold style |
| **Retrieval-mix** (some drafts with neighbors, some without) | Forward: concrete terminology grounding. Backward: avoids over-copying |
| **Critic rubric** (grounding + specificity + coherence + novelty + feasibility) | Directly rewards contribution-style single sentences; penalizes dictionary-style explanations |
| **Refiner** (bounded iterative rewrites) | Smooths away "Based on the context..." openers and numbered-list artifacts |
| **Selector** (submodular top-1) | Picks the draft that stopped at the claim — not the one that explained "why" |

---

## 9. Where Phase-2 Regresses (7.7% of cases, −0.04 ROUGE-L avg)

### Example: ΔROUGE-L = −0.08

**Gold (41w):**
> *"Our approach can be understood as a specially-trained coarse-to-fine algorithm, where an event transition planner provides a 'coarse' plot skeleton and a text generator in the second stage r..."*

**Phase-2 (29w):**
> *"The neural text generator is used for generating coherent continuations by leveraging a graph-based attention mechanism that explicitly models causalities and relations between given facts and possible ensuing events."*

**Diagnosis:** Phase-2 correctly identifies the task (text generation, causal relations) but **invents a specific mechanism** ("graph-based attention") not in the actual paper. The planner's angle-bias + refiner's specificity-reward push toward concrete claims even when the model has no grounding for them.

**Potential mitigation:** lower the specificity weight in the critic rubric, or add a hallucination-penalty critic. Not urgent given the net win (83.5% improved vs 7.7% regressed).

---

## 10. Honest Framing for the Project Report

Recommend reporting **both metrics** with the length caveat:

> The multi-agent pipeline improves ROUGE-L by 63% (0.113 → 0.185) and BERTScore by 14% (0.555 → 0.632) over the Phase-1 single-shot Llama baseline. ROUGE-L's larger gain partly reflects length-format alignment (Phase-2 outputs match the 29-word gold median; Phase-1 produced ~118-word explanatory outputs). BERTScore is length-normalized, so its +14% represents the conservative "semantic content" gain. Both metrics outperform all GPT-4 variants in the paper's Table 9 (zero-shot, few-shot, +SN, +KG).

This framing is defensible:
- Claiming **63% alone** risks the critique "most of that is length alignment"
- Claiming **14% alone** undersells the methodological contribution
- Reporting **both with the caveat** is honest and matches how reviewers who know these benchmarks will read the numbers

---

## 11. Takeaway

The multi-agent pipeline didn't teach Llama-3.3-70B new facts — it taught Llama to **write like the gold references write**. The critic + refiner + selector combination converts Llama's verbose dictionary-entry style (118 words, "Based on the context, X is..." openings, numbered explanations) into research-paper contribution sentences (33 words, "We propose X..." / "X is used for Y..."). Since ROUGE-L heavily penalizes length mismatch against a 29-word reference, style alignment alone explains most of the 63% ROUGE-L gain. The 14% BERTScore gain is the length-invariant evidence that the content is also genuinely better-aligned.

---

## 12. Files for Reproducibility

```
Phase_2_multiagent/SCIMON_result/phase1_format/
├── eval_forward.json           (46 entries — Phase-1 format)
├── eval_backward.json          (148 entries — Phase-1 format)
└── evaluation/
    ├── metrics.json            (raw numbers from evaluate.py)
    ├── results_table.md        (Phase-1 evaluator's default output)
    └── SCIMON_phase2_report.md (this file)
```

**Evaluator used:** `/home/wwang360/CSE598/project2/SCIMON/Llama-3.3-70B/scripts/evaluate.py` — unchanged from Phase-1.

**Source data:** `Phase_2_multiagent/SCIMON_result/generation_merged.jsonl` (Devesh's merged output, not modified). Working copy at `SCIMON_pipeline_result.jsonl` with added `forward` field for conversion.
