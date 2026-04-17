# External dependencies

Clone the two original paper repos here before running. They are *not* checked
into this repo to keep it clean and avoid license issues.

```bash
cd external
git clone https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty SCIMON
git clone https://github.com/amir-hassan25/IdeaBench IdeaBench
```

Our dataset adapters auto-detect common file layouts inside each repo's `data/`
directory. If layouts differ in your clone, override paths in:

- `configs/scimon.yaml` → `data.gold_test_path`, `data.train_corpus_path`
- `configs/ideabench.yaml` → `data.papers_path`, `data.references_path`

Nothing else needs editing.
