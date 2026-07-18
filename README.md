# Explaining Predictions on the AIFB Knowledge Graph

**Course:** Explainable Artificial Intelligence — Summer Term 2026

**University:** Paderborn University

**Instructor:** Dr. Stefan Heindorf

**Author:** Atchiyya Naidu Chitikela:4064214

**Group:** 1 (individual submission)

**Test Accuracy (AIFB, held-out test set): 77.8%**
**Explanation Fidelity (surrogate decision tree): 77.2%**

---

## What this project does

This is a **knowledge-graph explainability pipeline**. Given the AIFB
RDF knowledge graph (researchers, publications, and research groups at
AIFB Karlsruhe), it predicts which **research group** a person belongs
to, and explains *why* the model made that prediction in human-readable
terms.

The pipeline:

1. Parses the AIFB RDF graph with **rdflib** (29,226 triples, 178
   labeled persons).
2. Converts each labeled person into a tabular feature row via one-hot
   encoding of their 1-hop outgoing/incoming RDF edges, keeping the 500
   most frequent (predicate, object) pairs.
3. Excludes the `swrc:member` relation from research groups, since it
   is the exact inverse of the prediction target (a data-leakage fix,
   documented in the report).
4. Trains a **RandomForestClassifier** (scikit-learn) on the tabular
   data.
5. Explains individual predictions with **SHAP** (`TreeExplainer`) and
   a **global surrogate decision tree** that approximates the random
   forest in human-readable, rule-based form.
6. Quantitatively evaluates explanations with **fidelity** and
   **sparsity** metrics, swept over multiple top-k values.
7. All entity IDs are resolved to **human-readable labels** (person
   names, research-group names, publication titles) via the RDF
   `swrc:name` / `swrc:title` properties, with the raw id kept in
   parentheses for traceability.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **77.8%** |
| Test macro-F1 | 0.729 |
| Train accuracy | 90.2% |
| Surrogate decision tree fidelity to Random Forest | **77.2%** |
| SHAP (top-5 features) fidelity | 53.7% |
| SHAP average explanation sparsity | 99.0% |

Full breakdown (per-class precision/recall, fidelity/sparsity sweep
across k=5,10,20,50,100) is in `data/model_performance.json` and
`data/explanation_evaluation_sweep.json`, and discussed in
`report/report.pdf`.

---

## Quick start

```bash
# 1. Set up the environment
conda create -n xai-mini python=3.12 -y
conda activate xai-mini
pip install -r requirements.txt

# 2. Run the full pipeline (dataset -> tabular -> train -> explain)
python src/pipeline.py --step all

# 3. Quantitative evaluation of explanation fidelity/sparsity
python src/evaluate_explanations.py
```

The dataset is fetched automatically. If the host is unreachable in
your environment, the raw file is already bundled at
`data/aifbfixed_complete.n3` — `pipeline.py` uses it automatically as a
fallback, no manual steps needed.

## Command-line options

```bash
# Full pipeline: download, analyze, tabularize, train, explain
python src/pipeline.py --step all

# Run only one stage
python src/pipeline.py --step download   # fetch AIFB dataset
python src/pipeline.py --step analyze    # dataset statistics + plots
python src/pipeline.py --step train      # train + evaluate the model
python src/pipeline.py --step explain    # SHAP + surrogate tree explanations

# Quantitative fidelity/sparsity evaluation (sweeps top-k internally)
python src/evaluate_explanations.py
```

## Requirements

- Python 3.10+
- ~50 MB disk (dataset + generated artifacts)
- No GPU required — all models are classical scikit-learn estimators

Libraries: `rdflib`, `pandas`, `numpy`, `scikit-learn`, `shap`,
`matplotlib` (see `requirements.txt` for pinned versions).

---

## For evaluators

To verify this project runs locally:

1. Clone the repo. `data/aifbfixed_complete.n3` is already included as
   a real file (not a pointer), so no separate download step is
   required.
2. Create a venv and install dependencies:
   `pip install -r requirements.txt`
3. Run: `python src/pipeline.py --step all`
4. Run: `python src/evaluate_explanations.py`
5. Check `data/model_performance.json` — expect test accuracy ≈ 0.778.

All generated artifacts (statistics, plots, model metrics,
explanations) are written to `data/`; see the table below.

---

## Output files (generated in `data/`)

| File | Content |
|---|---|
| `stats.json` | Dataset statistics (#triples, #entities, label distribution, top predicates) |
| `label_distribution.png` | Bar chart of research-group distribution (human-readable names) |
| `aifb_tabular.csv` | Tabular version of the RDF graph |
| `model_performance.json` | Train/test accuracy, precision, recall, F1 |
| `shap_summary.png` | Global SHAP feature-importance plot (human-readable labels) |
| `example_explanation.json` | Local explanation for one example person |
| `surrogate_tree.txt` | Global interpretable decision-tree surrogate + its fidelity to the RF |
| `explanation_evaluation.json` / `explanation_evaluation_sweep.json` | Fidelity & sparsity of SHAP explanations across the test set, swept over top-k |

---

## Project structure

```
mini-project/
├── README.md
├── requirements.txt
├── src/
│   ├── pipeline.py                # data loading, tabularization, training, explanation
│   └── evaluate_explanations.py   # quantitative explanation evaluation
├── data/
│   └── aifbfixed_complete.n3      # bundled AIFB dataset (real file, not a pointer)
└── report/
    ├── report.tex                 # LNCS-format report source
    ├── report.pdf                 # final compiled report
    └── figures/                   # plots embedded in the report
```

---

## Notes / Own contribution

- Designed our own RDF→tabular feature-extraction scheme (1-hop
  incoming + outgoing predicate-object pairs, human-readable label
  resolution), rather than reusing an existing library's default
  encoding.
- Identified and fixed a data-leakage issue: the `swrc:member` relation
  is the exact inverse of the prediction target and was excluded.
- Combined a local explanation method (SHAP) with a global surrogate
  decision tree, and evaluated both quantitatively using fidelity and
  sparsity metrics adapted from the GNN-explanation literature (Yuan et
  al., 2022) to the tabular-surrogate setting, swept across multiple
  top-k values to characterize the fidelity/sparsity tradeoff.

## Contribution of team members

This project was completed **individually** by a single author
(**Atchiyya Naidu Chitikela**), responsible for all aspects: dataset
analysis, RDF-to-tabular feature engineering, model training and
evaluation, SHAP/surrogate-tree explanation implementation, and the
final report.
