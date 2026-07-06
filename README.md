# Explainable AI Mini Project — Explaining Predictions on the AIFB Knowledge Graph

**Course:** Explainable Artificial Intelligence, Dr. Stefan Heindorf, Paderborn University

## 1. Overview

We predict which **research group** a person belongs to in the **AIFB**
knowledge graph (RDF dataset describing researchers, publications, and
research groups at AIFB Karlsruhe), and explain the predictions of the
trained model.

**Approach (Strategy 2a from the assignment):**
1. Parse the AIFB RDF graph with `rdflib`.
2. Convert each labeled person into a tabular feature row via one-hot
   encoding of their outgoing/incoming RDF edges (`predicate::object` pairs).
3. Train a `RandomForestClassifier` (scikit-learn) on the tabular data.
4. Explain individual predictions with **SHAP** (`TreeExplainer`) and
   provide a **global surrogate decision tree** as an interpretable
   approximation of the random forest.
5. Quantitatively evaluate explanations with **fidelity** and **sparsity**
   metrics (as defined in the lecture).

## 2. Dataset

- Original source: AIFB dataset, https://figshare.com/articles/dataset/AIFB_DataSet/745364/1
  (also mirrored at https://data.dgl.ai/dataset/rdf/aifb-hetero.zip)
- Task: node classification — predict `affiliation` (research group) of a `Person`.
- `src/pipeline.py` will try to download the dataset automatically. If that
  host is unreachable in your environment, we also ship the raw file
  directly at `data/aifbfixed_complete.n3` (29,226 triples, 178 labeled
  persons) so the pipeline works out of the box — `find_n3_file()` checks
  for this file first before attempting a download.

## 2b. Results summary (already generated, see `data/`)

| Metric | Value |
|---|---|
| Test accuracy | 77.8% |
| Test macro-F1 | 0.729 |
| Surrogate decision tree fidelity to RF | 77.2% |
| SHAP (top-5 features) fidelity | 53.7% |

**Note on data leakage:** the RDF predicate `swrc:member`
(ResearchGroup→Person) is the exact inverse of the `affiliation` label
we predict. We explicitly exclude it from the feature set (see comments
in `rdf_to_dataframe()` in `pipeline.py`) — including it inflates test
accuracy to ~92% in a misleading way. This is documented and discussed
in the report.

## 3. Setup

```bash
conda create -n xai-mini python=3.12 -y
conda activate xai-mini
pip install -r requirements.txt
```

## 4. Reproduce results

```bash
# Full pipeline: download data, analyze, convert to tabular, train, explain
python src/pipeline.py --step all

# Quantitative evaluation of explanation fidelity/sparsity
python src/evaluate_explanations.py
```

All outputs (statistics, plots, trained-model metrics, explanations) are
written to the `data/` folder:

| File | Content |
|---|---|
| `stats.json` | Dataset statistics (#triples, #entities, label distribution, top predicates) |
| `label_distribution.png` | Bar chart of research-group distribution |
| `aifb_tabular.csv` | Tabular version of the RDF graph |
| `model_performance.json` | Train/test accuracy, precision, recall, F1 |
| `shap_summary.png` | Global SHAP feature-importance plot |
| `example_explanation.json` | Local explanation for one example person |
| `surrogate_tree.txt` | Global interpretable decision-tree surrogate + its fidelity to the RF |
| `explanation_evaluation.json` | Fidelity & sparsity of SHAP explanations across the test set |

## 5. Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── pipeline.py                # data loading, tabularization, training, explanation
│   └── evaluate_explanations.py   # quantitative explanation evaluation
├── data/                          # generated data & results (created at runtime)
└── report/
    └── report.pdf                 # final LNCS-format report
```

## 6. Team / Contributions

| Name | imt account | Matriculation Nr. | Contribution |
|---|---|---|---|
| ... | ... | ... | Data analysis, RDF-to-tabular conversion |
| ... | ... | ... | Model training & evaluation |
| ... | ... | ... | Explanation methods & report writing |

## 7. Notes / Own contribution

- We designed our own RDF→tabular feature-extraction scheme (1-hop
  incoming + outgoing predicate-object pairs), rather than reusing an
  existing library's default encoding.
- We combine a local explanation method (SHAP) with a global surrogate
  decision tree, and evaluate both quantitatively using fidelity/sparsity
  metrics adapted from the GNN-explanation literature (Yuan et al., 2022)
  to the tabular-surrogate setting.
