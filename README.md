# Report skeleton (paste content into the LNCS Overleaf template)

Template: https://www.overleaf.com/latex/templates/springer-lecture-notes-incomputer-science/kzwwpvhwnvfj
Max 10 pages + unlimited references.

---

## 1. Abstract (~1 paragraph)
One paragraph: task (predict AIFB research group), method (RDF→tabular,
Random Forest, SHAP + surrogate decision tree), and headline result
(test accuracy + fidelity of explanations).

## 2. Introduction (~1 page)
- Motivation: why explainability matters for KG-based ML models.
- Task definition: node classification on AIFB — predict a person's
  research group affiliation from the knowledge graph.
- Brief roadmap of the paper.
- State your chosen strategy (Strategy 2a) and why (interpretability,
  simplicity, direct compatibility with SHAP/decision trees).

## 3. Data analysis (~1 page)
Include a **table** with:
- # triples, # entities, # predicates (from `data/stats.json`)
- label distribution table (5 research groups + counts)
Include the **figure** `data/label_distribution.png`.
Discuss class imbalance (Business Info & Comm. Systems has 73 members,
Usability Engineering only 1 — mention how this affects modeling/metrics
choice, e.g., macro-F1, class_weight="balanced").

## 4. Model Training & Evaluation (~1 page)
- Describe RDF→tabular conversion in your own words (one-hot encoding of
  (predicate, object) pairs, 1-hop neighborhood, top-N most frequent
  features kept).
- Describe the Random Forest model & hyperparameters.
- **Table**: train vs. test accuracy/precision/recall/F1 (from
  `data/model_performance.json`).
- Discuss over/underfitting, and the effect of class imbalance on results.

## 5. Model Explanation (~3 pages)
### 5.1 Local explanation (SHAP)
- Explain the SHAP TreeExplainer method briefly (own words).
- Show `data/shap_summary.png` (global feature importance).
- Show one **concrete example** from `data/example_explanation.json`,
  rewritten in human-readable form, e.g.:
  > "The model predicts that **Person X** belongs to the *Knowledge
  > Management* group, mainly because they **publish with id3instance**
  > (research group) and **co-author with id245instance**."
  (Do NOT show raw node IDs only — the assignment requires human-readable
  labels!)

### 5.2 Global explanation (surrogate decision tree)
- Show (a shortened, readable version of) `data/surrogate_tree.txt`.
- State its fidelity to the Random Forest.
- Interpret 2-3 of the top splits in natural language.

### 5.3 Evaluation of explanations
- Report `data/explanation_evaluation.json`: fidelity and sparsity for
  top-k=5 SHAP features.
- Discuss the fidelity/sparsity tradeoff (cf. lecture slide 24): does a
  smaller top-k lower fidelity? (Optionally rerun `evaluate_explanations.py`
  with top_k = 3, 5, 10, 20 and plot the tradeoff curve — this is a good
  "own contribution".)

## 6. Conclusion (~1 paragraph)
Summarize what worked, key findings (e.g., which relations were most
predictive of research group), and limitations (e.g., only 1-hop features
used; small dataset size; class imbalance).

## 7. Contributions of team members (~1 paragraph)
State per-member contributions explicitly (data analysis / modeling /
explanation / writing).

## 8. Optional: Acknowledgement

---

## Suggested "own contribution" additions (pick at least one, per grading criteria)
- [ ] Sweep the fidelity/sparsity tradeoff over multiple top-k values and plot it.
- [ ] Compare Random Forest vs. a Logistic Regression baseline in terms of
      accuracy AND interpretability.
- [ ] Extend feature extraction to 2-hop neighborhoods and compare fidelity.
- [ ] Try an additional dataset (e.g., MUTAG) with the same pipeline.
- [ ] Compare SHAP explanations against the surrogate decision tree's
      explanations for agreement (do they highlight the same features?).
