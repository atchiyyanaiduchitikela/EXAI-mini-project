"""
Evaluate explanation quality using the fidelity & sparsity metrics
introduced in the lecture (Explaining Graph Neural Networks, slides 23-24),
adapted to the tabular surrogate setting:

  fidelity(S) = fraction of instances where the prediction using only the
                top-k SHAP features (masking all others to 0) matches the
                original model's prediction.

  sparsity(S) = 1 - |S| / |F|   where |S| = number of features used in the
                explanation, |F| = total number of features.

Run after pipeline.py has produced data/aifb_tabular.csv and a trained model.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main(top_k=5, seed=42):
    df = pd.read_csv(os.path.join(DATA_DIR, "aifb_tabular.csv"))
    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("person", "label")]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    n_features = X_test.shape[1]
    original_preds = clf.predict(X_test)

    matches = 0
    sizes = []
    for i in range(len(X_test)):
        pred_class_idx = list(clf.classes_).index(original_preds[i])
        contribs = shap_values[pred_class_idx][i] if isinstance(shap_values, list) else shap_values[i]
        top_idx = np.argsort(-np.abs(contribs))[:top_k]

        masked = np.zeros_like(X_test[i])
        masked[top_idx] = X_test[i][top_idx]
        masked_pred = clf.predict(masked.reshape(1, -1))[0]

        matches += int(masked_pred == original_preds[i])
        sizes.append(len(top_idx))

    fidelity = matches / len(X_test)
    avg_size = float(np.mean(sizes))
    sparsity = 1 - avg_size / n_features

    results = {
        "top_k": top_k,
        "num_test_instances": len(X_test),
        "fidelity": fidelity,
        "avg_explanation_size": avg_size,
        "total_features": n_features,
        "sparsity": sparsity,
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(DATA_DIR, "explanation_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
