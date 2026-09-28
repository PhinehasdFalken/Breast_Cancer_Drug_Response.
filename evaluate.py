import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    accuracy_score,
    precision_recall_fscore_support, 
    roc_auc_score, 
    roc_curve, 
)
# from tensorflow import keras
import keras

from prepocessing import LABELS

CLASS_NAMES = [LABELS[i] for i in range(3)]

# ------- 1. Load the trained model and the test set -------
model = keras.models.load_model("model/cnn_model.keras")
expr = pd.read_csv("data/expression_processed.csv", index_col="COSMIC_ID")
responses = pd.read_csv("data/responses.csv")
genes = pd.read_csv("model/genes.txt", header=None)[0].tolist()
drugs = pd.read_csv("model/drugs.txt", header=None)[0].tolist()

test = responses[responses["split"] == "test"].reset_index(drop=True)
x_genes = expr.loc[test["COSMIC_ID"], genes].to_numpy(dtype="float32")[..., np.newaxis]
x_drug = pd.get_dummies(pd.Categorical(test["DRUG_NAME"], categories=drugs)).to_numpy(dtype="float32")
y_true = test["label"].to_numpy()

# ------- 2. Predict -------
probs = model.predict([x_genes, x_drug], verbose=0)   # three probabilities per row
y_pred = probs.argmax(axis=1)                          # the class with the highest probability


def score(mask):
    """All the headline metrics for a subset of the test rows."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true[mask], y_pred[mask], average="macro", zero_division=0
    )
    return {
        "samples": int(mask.sum()), 
        "accuracy": accuracy_score(y_true[mask], y_pred[mask]), 
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1, 
        "auc_roc": roc_auc_score(y_true[mask], probs[mask], multi_class="ovr", average="macro"),
    }


everyone = np.ones(len(test), dtype=bool)
breast = (test["CANCER_TYPE"] == "Breast Carcinoma").to_numpy()

# ------- 3. Headline metrics: all cancer types vs breast only -------
metrics = pd.DataFrame(
    [score(everyone), score(breast)], index=["all_cancer_types", "breast_only"]
).round(3)
metrics.to_csv("results/performance_metrics.csv", index_label="test_set")

# ------- 4. Metrics per drug (feeds the Drug Comparison page) -------
per_drug = pd.DataFrame(
    {drug: score((test["DRUG_NAME"] == drug).to_numpy()) for drug in drugs}
).T
per_drug["samples"] = per_drug["samples"].astype(int)
per_drug = per_drug.round(3)
per_drug.to_csv("results/per_drug_metrics.csv", index_label="drug")

# ------- 5. Confusion matrices -------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mask, title in [(axes[0], everyone, "All cancer types"), (axes[1], breast, "Breast only")]:
    ConfusionMatrixDisplay.from_predictions(
        y_true[mask], y_pred[mask], labels=[0, 1, 2], 
        display_labels=CLASS_NAMES, cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{title} (n={mask.sum()})")
fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=150)
plt.close(fig)

# ------- 6. ROC curves (one per class) -------
fig, ax = plt.subplots(figsize=(6, 6))
for class_id, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true == class_id, probs[:, class_id])
    auc = roc_auc_score(y_true == class_id, probs[:, class_id])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves - test set, all cancer types")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/roc_curve.png", dpi=150)
plt.close(fig)

# ------- Summary -------
print(metrics.to_string())
print("\nPer drug:")
print(per_drug[["samples", "accuracy", "auc_roc"]].sort_values("auc_roc", ascending=False).to_string())
print("\nSaved to results/: performance_metrics.csv, per_drug_metrics.csv, confusion_matrix.png, roc_curve.png")
