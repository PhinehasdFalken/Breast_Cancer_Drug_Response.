"""Train and evaluate the MRI model that predicts pathological complete response (pCR).

Input: the tumour-centred images made by prepare_mri.py (5 slices per patient, with
the pre-, early and late-contrast scans as the three colour channels).
Model: EfficientNetB0, a CNN pre-trained on everyday photos (ImageNet), fine-tuned here.
A patient's prediction is the average over their 5 slices.

"python train_mri.py" trains on the MRI images alone.
"python train_mri.py --subtype" also gives the model the tumour's HR and HER2 status,
which are known before treatment starts.
"""
import json
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

SEED = 42
keras.utils.set_random_seed(SEED)

PROCESSED = "data/raw/mri/processed"
WEIGHTS = "data/raw/mri/efficientnetb0_notop.h5"
BATCH = 32
USE_SUBTYPE = "--subtype" in sys.argv

# ---------- 1. Load ----------
data = np.load(f"{PROCESSED}/mri_images.npz")
images, image_pids = data["images"], data["pids"]
patients = pd.read_csv(f"{PROCESSED}/mri_patients.csv", index_col="pid")
image_labels = patients.loc[image_pids, "pCR"].to_numpy()
image_split = patients.loc[image_pids, "split"].to_numpy()
# HR and HER2 status as 0/1 (an unknown status counts as 0.5)
subtype_values = patients[["HR", "HER2"]].fillna(0.5).astype("float32")
image_subtype = subtype_values.loc[image_pids].to_numpy()


def part(name):
    keep = image_split == name
    x = [images[keep], image_subtype[keep]] if USE_SUBTYPE else images[keep]
    return x, image_labels[keep], image_pids[keep]


x_train, y_train, _ = part("train")
x_val, y_val, pid_val = part("val")
x_test, y_test, pid_test = part("test")
print({name: int((patients["split"] == name).sum()) for name in ["train", "val", "test"]}, "patients")

# pCR is the minority outcome (about 30%), so its examples get more weight in training
weight = {0: 1.0, 1: float((y_train == 0).sum() / max((y_train == 1).sum(), 1))}

# ---------- 2. Build the model ----------
inputs = keras.Input(shape=images.shape[1:], name="mri")
x = layers.RandomFlip("horizontal_and_vertical")(inputs)      # augmentation: only active in training
x = layers.RandomRotation(0.08)(x)
x = layers.RandomZoom(0.1)(x)
base = keras.applications.EfficientNetB0(include_top=False, weights=WEIGHTS, input_tensor=x)
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.4)(x)
if USE_SUBTYPE:
    subtype_input = keras.Input(shape=(2,), name="subtype")
    x = layers.Concatenate()([x, subtype_input])
    inputs = [inputs, subtype_input]
outputs = layers.Dense(1, activation="sigmoid", name="pcr")(x)
model = keras.Model(inputs, outputs)


def compile_model(learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )


def fit(epochs, patience):
    stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=patience, restore_best_weights=True
    )
    return model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
        batch_size=BATCH, class_weight=weight, callbacks=[stop], verbose=2,
    )


# Stage 1: keep the pre-trained layers fixed and train only the new final layers
for layer in base.layers:
    layer.trainable = False
compile_model(1e-3)
fit(epochs=8, patience=3)

# Stage 2: also adjust the last part of the pre-trained network, with a small learning rate.
# Batch-normalisation layers stay fixed, which is standard practice for small datasets.
for layer in base.layers[-60:]:
    layer.trainable = not isinstance(layer, layers.BatchNormalization)
compile_model(1e-4)
fit(epochs=30, patience=6)


# ---------- 3. Patient-level predictions: average over each patient's slices ----------
def patient_probabilities(x, pids):
    slice_probs = model.predict(x, batch_size=64, verbose=0).ravel()
    return pd.Series(slice_probs).groupby(pids).mean()


val_probs = patient_probabilities(x_val, pid_val)
test_probs = patient_probabilities(x_test, pid_test)
val_true = patients.loc[val_probs.index, "pCR"].to_numpy()
test_true = patients.loc[test_probs.index, "pCR"].to_numpy()

# Decision threshold chosen on the VALIDATION set (Youden's J: best sensitivity + specificity),
# then applied unchanged to the test set
fpr, tpr, thresholds = roc_curve(val_true, val_probs)
threshold = float(thresholds[np.argmax(tpr - fpr)])
test_pred = (test_probs.to_numpy() >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(test_true, test_pred, labels=[0, 1]).ravel()
rows = {
    "test_all": {
        "patients": len(test_true),
        "pcr_patients": int(test_true.sum()),
        "auc_roc": roc_auc_score(test_true, test_probs),
        "accuracy": accuracy_score(test_true, test_pred),
        "balanced_accuracy": balanced_accuracy_score(test_true, test_pred),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }
}
# Results per breast cancer subtype (only where the test set has both outcomes)
subtype = patients.loc[test_probs.index, "HR_HER2_STATUS"].to_numpy()
for name in ["HRposHER2neg", "HER2pos", "TripleNeg"]:
    keep = subtype == name
    if keep.sum() and len(set(test_true[keep])) == 2:
        rows[f"test_{name}"] = {
            "patients": int(keep.sum()),
            "pcr_patients": int(test_true[keep].sum()),
            "auc_roc": roc_auc_score(test_true[keep], test_probs[keep]),
            "accuracy": accuracy_score(test_true[keep], test_pred[keep]),
            "balanced_accuracy": balanced_accuracy_score(test_true[keep], test_pred[keep]),
        }
rows["validation"] = {"patients": len(val_true), "auc_roc": roc_auc_score(val_true, val_probs)}

# Reference point: how well HR and HER2 status ALONE predict pCR (logistic regression)
train_ids = patients.index[patients["split"] == "train"]
reference = LogisticRegression().fit(subtype_values.loc[train_ids], patients.loc[train_ids, "pCR"])
reference_test = reference.predict_proba(subtype_values.loc[test_probs.index])[:, 1]
reference_val = reference.predict_proba(subtype_values.loc[val_probs.index])[:, 1]
rows["reference_subtype_only"] = {"patients": len(test_true), "auc_roc": roc_auc_score(test_true, reference_test)}
rows["reference_subtype_only_validation"] = {"patients": len(val_true), "auc_roc": roc_auc_score(val_true, reference_val)}
metrics = pd.DataFrame(rows).T.round(3)
metrics.to_csv("results/mri_metrics.csv", index_label="set")

# ---------- 4. Charts ----------
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    test_true, test_pred, labels=[0, 1], display_labels=["No pCR", "pCR"],
    cmap="Purples", ax=ax, colorbar=False,
)
ax.set_title(f"MRI model, test set (n={len(test_true)} patients)")
fig.tight_layout()
fig.savefig("results/mri_confusion_matrix.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
fpr, tpr, _ = roc_curve(test_true, test_probs)
ax.plot(fpr, tpr, color="#5B45A8", label=f"MRI model (AUC = {rows['test_all']['auc_roc']:.2f})")
fpr_ref, tpr_ref, _ = roc_curve(test_true, reference_test)
ax.plot(fpr_ref, tpr_ref, color="#D65A8C", linestyle="-.",
        label=f"HR/HER2 status alone (AUC = {rows['reference_subtype_only']['auc_roc']:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve, test set")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/mri_roc_curve.png", dpi=150)
plt.close(fig)

# ---------- 5. Save the model and the demonstration patients ----------
# Save without the optimiser's training state: the app only needs the network itself
keras.Model(model.inputs, model.outputs, name="mri_pcr_model").save("model/mri_model.keras")
json.dump({
    "threshold": threshold,
    "uses_subtype": USE_SUBTYPE,
    "image_size": int(images.shape[1]),
    "patients": patients["split"].value_counts().to_dict(),
}, open("model/mri_settings.json", "w"), indent=2)

# 12 test patients for the app's demo, picked at random (6 with pCR, 6 without)
rng = np.random.default_rng(SEED)
test_patients = patients.loc[test_probs.index]
demo_ids = []
for outcome in [1, 0]:
    pool = test_patients.index[test_patients["pCR"] == outcome]
    demo_ids += list(rng.choice(pool, min(6, len(pool)), replace=False))
demo_ids = sorted(demo_ids)
keep = np.isin(image_pids, demo_ids)
np.savez_compressed("data/mri_demo.npz", images=images[keep], pids=image_pids[keep])
patients.loc[demo_ids, ["dataset", "pCR", "HR", "HER2", "HR_HER2_STATUS", "age"]].to_csv(
    "data/mri_demo_patients.csv", index_label="pid"
)

print(metrics.to_string())
print(f"Decision threshold (from validation set): {threshold:.3f}")
