"""Prediction and explanation for the MRI model, used by the app."""
import gzip
import json

import keras
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

from mri_preprocessing import prepare_patient

OUTCOMES = ["No pCR", "pCR"]


def load_mri_assets():
    """The trained MRI model and its settings (decision threshold, whether it uses subtype)."""
    model = keras.models.load_model("model/mri_model.keras")
    settings = json.load(open("model/mri_settings.json"))
    return model, settings


def model_inputs(images, subtype=None):
    """The model's input: the images, plus HR/HER2 status for each slice if the model uses it."""
    images = images.astype("float32")
    if subtype is None:
        return images
    return [images, np.repeat(np.asarray(subtype, dtype="float32")[None, :], len(images), axis=0)]


def read_nifti(uploaded_bytes):
    """Read an uploaded .nii or .nii.gz file into a 3D array (slices, rows, cols)."""
    if uploaded_bytes[:2] == b"\x1f\x8b":            # gzip-compressed (.nii.gz)
        uploaded_bytes = gzip.decompress(uploaded_bytes)
    return np.asarray(nib.Nifti1Image.from_bytes(uploaded_bytes).dataobj)


def images_from_uploads(pre, early, late, mask):
    """Turn four uploaded files (three scans and the tumour mask) into model-ready images."""
    return prepare_patient(*(read_nifti(f) for f in (pre, early, late, mask)))


def predict_pcr(model, images, threshold, subtype=None):
    """The patient's probability of pCR: the average over their tumour slices.

    subtype: [HR, HER2] as 0/1, only for a model trained with subtype.
    """
    slice_probs = model.predict(model_inputs(images, subtype), verbose=0).ravel()
    probability = float(slice_probs.mean())
    return {
        "probability": probability,
        "slice_probabilities": slice_probs,
        "predicted": OUTCOMES[int(probability >= threshold)],
    }


def mri_gradcam(model, images, towards_pcr=True, subtype=None):
    """Grad-CAM for each slice: which parts of the image drove the prediction.

    Uses the last convolutional layer of EfficientNet ("top_activation"). Returns one
    heatmap per slice, from 0 (no influence) to 1 (most influence), at image size.
    """
    grad_model = keras.Model(model.inputs, [model.get_layer("top_activation").output, model.outputs[0]])
    with tf.GradientTape() as tape:
        features, prediction = grad_model(model_inputs(images, subtype), training=False)
        score = prediction[:, 0] if towards_pcr else 1 - prediction[:, 0]
    grads = tape.gradient(score, features)                     # (slices, 4, 4, 1280)
    weights = tf.reduce_mean(grads, axis=(1, 2))               # importance of each feature map
    cams = tf.nn.relu(tf.einsum("nhwc,nc->nhw", features, weights)).numpy()

    size = images.shape[1]
    heatmaps = []
    for cam in cams:
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmaps.append(np.clip(zoom(cam, size / cam.shape[0], order=1), 0, 1))
    return np.stack(heatmaps)


def plot_slices(images, heatmaps, slice_probabilities):
    """Top row: the model's input (red = pre, green = early, blue = late contrast).
    Bottom row: the early-contrast scan with the Grad-CAM heatmap on top."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    axes = np.array(axes).reshape(2, n)
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Slice {i + 1}: {slice_probabilities[i]:.0%}", fontsize=9)
        axes[1, i].imshow(images[i][..., 1], cmap="gray")
        axes[1, i].imshow(heatmaps[i], cmap="magma", alpha=0.45, vmin=0, vmax=1)
        for row in (0, 1):
            axes[row, i].axis("off")
    axes[0, 0].text(-0.08, 0.5, "Model input", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9)
    axes[1, 0].text(-0.08, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout()
    return fig
