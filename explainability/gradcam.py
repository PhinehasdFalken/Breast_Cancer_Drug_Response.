import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
import keras

REGION_SIZE = 20   # genes per highlighted region


def gradcam(model, gene_values, drugs, drug, class_id, layer_name="last_conv"):
    """Grad-CAM for the 1D CNN: how much attention each gene position received.

    Returns one score per gene, from 0 (ignored) to 1 (most important).
    """
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    sample = tf.constant(gene_values.astype("float32")[np.newaxis, :, np.newaxis])
    drug_vector = tf.constant(np.eye(len(drugs), dtype="float32")[[drugs.index(drug)]])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([sample, drug_vector])
        class_score = predictions[:, class_id]

    # How strongly each of the 64 pattern detectors affects the class score
    grads = tape.gradient(class_score, conv_output)[0]          # shape (250, 64)
    weights = tf.reduce_mean(grads, axis=0)                     # shape (64,)

    # Weighted sum of the detectors at each position, keeping only positive influence
    cam = tf.nn.relu(tf.reduce_sum(conv_output[0] * weights, axis=-1)).numpy()   # shape (250,)
    if cam.max() > 0:
        cam = cam / cam.max()

    # The last conv layer sees the genes after pooling by 4, so each position covers 4 genes
    return np.repeat(cam, len(gene_values) // len(cam))


def top_regions(scores: np.ndarray, genes: list[str], n_regions: int = 3) -> pd.DataFrame:
    """The gene groups (runs of neighbouring genes) with the highest average attention."""
    window = np.convolve(scores, np.ones(REGION_SIZE) / REGION_SIZE, mode="valid")
    regions = []
    for _ in range(n_regions):
        start = int(window.argmax())
        if window[start] <= 0:
            break
        end = start + REGION_SIZE
        regions.append({
            "region": len(regions) + 1,
            "positions": f"{start + 1}-{end}",
            "attention": round(float(window[start]), 2),
            "genes": ", ".join(genes[start:end]),
            "start": start,
            "end": end,
        })
        window[max(0, start - REGION_SIZE + 1):end] = -1   # don't pick overlapping regions
    return pd.DataFrame(regions)


def plot_gradcam(scores: np.ndarray, regions: pd.DataFrame, class_name: str):
    """Line chart of attention along the 1,000 ordered genes, with top regions shaded."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(scores)), scores, color="#1565c0", alpha=0.6)
    for _, region in regions.iterrows():
        ax.axvspan(region["start"], region["end"], color="#f9a825", alpha=0.3)
        ax.text((region["start"] + region["end"]) / 2, 1.02, str(region["region"]), ha="center")
    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Gene position (related genes sit next to each other)")
    ax.set_ylabel("Attention")
    ax.set_title(f"Grad-CAM: where the CNN looked when predicting '{class_name}'")
    fig.tight_layout()
    return fig
