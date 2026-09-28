import numpy as np
import pandas as pd
# from tensorflow import keras
import keras

from prepocessing import LABELS, rank_normalize

CLASS_NAMES = [LABELS[i] for i in range(3)]
MAX_MISSING = 0.25   # refuse samples missing more than a quarter of the model's genes


def load_assets():
    """Load the trained model plus the gene and drug lists it was trained with."""
    model = keras.models.load_model("model/cnn_model.keras")
    genes = pd.read_csv("model/genes.txt", header=None)[0].tolist()
    drugs = pd.read_csv("model/drugs.txt", header=None)[0].tolist()
    return model, genes, drugs


def read_samples(raw: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Turn an uploaded table into rows = samples, columns = genes.

    Accepts either layout:
      - one row per sample, genes as columns (first column = sample ID), or
      - one row per gene, samples as columns (first column = gene name),
        which is how most public patient datasets are shared.
    """
    first_col = raw.columns[0]
    genes_in_header = len(set(raw.columns) & set(genes))
    genes_in_first_col = len(set(raw[first_col].astype(str)) & set(genes))

    table = raw.set_index(first_col)
    if genes_in_first_col > genes_in_header:
        table = table.T                       # genes were rows -> flip them into columns
    table.index = table.index.astype(str)
    return table.apply(pd.to_numeric, errors="coerce")


def prepare_samples(samples: pd.DataFrame, genes: list[str]):
    """Apply the same preprocessing used in training.

    Returns the model-ready values and the list of model genes missing from the upload.
    Missing genes are set to 0.5, the middle of the 0-1 scale.
    """
    missing = [g for g in genes if g not in samples.columns]
    if len(missing) > MAX_MISSING * len(genes):
        raise ValueError(
            f"The file is missing {len(missing)} of the model's {len(genes)} genes. "
            f"At most {int(MAX_MISSING * len(genes))} can be missing. "
            "Check that the gene columns use standard gene symbols such as GATA3 or ESR1."
        )

    present = [g for g in genes if g in samples.columns]
    values = samples[present]
    values = values.T.fillna(values.median(axis=1)).T   # blank cells -> that sample's median
    ranked = rank_normalize(values)
    return ranked.reindex(columns=genes, fill_value=0.5), missing


def predict_all_drugs(model, sample_values: np.ndarray, drugs: list[str]) -> pd.DataFrame:
    """Predict one sample's response to every drug.

    sample_values: the 1,000 prepared gene values for one sample.
    Returns one row per drug, sorted so the most promising drug comes first.
    """
    x_genes = np.repeat(sample_values[np.newaxis, :, np.newaxis], len(drugs), axis=0).astype("float32")
    x_drug = np.eye(len(drugs), dtype="float32")        # row i = "drug i is being tested"
    probs = model.predict([x_genes, x_drug], verbose=0)

    result = pd.DataFrame(probs, columns=CLASS_NAMES, index=pd.Index(drugs, name="Drug"))
    result["Predicted response"] = [CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    result["Confidence"] = probs.max(axis=1)
    return result.sort_values("Sensitive", ascending=False)
