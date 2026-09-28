import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_background(genes: list[str]) -> np.ndarray:
    """The breast cancer TRAINING cell lines: SHAP's picture of a 'typical breast cancer'.

    SHAP explains a prediction by comparing the sample against these reference samples.
    """
    expr = pd.read_csv("data/expression_processed.csv", index_col="COSMIC_ID")
    responses = pd.read_csv("data/responses.csv")
    is_reference = (responses["split"] == "train") & (responses["CANCER_TYPE"] == "Breast Carcinoma")
    reference_ids = responses.loc[is_reference, "COSMIC_ID"].unique()
    return expr.loc[reference_ids, genes].to_numpy(dtype="float32")[..., np.newaxis]


def explain_with_shap(model, background, gene_values, genes, drugs, drug, class_id):
    """How much each gene pushed this sample towards (or away from) one response class.

    Returns:
      - a table with one row per gene: its SHAP value and the sample's gene value.
        A positive SHAP value pushes the prediction towards the class; negative pushes away.
      - the baseline: the average probability of the class for a typical breast cancer line.
        Baseline + all SHAP values is approximately the sample's predicted probability.
    """
    drug_vector = np.eye(len(drugs), dtype="float32")[[drugs.index(drug)]]
    sample = gene_values.astype("float32")[np.newaxis, :, np.newaxis]

    # The reference samples get the same drug, so only differences in genes are explained
    reference_drug = np.repeat(drug_vector, len(background), axis=0)
    baseline = float(model.predict([background, reference_drug], verbose=0)[:, class_id].mean())

    explainer = shap.GradientExplainer(
        model, [background, reference_drug]
    )
    gene_shap, _ = explainer.shap_values([sample, drug_vector], nsamples=200, rseed=42)

    table = pd.DataFrame({
        "gene": genes,
        "shap_value": gene_shap[0, :, 0, class_id],
        "gene_value": gene_values,
    })
    return table, baseline


def plot_top_genes(shap_table: pd.DataFrame, class_name: str, top_n: int = 15):
    """Bar chart of the genes with the largest effect, positive or negative."""
    top = shap_table.reindex(shap_table["shap_value"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]   # largest at the top of the chart

    fig, ax = plt.subplots(figsize=(7, 0.35 * top_n + 1))
    colours = ["#2e7d32" if v > 0 else "#c62828" for v in top["shap_value"]]
    ax.barh(top["gene"], top["shap_value"], color=colours)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"Effect on probability of '{class_name}'")
    ax.set_title(f"Top {top_n} genes behind the prediction (SHAP)")
    fig.tight_layout()
    return fig
