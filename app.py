import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from explainability.gradcam import gradcam, plot_gradcam, top_regions
from explainability.shap_analysis import explain_with_shap, load_background, plot_top_genes
from predict import CLASS_NAMES, load_assets, predict_all_drugs, prepare_samples, read_samples
from report import DISCLAIMER, build_pdf, report_tables, top_shap_genes

st.set_page_config(page_title="Breast Cancer Drug Response", layout="wide")

CLASS_COLOURS = ["#2e7d32", "#f9a825", "#c62828"]   # Sensitive, Intermediate, Resistant
SUBTYPES = ["Not specified", "Luminal A", "Luminal B", "HER2-enriched", "Triple-negative"]


@st.cache_resource
def get_assets():
    """Load the model once and reuse it, instead of reloading on every click."""
    return load_assets()


@st.cache_resource
def get_background(genes):
    """Load SHAP's reference samples once."""
    return load_background(genes)


@st.cache_data
def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


model, genes, drugs = get_assets()
background = get_background(genes)

st.sidebar.title("Drug Response Prediction")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict", "Report", "Model performance"])
st.sidebar.caption(
    "Research prototype for a thesis project. Not for clinical decisions."
)


# ============================== Dashboard ==============================
def show_dashboard():
    st.title("Breast Cancer Drug Response Prediction")
    st.write(
        "A convolutional neural network that predicts whether breast cancer cells will be "
        "**Sensitive**, **Intermediate** or **Resistant** to ten anti-cancer drugs, "
        "based on the activity of 1,000 genes."
    )

    responses = load_csv("data/responses.csv")
    metrics = load_csv("results/performance_metrics.csv", index_col="test_set")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cell lines", responses["COSMIC_ID"].nunique())
    col2.metric("Genes per sample", len(genes))
    col3.metric("Drugs", len(drugs))
    col4.metric("Training pairs", (responses["split"] == "train").sum())

    st.subheader("Test results")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy, all cancer types", f"{metrics.loc['all_cancer_types', 'accuracy']:.1%}")
    col2.metric("Accuracy, breast cancer only", f"{metrics.loc['breast_only', 'accuracy']:.1%}")
    st.caption("Random guessing between three classes would score about 33%.")

    st.subheader("How it works")
    st.markdown(
        "1. **Input:** upload gene expression data for a sample, or use a demonstration sample.\n"
        "2. **Preprocessing:** each gene's value is converted to its position within the sample (0 to 1).\n"
        "3. **Prediction:** the CNN estimates the probability of each response class for every drug.\n"
        "4. **Results:** predicted response, probabilities, and a comparison across all ten drugs."
    )
    st.info(
        "The model was trained on laboratory cell lines (GDSC dataset). It accepts patient data, "
        "but its accuracy on real patients has not been validated."
    )


# ============================== Predict ==============================
def show_predict():
    st.title("Predict drug response")

    # ---- Patient details (for the report only; the model does not use them) ----
    st.subheader("1. Patient details")
    st.caption("Recorded on the report only. The prediction uses gene expression data.")
    col1, col2, col3 = st.columns(3)
    patient_id = col1.text_input("Patient ID", value="DEMO-001")
    age = col2.number_input("Age", min_value=0, max_value=120, value=50)
    subtype = col3.selectbox("Cancer subtype", SUBTYPES)

    # ---- Gene expression data ----
    st.subheader("2. Gene expression data")
    source = st.radio(
        "Data source", ["Use a demonstration sample", "Upload a CSV file"], horizontal=True
    )
    if source == "Upload a CSV file":
        uploaded = st.file_uploader("Gene expression CSV", type="csv")
        st.caption(
            "Either one row per sample with gene symbols as column names, or one row per gene "
            "with samples as columns. The first column holds the sample IDs or gene symbols."
        )
        if uploaded is None:
            return
        raw = pd.read_csv(uploaded)
    else:
        raw = load_csv("data/demonstration_data.csv")
        st.caption(
            "Demonstration samples are breast cancer cell lines from the test set. "
            "The model never saw them during training."
        )

    samples = read_samples(raw, genes)
    sample_id = st.selectbox("Sample", samples.index)

    # ---- Drug ----
    st.subheader("3. Drug")
    drug = st.selectbox("Drug to focus on", drugs)

    if st.button("Predict response", type="primary"):
        try:
            prepared, missing = prepare_samples(samples.loc[[sample_id]], genes)
        except ValueError as error:
            st.error(str(error))
            return
        gene_values = prepared.iloc[0].to_numpy()
        predictions = predict_all_drugs(model, gene_values, drugs)

        # Explain the predicted class for the chosen drug
        class_id = CLASS_NAMES.index(predictions.loc[drug, "Predicted response"])
        shap_table, shap_baseline = explain_with_shap(
            model, background, gene_values, genes, drugs, drug, class_id
        )
        attention = gradcam(model, gene_values, drugs, drug, class_id)

        st.session_state["result"] = {
            "patient_id": patient_id,
            "age": age,
            "subtype": subtype,
            "sample_id": sample_id,
            "drug": drug,
            "missing": missing,
            "gene_values": gene_values,
            "predictions": predictions,
            "class_id": class_id,
            "shap_table": shap_table,
            "shap_baseline": shap_baseline,
            "attention": attention,
            "regions": top_regions(attention, genes),
        }

    result = st.session_state.get("result")
    if result is None:
        return
    show_results(result)


def show_results(result):
    predictions = result["predictions"]
    drug = result["drug"]
    row = predictions.loc[drug]

    st.divider()
    st.subheader(f"Prediction for {drug}")
    st.caption(f"Patient {result['patient_id']}, sample {result['sample_id']}")
    if result["missing"]:
        st.warning(
            f"{len(result['missing'])} of the model's genes were missing from the file and were "
            "filled with a neutral value. Predictions may be less reliable."
        )

    col1, col2 = st.columns([1, 2])
    col1.metric("Predicted response", row["Predicted response"], f"{row['Confidence']:.0%} probability", delta_color="off")
    col2.bar_chart(
        pd.DataFrame({"Probability": row[CLASS_NAMES].astype(float)}),
        horizontal=True, height=200,
    )

    # ---- Drug comparison ----
    st.subheader("Comparison across all drugs")
    st.caption("Sorted by probability of being Sensitive, most promising first.")
    st.bar_chart(
        predictions[CLASS_NAMES], horizontal=True, color=CLASS_COLOURS, height=400,
    )

    table = predictions.copy()
    truth = load_csv("data/demonstration_truth.csv")
    truth = truth[truth["sample_id"] == result["sample_id"]].set_index("DRUG_NAME")["lab_result"]
    if not truth.empty:
        table["Lab result"] = truth
    st.dataframe(
        table.style.format({name: "{:.0%}" for name in CLASS_NAMES + ["Confidence"]}),
        width="stretch",
    )
    if not truth.empty:
        matches = (table["Predicted response"] == table["Lab result"]).sum()
        st.caption(
            f"This is a demonstration cell line, so its real lab results are known: "
            f"the model matched {matches} of {len(table)}."
        )

    show_explanations(result)


def show_explanations(result):
    class_name = CLASS_NAMES[result["class_id"]]
    probability = result["predictions"].loc[result["drug"], class_name]

    st.subheader("Why this prediction?")
    st.caption(f"Explaining the '{class_name}' prediction for {result['drug']}.")
    shap_tab, gradcam_tab = st.tabs(["Gene effects (SHAP)", "CNN attention (Grad-CAM)"])

    with shap_tab:
        st.write(
            f"A typical breast cancer cell line has a **{result['shap_baseline']:.0%}** chance of "
            f"being '{class_name}' for {result['drug']}. This sample's genes move that to "
            f"**{probability:.0%}**. Green bars pushed the prediction towards '{class_name}', "
            "red bars pushed it away."
        )
        fig = plot_top_genes(result["shap_table"], class_name)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Show the numbers"):
            table = result["shap_table"]
            top = table.reindex(table["shap_value"].abs().sort_values(ascending=False).index).head(15)
            st.dataframe(
                top.rename(columns={
                    "gene": "Gene",
                    "shap_value": "Effect on probability",
                    "gene_value": "Level in sample (0 = lowest, 1 = highest)",
                }).style.format(precision=3),
                hide_index=True, width="stretch",
            )

    with gradcam_tab:
        st.write(
            "Grad-CAM shows which parts of the gene list the CNN focused on. Because related genes "
            "were placed next to each other before training, each highlighted region is a group "
            "of genes that tend to rise and fall together."
        )
        fig = plot_gradcam(result["attention"], result["regions"], class_name)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            result["regions"][["region", "positions", "attention", "genes"]].rename(columns=str.capitalize),
            hide_index=True, width="stretch",
        )
        st.caption(
            "SHAP and Grad-CAM answer different questions: SHAP shows which individual genes "
            "pushed the prediction and in which direction; Grad-CAM shows where the CNN's "
            "attention was concentrated."
        )


# ============================== Model performance ==============================
def show_performance():
    st.title("Model performance")
    st.write("All results are from the test set: cell lines the model never saw during training.")

    metrics = load_csv("results/performance_metrics.csv", index_col="test_set")
    metrics.index = ["All cancer types", "Breast cancer only"]
    st.subheader("Overall metrics")
    st.dataframe(metrics, width="stretch")
    st.caption(
        "Precision, recall and F1 are averaged equally across the three classes. "
        "AUC-ROC measures how well the model ranks samples: 0.5 is random, 1.0 is perfect."
    )

    col1, col2 = st.columns([3, 2])
    col1.subheader("Confusion matrix")
    col1.image("results/confusion_matrix.png")
    col2.subheader("ROC curves")
    col2.image("results/roc_curve.png")

    st.subheader("Performance per drug")
    per_drug = load_csv("results/per_drug_metrics.csv", index_col="drug")
    st.dataframe(per_drug.sort_values("auc_roc", ascending=False), width="stretch")


# ============================== Report ==============================
def show_report():
    st.title("Patient report")
    result = st.session_state.get("result")
    if result is None:
        st.info("Run a prediction on the Predict page first. The report uses the latest prediction.")
        return

    drug = result["drug"]
    class_name = CLASS_NAMES[result["class_id"]]
    row = result["predictions"].loc[drug]
    metrics = load_csv("results/performance_metrics.csv", index_col="test_set")
    truth = load_csv("data/demonstration_truth.csv")
    lab_results = truth[truth["sample_id"] == result["sample_id"]].set_index("DRUG_NAME")["lab_result"]

    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(result["patient_id"]))
    st.download_button(
        "Download PDF report",
        data=build_pdf(result, metrics, lab_results),
        file_name=f"report_{safe_id}_{drug}.pdf",
        mime="application/pdf",
        type="primary",
    )
    st.warning(DISCLAIMER)

    st.subheader("Patient details")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patient ID", result["patient_id"])
    col2.metric("Age", result["age"])
    col3.metric("Cancer subtype", result["subtype"])
    col4.metric("Sample", result["sample_id"])

    st.subheader(f"Prediction for {drug}")
    st.markdown(
        f"**Predicted response: {row['Predicted response']}** ({row['Confidence']:.0%} probability)  \n"
        + " | ".join(f"{name}: {row[name]:.0%}" for name in CLASS_NAMES)
    )
    if result["missing"]:
        st.caption(f"{len(result['missing'])} of the model's genes were missing and filled with a neutral value.")

    st.subheader("Comparison across all drugs")
    comparison = report_tables(result, lab_results)
    st.dataframe(comparison.style.format({name: "{:.0%}" for name in CLASS_NAMES}), width="stretch")

    st.subheader("Genes behind the prediction (SHAP)")
    st.write(
        f"A typical breast cancer cell line has a {result['shap_baseline']:.0%} chance of being "
        f"'{class_name}' for {drug}; this sample's genes move that to {row[class_name]:.0%}."
    )
    col1, col2 = st.columns([3, 2])
    fig = plot_top_genes(result["shap_table"], class_name)
    col1.pyplot(fig)
    plt.close(fig)
    col2.dataframe(
        top_shap_genes(result)[["gene", "shap_value"]]
        .rename(columns={"gene": "Gene", "shap_value": "Effect"})
        .style.format(precision=3),
        hide_index=True, width="stretch",
    )

    st.subheader("Gene groups the CNN focused on (Grad-CAM)")
    fig = plot_gradcam(result["attention"], result["regions"], class_name)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        result["regions"][["region", "positions", "attention", "genes"]].rename(columns=str.capitalize),
        hide_index=True, width="stretch",
    )


if page == "Dashboard":
    show_dashboard()
elif page == "Predict":
    show_predict()
elif page == "Report":
    show_report()
else:
    show_performance()
