import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from explainability.gradcam import gradcam, plot_gradcam, top_regions
from explainability.shap_analysis import explain_with_shap, load_background, plot_top_genes
from predict import CLASS_NAMES, load_assets, predict_all_drugs, prepare_samples, read_samples
from report import DISCLAIMER, build_pdf, report_tables, top_shap_genes

MRI_READY = os.path.exists("model/mri_model.keras")

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


@st.cache_resource
def get_mri_assets():
    """Load the MRI model only when the MRI page is first opened."""
    from mri_predict import load_mri_assets
    return load_mri_assets()


@st.cache_data
def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_mri_demo():
    demo = np.load("data/mri_demo.npz")
    return demo["images"], demo["pids"]


model, genes, drugs = get_assets()
background = get_background(genes)

st.sidebar.title("Drug Response Prediction")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict", "MRI prediction", "Report", "Model performance"])
st.sidebar.caption(
    "Research prototype for a thesis project. Not for clinical decisions."
)


# ============================== Dashboard ==============================
def show_dashboard():
    st.title("Breast Cancer Drug Response Prediction")
    st.write(
        "A convolutional neural network that predicts whether breast cancer cells will be "
        "**Sensitive**, **Intermediate** or **Resistant** to ten anti-cancer drugs, "
        "based on the activity of 1,000 genes. A second model estimates the chance of a complete "
        "response to chemotherapy from breast MRI scans."
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

    if MRI_READY:
        mri = load_csv("results/mri_metrics.csv", index_col="set")
        mri_patients = json.load(open("model/mri_settings.json"))["patients"]
        st.subheader("MRI model")
        col1, col2, col3 = st.columns(3)
        col1.metric("MRI patients", sum(mri_patients.values()))
        col2.metric("Test patients", int(mri.loc["test_all", "patients"]))
        col3.metric("Test AUC-ROC", f"{mri.loc['test_all', 'auc_roc']:.2f}")
        st.caption(
            "Predicts complete response to chemotherapy from breast MRI. "
            "AUC-ROC measures how well the model ranks patients: 0.5 is random, 1.0 is perfect."
        )

    st.subheader("How it works")
    st.markdown(
        "1. **Input:** upload gene expression data for a sample, or use a demonstration sample.\n"
        "2. **Preprocessing:** each gene's value is converted to its position within the sample (0 to 1).\n"
        "3. **Prediction:** the CNN estimates the probability of each response class for every drug.\n"
        "4. **Results:** predicted response, probabilities, and a comparison across all ten drugs."
    )
    st.info(
        "The model was trained on laboratory cell lines (GDSC dataset). It accepts patient data."
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


# ============================== MRI prediction ==============================
def show_mri():
    st.title("Predict response from an MRI scan")
    st.write(
        "This model looks at a patient's breast MRI and estimates the chance of a "
        "**pathological complete response (pCR)**: no invasive cancer left at surgery after "
        "chemotherapy given before the operation. It predicts the response to the chemotherapy "
        "treatment as a whole, not to individual drugs."
    )
    if not MRI_READY:
        st.info("The MRI model has not been trained yet.")
        return
    st.caption(
        "Besides the scans, the model uses the tumour's hormone receptor (HR) and HER2 status, "
        "which are known before treatment starts."
    )
    from mri_predict import images_from_uploads, mri_gradcam, predict_pcr

    mri_model, mri_settings = get_mri_assets()
    threshold = mri_settings["threshold"]
    uses_subtype = mri_settings.get("uses_subtype", False)
    status = ["Negative", "Positive"]

    st.subheader("1. Scans")
    source = st.radio(
        "Data source", ["Use a demonstration patient", "Upload scan files"],
        horizontal=True, key="mri_source",
    )
    truth = None
    if source == "Upload scan files":
        st.caption(
            "Upload the patient's three DCE-MRI scans and the tumour mask as NIfTI files "
            "(.nii or .nii.gz), all the same size, as in the BreastDCEDL dataset. "
            "Photos or screenshots of scans cannot be used."
        )
        col1, col2 = st.columns(2)
        files = [
            col1.file_uploader("Before contrast", type=["nii", "gz"]),
            col2.file_uploader("Early after contrast", type=["nii", "gz"]),
            col1.file_uploader("Late after contrast", type=["nii", "gz"]),
            col2.file_uploader("Tumour mask", type=["nii", "gz"]),
        ]
        subtype = None
        if uses_subtype:
            col1, col2 = st.columns(2)
            hr = col1.selectbox("Hormone receptor (HR) status", status)
            her2 = col2.selectbox("HER2 status", status)
            subtype = [status.index(hr), status.index(her2)]
        if not all(files):
            return
        patient = files[0].name.split(".")[0]
    else:
        demo_images, demo_pids = load_mri_demo()
        demo_patients = load_csv("data/mri_demo_patients.csv", index_col="pid")
        patient = st.selectbox("Patient", demo_patients.index)
        truth = "pCR" if demo_patients.loc[patient, "pCR"] == 1 else "No pCR"
        subtype = None
        if uses_subtype:
            subtype = [float(demo_patients.loc[patient, "HR"]), float(demo_patients.loc[patient, "HER2"])]
            st.write(
                f"HR status: **{status[int(subtype[0])]}**, HER2 status: "
                f"**{status[int(subtype[1])]}** (from the trial records)"
            )
        st.caption(
            "Demonstration patients come from the test set of the I-SPY clinical trials. "
            "The model never saw them during training."
        )

    if st.button("Predict response", type="primary", key="mri_predict"):
        try:
            if source == "Upload scan files":
                images = images_from_uploads(*(f.getvalue() for f in files))
            else:
                images = demo_images[demo_pids == patient]
        except Exception as error:
            st.error(f"These files could not be read as matching MRI scans and a tumour mask: {error}")
            return
        result = predict_pcr(mri_model, images, threshold, subtype)
        result.update({
            "patient": patient,
            "truth": truth,
            "images": images,
            "heatmaps": mri_gradcam(
                mri_model, images, towards_pcr=result["predicted"] == "pCR", subtype=subtype
            ),
        })
        st.session_state["mri_result"] = result

    result = st.session_state.get("mri_result")
    if result is not None:
        show_mri_result(result, threshold)


def show_mri_result(result, threshold):
    from mri_predict import plot_slices

    st.divider()
    st.subheader(f"Prediction for {result['patient']}")
    col1, col2 = st.columns([1, 2])
    col1.metric(
        "Predicted outcome", result["predicted"],
        f"{result['probability']:.0%} chance of pCR", delta_color="off",
    )
    col2.write(
        f"Patients with an estimated chance of **{threshold:.0%} or more** are classed as pCR. "
        "This cut-off was chosen on the validation patients. In the training trials, about "
        "3 in 10 patients had a pCR."
    )
    if result["truth"]:
        verdict = "matches" if result["truth"] == result["predicted"] else "does not match"
        col2.caption(f"Actual outcome in the trial: **{result['truth']}**. The prediction {verdict}.")

    st.subheader("Where the model looked")
    fig = plot_slices(result["images"], result["heatmaps"], result["slice_probabilities"])
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Top row: the five slices where the tumour is largest, with the chance of pCR from each. "
        "Colours combine three scans: red before contrast, green early after contrast, blue late "
        "after contrast. Bottom row: Grad-CAM; bright areas influenced the "
        f"'{result['predicted']}' prediction most."
    )


# ============================== Model performance ==============================
def show_performance():
    st.title("Model performance")
    st.header("Gene expression model")
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

    if MRI_READY:
        st.divider()
        st.header("MRI model")
        st.write(
            "All results are from the test set: I-SPY patients the model never saw during "
            "training. There is one prediction per patient, the average over five tumour slices."
        )
        mri = load_csv("results/mri_metrics.csv", index_col="set")
        mri.index = [
            {"test_all": "All test patients", "test_HRposHER2neg": "HR+/HER2-",
             "test_HER2pos": "HER2+", "test_TripleNeg": "Triple-negative",
             "validation": "Validation set",
             "reference_subtype_only": "Reference: HR/HER2 status alone, test set",
             "reference_subtype_only_validation": "Reference: HR/HER2 status alone, validation set",
             }.get(name, name)
            for name in mri.index
        ]
        st.dataframe(mri.style.format(na_rep="–", precision=3), width="stretch")
        st.caption(
            "Sensitivity: share of pCR patients the model found. Specificity: share of non-pCR "
            "patients it correctly ruled out. Balanced accuracy averages the two. The reference "
            "rows show how well the tumour's HR and HER2 status alone predict pCR, without any "
            "images: the MRI model does not do better than this reference."
        )
        col1, col2 = st.columns(2)
        col1.subheader("Confusion matrix")
        col1.image("results/mri_confusion_matrix.png")
        col2.subheader("ROC curve")
        col2.image("results/mri_roc_curve.png")


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
elif page == "MRI prediction":
    show_mri()
elif page == "Report":
    show_report()
else:
    show_performance()
