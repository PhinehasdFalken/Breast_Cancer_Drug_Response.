from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

from explainability.gradcam import plot_gradcam
from explainability.shap_analysis import plot_top_genes
from predict import CLASS_NAMES

DISCLAIMER = (
    "Research prototype. This report was produced by a model trained on laboratory cancer cell "
    "lines (GDSC). Its accuracy on patient samples has not been validated. It must not be used "
    "to make treatment decisions."
)


def report_tables(result, lab_results: pd.Series | None = None):
    """The drug comparison table used by both the on-screen and PDF report."""
    table = result["predictions"][CLASS_NAMES + ["Predicted response"]].copy()
    if lab_results is not None and not lab_results.empty:
        table["Lab result"] = lab_results
    return table


def top_shap_genes(result, top_n: int = 10) -> pd.DataFrame:
    table = result["shap_table"]
    return table.reindex(table["shap_value"].abs().sort_values(ascending=False).index).head(top_n)


def safe(text) -> str:
    """The PDF's built-in font only supports basic Latin characters; replace anything else."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def figure_to_png(fig) -> BytesIO:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def build_pdf(result, metrics: pd.DataFrame, lab_results: pd.Series | None = None) -> bytes:
    """Build the patient prediction report as PDF bytes, ready to download."""
    drug = result["drug"]
    class_name = CLASS_NAMES[result["class_id"]]
    row = result["predictions"].loc[drug]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Title ----
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Drug Response Prediction Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Generated {datetime.now():%d %B %Y, %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 0, 0)
    paragraph(pdf, 4, DISCLAIMER)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ---- Patient details ----
    section(pdf, "Patient details")
    with pdf.table(col_widths=(40, 150), first_row_as_headings=False) as table:
        for label, value in [
            ("Patient ID", result["patient_id"]),
            ("Age", result["age"]),
            ("Cancer subtype", result["subtype"]),
            ("Sample", result["sample_id"]),
        ]:
            table.row([label, safe(value)])
    pdf.ln(4)

    # ---- Prediction ----
    section(pdf, f"Prediction for {drug}")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Predicted response: {row['Predicted response']} ({row['Confidence']:.0%} probability)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "   ".join(f"{name}: {row[name]:.0%}" for name in CLASS_NAMES),
             new_x="LMARGIN", new_y="NEXT")
    if result["missing"]:
        pdf.set_font("Helvetica", "I", 9)
        paragraph(pdf, 5, f"Note: {len(result['missing'])} of the model's genes were missing from "
                             "the input and were filled with a neutral value.")
    pdf.ln(4)

    # ---- Drug comparison ----
    section(pdf, "Comparison across all drugs (most promising first)")
    comparison = report_tables(result, lab_results)
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(text_align="CENTER") as table:
        table.row(["Drug"] + list(comparison.columns))
        for drug_name, values in comparison.iterrows():
            table.row([drug_name] + [
                f"{v:.0%}" if isinstance(v, float) else safe(v) for v in values
            ])
    if "Lab result" in comparison.columns:
        matches = (comparison["Predicted response"] == comparison["Lab result"]).sum()
        pdf.set_font("Helvetica", "I", 9)
        paragraph(pdf, 5, f"This sample is a demonstration cell line with known lab results. "
                             f"The model matched {matches} of {len(comparison)}.")
    pdf.ln(4)

    # ---- Model information ----
    section(pdf, "About the model")
    pdf.set_font("Helvetica", "", 9)
    paragraph(pdf, 5, (
        "1D convolutional neural network trained on 1,000 genes and 10 drugs from the GDSC "
        "cell line dataset. Test set results (cell lines not seen during training): "
        f"accuracy {metrics.loc['all_cancer_types', 'accuracy']:.1%} and AUC-ROC "
        f"{metrics.loc['all_cancer_types', 'auc_roc']:.2f} across all cancer types; "
        f"accuracy {metrics.loc['breast_only', 'accuracy']:.1%} and AUC-ROC "
        f"{metrics.loc['breast_only', 'auc_roc']:.2f} for breast cancer. "
        "Random guessing between three classes would score about 33%."
    ))

    # ---- SHAP ----
    pdf.add_page()
    section(pdf, "Why this prediction? Gene effects (SHAP)")
    pdf.set_font("Helvetica", "", 10)
    paragraph(pdf, 5, (
        f"A typical breast cancer cell line has a {result['shap_baseline']:.0%} chance of being "
        f"'{class_name}' for {drug}. This sample's genes move that to {row[class_name]:.0%}. "
        f"Green bars pushed the prediction towards '{class_name}'; red bars pushed it away."
    ))
    pdf.image(figure_to_png(plot_top_genes(result["shap_table"], class_name)), w=150)
    pdf.ln(2)

    # ---- Grad-CAM ----
    section(pdf, "Where the CNN looked (Grad-CAM)")
    pdf.set_font("Helvetica", "", 10)
    paragraph(pdf, 5, (
        "Highlighted regions are groups of related genes that received the most attention from "
        "the CNN when making this prediction."
    ))
    pdf.image(figure_to_png(plot_gradcam(result["attention"], result["regions"], class_name)), w=180)
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=(15, 20, 20, 135)) as table:
        table.row(["Region", "Positions", "Attention", "Genes"])
        for _, region in result["regions"].iterrows():
            table.row([str(region["region"]), region["positions"], f"{region['attention']:.2f}",
                       safe(region["genes"])])
    pdf.ln(4)

    return bytes(pdf.output())


def paragraph(pdf: FPDF, line_height: float, text: str):
    """A block of wrapped text; the cursor returns to the left margin afterwards."""
    pdf.multi_cell(0, line_height, safe(text), new_x="LMARGIN", new_y="NEXT")


def section(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 236, 245)
    pdf.cell(0, 8, safe(title), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
