import pandas as pd

from prepocessing import LABELS

RAW = "data/raw"

# The breast cancer cell lines in the TEST set -- the model never saw them in training
responses = pd.read_csv("data/responses.csv")
demo = responses[(responses["split"] == "test") & (responses["CANCER_TYPE"] == "Breast Carcinoma")]
genes = pd.read_csv("model/genes.txt", header=None)[0].tolist()

# Raw (not yet normalised) expression values, the same way a real user would upload them
expr = pd.read_csv(f"{RAW}/Cell_line_RMA_proc_basalExp.txt", sep="\t")
expr = expr.dropna(subset=["GENE_SYMBOLS"]).drop(columns=["GENE_title"])
expr = expr.groupby("GENE_SYMBOLS").mean().T
expr.index = [int(c.split(".")[1]) for c in expr.index]
expr = expr[~expr.index.duplicated(keep="first")]

names = demo.drop_duplicates("COSMIC_ID").set_index("COSMIC_ID")["CELL_LINE_NAME"]
samples = expr.loc[names.index, genes]
samples.index = names.values
samples.index.name = "sample_id"
samples.to_csv("data/demonstration_data.csv")

# The real lab results for those cell lines, so the demo can show prediction vs reality
truth = demo[["CELL_LINE_NAME", "DRUG_NAME", "label"]].rename(columns={"CELL_LINE_NAME": "sample_id"})
truth["lab_result"] = truth["label"].map(LABELS)
truth.drop(columns="label").to_csv("data/demonstration_truth.csv", index=False)

print(f"Saved {len(samples)} demonstration samples: {', '.join(samples.index)}")
