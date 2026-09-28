import pandas as pd

RAW = "data/raw"
CANDIDATE_DRUGS = [
    "docetaxel", "paclitaxel", "lapatinib", "olaparib", "palbociclib",
    "fulvestrant", "epirubicin", "5-fluorouracil", "cisplatin", "gemcitabine",
]

response = pd.read_excel(f"{RAW}/GDSC2_fitted_dose_response.xlsx")
expr = pd.read_csv(f"{RAW}/Cell_line_RMA_proc_basalExp.txt", sep="\t")
models = pd.read_csv(f"{RAW}/model_list.csv", usecols=["model_id", "COSMIC_ID"], dtype=str)

# The response file uses Sanger IDs; the expression file uses COSMIC IDs.
# The model list links the two, so we attach a COSMIC_ID to every response row.
# A few cell lines list two COSMIC IDs like "907284;1330932", so we split them
# into separate rows, one ID each, before converting to numbers.
models = models.dropna(subset=["COSMIC_ID"])
models["COSMIC_ID"] = models["COSMIC_ID"].str.split(";")
models = models.explode("COSMIC_ID")
models["COSMIC_ID"] = models["COSMIC_ID"].str.strip().astype(int)
response = response.merge(models, left_on="SANGER_MODEL_ID", right_on="model_id")

# Expression columns are named like "DATA.906826" -- the number is the cell line's COSMIC ID
expr_ids = {int(c.split(".")[1]) for c in expr.columns if c.startswith("DATA.")}

# Keep only cell lines that both drug response AND expression data
response = response[response["COSMIC_ID"].isin(expr_ids)]
response["drug_lower"] = response["DRUG_NAME"].str.lower()
found = response[response["drug_lower"].isin(CANDIDATE_DRUGS)]
breast = found[found["CANCER_TYPE"] == "Breast Carcinoma"]

summary = pd.DataFrame({
    "all_cell_lines": found.groupby("DRUG_NAME")["COSMIC_ID"].nunique(),
    "breast_cell_lines": breast.groupby("DRUG_NAME")["COSMIC_ID"].nunique(),
}).fillna(0).astype(int)

print(summary.sort_values("breast_cell_lines", ascending=False))
print("\nNot found in GDSC2:", set(CANDIDATE_DRUGS) - set(found["drug_lower"]) or "none")
print(f"Expression data: {len(expr)} genes, {len(expr_ids)} cell lines")
