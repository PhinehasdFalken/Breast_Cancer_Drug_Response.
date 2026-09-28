import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split

from prepocessing import rank_normalize

RAW = "data/raw"
DRUGS = [
    "docetaxel", "paclitaxel", "lapatinib", "olaparib", "palbociclib",
    "fulvestrant", "epirubicin", "5-fluorouracil", "cisplatin", "gemcitabine",
]
N_GENES = 1000
SEED = 42

# Genes that mostly reveal whether a cell line came from a male or female.
# Breast cancer patients are almost all female, so these would mislead the model.
SEX_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "EIF1AY", "UTY", "USP9Y", "ZFY", "NLGN4Y", "TXLNGY", "PRKY", "XIST", "TSIX"]

# ---------- 1. Drug responses ----------
response = pd.read_excel(f"{RAW}/GDSC2_fitted_dose_response.xlsx")
models = pd.read_csv(f"{RAW}/model_list.csv", usecols=["model_id", "COSMIC_ID"], dtype=str)
models = models.dropna(subset=["COSMIC_ID"])
models["COSMIC_ID"] = models["COSMIC_ID"].str.split(";")
models = models.explode("COSMIC_ID")
models["COSMIC_ID"] = models["COSMIC_ID"].str.strip().astype(int)

response = response[response["DRUG_NAME"].str.lower().isin(DRUGS)]
response = response[response["CANCER_TYPE"] != "Non-Cancerous"]
response = response.merge(models, left_on="SANGER_MODEL_ID", right_on="model_id")

# ---------- 2. Gene expression ----------
expr = pd.read_csv(f"{RAW}/Cell_line_RMA_proc_basalExp.txt", sep="\t")
expr = expr.dropna(subset=["GENE_SYMBOLS"])            # rows with no gene name
expr = expr.drop(columns=["GENE_title"])
expr = expr.groupby("GENE_SYMBOLS").mean()              # same gene listed twice -> average
expr = expr.T                                           # rows = cell lines, columns = genes
expr.index = [int(c.split(".")[1]) for c in expr.index] # "DATA.906826" -> 906826
expr = expr[~expr.index.duplicated(keep="first")]       # cell line listed twice -> keep one
expr.index.name = "COSMIC_ID"

# Keep only cell lines that have both responses and expression
response = response[response["COSMIC_ID"].isin(expr.index)]
expr = expr.loc[sorted(response["COSMIC_ID"].unique())]

# ---------- 3. One value per cell line + drug ----------
# Docetaxel and fulvestrant were screened twice, so average their LN_IC50
response = (
    response.groupby(["COSMIC_ID", "CELL_LINE_NAME", "CANCER_TYPE", "DRUG_NAME"], as_index=False)
    ["LN_IC50"].mean()
)

# ---------- 4. Pick the most informative genes, then scale ----------
expr = expr.drop(columns=SEX_GENES, errors="ignore")
top_genes = expr.var().sort_values(ascending=False).head(N_GENES).index
expr = rank_normalize(expr[top_genes])

# ---------- 5. Split by cell line (never the same cell line in train and test) ----------
lines = response.drop_duplicates("COSMIC_ID")[["COSMIC_ID", "CANCER_TYPE"]]
is_breast = lines["CANCER_TYPE"] == "Breast Carcinoma"
train_ids, rest_ids = train_test_split(lines["COSMIC_ID"], test_size=0.3, stratify=is_breast, random_state=SEED)
rest_breast = is_breast[lines["COSMIC_ID"].isin(rest_ids).values]
val_ids, test_ids = train_test_split(rest_ids, test_size=0.5, stratify=rest_breast, random_state=SEED)

split = pd.Series("train", index=lines["COSMIC_ID"])
split[val_ids] = "val"
split[test_ids] = "test"
response["split"] = response["COSMIC_ID"].map(split)

# ---------- 6. Labels: Sensitive / Intermediate / Resistant ----------
# For each drug, find the two IC50 values that split the BREAST TRAINING cell lines
# into thirds, then apply those same cutoffs to every cell line. So "Sensitive"
# means sensitive compared to other breast cancers. Only training lines are used,
# so the test set has no say in where the cutoffs fall.
breast_train = response[(response["CANCER_TYPE"] == "Breast Carcinoma") & (response["split"] == "train")]
cutoffs = breast_train.groupby("DRUG_NAME")["LN_IC50"].quantile([1 / 3, 2 / 3]).unstack()
cutoffs.columns = ["low", "high"]

response = response.join(cutoffs, on="DRUG_NAME")
response["label"] = 0                                                   # Sensitive
response.loc[response["LN_IC50"] > response["low"], "label"] = 1        # Intermediate
response.loc[response["LN_IC50"] > response["high"], "label"] = 2       # Resistant
response = response.drop(columns=["low", "high"])

# ---------- 7. Put related genes next to each other ----------
# A CNN looks at small windows of neighbouring genes. Ordering the genes so that
# ones that rise and fall together sit side by side gives each window a group of
# related genes to learn from. Only training cell lines are used to decide the order.
train_expr = expr.loc[train_ids]
distance = 1 - train_expr.corr().to_numpy()
np.fill_diagonal(distance, 0)
order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
top_genes = train_expr.columns[order]
expr = expr[top_genes]

# ---------- 8. Save ----------
expr.to_csv("data/expression_processed.csv")
response.to_csv("data/responses.csv", index=False)
pd.Series(top_genes).to_csv("model/genes.txt", index=False, header=False)
cutoffs.to_csv("model/label_cutoffs.csv")

# ---------- Summary ----------
print(f"Expression: {expr.shape[0]} cell lines x {expr.shape[1]} genes")
print(f"Responses: {len(response)} cell line + drug pairs\n")
print("Pairs per split (breast in brackets):")
for name in ["train", "val", "test"]:
    part = response[response["split"] == name]
    n_breast = (part["CANCER_TYPE"] == "Breast Carcinoma").sum()
    print(f"  {name:5}: {len(part):5}  ({n_breast})")
print("\nLabel counts (0=Sensitive, 1=Intermediate, 2=Resistant):")
counts = pd.DataFrame({
    "all": response["label"].value_counts(),
    "breast": response.loc[response["CANCER_TYPE"] == "Breast Carcinoma", "label"].value_counts(),
}).sort_index()
print(counts.to_string())
