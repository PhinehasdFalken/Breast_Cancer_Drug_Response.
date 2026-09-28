# Breast Cancer Drug Response Prediction — Project Overview

This document explains what this project is, why it was built the way it was, and the steps taken from an empty folder to a working, defensible thesis prototype. It is written for the developer who built it: someone comfortable with Python and backend work, but new to machine learning.

---

## 1. What the thesis is about

The thesis title is *Development of a Drug Response Prediction Model for Breast Cancer Patients Using Convolutional Neural Networks*.

The problem it addresses is simple to state. Two breast cancers can respond very differently to the same drug, and giving a patient a drug that doesn't work costs them time and side effects. If you could look at a tumour's gene activity and predict which drugs it is likely to respond to, you could help rank treatment options.

The thesis asks whether a convolutional neural network (CNN), a type of deep learning model, can make that prediction from gene expression data. It also requires the prediction to be explainable (why did the model say this?) and delivered through a working software prototype that someone could demonstrate at a defence.

The final deliverable is therefore two things together: a trained model with honest evaluation results, and a Streamlit web application that uses it. The application takes a sample's gene data, predicts its response to ten breast cancer drugs, explains the prediction, and generates a PDF report.

The project is explicitly a **research prototype**, not a clinical tool. That framing runs through every decision below.

---

## 2. The shape of the problem in ML terms

In machine learning language, this is a **supervised, three-class classification** problem.

*Supervised* means the model learns from examples where the right answer is already known. *Classification* means the answer is a category rather than a number. *Three-class* means there are three possible answers: **Sensitive**, **Intermediate** or **Resistant**.

Each training example is a pair: one cancer sample and one drug. The input is the sample's gene activity plus which drug is being tested. The output is the response class.

If you think of it in backend terms, the trained model is a pure function: `predict(gene_values, drug) -> {Sensitive: p1, Intermediate: p2, Resistant: p3}`, where the three probabilities add up to 1. Everything else in the project either builds that function (the training pipeline) or wraps it in a usable interface (the app).

---

## 3. Where the data came from

Real patient data with recorded drug responses is scarce and hard to access, so the project uses **laboratory cell lines**: cancer cells originally taken from patients and grown in labs for years. Drugs can be tested on them directly and the results measured precisely.

Three public files from the Wellcome Sanger Institute were used:

| File | What it contains | What we used it for |
| --- | --- | --- |
| `GDSC2_fitted_dose_response.xlsx` | For each cell line and drug, the measured response (LN_IC50) and the cell line's cancer type | The answers (labels) |
| `Cell_line_RMA_proc_basalExp.txt` | Gene expression for about 17,700 genes across about 1,000 cell lines | The inputs (features) |
| `model_list.csv` (Cell Model Passports) | Each cell line's ID in two different ID systems | Joining the two files above |

**IC50** is the drug dose needed to stop half the cells from growing. A low IC50 means the drug works at a low dose, so the cells are sensitive. GDSC reports it as **LN_IC50**, its natural logarithm.

The ten drugs chosen are all used in, or studied for, breast cancer: docetaxel, paclitaxel, epirubicin, 5-fluorouracil, gemcitabine, cisplatin, lapatinib, olaparib, palbociclib and fulvestrant.

---

## 4. The steps we took

The work followed a four-week plan: data in week 1, model in week 2, app in week 3, polish and defence preparation in week 4. It finished ahead of schedule.

### Step 1: Scoping and early decisions

Before writing code, a few decisions shaped everything:

**Cell lines instead of patients.** Accepted by the supervisor. Patient data became a *compatibility* goal (the app can accept it) rather than a *validation* goal (proving accuracy on patients), because validation needs patient data with known outcomes, which was out of scope for four weeks.

**Streamlit for the interface.** It is pure Python, so no React or JavaScript was needed.

**Drop Grad-CAM, then bring it back.** Grad-CAM is an explanation technique designed for images. It highlights *regions* of the input, which only means something if neighbouring inputs are related. At first the gene order was arbitrary, so Grad-CAM was dropped. Later a gene-ordering step (Step 5) made neighbouring genes related, and Grad-CAM became meaningful again.

### Step 2: Environment setup

Python 3.11 was installed alongside the existing Python 3.14, because TensorFlow does not yet support 3.14. On Windows, the `py` launcher lets several versions coexist, so `py -3.11 -m venv venv` created a project environment on 3.11 without touching anything else.

All packages were installed at once: TensorFlow, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Streamlit and openpyxl (for reading Excel). fpdf2 was added later for PDF reports.

### Step 3: Exploring the data (and the bugs it surfaced)

A small script, `explore_data.py`, checked that all ten drugs existed in the data and counted how many cell lines had been tested with each. Getting it to run surfaced three real-world data problems, which are worth remembering because this is what data work usually looks like:

1. **`KeyError: 'COSMIC_ID'`.** The newer GDSC release identifies cell lines by *Sanger model ID*, while the expression file still uses *COSMIC IDs*. There was no shared column to join on. The fix was a third file, the Cell Model Passports model list, which maps one ID system to the other.

2. **`ValueError: invalid literal for int(): '1331031;1331030'`.** Two cell lines in the model list had two COSMIC IDs packed into one cell, separated by a semicolon. The fix was to read the IDs as text, `split(";")`, then `explode()` so each ID gets its own row.

3. **`KeyError: 'TCGA_DESC'`.** The newer release also renamed the cancer-type column to `CANCER_TYPE`, and breast cancer appears as `Breast Carcinoma` instead of `BRCA`.

Once it ran, the result was encouraging: all ten drugs were present, each tested on 49–50 breast cancer cell lines and about 930 cell lines overall.

### Step 4: Building the training dataset (`build_dataset.py`)

This script turns the three raw files into clean, model-ready data. In order, it:

1. **Joins the files** using the ID mapping from Step 3.
2. **Cleans the responses:** keeps the ten drugs, removes non-cancerous cell lines, and averages docetaxel and fulvestrant, which were each tested twice.
3. **Cleans the expression data:** removes 318 rows with no gene name, averages genes that appear twice, and keeps one copy of cell lines that appear twice.
4. **Removes sex-linked genes.** The single most variable gene was `RPS4Y1`, a Y-chromosome gene that mostly reveals whether the cell line came from a man. Breast cancer patients are almost all women, so such genes would teach the model a pattern that is useless for the real use case.
5. **Selects the 1,000 most variable genes.** Genes that barely change between samples carry little information.
6. **Normalises each sample on its own** by ranking its genes from lowest (0) to highest (1). This is covered in Section 5.
7. **Splits the data by cell line** into 70% training, 15% validation and 15% test. Also covered in Section 5.
8. **Assigns labels** using breast-based cutoffs. Covered in Section 5.
9. **Orders the genes by similarity** so related genes sit next to each other. Covered in Section 5.

The result: **933 cell lines, 1,000 genes, 9,114 cell line and drug pairs.**

### Step 5: Training the CNN (`train_model.py`)

The model is small by deep learning standards (about 265,000 parameters), because the dataset is small.

```
Gene input (1000 × 1)
  → Conv1D: 32 filters, window of 7 genes
  → MaxPooling: keep the strongest of every 4
  → Conv1D: 64 filters, window of 5          ← named "last_conv", used by Grad-CAM
  → MaxPooling: keep the strongest of every 4
  → Flatten → Dropout (50%)
  → join with the drug input (one-hot, 10 values)
  → Dense (64) → Dropout (30%)
  → Output: 3 probabilities (softmax)
```

The **convolution layers** slide a small window along the gene list and learn local patterns, such as "these seven neighbouring genes are all high". **Pooling** shrinks the data by keeping only the strongest signals. **Dropout** randomly switches off connections during training so the model can't memorise individual cell lines. The **drug** joins late, so the gene patterns are learned once and shared across all drugs.

Training used **early stopping**. After each pass over the training data (an *epoch*), the model is scored on the validation set. When that score stops improving for 10 epochs, training ends and the best version is restored. In the actual training run, training accuracy kept rising while validation accuracy flattened after only a few epochs. That is the classic sign of overfitting, and early stopping caught it.

### Step 6: Evaluating honestly (`evaluate.py`)

The trained model was scored once on the **test set**: 140 cell lines it had never seen, including 8 breast cancer lines. The script saves the metrics and charts to `results/`.

The first evaluation revealed a real problem. The labels had originally been defined by splitting each drug's IC50 values into thirds across *all* cancer types. Breast cancers are generally more drug-resistant than, say, blood cancers, so very few breast lines counted as "Sensitive". The model learned this and **never predicted Sensitive for a breast sample, not once**. Accuracy looked acceptable at 54%, but that was partly because the model mostly guessed "Resistant". The macro-averaged F1 score exposed it: 0.37 for breast.

The fix was to **define the thirds using breast cancer training lines only**, then apply those cutoffs to every cell line. "Sensitive" now means "sensitive compared with other breast cancers", which is what a breast cancer tool should mean anyway. After retraining, breast predictions spread across all three classes, and breast F1 rose to about 0.49.

This is also why the project reports **macro** metrics. They average the three classes equally, so a class the model ignores drags the score down instead of hiding.

### Step 7: The prediction layer (`predict.py`, `make_demo_data.py`)

`predict.py` holds the prediction logic, separate from the UI. It does three jobs:

1. **Reads uploads in either layout.** It accepts one row per sample (genes as columns) or one row per gene (samples as columns, the usual layout for public patient datasets), and detects which it has been given.
2. **Applies the exact training preprocessing.** It matches the 1,000 genes in the saved order and rank-normalises the sample. Missing genes get a neutral 0.5 with a warning; if more than 25% are missing, it refuses to predict. The pipeline was checked against the training data, and a demo sample came out *identical* to its training values (difference of about 10⁻¹⁶).
3. **Predicts all ten drugs in one call,** which feeds the drug comparison.

`make_demo_data.py` exports the **8 breast cancer test lines** as demonstration samples, with their raw gene values and their real lab results. Because the model never saw them in training, the demo can honestly show "here's the prediction, here's what actually happened".

### Step 8: The Streamlit app (`app.py`)

The app has four pages:

| Page | What it shows |
| --- | --- |
| Dashboard | Project overview, dataset size, headline test results |
| Predict | Patient details form, data input, prediction, drug comparison, explanations |
| Report | Summary of the latest prediction, with a PDF download button |
| Model performance | Metrics tables, confusion matrices, ROC curves, per-drug results |

Coming from Django, the main mental shift is that **Streamlit re-runs the whole script on every click**. Two tools make that workable. `@st.cache_resource` loads the model once instead of on every click. `st.session_state` is a per-user dictionary that survives re-runs, much like a Django session; it holds the latest prediction so it stays on screen.

The patient details (ID, age, subtype) go on the report only. The model never uses them.

### Step 9: Explainability (`explainability/`)

Two methods explain each prediction, because they answer different questions.

**SHAP** answers *"which genes pushed this prediction, and in which direction?"* It compares the sample against a reference group and gives each gene a number. Positive pushes towards the predicted class, negative pushes away. The reference group is the 35 breast cancer training lines, so the explanation reads as "what makes this breast cancer different from a typical one". An earlier test against all cancer types made every breast sample look "less sensitive than average", which was true but useless.

**Grad-CAM** answers *"where in the gene list was the CNN looking?"* It uses the gradients flowing into the last convolution layer to score each position. Thanks to the gene ordering, each highlighted region is a group of co-expressed genes.

### Step 10: The report (`report.py`)

The report is a two-page PDF built with fpdf2. Page one has the disclaimer, patient details, prediction, drug comparison and model performance summary. Page two has the SHAP and Grad-CAM charts. It reuses the same chart functions as the app, so the PDF and the screen always match.

### Step 11: Thesis write-up and defence preparation

The last step produced two documents for the friend whose thesis this is:

- **Chapter 3 (Methodology):** every design decision written up in thesis form, with an acknowledgement of technical assistance.
- **A defence guide:** a 60-second project summary, the numbers to memorise, key terms in plain words, a live demo script, likely examiner questions with answers, and the limitations to state up front.

---

## 5. The five decisions that matter most

These are the choices an examiner is most likely to probe, and the ones that separate a careful project from a naive one.

**1. Split by cell line, not by row.** Each cell line appears ten times, once per drug. A random row split would put the same cell line in both training and test data, and the model would score well simply by recognising cell lines it had already seen. This is called **data leakage**. Splitting whole cell lines into one group each prevents it.

**2. Breast-based label cutoffs.** Explained in Step 6. This fixed the model never predicting Sensitive for breast samples, and gave the labels the meaning a breast cancer tool needs.

**3. Per-sample rank normalisation.** The app receives one sample at a time, so you can't scale "across the dataset". Ranking genes *within* each sample works on a single sample, and it cancels out differences in scale between lab machines. That is what makes patient data usable at all.

**4. Ordering genes by similarity.** A CNN learns from neighbouring inputs. Gene lists have no natural order, so the genes were clustered by how similarly they behave across training cell lines, and arranged so similar genes sit side by side. This is the core justification for using a CNN on gene data, and it is what makes Grad-CAM meaningful. As an example, `GATA3`, a well-known breast cancer gene, ended up near the top alongside genes that behave like it.

**5. Train on all cancers, report breast separately.** About 50 breast cancer lines is too few to train a CNN. Training on all 933 lines teaches general gene–drug patterns, and breast results are reported on their own so they aren't hidden by the rest.

---

## 6. Results

All numbers are from the test set: cell lines the model never saw.

| Metric | All cancer types (1,370 pairs) | Breast cancer only (80 pairs) |
| --- | --- | --- |
| Accuracy | 58.0% | 52.5% |
| Precision (macro) | 0.52 | 0.50 |
| Recall (macro) | 0.52 | 0.50 |
| F1-score (macro) | 0.51 | 0.49 |
| AUC-ROC (macro) | 0.74 | 0.68 |

Random guessing between three classes scores about 33%, so the model has learned real signal. It is far from clinical quality, which is typical for drug-response prediction and is why the project is framed as a prototype.

**AUC** is the more encouraging number. It measures whether the model ranks samples in the right order, from more to less sensitive, regardless of where the class boundaries fall. The model is better at ranking than at landing each sample in exactly the right one of three bins. The **Intermediate** class is the hardest, because it sits between the other two by definition.

Per drug, AUC ranged from 0.66 (paclitaxel, lapatinib) to 0.72 (palbociclib, fulvestrant). Individual demo samples vary a lot: on the development machine, the model matched between 1 and 7 of the 10 lab results depending on the sample.

---

## 7. Limitations

These are stated openly in the thesis and the defence guide:

- **Cell lines only.** The app accepts patient data, but accuracy on patients has not been tested.
- **Small breast test set.** With 8 cell lines and 80 predictions, the breast figures are uncertain.
- **No baseline comparison.** The CNN was not compared against simpler models such as logistic regression or random forest.
- **Some choices were informed by test results.** The label cutoffs and gene ordering were compared in preliminary runs that looked at the test set. They were justified mainly on conceptual grounds, but strictly the reported test results may be slightly optimistic. A cleaner process would use the validation set only.
- **Explanations show correlation, not biology.** SHAP and Grad-CAM show what the model relied on, not proven biological causes.
- **Gene names must match exactly.** Older gene symbols in uploaded files are not translated.

---

## 8. Project files

```
Breast_Cancer_Drug_Response/
  build_dataset.py      Offline: raw files → clean, labelled, split dataset
  train_model.py        Offline: builds and trains the CNN
  evaluate.py           Offline: test-set metrics and charts
  make_demo_data.py     Offline: exports the 8 breast test lines for demos
  preprocessing.py      Shared: rank normalisation + class names
  predict.py            App: reads uploads, preprocesses, predicts all drugs
  report.py             App: builds the PDF report
  app.py                App: Streamlit interface
  explainability/
    shap_analysis.py    SHAP explanations and chart
    gradcam.py          Grad-CAM map, top regions and chart
  data/
    raw/                The three original downloads
    expression_processed.csv, responses.csv
    demonstration_data.csv, demonstration_truth.csv
  model/
    cnn_model.keras     The trained model
    genes.txt           The 1,000 genes, in the order the model expects
    drugs.txt           The 10 drugs, in the order the model expects
    label_cutoffs.csv   The IC50 cutoffs that define each class
  results/
    performance_metrics.csv, per_drug_metrics.csv
    confusion_matrix.png, roc_curve.png
  requirements.txt
```

The `model/` folder holds more than the model. `genes.txt`, `drugs.txt` and `label_cutoffs.csv` are the **contract** between training and prediction. The model only makes sense if its inputs arrive in exactly the order it was trained on, so these files are saved alongside it and reused everywhere.

---

## 9. How to run it

**Rebuild everything from scratch** (only needed if the data or model design changes):

```
python build_dataset.py
python train_model.py
python evaluate.py
python make_demo_data.py
```

Order matters, because each script reads what the previous one saved. Retraining changes the numbers slightly, and the thesis, defence guide and app should always come from the same trained model, so don't retrain casually.

**Run the app:**

```
venv\Scripts\activate
streamlit run app.py
```

It opens at `http://localhost:8501`.

---

## 10. What would come next

With more time, the natural next steps are:

1. **Validate on patient data.** Find a public breast cancer dataset with recorded treatment outcomes and test the model on it.
2. **Add a baseline comparison** to show whether the CNN beats simpler models.
3. **Move design comparisons to the validation set,** so the test set is touched only once.
4. **Add more data types,** such as mutations and copy number, alongside gene expression.
5. **Translate old gene names** in uploaded files to current symbols.

---

## 11. The ML concepts this project covered

For your own learning, this project touched most of the core ideas of applied machine learning:

- **Train / validation / test splits,** and why the test set must stay untouched.
- **Data leakage,** and how splitting by entity (cell line) rather than by row prevents it.
- **Label design:** how you define the classes changes what the model learns.
- **Overfitting,** spotted by training accuracy rising while validation accuracy stalls, and handled with dropout and early stopping.
- **Metric choice:** why accuracy alone can hide a failure that macro F1 reveals, and what AUC adds.
- **Train/serve consistency:** the exact same preprocessing code must run in training and in the app.
- **Explainability:** SHAP for per-feature attribution, Grad-CAM for model attention.
- **Honest reporting:** stating limitations clearly instead of overclaiming.
