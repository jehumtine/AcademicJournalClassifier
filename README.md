# AcademicJournalClassifier

A project for classifying academic journal articles using data mining and warehousing techniques.

# 1. Business Understanding

## Problem Statement

The University of Zambia (UNZA) hosts a growing repository of academic journal articles across multiple disciplines. However, these articles are not systematically categorized according to Zambia’s Vision 2030 development sectors. This lack of alignment presents a missed opportunity to leverage UNZA’s intellectual output for national strategic planning, policy formulation, and sectoral development monitoring.

This project aims to develop a data-driven classification system that maps UNZA journal articles to the appropriate Vision 2030 sectors using machine learning techniques. By automating this classification, we intend to bridge the gap between academic research and national development priorities, enabling policymakers, researchers, and institutions to better identify and track sectoral contributions and trends.

## Objectives

1. **Align UNZA’s research with national priorities:**  
    Systematically map academic journal articles to Zambia’s Vision 2030 development sectors to highlight how UNZA’s intellectual output contributes to national development goals.

2. **Enable evidence-based decision-making:**  
    Provide policymakers, researchers, and development stakeholders with an accessible, data-driven tool for identifying sectoral trends and gaps in research, supporting targeted policy formulation and strategic resource allocation.

3. **Automate and scale research classification:**  
    Develop a machine learning–powered system to efficiently classify and update research article categorization, ensuring scalability as UNZA’s repository grows and enabling continuous monitoring of sectoral contributions.

## Data Mining Goals

1. **Design a supervised multi-class classification model**  
    Assign each UNZA journal article to one of Zambia’s Vision 2030 sectors based on metadata (title, abstract, keywords).

    - *Purpose*: Reveal alignment between academic output and national development areas.
    - *Method*: Use labeled training data mapped to Vision 2030 sectors, extracted from a subset of articles.
    - *Expected Output*: Accurate labels such as “Education,” “Agriculture,” “Health,” “Infrastructure,” etc.

2. **Identify latent research clusters and anomalies**  
    Use unsupervised learning (e.g., clustering or topic modeling) to uncover emerging themes or neglected areas.

    - *Purpose*: Help decision-makers identify new or missing areas of national interest not currently emphasized in the Vision 2030 framework.
    - *Method*: Apply techniques like K-Means, DBSCAN, or LDA topic modeling on text embeddings.
    - *Expected Output*: Visual or descriptive reports of discovered themes or outliers.

3. **Deploy a scalable, retrainable classification pipeline**  
    Use modern ML techniques and modular design.

    - *Purpose*: Automate the tagging process for future UNZA research uploads.
    - *Method*: Build a modular pipeline for preprocessing, vectorization (e.g., TF-IDF or BERT), training, evaluation, and inference.
    - *Expected Output*: A script or web app that classifies new articles on upload.

4. **Continuously evaluate model performance**  
    Use metrics such as F1-score, accuracy, and confusion matrices.

    - *Purpose*: Ensure system reliability and adaptiveness as language and research topics evolve.
    - *Method*: Establish a validation framework and regularly benchmark models.
    - *Expected Output*: Monitoring logs or retraining criteria to prevent model drift.

## Initial Project Success Criteria

The project will be considered initially successful if the supervised classification model achieves at least **60% accuracy** in assigning UNZA journal articles to the correct Zambia Vision 2030 development sectors.

This baseline is realistic for a first iteration, considering:

- Data quality issues (e.g., incomplete or inconsistent titles, abstracts, or keywords)
- Sector overlap, where some research spans multiple development areas
- Model maturity, as this is the initial deployment and will improve with further training and tuning

Achieving this baseline will:

- Demonstrate that the model performs significantly above random guessing
- Provide policymakers and researchers with a usable starting point for tracking sectoral research contributions
- Establish a functional foundation for refining the system toward higher accuracy and more adoption

# Model Training (leak-proof & reproducible)


## 4.1 Split first, then fit (no leakage)

Always split before any fitting or vectorizing.

```python
from sklearn.model_selection import train_test_split
X_text = df["combined_text"]
y      = df["query_sector"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.20, stratify=y, random_state=42
)
```

## 4.2 End-to-end Pipeline (sparse, scalable)

Keep TF-IDF and feature scaling inside a single `Pipeline` to avoid leakage, ensure deployability, and preserve sparsity.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC   # or LogisticRegression
import numpy as np

num_cols = ["title_length","abstract_length","total_text_length",
            "published_year","publication_decade","has_doi","has_pdf"]
cat_cols = ["source","journal","provenance_sources","main_topic"]

pre = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2)), "combined_text"),
        ("num",  StandardScaler(with_mean=False), num_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    sparse_threshold=1.0,  # keep it sparse
    remainder="drop"
)

svm_clf = Pipeline(steps=[
    ("prep", pre),
    ("clf",  LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
])

logreg_clf = Pipeline(steps=[
    ("prep", pre),
    ("clf",  LogisticRegression(max_iter=200, multi_class="multinomial",
                                solver="saga", class_weight="balanced"))
])
```

> **When to use which**
>
> * **LinearSVC**: fast, strong with high-dim TF-IDF; **no probabilities** (use `CalibratedClassifierCV` if you need them).
> * **LogisticRegression**: competitive baseline, gives calibrated probabilities out of the box.

## 4.3 Train & evaluate (accuracy isn’t enough)

Track macro/weighted F1 for class imbalance; show a confusion matrix.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

svm_clf.fit(pd.DataFrame({"combined_text": X_tr}).assign(**df.loc[X_tr.index, num_cols+cat_cols]), y_tr)
y_pr = svm_clf.predict(pd.DataFrame({"combined_text": X_te}).assign(**df.loc[X_te.index, num_cols+cat_cols]))

print("Accuracy:", accuracy_score(y_te, y_pr))
print(classification_report(y_te, y_pr, digits=3))
# Optional viz: confusion matrix heatmap
```

**Minimum bar**: meet or beat the baseline in the project’s success criteria and prioritize **macro-F1** for fairness across sectors.&#x20;

## 4.4 Hyperparameters worth tuning

* **TF-IDF**: `max_features` (3k–20k), `ngram_range` ((1,1) vs (1,2)), `min_df`, `max_df`
* **LinearSVC**: `C` (0.1–10), `max_iter` (≥5000)
* **LogReg**: `C` (0.1–10), `penalty='l2'`, `class_weight`
  Use **stratified CV**; report mean ± std of macro-F1.

## 4.5 Persistence (ship one artifact)

Save the **entire pipeline** so preprocessing + model stay in sync.

```python
import joblib
joblib.dump(svm_clf, "vision2030_linear_svm_pipeline.pkl")
# or
joblib.dump(logreg_clf, "vision2030_logreg_pipeline.pkl")
```

## 4.6 Inference (single call)

```python
pipe = joblib.load("vision2030_linear_svm_pipeline.pkl")
new = pd.DataFrame([{
  "combined_text": "<title> ... </title> <abstract> ...",
  **{k: v for k,v in engineered_numeric_and_cats.items()}
}])
pred = pipe.predict(new)[0]
```

## 4.7 Gaps we’ve closed

* **Leakage**: vectorizers/encoders now fit on **train only** (inside Pipeline).
* **Scaling**: numeric features standardized to play nice with TF-IDF magnitudes.
* **Categoricals**: one-hot encoded (no fake ordinality from label encoding).
* **Sparsity**: no `.toarray()`; memory stays tame.
* **Imbalance**: `class_weight='balanced'` + macro-F1 reporting.
* **Convergence**: boosted `max_iter` for LinearSVC.
* **Reproducibility**: fixed `random_state`; recommend logging library versions.

> one line of steel: **Split early, pipeline everything, tune `C`, report macro-F1, and save the pipeline.**

Evaluation
5.1 What this stage does

Evaluates the Linear SVM on the engineered feature set:

Computes accuracy, full classification report, confusion matrix.

Adds weighted P/R/F1, per-class accuracy, and a confidence (margin) histogram.

Prints train vs test accuracy and the generalization gap.

Assembles an evaluation_results dictionary for downstream logging/exports.

5.2 How to run

Ensure you’ve already generated and saved:

../data/processed_features.csv, ../data/target.csv, ../data/cleaned_dataset.csv

../data/label_encoders.pkl, ../data/target_encoder.pkl, and (optionally) ../data/tfidf_vectorizer.pkl

Open the notebook, run all cells in order (A → C).
(If you see port/kernel issues in VS Code, select the .venv interpreter and the academic-journal kernel.)

5.3 Recommended persistence (add to the last cell if you want files)
import os, json, joblib
os.makedirs("../data", exist_ok=True)

# Save the trained model (optional here if you trained inline)
joblib.dump(svm, "../data/model_linear_svc.pkl")

# Save the detailed classification report too
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

evaluation_results.update({"report": report})
with open("../data/evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)
print("[INFO] Saved model_linear_svc.pkl and evaluation_results.json")

5.4 Interpreting results (fast)

Macro-F1 (add it if not printed yet) treats classes equally; use it alongside weighted F1.

Confusion matrix: normalize by true class when diagnosing systematic confusion.

Generalization gap: large positive gap ⇒ likely overfit; revisit features/regularization.

5.5 Next steps (optional but wise)

Move TF-IDF + encoders + scaling into a single Pipeline trained on train only.

Tune C and max_iter, try class_weight="balanced" if imbalance hurts minority recall.

If you need probabilities for ranking: CalibratedClassifierCV or switch to multinomial LogisticRegression.

Deployment
6.1 Artifacts needed

Make sure these exist (produced in training/evaluation):

../data/model_linear_svc.pkl              # trained LinearSVC
../data/target_encoder.pkl                # LabelEncoder for target classes
../data/label_encoders.pkl                # dict of LabelEncoders for cats
../data/tfidf_vectorizer.pkl              # TF-IDF vectorizer (if used)
../data/feature_order.json                # list of final feature column names (save this from training)


If you didn’t save feature_order.json earlier, export it once from the training notebook:

import json, os
os.makedirs("../data", exist_ok=True)
with open("../data/feature_order.json", "w") as f:
    json.dump(list(final_features.columns), f)

6.2 Inference utilities (local)

Create predict.py:

# predict.py
import os, json, joblib, argparse
import numpy as np
import pandas as pd

NUMERIC = ['title_length','abstract_length','total_text_length',
           'published_year','publication_decade','has_doi','has_pdf']
CAT_KEYS = ['source','journal','provenance_sources','main_topic']  # adjust to your project
TOPIC_PREFIX = 'topic_'

def load_artifacts(data_dir="../data"):
    model = joblib.load(os.path.join(data_dir, "model_linear_svc.pkl"))
    tgt = joblib.load(os.path.join(data_dir, "target_encoder.pkl"))
    cats = joblib.load(os.path.join(data_dir, "label_encoders.pkl"))
    tfidf = joblib.load(os.path.join(data_dir, "tfidf_vectorizer.pkl"))
    with open(os.path.join(data_dir, "feature_order.json")) as f:
        feat_order = json.load(f)
    return model, tgt, cats, tfidf, feat_order

# minimal text preproc consistent with training
import re
try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
    LEM  = WordNetLemmatizer()
except Exception:
    STOP, LEM = set(), None

def preprocess_text(s: str) -> str:
    if not isinstance(s, str): s = ""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in s.split() if len(t) > 2 and t not in STOP]
    if LEM:
        tokens = [LEM.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def build_features(payload: dict, tfidf, cats: dict, feat_order: list) -> pd.DataFrame:
    # text block
    title = preprocess_text(payload.get("title",""))
    abstract = preprocess_text(payload.get("abstract",""))
    combined = f"{title} {abstract}"
    X_text = tfidf.transform([combined])
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_text, columns=tfidf.get_feature_names_out())

    # numeric block
    num_vals = {k: payload.get(k, 0) for k in NUMERIC}
    num_df = pd.DataFrame([num_vals])

    # cat block (label-encode with fallbacks)
    enc_vals = {}
    for k in CAT_KEYS:
        v = str(payload.get(k, "Unknown"))
        if k in cats:
            le = cats[k]
            if v in le.classes_:
                enc_vals[f"{k}_encoded"] = int(le.transform([v])[0])
            else:
                # fallback: map to 'Unknown' if present, else 0
                if "Unknown" in le.classes_:
                    enc_vals[f"{k}_encoded"] = int(le.transform(["Unknown"])[0])
                else:
                    enc_vals[f"{k}_encoded"] = 0
        else:
            enc_vals[f"{k}_encoded"] = 0
    enc_df = pd.DataFrame([enc_vals])

    # topic_* one-hots if provided
    topics = payload.get("topics", []) or []
    topic_cols = {f"{TOPIC_PREFIX}{t}": 1 for t in topics}
    topic_df = pd.DataFrame([topic_cols])

    # combine & align
    X = pd.concat([num_df, enc_df, tfidf_df, topic_df], axis=1)
    X = X.reindex(columns=feat_order, fill_value=0)
    return X

def top_k_from_margins(margins: np.ndarray, classes: list, k: int = 3):
    idx = np.argsort(-margins)[:k]
    return [{"label": classes[i], "margin": float(margins[i])} for i in idx]

def predict(payloads, k=3, data_dir="../data"):
    model, tgt, cats, tfidf, feat_order = load_artifacts(data_dir)
    single = isinstance(payloads, dict)
    payloads = [payloads] if single else payloads

    results = []
    for p in payloads:
        X = build_features(p, tfidf, cats, feat_order)
        pred_id = int(model.predict(X)[0])
        label = tgt.inverse_transform([pred_id])[0]
        margins = model.decision_function(X)
        if margins.ndim == 2:
            margins = margins[0]
        topk = top_k_from_margins(margins, list(tgt.classes_), k)
        results.append({"prediction": label, "topk": topk})
    return results[0] if single else results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Path to JSON file with a list of records")
    ap.add_argument("--title", default="")
    ap.add_argument("--abstract", default="")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    if args.json:
        data = json.load(open(args.json))
        out = predict(data, k=args.topk)
    else:
        out = predict({"title": args.title, "abstract": args.abstract}, k=args.topk)
    print(json.dumps(out, indent=2))


Usage:

# single
python predict.py --title "Maize yield..." --abstract "We study rainfall shocks..."
# batch
python predict.py --json samples.json --topk 5

6.3 FastAPI microservice

Create app.py:

# app.py
import time, os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict  # reuses loaders/builders

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DATA_DIR    = os.getenv("DATA_DIR", "../data")

class Item(BaseModel):
    title: str = ""
    abstract: str = ""
    source: Optional[str] = None
    journal: Optional[str] = None
    provenance_sources: Optional[str] = None
    main_topic: Optional[str] = None
    published_year: Optional[int] = None
    publication_decade: Optional[int] = None
    has_doi: Optional[int] = 0
    has_pdf: Optional[int] = 0
    topics: Optional[List[str]] = None

app = FastAPI(title="Vision2030 Classifier", version=APP_VERSION)

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict")
def do_predict(items: List[Item]):
    t0 = time.time()
    payloads = [i.dict() for i in items]
    out = predict(payloads, k=3, data_dir=DATA_DIR)
    return {"predictions": out, "latency_ms": round((time.time()-t0)*1000, 2)}


Run locally:

pip install fastapi uvicorn pydantic
uvicorn app:app --host 0.0.0.0 --port 8000
# POST to http://localhost:8000/predict with a JSON array of Item

6.4 Docker (optional)

requirements.txt (pin as needed):

fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
joblib
nltk


Dockerfile:

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch NLTK data if your preproc uses it
RUN python - <<'PY'
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
PY

COPY . .
EXPOSE 8000
ENV DATA_DIR=/app/data APP_VERSION=1.0.0

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


Build & run:

docker build -t vision2030-classifier:1.0.0 .
docker run -p 8000:8000 -v %cd%/data:/app/data vision2030-classifier:1.0.0

6.5 Post-deploy checks

/health returns {"status":"ok"} and version.

/predict returns prediction and topk margins; labels match target_encoder.classes_.

Log the model hash (e.g., SHA256 of model_linear_svc.pkl) and feature_order.json length on startup.

6.6 Notes & limits

No probabilities from LinearSVC; margins only. If you must have calibrated probabilities, retrain with CalibratedClassifierCV or switch to multinomial LogisticRegression.

Strict column order: always reindex to feature_order.json. Any new topics or unseen categories become zeros or “Unknown”.