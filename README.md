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