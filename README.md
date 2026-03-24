# 🩺 Diabetes Risk Predictor — Optimized ML Pipeline + Streamlit App

> An end-to-end machine learning project for diabetes risk prediction, built on the Pima Indians Diabetes Database.  
> Features a fully optimized pipeline (stratified imputation → feature engineering → tuned SVM / Random Forest → threshold optimization) and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Acknowledgment](#acknowledgment)
3. [Dataset](#dataset)
4. [Original Repository Analysis](#original-repository-analysis)
5. [Improvements in This Version](#improvements-in-this-version)
6. [Results Comparison](#results-comparison)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Model Details](#model-details)
11. [Future Improvements](#future-improvements)
12. [Author](#author)

---

## Project Overview

This project predicts the likelihood of diabetes in female patients based on diagnostic measurements from the Pima Indians Diabetes Database. It addresses key weaknesses found in baseline implementations — including biologically invalid zero values, default hyperparameters, single-metric evaluation, and the complete absence of a user-facing interface — and delivers:

- A **rigorously engineered ML pipeline** (data cleaning → feature engineering → robust scaling → hyperparameter tuning → threshold optimization)
- A **production-ready Streamlit application** with a dark-themed dashboard, risk gauge, patient radar chart, and clinical factor analysis

---

## Acknowledgment

This project builds upon the original work by [**TensorTitans01**](https://github.com/TensorTitans01/Diabetes-prediction). Their repository provided a clean and accessible foundation using SVM on the Pima Indians dataset and served as a strong starting point for this extended implementation. The core dataset and model type were retained; all preprocessing, evaluation, and deployment components were redesigned and significantly improved. Many thanks for making the original work publicly available.

---

## Dataset

**Pima Indians Diabetes Database**  
Source: [Kaggle — UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Why This Dataset Matters

The Pima Indians Diabetes Database is one of the most widely cited datasets in medical machine learning research. It originates from a study conducted by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) on Pima Indian women aged 21 and above — a population with a clinically documented high prevalence of Type 2 diabetes, making it ideal for studying risk factors.

| Feature | Description | Clinical Relevance |
|---|---|---|
| `Pregnancies` | Number of times pregnant | Gestational diabetes risk factor |
| `Glucose` | Plasma glucose concentration (2h OGTT) | Primary diagnostic indicator |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Hypertension co-morbidity marker |
| `SkinThickness` | Triceps skin fold thickness (mm) | Proxy for body fat |
| `Insulin` | 2-hour serum insulin (mu U/ml) | Insulin resistance indicator |
| `BMI` | Body mass index (weight/height²) | Obesity risk factor |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood score | Hereditary predisposition |
| `Age` | Age in years | Age-related metabolic decline |
| `Outcome` | Binary label: 1 = Diabetic, 0 = Non-diabetic | Target variable |

**Key statistics:**
- 768 records · 8 input features · 1 binary target
- Class distribution: ~65% non-diabetic, ~35% diabetic (class imbalance present)
- Several columns contain biologically impossible zero values (`Glucose`, `Insulin`, `BMI`, `BloodPressure`, `SkinThickness`) — these represent missing data encoded as 0 and require careful treatment before modeling

This dataset is valuable because it covers the full spectrum of clinically relevant risk factors for Type 2 diabetes, is compact enough for rapid prototyping, and is a standard benchmark that allows direct comparison with published research.

---

## Original Repository Analysis

### Methodology

The original repository implements a straightforward **Support Vector Machine (SVM)** classifier with the following pipeline:

1. **Data loading** — reads `diabetes.csv` with no cleaning step
2. **Preprocessing** — applies `StandardScaler` to all 8 original features
3. **Model** — `SVC(kernel='linear')` with default hyperparameters (`C=1.0`, no class weighting)
4. **Evaluation** — reports training accuracy and test accuracy only
5. **Deployment** — none (notebook only, no UI)

### Identified Limitations

| Area | Limitation |
|---|---|
| **Data quality** | Zero values in `Glucose`, `BMI`, `Insulin`, `BloodPressure`, `SkinThickness` are biologically impossible and were left untreated, introducing significant noise |
| **Feature engineering** | Only the 8 original features are used; no domain-informed derived features |
| **Scaler robustness** | `StandardScaler` is sensitive to the outliers present in this dataset |
| **Class imbalance** | No handling of the ~35/65 class split; the model is biased toward the majority class |
| **Hyperparameters** | Default `C=1.0`, linear kernel only — no search over regularization strength or kernel type |
| **Evaluation depth** | Only accuracy is reported; Recall (critical in medical diagnosis), F1, and ROC-AUC are absent |
| **Decision threshold** | Hard-coded at 0.5; not optimized for the medical context where false negatives (missed diabetics) are costly |
| **Deployment** | No web interface, no saved model artifact, no prediction function for new patients |

**Baseline metrics (original SVM, v1):**

| Metric | Score |
|---|---|
| Accuracy | 77.27% |
| Precision | 75.68% |
| Recall | 51.85% |
| F1 Score | 61.54% |
| ROC-AUC | Not computed |

The recall of **51.85%** is the most critical failure — the model missed nearly half of all diabetic patients, which is clinically unacceptable.

---

## Improvements in This Version

### 1. Deep Data Cleaning — Stratified Median Imputation

Zero values in clinically impossible columns are replaced with `NaN`, then imputed using the **median per outcome class** (not the global median). Diabetic and non-diabetic patients have systematically different median values — using a single global median introduces label-dependent bias.

```python
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean[zero_invalid_cols] = df_clean[zero_invalid_cols].replace(0, np.nan)

for col in zero_invalid_cols:
    median_0 = df_clean.loc[df_clean['Outcome'] == 0, col].median()
    median_1 = df_clean.loc[df_clean['Outcome'] == 1, col].median()
    df_clean.loc[(df_clean['Outcome'] == 0) & (df_clean[col].isna()), col] = median_0
    df_clean.loc[(df_clean['Outcome'] == 1) & (df_clean[col].isna()), col] = median_1
```

Especially impactful for `Insulin` (48.7% zeros) and `SkinThickness` (29.6% zeros).

---

### 2. Feature Engineering — 6 Domain-Informed Features

| New Feature | Formula | Rationale |
|---|---|---|
| `Glucose_BMI` | `Glucose × BMI / 1000` | Combined metabolic risk |
| `Age_Glucose` | `Age × Glucose / 1000` | Older patients with high glucose carry disproportionate risk |
| `Glucose_squared` | `(Glucose / 100)²` | Non-linear risk acceleration at high glucose |
| `Insulin_Glucose_ratio` | `Glucose / (Insulin + 1)` | HOMA-IR insulin resistance proxy |
| `BMI_category` | WHO thresholds: 0/1/2/3 | Clinical obesity classification |
| `Age_group` | Bins: <30 / 30–45 / 45–60 / 60+ | Age-group risk stratification |

Feature space expanded from **8 to 14**, all grounded in clinical knowledge.

---

### 3. RobustScaler

Replaced `StandardScaler` with `RobustScaler` (median/IQR). Significantly less distorted by extreme outlier values present in this dataset (e.g., Insulin up to 846 mu U/ml).

---

### 4. Class Imbalance Handling

`class_weight='balanced'` applied to both SVM and Random Forest. Automatically adjusts sample weights inversely proportional to class frequency, correcting the 35/65 split without oversampling.

---

### 5. Outlier Removal

Rows where 2 or more features exceed 3 standard deviations are removed as likely data entry errors, improving SVM margin quality and overall signal clarity.

---

### 6. Hyperparameter Tuning — GridSearchCV (5-fold Stratified CV)

**SVM search space:**
```
C:      [0.01, 0.1, 1, 10, 100]
kernel: ['linear', 'rbf']
gamma:  ['scale', 'auto', 0.001, 0.01, 0.1]
```

**Random Forest search space:**
```
n_estimators:      [100, 200, 300]
max_depth:         [None, 5, 10, 15]
min_samples_split: [2, 5, 10]
min_samples_leaf:  [1, 2, 4]
max_features:      ['sqrt', 'log2']
```

Both optimized for **F1 score** using stratified cross-validation.

---

### 7. Decision Threshold Optimization

Default 0.5 threshold replaced with a CV-optimized threshold that maximizes F1 on the training distribution. Allows trading a small precision reduction for a meaningful recall gain — catching more diabetic patients.

---

### 8. Comprehensive Evaluation

Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix · Learning Curves · Feature Importance · ROC Curves.

Best model selected using composite score: `F1 × 0.5 + AUC × 0.3 + Recall × 0.2`.

---

### 9. Streamlit Web Application

- Dark-themed responsive dashboard
- Live risk gauge with optimized threshold marker (Plotly)
- Patient profile radar chart across 6 normalized dimensions
- Feature value bar chart (base vs. engineered, color-coded)
- Clinical risk factor pills (risk / warning / positive)
- Personalized recommendations based on prediction outcome
- Model metadata (accuracy, F1, AUC, threshold) displayed live in sidebar

---

## Results Comparison

### Full Metrics Table

| Metric | v1 — Original SVM | v2 — Tuned SVM | v2 — Tuned Random Forest ✅ |
|---|---|---|---|
| **Accuracy** | 77.27% | 82.35% | **86.27%** |
| **Precision** | 75.68% | 72.41% | **76.67%** |
| **Recall** | 51.85% | 79.25% | **86.79%** |
| **F1 Score** | 61.54% | 75.68% | **81.42%** |
| **ROC-AUC** | N/A | 0.8802 | **0.9396** |

✅ **Random Forest (tuned) selected as the final model** — composite score `F1 × 0.5 + AUC × 0.3 + Recall × 0.2`.

---

### Improvement Over Baseline (v1 SVM → v2 Random Forest)

| Metric | v1 Original | v2 Final | Gain |
|---|---|---|---|
| Accuracy | 77.27% | 86.27% | **+8.99 pp** |
| Precision | 75.68% | 76.67% | +0.99 pp |
| Recall | 51.85% | 86.79% | **+34.94 pp** ← most critical |
| F1 Score | 61.54% | 81.42% | **+19.88 pp** |
| ROC-AUC | N/A | 0.9396 | — |

**The most significant gain is Recall: +34.94 percentage points.**  
In clinical terms — the original model missed nearly 1 in 2 diabetic patients. The optimized model now correctly identifies **86.79% of all diabetic cases**.

---

## Project Structure

```
diabetes-risk-predictor/
│
├── diabetes.ipynb                  # Full ML pipeline — run this first
├── app.py                          # Streamlit web application
├── diabetes.csv                    # Pima Indians dataset
├── diabetes_optimized_model.pkl    # Saved model bundle (generated by notebook)
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train and save the model

Run all cells in `diabetes.ipynb`. This executes the full pipeline and saves `diabetes_optimized_model.pkl`.

```bash
jupyter notebook diabetes.ipynb
```

> GridSearchCV is exhaustive — expect **2–5 minutes** runtime on a standard machine.

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

### Step 3 — Make a prediction

1. Enter patient values in the left sidebar
2. Click **⚡ Run Prediction**
3. Review the risk gauge, radar profile, feature breakdown, and clinical recommendations

---

## Model Details

### Pipeline Architecture

```
Raw Input (8 features)
        ↓
Stratified Median Imputation
        ↓
Feature Engineering  →  14 features total
        ↓
Outlier Removal (z-score filter)
        ↓
RobustScaler
        ↓
GridSearchCV — SVM + Random Forest (5-fold CV, F1 scoring)
        ↓
Threshold Optimization (maximize F1 via CV)
        ↓
Model Selection  (F1×0.5 + AUC×0.3 + Recall×0.2)
        ↓
Saved Bundle → diabetes_optimized_model.pkl
```

### Saved Model Bundle

```python
{
    'model':         <Pipeline: RobustScaler + RandomForestClassifier>,
    'threshold':     <float>,          # optimized decision threshold
    'feature_names': <list>,           # 14 feature names
    'model_name':    'Random Forest',
    'test_accuracy': 0.8627,
    'test_f1':       0.8142,
    'test_auc':      0.9396,
}
```

---

## Future Improvements

- **SHAP explainability** — per-prediction feature attribution to explain why a patient is high risk
- **SMOTE oversampling** — alternative to `class_weight` for handling class imbalance
- **XGBoost / LightGBM** — gradient boosting models that frequently outperform RF on tabular medical data
- **Calibrated probabilities** — `CalibratedClassifierCV` for more reliable probability estimates
- **Docker + cloud deployment** — containerize for Streamlit Cloud, Render, or HuggingFace Spaces
- **Extended datasets** — test generalizability on BRFSS or CDC Diabetes Health Indicators dataset
- **Longitudinal tracking** — store patient history to monitor risk change over time

---

## Author

**MILOUDI Meroua Amria**  
📧 miloudymiloudy3@gmail.com

---

## License

This project is licensed under the **MIT License**.

---

*Built with Python · scikit-learn · Streamlit · Plotly*
