# 🔧 Predictive Maintenance of Industrial Machinery Using Machine Learning

> **AICTE 2026 · IBM SkillsBuild for University Engagements · Problem Statement No. 39**
> Edunet Foundation × IBM Cloud

[![IBM Cloud](https://img.shields.io/badge/IBM%20Cloud-Lite-054ADA?style=flat&logo=ibm&logoColor=white)](https://cloud.ibm.com)
[![Platform](https://img.shields.io/badge/Platform-watsonx.ai%20Studio-1261FE?style=flat&logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-222?style=flat&logo=github)](https://YOUR_USERNAME.github.io/YOUR_REPO)

---

## 🌐 Live Demo

**[View Project Website →](https://YOUR_USERNAME.github.io/YOUR_REPO)**

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Setup Instructions](#-setup-instructions)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 📌 Problem Statement

Industrial machines such as lathes, mills, pumps, and CNC systems are subject to continuous operational stress. Unplanned failures lead to costly downtime, safety hazards, and lost productivity. Traditional maintenance is either **reactive** (fix after failure) or **schedule-based** (replace at fixed intervals) — both inefficient and unable to use the rich sensor data modern machines already generate.

**Goal:** Build a multi-class ML classification model that predicts the **type of failure** (Tool Wear, Heat Dissipation, Power Failure, Overstrain, Random, or No Failure) from sensor readings — enabling proactive, data-driven maintenance.

---

## 🎯 Project Overview

| Item | Detail |
|---|---|
| **Problem Statement** | No. 39 — Predictive Maintenance of Industrial Machinery |
| **Programme** | IBM SkillsBuild for University Engagements — AICTE 2026 |
| **Organisation** | Edunet Foundation |
| **Domain** | Mechanical Engineering / Industrial ML / IoT |
| **Dataset** | AI4I 2020 Predictive Maintenance Dataset (Kaggle) |
| **Algorithm** | Random Forest Classifier + SMOTE rebalancing |
| **Platform** | IBM Cloud Lite + watsonx.ai Studio (Jupyter Notebook) |
| **Language** | Python 3.11 |
| **Baseline Accuracy** | **98.05%** |
| **SMOTE Model Accuracy** | **97.2%** |
| **Best GridSearchCV Score** | **99.48%** |

---

## 📊 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset — Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

| Feature | Description |
|---|---|
| `Type` | Product quality variant — Low (L), Medium (M), High (H) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Manufacturing process temperature |
| `Rotational speed [rpm]` | Machine rotation speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool usage time |
| `Failure Type` | **Target** — 6 classes (see below) |

**Class Distribution:**

```
No Failure                9,652  (96.5%)
Heat Dissipation Failure    112
Power Failure                95
Overstrain Failure           78
Tool Wear Failure            45
Random Failures              18   ← extreme minority
```

---

## 🔬 Methodology

```
Raw CSV (Kaggle)
      ↓
IBM Cloud Object Storage (au-syd region)
      ↓
watsonx.ai Studio — Jupyter Notebook (Python 3.11)
      ↓
EDA: null check, class distribution, correlation heatmap, feature histograms
      ↓
Preprocessing: Drop UDI/Product ID, LabelEncode Type + Failure Type, drop Target
      ↓
Stratified Train/Test Split — 80% train (8,000) / 20% test (2,000), random_state=42
      ↓
┌────────────────────────────────┐   ┌──────────────────────────────────┐
│  Baseline Model                │   │  Enhanced Model (SMOTE)           │
│  RandomForestClassifier        │   │  SMOTE(k_neighbors=3) on train   │
│  class_weight='balanced'       │   │  RandomForestClassifier           │
│  n_estimators=200              │   │  n_estimators=200                 │
│  random_state=42               │   │  random_state=42                  │
└────────────────────────────────┘   └──────────────────────────────────┘
      ↓
Evaluate on same 2,000-record test set
Accuracy · Confusion Matrix · Classification Report · Feature Importance
      ↓
5-Fold Cross Validation + GridSearchCV Hyperparameter Tuning
      ↓
Save best model → predictive_maintenance_best_model.pkl
```

---

## 📈 Results

### Baseline Model (`class_weight='balanced'`)

| Metric | Value |
|---|---|
| Overall Accuracy | **98.05%** |
| Weighted Precision | 0.97 |
| Weighted Recall | 0.98 |
| Weighted F1-score | 0.98 |

### SMOTE Model

| Metric | Value |
|---|---|
| Overall Accuracy | **97.2%** |
| Weighted Precision | 0.98 |
| Weighted Recall | 0.97 |
| Weighted F1-score | 0.97 |

### Per-Class Recall — Before vs. After SMOTE

| Failure Type | Training Examples | Recall (Baseline) | Recall (SMOTE) | Change |
|---|---|---|---|---|
| No Failure | 7,722 | 1.00 | 0.98 | — |
| Heat Dissipation Failure | 90 | 0.73 | **0.95** | ▲ +0.22 |
| Overstrain Failure | 62 | 0.50 | **0.88** | ▲ +0.38 |
| Power Failure | 76 | 0.42 | **0.79** | ▲ +0.37 |
| Random Failures | 14 | 0.00 | 0.00 | — (insufficient data) |
| Tool Wear Failure | 36 | 0.00 | 0.00 | — (insufficient data) |

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Tool wear [min] | **0.2735** |
| 2 | Torque [Nm] | **0.2714** |
| 3 | Rotational speed [rpm] | 0.2055 |
| 4 | Air temperature [K] | 0.1276 |
| 5 | Process temperature [K] | 0.0996 |
| 6 | Type (L/M/H) | 0.0224 |

### Cross Validation & Hyperparameter Tuning

| Metric | Value |
|---|---|
| 5-Fold CV Mean Accuracy | 89.12% (±16.96%) |
| GridSearchCV Best Score | **99.48%** |
| Best Parameters | `max_depth=None`, `n_estimators=200` |

### Sample Prediction

```
Sensor readings: Type=2, Air temp=302.1K, Process temp=310.7K,
                 Speed=1608rpm, Torque=30.3Nm, Tool wear=81min

Actual failure type:    No Failure
Predicted failure type: No Failure ✓
```

---

## 🔍 Key Findings

1. **Tool wear and torque dominate** — together accounting for ~54% of the model's decision weight, consistent with mechanical engineering understanding of machine stress.

2. **SMOTE meaningfully improved minority-class recall** — Heat Dissipation (+22pts), Overstrain (+38pts), Power Failure (+37pts).

3. **SMOTE cannot substitute for real data** — Random Failures (14 training examples) and Tool Wear Failure (36 examples) remained at 0% recall. SMOTE interpolates between existing points; it cannot create signal that was never captured by sensors.

4. **All misclassifications defaulted to "No Failure"** — every incorrect prediction defaulted to the majority class, a textbook signature of class-imbalance dominance.

5. **GridSearchCV confirmed optimal parameters** — `max_depth=None` and `n_estimators=200` with a best CV score of 99.48%, validating the hyperparameter choices made in the baseline model.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| IBM Cloud Lite | Free-tier cloud platform |
| watsonx.ai Studio (Lite) | Hosted Jupyter Notebook environment |
| IBM Cloud Object Storage (Lite) | Dataset and project file storage |
| Python 3.11 | Core programming language |
| pandas, NumPy | Data manipulation |
| scikit-learn | RandomForestClassifier, metrics, GridSearchCV |
| imbalanced-learn | SMOTE oversampling |
| matplotlib, seaborn | Visualisation |
| joblib | Model serialisation |

---

## 📁 Repository Structure

```
YOUR_REPO/
│
├── index.html                               ← GitHub Pages frontend
├── README.md                                ← This file
│
├── Predictive_Maintenance_FINAL.ipynb       ← Main Jupyter Notebook
├── predictive_maintenance.csv               ← Dataset (from Kaggle)
│
├── results_all_screenshots.jpg              ← All notebook output charts compiled
│
└── docs/                                    ← (optional)
    ├── Predictive_Maintenance_Project_Report.docx
    └── Predictive_Maintenance_Final.pptx
```

---

## ⚙️ Setup Instructions

### Run on IBM watsonx.ai Studio (Recommended — Required for AICTE)

1. Create a free [IBM Cloud account](https://cloud.ibm.com/registration)
2. Provision **watsonx.ai Studio** (Lite) + **Cloud Object Storage** (Lite)
3. Create a new project and upload `predictive_maintenance.csv` as a Data Asset
4. Upload `Predictive_Maintenance_FINAL.ipynb` via **Assets → New asset → Jupyter Notebook editor → From file**
5. Open the notebook, use **"Insert to code"** on the CSV asset to replace the data-loading cell with your own COS credentials
6. Run all cells: **Run → Run All Cells**

### Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

jupyter notebook Predictive_Maintenance_FINAL.ipynb
```

> Replace the IBM COS loading cell (Cell 4) with:
> ```python
> import pandas as pd
> df = pd.read_csv('predictive_maintenance.csv')
> ```

### Deploy Frontend to GitHub Pages

1. Push all files to your GitHub repo
2. **Settings → Pages → Source → Deploy from branch → main → / (root)**
3. Live at: `https://YOUR_USERNAME.github.io/YOUR_REPO`

---

## 🖼 Screenshots

All notebook output charts compiled in one image:

![Results Collage](results_all_screenshots.jpg)

---

## 🚀 Future Scope

- **Collect more real-world data** for Random Failures and Tool Wear Failure classes
- **XGBoost / Gradient Boosting** comparison against the Random Forest baseline
- **Deploy via IBM watsonx.ai Runtime** as a live REST API for real-time IoT sensor inference
- **Real-time dashboard** showing live machine health and predicted failure risk
- **Multi-sensor fusion** — integrating vibration, acoustic, and power draw sensors

---

## 👤 Author

**Syed Musaib**

- IBM Cloud Account: `Syed Musaib's Account` (Region: au-syd)
- Programme: IBM SkillsBuild for University Engagements — AICTE 2026
- Problem Statement: No. 39 — Predictive Maintenance of Industrial Machinery
- Platform: IBM Cloud Lite + watsonx.ai Studio

---

## 📄 References

1. [AI4I 2020 Predictive Maintenance Dataset — Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
2. [IBM watsonx.ai Studio Documentation](https://www.ibm.com/docs/en/watsonx)
3. [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. [imbalanced-learn — SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
5. IBM SkillsBuild for University Engagements — AICTE 2026, Problem Statement No. 39

---

*Built with ❤️ on IBM Cloud Lite — no paid infrastructure required.*
