# ❤️ Heart Disease Prediction System

> **AI-powered cardiovascular risk assessment using an advanced ML pipeline**
> Built by Eman Fatima | BS-AI @ PAF-IAST

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo
**[👉 Click here to try the live app](YOUR_STREAMLIT_URL_HERE)**

---

## 📌 Project Overview

This project builds a complete, production-ready ML pipeline to predict whether a patient
is at risk of heart disease based on clinical measurements. The system compares 5 machine
learning models, applies SMOTE for class imbalance, engineers 4 new features, and deploys
the best model as an interactive Streamlit web application.

---

## 🏆 Results

| Model                  | Accuracy | ROC-AUC    | F1 Score | CV Score        |
|------------------------|----------|------------|----------|-----------------|
| Logistic Regression    | 83.15%   | 0.9073     | 0.8314   | 81.20% ± 2.28%  |
| **Random Forest ✅**   | **83.15%** | **0.9179** | **0.8314** | **82.43% ± 0.61%** |
| XGBoost                | 84.24%   | 0.9057     | 0.8421   | 80.96% ± 2.16%  |
| Gradient Boosting      | 82.61%   | 0.8984     | 0.8259   | 81.82% ± 2.14%  |
| SVM                    | 83.70%   | 0.9097     | 0.8370   | 81.45% ± 1.54%  |
| Ensemble (RF+XGB+GB)   | 84.78%   | 0.9132     | —        | —               |

**Final Model: Random Forest | ROC-AUC: 0.9179 | Accuracy: 83.15%**
> Random Forest selected as final model due to highest ROC-AUC score (0.9179) and most stable cross-validation performance (±0.61%)

---

## 🔬 ML Pipeline

```
Raw Data (920 rows, 16 features)
       ↓
Missing Value Imputation (Median/Mode)
       ↓
Feature Engineering (+4 new features)
       ↓
Label Encoding (Categorical → Numeric)
       ↓
Train/Test Split (80/20, Stratified)
       ↓
SMOTE (Class Imbalance: 736 → 814 samples)
       ↓
Standard Scaling
       ↓
5 Models Trained + 5-Fold Cross Validation
       ↓
Ensemble (RF + XGBoost + Gradient Boosting)
       ↓
Best Model Selected by ROC-AUC → Deployed on Streamlit
```

---

## ✨ Key Features

- **5 ML models** trained and compared with cross-validation
- **SMOTE** applied to handle class imbalance (736 → 814 training samples)
- **Feature Engineering** — 4 new clinically meaningful features:
  - `age_group` — Young / Middle / Senior / Elderly
  - `bp_category` — Normal / Elevated / High Stage 1 & 2
  - `chol_risk` — Binary flag for cholesterol > 240 mg/dL
  - `hr_reserve` — Max heart rate minus age (cardiac reserve indicator)
- **Ensemble Voting Classifier** (RF + XGBoost + Gradient Boosting)
- **Interactive Streamlit app** with real-time prediction and risk gauge
- **6 publication-quality visualizations**

---

## 📊 Dataset

- **Source:** UCI Heart Disease Dataset
- **Samples:** 920 patient records
- **Hospitals:** Cleveland, Hungary, Switzerland, VA Long Beach
- **Features:** 13 clinical features + 4 engineered features
- **Target:** Binary — Heart Disease (1) / No Disease (0)
- **Class Distribution:** No Disease: 44.7% | Disease: 55.3%

---

## 🛠️ Tech Stack

| Category        | Tools                                      |
|-----------------|--------------------------------------------|
| Language        | Python 3.13                                |
| ML Models       | Scikit-learn, XGBoost                      |
| Imbalance       | imbalanced-learn (SMOTE)                   |
| Data Processing | Pandas, NumPy                              |
| Visualization   | Matplotlib, Seaborn                        |
| Web App         | Streamlit                                  |
| Model Saving    | Joblib                                     |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/EmanFatima00/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates outputs/ folder with model + charts)
python model_training.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py                    # Streamlit web application
├── model_training.py         # Full ML pipeline
├── heart_disease_uci.csv     # Dataset (UCI Heart Disease)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
└── outputs/                  # Auto-generated after running model_training.py
    ├── heart_disease_model.pkl     # Trained Random Forest model
    ├── scaler.pkl                  # StandardScaler
    ├── feature_names.pkl           # Feature list
    ├── model_comparison.png        # Accuracy / ROC-AUC / F1 comparison
    ├── roc_curves.png              # ROC curves for all models
    ├── confusion_matrix.png        # Confusion matrix
    ├── feature_importance.png      # Top features by importance
    ├── correlation_heatmap.png     # Feature correlation heatmap
    └── age_chol_distribution.png   # EDA — age and cholesterol
```

---

## ⚕️ Disclaimer

This tool is developed for **educational and research purposes only**.
It does not replace professional medical diagnosis.
Always consult a qualified healthcare professional for medical advice.

---

## 👩‍💻 Author

**Eman Fatima**
BS Artificial Intelligence — Semester 6 | PAF-IAST, Pakistan
ML Intern @ ProSensia | HR Manager @ CtrlAltCrew | Dean's List (SGPA: 3.72)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/eman-fatima-99962230b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/EmanFatima00)

---

*⭐ If you found this project useful, please star the repository!*
