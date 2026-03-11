# ❤️ CardioSense — Heart Disease Risk Prediction System

> An end-to-end Machine Learning web application for real-time heart disease risk assessment using the Cleveland Heart Disease Dataset.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-85%25-brightgreen?style=flat)
![AUC](https://img.shields.io/badge/ROC--AUC-0.95-brightgreen?style=flat)

---

## 📌 Overview

CardioSense is a full-stack ML-powered web application that predicts the likelihood of heart disease in a patient based on 13 clinical parameters. It features:

- 🎯 **Real-time AI inference** via a FastAPI REST backend
- 📊 **Interactive dashboard** with EDA and model performance visualizations
- 🌐 **Responsive modern UI** with glassmorphism design and dark mode
- 🤖 **Tuned Random Forest classifier** achieving 85% accuracy and 0.95 ROC-AUC

---

## 🖥️ Demo

| Diagnostics Tab | Data Analysis Tab | ML Models Tab |
|:-:|:-:|:-:|
| Fill in clinical parameters & get instant AI risk assessment | Explore dataset distributions, correlation heatmaps & boxplots | View model performance: ROC curves, confusion matrices, feature importance |

---

## 📊 Model Performance

Trained and evaluated on the **UCI Cleveland Heart Disease Dataset** (297 patients, 13 features).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 83.33% | 0.846 | 0.786 | 0.815 | 0.950 |
| SVM (RBF Kernel) | 85.00% | 0.880 | 0.786 | 0.830 | 0.954 |
| Decision Tree | 70.00% | 0.750 | 0.536 | 0.625 | 0.745 |
| **Random Forest (Tuned)** ✅ | **85.00%** | **0.880** | **0.786** | **0.830** | **0.951** |

> Best model selected via 5-Fold GridSearchCV. No PCA applied — models trained directly on scaled features for consistent inference.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML / Data** | scikit-learn, pandas, numpy, matplotlib, seaborn |
| **Backend API** | FastAPI, uvicorn, pydantic |
| **Frontend** | HTML5, CSS3 (glassmorphism), Vanilla JS |
| **Serving** | FastAPI static file serving (single-port deployment) |

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
├── backend.py              # FastAPI application (API + static file server)
├── index.html              # Frontend UI (3-tab dashboard)
├── style.css               # UI styles (glassmorphism dark theme)
├── app.js                  # Frontend logic + tab navigation
├── extracted_code.py       # Original ML training + EDA notebook script
├── best_rf_model.pkl       # Saved tuned Random Forest model
├── scaler.pkl              # Fitted StandardScaler
├── model_results.csv       # Model evaluation results
├── eda_overview.png        # EDA: class distribution + age histogram
├── eda_correlation.png     # EDA: correlation heatmap
├── eda_distributions.png   # EDA: feature density plots
├── eda_boxplots.png        # EDA: boxplots by class
├── model_comparison.png    # All model metrics comparison chart
├── roc_curves.png          # ROC curves for all classifiers
├── confusion_matrices.png  # Confusion matrices for all classifiers
├── feature_importance.png  # Random Forest feature importance
├── learning_curves.png     # Learning curves (bias-variance tradeoff)
├── cv_boxplot.png          # 10-Fold CV accuracy distribution
└── summary_dashboard.png   # Full executive summary dashboard
```

---

## ⚙️ Setup & Run Locally

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/lil-kash/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn pydantic scikit-learn pandas numpy matplotlib seaborn
```

### 3. Start the Application
```bash
python3 -m uvicorn backend:app --reload --port 8000
```

### 4. Open in Browser
```
http://127.0.0.1:8000/
```

The backend serves both the API (`/predict`) and the frontend UI on the same port.

---

## 🔌 API Usage

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

**Response:**
```json
{
  "Prediction": "Heart Disease",
  "Probability": 0.7842,
  "Risk_Level": "HIGH RISK 🔴"
}
```

### Input Features

| Feature | Description | Range |
|---|---|---|
| `age` | Age in years | 29–77 |
| `sex` | Sex (1=Male, 0=Female) | 0, 1 |
| `cp` | Chest pain type | 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) | 94–200 |
| `chol` | Serum cholesterol (mg/dl) | 126–564 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0, 1 |
| `restecg` | Resting ECG results | 0–2 |
| `thalach` | Max heart rate achieved | 71–202 |
| `exang` | Exercise-induced angina | 0, 1 |
| `oldpeak` | ST depression induced by exercise | 0–6.2 |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Number of major vessels (fluoroscopy) | 0–3 |
| `thal` | Thalassemia type | 0–3 |

---

## 👨‍💻 Team

| Name | Role |
|---|---|
| **Kashish Mohammad** | ML Engineering, Backend API, Frontend |
| **Vridhi Vazirani** | Data Analysis, EDA |
| **Jaaswanth Chikkala** | Model Training & Evaluation |
| **Mahadev** | Data Preprocessing |
| **Mohith** | Visualization & Dashboard |

**Woxsen University — B.Tech CSE (Data Science) | 2026**

---

## 📜 Dataset

**UCI Cleveland Heart Disease Dataset**
- 303 instances (297 after cleaning)
- 13 clinical features + 1 target variable
- Source: UCI Machine Learning Repository

---

## 📄 License

This project is for academic purposes — Woxsen University B.Tech Final Year Project, 2026.