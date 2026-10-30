# 🫁 Asthma Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asthma-prediction-system-h7h7gikcfxenqx3wcnrkm5.streamlit.app/)

🔗 **Live demo:** https://asthma-prediction-system-h7h7gikcfxenqx3wcnrkm5.streamlit.app/

A Data Science & Big Data (DSBD) mini project that predicts whether a patient has asthma from demographic, lifestyle, environmental, medical-history, lung-function and symptom data. The data pipeline is written in **PySpark**, models are compared with **scikit-learn / XGBoost**, and the best model is served through a **Streamlit** web app.

> ⚠️ Educational project built on a public synthetic dataset. It is **not** a medical device and must not be used for real diagnosis.

## Problem statement
Asthma symptoms (wheezing, cough, breathlessness) overlap with other conditions, which makes early detection hard. This project builds a binary classifier (asthma / no asthma) and a simple web interface around it.

## Dataset
- Kaggle: [Asthma Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoubi/asthma-disease-dataset) (`rabieelkharoubi/asthma-disease-dataset`)
- ~2,400 patients, ~30 columns, target = `Diagnosis` (1 = asthma, 0 = no asthma)
- The raw CSV is not committed. Download it from Kaggle (the notebook does this automatically with `kagglehub`).

## Tech stack
| Area | Tools |
|---|---|
| Language | Python 3 |
| Big Data | PySpark (DataFrame API, Imputer, VectorAssembler, StandardScaler, MLlib, CrossValidator) |
| Data / EDA | Pandas, NumPy, Matplotlib, Seaborn |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Models | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| App | Streamlit, joblib |
| Tools | Google Colab, Git, GitHub |

## Pipeline
1. **Load + clean (PySpark):** drop ID columns, remove duplicates, median imputation, IQR outlier capping
2. **Feature engineering:** `FEV1_FVC_Ratio`, `SymptomScore`, `AllergyScore`
3. **Spark MLlib baseline:** VectorAssembler + StandardScaler, Logistic Regression / Decision Tree / Random Forest, CrossValidator tuning
4. **EDA:** class balance, distributions, symptom prevalence, correlation heatmap
5. **Preprocessing:** one-hot encoding, scaling, stratified 80/20 split, SMOTE on the training set only
6. **Models + tuning:** GridSearchCV (3-fold, ROC-AUC) on 4 models
7. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves
8. **Deployment:** best pipeline saved with joblib, used by the Streamlit app

## Folder structure
```
asthma-prediction-system/
├── data/            # cleaned data, model comparison table
├── notebooks/       # Asthma_Prediction_DSBD.ipynb (Colab-ready)
├── app/             # Streamlit app (+ light requirements for deployment)
├── models/          # asthma_model.joblib
├── images/          # plots used in this README
├── requirements.txt
└── README.md
```

## Results
Run the notebook and paste your own numbers here (from `data/model_comparison.csv`).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| _fill after running_ | | | | | |

Best model: _fill after running_

## Screenshots
| Class balance | Correlation heatmap | Model comparison |
|---|---|---|
| ![](images/class_balance.png) | ![](images/correlation_heatmap.png) | ![](images/model_comparison.png) |

Add a screenshot of the running Streamlit app here too.

## How to run

### Option A: Google Colab (easiest)
1. Open `notebooks/Asthma_Prediction_DSBD.ipynb` in Colab (File > Open notebook > GitHub).
2. Runtime > Run all.
3. The notebook creates `models/asthma_model.joblib` and `app/app.py`.

### Option B: Local machine
```bash
git clone https://github.com/<your-username>/asthma-prediction-system.git
cd asthma-prediction-system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # PySpark also needs Java (JDK 8/11/17)
jupyter notebook notebooks/Asthma_Prediction_DSBD.ipynb
```

### Run the web app
```bash
streamlit run app/app.py
```
Use the same scikit-learn version for training and running the app (a model saved with one version may not load on another).

### Deploy (optional)
Push the repo to GitHub, then deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) with main file path `app/app.py`. Add the live link here.

## Future scope
- Real clinical data (with ethical approval) instead of a synthetic dataset
- Explainability with SHAP
- Spark Structured Streaming for real-time prediction
- Mobile-friendly UI and multilingual support

## Author
Tosha Choudhary, T.Y. B.Tech CSD, K.K. Wagh Institute of Engineering Education and Research, Nashik


