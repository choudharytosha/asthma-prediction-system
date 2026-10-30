"""Asthma Prediction System - Streamlit web app.

Run from the project root:
    streamlit run app/app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Asthma Prediction System", page_icon="🫁", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "asthma_model.joblib"


@st.cache_resource
def load_bundle():
    # bundle = {"model": pipeline, "model_name": str, "features": [...], "metrics": {...}}
    return joblib.load(MODEL_PATH)


def yes_no(label, key):
    # Streamlit checkbox -> 1 / 0
    return 1 if st.checkbox(label, key=key) else 0


bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]

st.title("🫁 Asthma Disease Prediction System")
st.caption(f"Model: {bundle['model_name']}  |  DSBD mini project (educational use only)")

st.subheader("1. Patient details")
c1, c2 = st.columns(2)
age = c1.slider("Age (years)", 5, 80, 30)
gender = c2.selectbox("Gender", ["Male", "Female"])
ethnicity = c1.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
education = c2.selectbox("Education level", ["None", "High School", "Bachelor's", "Higher"])
bmi = c1.slider("BMI", 15.0, 40.0, 24.0, 0.1)
smoking = c2.selectbox("Smoker?", ["No", "Yes"])

st.subheader("2. Lifestyle and environment (0 = low, 10 = high)")
c1, c2 = st.columns(2)
physical = c1.slider("Physical activity", 0.0, 10.0, 5.0, 0.1)
diet = c2.slider("Diet quality", 0.0, 10.0, 5.0, 0.1)
sleep = c1.slider("Sleep quality (4-10)", 4.0, 10.0, 7.0, 0.1)
pollution = c2.slider("Pollution exposure", 0.0, 10.0, 5.0, 0.1)
pollen = c1.slider("Pollen exposure", 0.0, 10.0, 5.0, 0.1)
dust = c2.slider("Dust exposure", 0.0, 10.0, 5.0, 0.1)

st.subheader("3. Medical history")
c1, c2 = st.columns(2)
with c1:
    family = yes_no("Family history of asthma", "family")
    allergies = yes_no("History of allergies", "allergies")
    pet = yes_no("Pet allergy", "pet")
with c2:
    eczema = yes_no("Eczema", "eczema")
    hay = yes_no("Hay fever", "hay")
    gerd = yes_no("Acid reflux (GERD)", "gerd")

st.subheader("4. Lung function")
c1, c2 = st.columns(2)
fev1 = c1.slider("FEV1 (litres)", 1.0, 4.0, 2.5, 0.01)
fvc = c2.slider("FVC (litres)", 1.5, 6.0, 3.5, 0.01)

st.subheader("5. Symptoms")
c1, c2 = st.columns(2)
with c1:
    wheezing = yes_no("Wheezing", "wheezing")
    breath = yes_no("Shortness of breath", "breath")
    chest = yes_no("Chest tightness", "chest")
with c2:
    cough = yes_no("Coughing", "cough")
    night = yes_no("Nighttime symptoms", "night")
    exercise = yes_no("Exercise-induced symptoms", "exercise")

if st.button("Predict", type="primary"):
    row = {
        "Age": age,
        "Gender": 0 if gender == "Male" else 1,
        "Ethnicity": ["Caucasian", "African American", "Asian", "Other"].index(ethnicity),
        "EducationLevel": ["None", "High School", "Bachelor's", "Higher"].index(education),
        "BMI": bmi,
        "Smoking": 1 if smoking == "Yes" else 0,
        "PhysicalActivity": physical,
        "DietQuality": diet,
        "SleepQuality": sleep,
        "PollutionExposure": pollution,
        "PollenExposure": pollen,
        "DustExposure": dust,
        "PetAllergy": pet,
        "FamilyHistoryAsthma": family,
        "HistoryOfAllergies": allergies,
        "Eczema": eczema,
        "HayFever": hay,
        "GastroesophagealReflux": gerd,
        "LungFunctionFEV1": fev1,
        "LungFunctionFVC": fvc,
        "Wheezing": wheezing,
        "ShortnessOfBreath": breath,
        "ChestTightness": chest,
        "Coughing": cough,
        "NighttimeSymptoms": night,
        "ExerciseInduced": exercise,
    }
    # Same engineered features as the notebook
    row["FEV1_FVC_Ratio"] = fev1 / fvc
    row["SymptomScore"] = wheezing + breath + chest + cough + night + exercise
    row["AllergyScore"] = pet + allergies + eczema + hay

    X = pd.DataFrame([row])[FEATURES].astype(float)
    prob = float(model.predict_proba(X)[0, 1])

    st.divider()
    st.metric("Asthma probability", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        st.error("Higher risk of asthma. Please consult a doctor for proper diagnosis.")
    else:
        st.success("Lower risk of asthma based on the entered values.")

st.info("This is a student project built on a public dataset. It is not a medical device and "
        "must not be used for real diagnosis.")
