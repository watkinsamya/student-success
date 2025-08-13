import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Project Root & Models Directory ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parent of "app"
MODELS_DIR = PROJECT_ROOT / "models"

KAGGLE_PATH = MODELS_DIR / "kaggle_classifier.joblib"
UCI_PATH = MODELS_DIR / "uci_regressor.joblib"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Student Success Predictor", layout="centered")
st.title("🎓 Student Grades & Academic Success Predictor")

# --- Load Models ---
def load_kaggle():
    if not KAGGLE_PATH.exists():
        st.warning(f"Train & save the Kaggle classifier first ({KAGGLE_PATH}).")
        return None, None
    obj = joblib.load(KAGGLE_PATH)
    if isinstance(obj, dict):
        return obj.get("pipeline"), obj.get("label_encoder")
    return obj, None

def load_uci():
    if not UCI_PATH.exists():
        st.warning(f"Train & save the UCI regressor first ({UCI_PATH}).")
        return None
    return joblib.load(UCI_PATH)

kaggle_pipe, kaggle_le = load_kaggle()
uci_model = load_uci()

# Debug info (optional, can remove later)
st.caption(f"Kaggle model found: {KAGGLE_PATH.exists()} | UCI model found: {UCI_PATH.exists()}")

# --- Tabs ---
tab1, tab2 = st.tabs(["Kaggle: Risk Category", "UCI: Final Grade (G3)"])

# Tab 1 — Kaggle Classification
with tab1:
    st.subheader("Risk Category (Kaggle)")
    race_ethnicity = st.selectbox(
        "Race/Ethnicity (dataset groups)",
        ["group A", "group B", "group C", "group D", "group E"],
        help=("These are anonymized categories from the Kaggle dataset. "
              "They are coded as Group A–E and do not correspond to real-world ethnicities."),
    )
    parent_education = st.selectbox(
        "Parental level of education",
        [
            "bachelor's degree", "some college", "master's degree",
            "associate's degree", "high school", "some high school"
        ],
    )
    gender = st.selectbox("Gender", ["female", "male"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    prep = st.selectbox("Test Prep Course", ["none", "completed"])
    math = st.number_input("Math score", 0, 100, 65)
    reading = st.number_input("Reading score", 0, 100, 70)
    writing = st.number_input("Writing score", 0, 100, 68)

    if st.button("Predict Risk Category"):
        x = pd.DataFrame([{
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "parent_education": parent_education,
            "lunch": lunch,
            "test_prep": prep,
            "math": math,
            "reading": reading,
            "writing": writing
        }])

        if kaggle_pipe is None or kaggle_le is None:
            st.warning("Kaggle model not loaded. Please train & save it first.")
        else:
            enc_pred = kaggle_pipe.predict(x)[0]
            label = kaggle_le.inverse_transform([enc_pred])[0]
            st.success(f"Predicted category: **{label}**")

# Tab 2 — UCI Regression
with tab2:
    st.subheader("Predict Final Grade (UCI G3)")
    g1 = st.number_input("G1 (0–20)", 0, 20, 10)
    g2 = st.number_input("G2 (0–20)", 0, 20, 12)
    absences = st.number_input("Absences", 0, 100, 3)
    studytime = st.number_input("Study time (1–4)", 1, 4, 2)
    higher = st.selectbox("Wants higher education?", ["yes", "no"])
    internet = st.selectbox("Internet access?", ["yes", "no"])
    romantic = st.selectbox("Romantic relationship?", ["no", "yes"])

    if st.button("Predict G3"):
        x = pd.DataFrame([{
            "G1": g1,
            "G2": g2,
            "absences": absences,
            "studytime": studytime,
            "grade_trend": g2 - g1,
            "higher": higher,
            "internet": internet,
            "romantic": romantic
        }])

        if uci_model is None:
            st.warning("UCI model not loaded. Please train & save it first.")
        else:
            yhat = uci_model.predict(x)[0]
            st.success(f"Predicted final grade (G3): **{yhat:.1f} / 20**")
