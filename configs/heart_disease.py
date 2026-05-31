"""
Dataset 2: Heart Disease Prediction
Source: kaggle.com/datasets/fedesoriano/heart-failure-prediction
Task:   Binary classification — predict heart disease (0=No, 1=Yes)
Note:   920 patients, 11 features, roughly balanced classes. n_splits=5 recommended.
        Categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
        are one-hot encoded automatically; not shown in the Streamlit form (default=0).
"""
import pandas as pd

# --- Pipeline settings ---
N_SPLITS = 5

# --- Preprocessing ---
CONDITION_COLS = []   # no condition columns to derive target from
DROP_COLS      = []   # no extra columns to drop

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Renames the existing HeartDisease column to 'target'."""
    df = df.copy()
    df["target"] = df["HeartDisease"].astype(str)
    df = df.drop(columns=["HeartDisease"])
    return df

# --- Streamlit app ---
PAGE_TITLE     = "Heart Disease Classifier"
PAGE_ICON      = "❤️"
APP_TITLE      = "❤️ Heart Disease Prediction"
APP_DESC       = "Enter the patient's clinical data to predict the likelihood of heart disease."
POSITIVE_CLASS = "0"   # "0" = No heart disease → shown as ✅ green
FOOTER_TEXT    = "Heart Disease Prediction | Artificial Intelligence Methods 2026 | Wrocław"

FEATURE_LABELS = {
    "Age":         ("Age",                                  18,  80,   50,  1),
    "RestingBP":   ("Resting Blood Pressure (mmHg)",        80,  200,  120, 1),
    "Cholesterol": ("Serum Cholesterol (mg/dL)",            0,   600,  200, 1),
    "FastingBS":   ("Fasting Blood Sugar >120 mg/dL (0/1)", 0,   1,    0,   1),
    "MaxHR":       ("Maximum Heart Rate Achieved",          60,  220,  150, 1),
    "Oldpeak":     ("ST Depression (Oldpeak)",              0.0, 10.0, 1.0, 0.1),
}

CLASS_LABELS = {
    "0": "No Heart Disease",
    "1": "Heart Disease Detected",
}

