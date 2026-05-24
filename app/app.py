"""
app/app.py — Streamlit app: BloodTest Classification

Run with:
    streamlit run app/app.py

Requires running main.py first to train and save the model.
"""
import os
import joblib
import pandas as pd
import streamlit as st

# --- Paths ---
MODEL_PATH         = os.path.join("models", "best_model.pkl")
SCALER_PATH        = os.path.join("models", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("models", "feature_names.pkl")

# --- Page config ---
st.set_page_config(
    page_title="BloodTest Classifier",
    page_icon="🩸",
    layout="wide",
)

st.title("🩸 Blood Test Classification")
st.markdown("Enter the patient's blood test results to receive a health status prediction.")

# --- Load model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

if model is None:
    st.error(
        "⚠️ Model not found. Please run `python main.py --data <file.csv>` first "
        "to train and save the model."
    )
    st.stop()

# --- Feature labels: name, min, max, default, step ---
FEATURE_LABELS = {
    "age":                    ("Age",                        10,  100, 40,   1),
    "hemoglobin_gdl":         ("Hemoglobin (g/dL)",          5.0, 20.0, 13.5, 0.1),
    "wbc_count_thou_ul":      ("WBC count (thou/µL)",        1.0, 30.0, 7.0,  0.1),
    "rbc_count_mill_ul":      ("RBC count (mill/µL)",        2.0, 7.0,  4.8,  0.1),
    "platelets_thou_ul":      ("Platelets (thou/µL)",        50,  600,  250,  1),
    "hematocrit_pct":         ("Hematocrit (%)",             20,  60,   42,   0.1),
    "glucose_mgdl":           ("Glucose (mg/dL)",            50,  400,  95,   1),
    "creatinine_mgdl":        ("Creatinine (mg/dL)",         0.3, 10.0, 1.0,  0.1),
    "alt_ul":                 ("ALT (U/L)",                  5,   200,  30,   1),
    "ast_ul":                 ("AST (U/L)",                  5,   200,  28,   1),
    "tsh_uiul":               ("TSH (µIU/L)",                0.1, 10.0, 2.0,  0.1),
    "vitamin_d_ngml":         ("Vitamin D (ng/mL)",          5,   100,  30,   1),
    "vitamin_b12_pgml":       ("Vitamin B12 (pg/mL)",        100, 1000, 400,  1),
    "cholesterol_total_mgdl": ("Total Cholesterol (mg/dL)",  100, 400,  190,  1),
    "hdl_mgdl":               ("HDL (mg/dL)",                20,  100,  55,   1),
    "ldl_mgdl":               ("LDL (mg/dL)",                50,  300,  110,  1),
    "triglycerides_mgdl":     ("Triglycerides (mg/dL)",      50,  500,  130,  1),
    "crp_mgdl":               ("CRP (mg/dL)",                0.0, 20.0, 0.5,  0.1),
    "ferritin_ngml":          ("Ferritin (ng/mL)",           5,   500,  80,   1),
}

CLASS_LABELS = {
    "healthy":                "Healthy",
    "cardiovascular":         "Cardiovascular disease",
    "diabetes":               "Diabetes",
    "metabolic_syndrome":     "Metabolic syndrome",
    "anemia":                 "Anemia",
    "thyroid":                "Thyroid disorder",
    "vitamin_d_deficiency":   "Vitamin D deficiency",
    "vitamin_b12_deficiency": "Vitamin B12 deficiency",
    "liver":                  "Liver disease",
    "kidney":                 "Kidney disease",
    "inflammation":           "Inflammation",
}

# --- Input form ---
st.subheader("📋 Patient data")

input_values = {}

numeric_features = [f for f in feature_names if f in FEATURE_LABELS]
other_features   = [f for f in feature_names if f not in FEATURE_LABELS]

cols = st.columns(3)
for i, feat in enumerate(numeric_features):
    label, mn, mx, default, step = FEATURE_LABELS[feat]
    with cols[i % 3]:
        input_values[feat] = st.number_input(
            label,
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step),
        )

# Fill remaining features (dummy columns) with zeros
for feat in other_features:
    input_values[feat] = 0.0

# --- Prediction ---
st.markdown("---")
col_btn, col_result = st.columns([1, 3])

with col_btn:
    predict_clicked = st.button("🔍 Classify", use_container_width=True, type="primary")

if predict_clicked:
    # Build feature vector in the correct order
    row = pd.DataFrame([{f: input_values.get(f, 0.0) for f in feature_names}])
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    label      = CLASS_LABELS.get(prediction, prediction)

    with col_result:
        if prediction == "healthy":
            st.success(f"### ✅ Result: {label}")
        else:
            st.warning(f"### ⚠️ Result: {label}")

        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(row_scaled)[0]
            classes = model.classes_
            proba_df = (
                pd.DataFrame({"Class": classes, "Probability": proba})
                .assign(Class=lambda d: d["Class"].map(lambda x: CLASS_LABELS.get(x, x)))
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            proba_df["Probability"] = proba_df["Probability"].map("{:.1%}".format)

            st.markdown("**Probability distribution:**")
            st.dataframe(proba_df, hide_index=True, use_container_width=True)
        else:
            st.info("This model does not return probabilities.")

# --- Footer ---
st.markdown("---")
st.caption("BloodTest Classification | Artificial Intelligence Methods 2026 | Wrocław")
