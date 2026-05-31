"""
app/app.py — Streamlit classification app

Run with:
    streamlit run app/app.py

Requires running main.py first to train and save the model.
To switch datasets, edit config.py (one line change).
"""
import os
import joblib
import pandas as pd
import streamlit as st

import config

# --- Model paths ---
MODEL_PATH         = os.path.join("models", "best_model.pkl")
SCALER_PATH        = os.path.join("models", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("models", "feature_names.pkl")

# --- Page config (values come from the active config file) ---
st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON, layout="wide")
st.title(config.APP_TITLE)
st.markdown(config.APP_DESC)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(FEATURE_NAMES_PATH),
    )


model, scaler, feature_names = load_model()

if model is None:
    st.error(
        "Model not found. Run `python main.py --data <file.csv>` first "
        "to train and save the model."
    )
    st.stop()

# --- Input form ---
st.subheader("Input data")
input_values = {}

numeric_features = [f for f in feature_names if f in config.FEATURE_LABELS]
other_features   = [f for f in feature_names if f not in config.FEATURE_LABELS]

cols = st.columns(3)
for i, feat in enumerate(numeric_features):
    label, mn, mx, default, step = config.FEATURE_LABELS[feat]
    with cols[i % 3]:
        input_values[feat] = st.number_input(
            label,
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step),
        )

# Dummy/one-hot columns not shown in the form — default to 0
for feat in other_features:
    input_values[feat] = 0.0

# --- Prediction ---
st.markdown("---")
col_btn, col_result = st.columns([1, 3])

with col_btn:
    predict_clicked = st.button("Classify", use_container_width=True, type="primary")

if predict_clicked:
    row        = pd.DataFrame([{f: input_values.get(f, 0.0) for f in feature_names}])
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    label      = config.CLASS_LABELS.get(str(prediction), str(prediction))

    with col_result:
        if str(prediction) == str(config.POSITIVE_CLASS):
            st.success(f"### Result: {label}")
        else:
            st.warning(f"### Result: {label}")

        if hasattr(model, "predict_proba"):
            proba    = model.predict_proba(row_scaled)[0]
            proba_df = (
                pd.DataFrame({"Class": model.classes_, "Probability": proba})
                .assign(Class=lambda d: d["Class"].map(
                    lambda x: config.CLASS_LABELS.get(str(x), str(x))
                ))
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            proba_df["Probability"] = proba_df["Probability"].map("{:.1%}".format)
            st.markdown("**Probability distribution:**")
            st.dataframe(proba_df, hide_index=True, use_container_width=True)
        else:
            st.info("This model does not return probabilities.")

st.markdown("---")
st.caption(config.FOOTER_TEXT)
