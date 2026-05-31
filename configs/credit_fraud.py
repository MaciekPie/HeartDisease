"""
Dataset 3: Credit Card Fraud Detection
Source: kaggle.com/datasets/mlg-ulb/creditcardfraud
Task:   Binary classification — detect fraudulent transactions (0=Normal, 1=Fraud)
Note:   284k transactions, heavily imbalanced (0.17% fraud).
        V1-V28 are PCA-transformed (confidential). n_splits=5 recommended.
"""
import pandas as pd

# --- Pipeline settings ---
N_SPLITS = 5

# --- Preprocessing ---
CONDITION_COLS = []
DROP_COLS      = ["Time"]   # seconds since first transaction — not predictive

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Renames the existing Class column to 'target'."""
    df = df.copy()
    df["target"] = df["Class"].astype(str)
    df = df.drop(columns=["Class"])
    return df

# --- Streamlit app ---
PAGE_TITLE     = "Fraud Detector"
PAGE_ICON      = "💳"
APP_TITLE      = "💳 Credit Card Fraud Detection"
APP_DESC       = "Enter the transaction features to classify it as normal or fraudulent."
POSITIVE_CLASS = "0"   # "0" = Normal transaction → shown as ✅ green
FOOTER_TEXT    = "Credit Card Fraud Detection | Artificial Intelligence Methods 2026 | Wrocław"

FEATURE_LABELS = {
    "V1":     ("V1 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V2":     ("V2 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V3":     ("V3 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V4":     ("V4 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V5":     ("V5 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V6":     ("V6 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V7":     ("V7 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V8":     ("V8 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V9":     ("V9 (PCA)",               -30.0,  30.0,  0.0,   0.01),
    "V10":    ("V10 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V11":    ("V11 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V12":    ("V12 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V13":    ("V13 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V14":    ("V14 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V15":    ("V15 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V16":    ("V16 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V17":    ("V17 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V18":    ("V18 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V19":    ("V19 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V20":    ("V20 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V21":    ("V21 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V22":    ("V22 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V23":    ("V23 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V24":    ("V24 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V25":    ("V25 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V26":    ("V26 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V27":    ("V27 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "V28":    ("V28 (PCA)",              -30.0,  30.0,  0.0,   0.01),
    "Amount": ("Transaction Amount ($)",  0.0,  30000.0, 100.0, 0.01),
}

CLASS_LABELS = {
    "0": "Normal Transaction",
    "1": "Fraudulent Transaction",
}
