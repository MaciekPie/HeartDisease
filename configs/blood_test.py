"""
Dataset 1: Blood Test Classification
Source: kaggle.com/datasets/kantesti/global-blood-test-health-insights-2025-2026
Task:   Multi-class — predict dominant health condition from population blood markers
Note:   Very small dataset (~130 rows). n_splits=3 recommended.
"""
import pandas as pd

# --- Pipeline settings ---
N_SPLITS = 3   # k-fold cross-validation splits (keep low — tiny dataset)

# --- Preprocessing ---
CONDITION_COLS = [
    "condition_healthy_pct",
    "condition_cardiovascular_pct",
    "condition_diabetes_pct",
    "condition_metabolic_syndrome_pct",
    "condition_anemia_pct",
    "condition_thyroid_pct",
    "condition_vitamin_d_deficiency_pct",
    "condition_vitamin_b12_deficiency_pct",
    "condition_liver_pct",
    "condition_kidney_pct",
    "condition_inflammation_pct",
]
DROP_COLS = ["country_code", "country_name", "region", "sub_region", "period"]

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derives target from the dominant condition column."""
    df = df.copy()
    df["target"] = df[CONDITION_COLS].idxmax(axis=1)
    df["target"] = df["target"].str.replace("condition_", "").str.replace("_pct", "")
    return df

# --- Streamlit app ---
PAGE_TITLE     = "BloodTest Classifier"
PAGE_ICON      = "🩸"
APP_TITLE      = "🩸 Blood Test Classification"
APP_DESC       = "Enter the patient's blood test results to receive a health status prediction."
POSITIVE_CLASS = "healthy"   # shown as ✅ green
FOOTER_TEXT    = "Blood Test Classification | Artificial Intelligence Methods 2026 | Wrocław"

FEATURE_LABELS = {
    "age":                    ("Age",                        10,   100,  40,    1),
    "hemoglobin_gdl":         ("Hemoglobin (g/dL)",          5.0,  20.0, 13.5,  0.1),
    "wbc_count_thou_ul":      ("WBC count (thou/µL)",        1.0,  30.0, 7.0,   0.1),
    "rbc_count_mill_ul":      ("RBC count (mill/µL)",        2.0,  7.0,  4.8,   0.1),
    "platelets_thou_ul":      ("Platelets (thou/µL)",        50,   600,  250,   1),
    "hematocrit_pct":         ("Hematocrit (%)",             20,   60,   42,    0.1),
    "glucose_mgdl":           ("Glucose (mg/dL)",            50,   400,  95,    1),
    "creatinine_mgdl":        ("Creatinine (mg/dL)",         0.3,  10.0, 1.0,   0.1),
    "alt_ul":                 ("ALT (U/L)",                  5,    200,  30,    1),
    "ast_ul":                 ("AST (U/L)",                  5,    200,  28,    1),
    "tsh_uiul":               ("TSH (µIU/L)",                0.1,  10.0, 2.0,   0.1),
    "vitamin_d_ngml":         ("Vitamin D (ng/mL)",          5,    100,  30,    1),
    "vitamin_b12_pgml":       ("Vitamin B12 (pg/mL)",        100,  1000, 400,   1),
    "cholesterol_total_mgdl": ("Total Cholesterol (mg/dL)",  100,  400,  190,   1),
    "hdl_mgdl":               ("HDL (mg/dL)",                20,   100,  55,    1),
    "ldl_mgdl":               ("LDL (mg/dL)",                50,   300,  110,   1),
    "triglycerides_mgdl":     ("Triglycerides (mg/dL)",      50,   500,  130,   1),
    "crp_mgdl":               ("CRP (mg/dL)",                0.0,  20.0, 0.5,   0.1),
    "ferritin_ngml":          ("Ferritin (ng/mL)",           5,    500,  80,    1),
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

