import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

DROP_COLS = ["country_code", "country_name", "region", "sub_region", "period", "primary_language"]


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df[CONDITION_COLS].idxmax(axis=1)
    df["target"] = df["target"].str.replace("condition_", "").str.replace("_pct", "")
    return df


def preprocess_data(path: str):
    """
    Loads CSV, creates target, drops unnecessary columns, encodes categoricals,
    splits into train/val/test (60/20/20) and scales features.

    Returns:
        X_train, X_val, X_test  - pd.DataFrame (column names preserved!)
        y_train, y_val, y_test  - pd.Series
        scaler                  - fitted StandardScaler
        feature_names           - list of feature names
    """
    df = pd.read_csv(path)

    # --- create target ---
    df = create_target(df)

    # --- drop unnecessary columns ---
    cols_to_drop = CONDITION_COLS + [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=["target"])
    y = df["target"]

    # --- encode categorical variables ---
    X = pd.get_dummies(X)
    feature_names = X.columns.tolist()

    # --- split: 60% train / 20% val / 20% test ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Second split without stratify — some classes may have too few samples
    # after the first split, making stratification impossible
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # --- scaling (fit only on train!) ---
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=feature_names, index=X_val.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    print(f"Class distribution (train):\n{y_train.value_counts()}\n")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
