import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def create_target(df):
    condition_cols = [
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
        "condition_inflammation_pct"
    ]

    df["target"] = df[condition_cols].idxmax(axis=1)
    df["target"] = df["target"].str.replace("condition_", "").str.replace("_pct", "")

    return df


def preprocess_data(path):
    df = pd.read_csv(path)

    df = create_target(df)

    # usuwamy kolumny condition
    condition_cols = [col for col in df.columns if "condition_" in col]
    df = df.drop(columns=condition_cols)

    # usuwamy zbędne kolumny
    df = df.drop(columns=[
        "country_code", "country_name", "region", "sub_region", "period"
    ])

    X = df.drop(columns=["target"])
    y = df["target"]

    # encoding
    X = pd.get_dummies(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

