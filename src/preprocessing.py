import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def preprocess_data(path: str):
    """
    Loads CSV, creates target, drops unnecessary columns, encodes categoricals,
    splits into train/val/test (60/20/20) and scales features.

    Target creation and columns to drop are defined in the active config file.

    Returns:
        X_train, X_val, X_test  — pd.DataFrame (column names preserved)
        y_train, y_val, y_test  — pd.Series
        scaler                  — fitted StandardScaler
        feature_names           — list of feature names
    """
    df = pd.read_csv(path)

    # Target column — logic defined in the active config
    df = config.create_target(df)

    # Drop dataset-specific unnecessary columns
    cols_to_drop = config.CONDITION_COLS + [c for c in config.DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Encode categorical variables
    X = pd.get_dummies(X)
    feature_names = X.columns.tolist()

    # Split: 60% train / 20% val / 20% test
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback for datasets with very few samples per class
        print("Warning: stratified split failed, falling back to random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

    # Second split without stratify — rare classes may be exhausted after first split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Scale: fit only on train, transform val and test
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_val   = pd.DataFrame(scaler.transform(X_val),       columns=feature_names, index=X_val.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_names, index=X_test.index)

    print(f"Class distribution (train):\n{y_train.value_counts()}\n")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
