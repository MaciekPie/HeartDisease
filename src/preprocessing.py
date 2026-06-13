import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    # Read the CSV file
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nClass distribution:")
    print(df["HeartDisease"].value_counts())
    print()

    # y = what we want to predict (0 = healthy, 1 = heart disease)
    y = df["HeartDisease"]

    # X = all other columns (features)
    X = df.drop(columns=["HeartDisease"])

    # Convert text columns to numbers using one-hot encoding
    # e.g. Sex: M/F -> Sex_M: 1/0, Sex_F: 1/0
    X = pd.get_dummies(X)

    print(f"Number of features after encoding: {len(X.columns)}")
    print(f"Feature names: {list(X.columns)}\n")

    # Scale all features so they are on a similar scale
    # (important for models like KNN and SVM)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y
