import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nClass distribution:")
    print(df["HeartDisease"].value_counts())
    print()

    # Zmienna docelowa (0 = zdrowy, 1 = choroba serca)
    y = df["HeartDisease"]

    # Wszystkie pozostałe kolumny jako cechy wejściowe
    X = df.drop(columns=["HeartDisease"])

    # Zamiana danych tekstowych na wartości numeryczne przy użyciu kodowania One-Hot Encoding
    X = pd.get_dummies(X)

    print(f"Number of features after encoding: {len(X.columns)}")
    print(f"Feature names: {list(X.columns)}\n")

    # Standaryzacja danych (istotna m.in. dla KNN i SVM)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y
