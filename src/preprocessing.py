import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    # Wczytanie danych z pliku CSV
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nClass distribution:")
    print(df["HeartDisease"].value_counts())
    print()

    # Zmienna docelowa y (0 = zdrowy, 1 = choroba serca)
    y = df["HeartDisease"]

    # Wszystkie pozostałe kolumny jako cechy wejściowe
    X = df.drop(columns=["HeartDisease"])

    # One-Hot Encoding: zamiana kolumn tekstowych (np. "Male"/"Female")
    # na kolumny binarne (0 lub 1). Modele ML wymagają danych liczbowych.
    X = pd.get_dummies(X)

    print(f"Number of features after encoding: {len(X.columns)}")
    print(f"Feature names: {list(X.columns)}\n")

    # Standaryzacja danych: przeskalowanie cech do średniej=0 i odchylenia=1 (istotna m.in. dla KNN i SVM)
    # Jest to ważne szczególnie dla KNN i SVM, które są czułe na skalę danych
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y
