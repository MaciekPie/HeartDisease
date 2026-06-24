"""
Uruchomienie:
    python main.py --data <path_to_csv>

Przykład:
    python main.py --data data/heart.csv
"""

import os
import argparse

from src.preprocessing import load_data
from src.models import get_models
from src.experiment1 import run_experiment1
from src.experiment2 import run_experiment2
from src.experiment3 import run_experiment3


def main(data_path):
    # Stwórz folder na wyniki jeśli nie istnieje
    os.makedirs("results", exist_ok=True)

    # 1. Wczytaj dane z pliku CSV
    print("Loading data...")
    X, y = load_data(data_path)

    # 2. Stwórz słownik z modelami do porównania
    models = get_models()

    # 3. Uruchom kolejno trzy eksperymenty
    run_experiment1(X, y, models)   # porównaj modele + test Wilcoxona
    run_experiment2(X, y)           # wpływ parametrów modelu na wyniki
    run_experiment3(X, y, models)   # wpływ wielkości zbioru treningowego

    print("\n")
    print("---------------------------------------------------------")
    print("All experiments finished!")
    print("Scores are saved in results/")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    # Obsługa argumentów z linii poleceń
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.data)
