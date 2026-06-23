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
    # Stwórz folder na resultaty
    os.makedirs("results", exist_ok=True)

    # 1. Załaduj dane
    print("Loading data...")
    X, y = load_data(data_path)

    # 2. Inicjalizacja modeli
    models = get_models()

    # 3. Uruchom eksperymenty
    run_experiment1(X, y, models)   # porównaj modele + test Wilcoxona
    run_experiment2(X, y)           # przetestuj różne parametry
    run_experiment3(X, y, models)   # przetestuj różne wielkości zbiorów

    print("\n")
    print("---------------------------------------------------------")
    print("All experiments done!")
    print("Results saved in the results/ folder.")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.data)
