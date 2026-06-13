"""
main.py — runs all 3 experiments for Heart Disease classification

Usage:
    python main.py --data <path_to_csv>

Example:
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
    # Create folder for results
    os.makedirs("results", exist_ok=True)

    # Step 1: Load and prepare the data
    print("Loading data...")
    X, y = load_data(data_path)

    # Step 2: Get all 6 models
    models = get_models()

    # Step 3: Run all experiments
    run_experiment1(X, y, models)   # compare all models + Wilcoxon test
    run_experiment2(X, y)           # test different model parameters
    run_experiment3(X, y, models)   # test different training data sizes

    print("\n" + "="*60)
    print("All experiments done!")
    print("Results saved in the results/ folder.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.data)
