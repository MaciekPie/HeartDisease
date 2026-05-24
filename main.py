"""
main.py — main pipeline for BloodTestClassification

Usage:
    python main.py --data <path_to_csv>

Pipeline:
    1. Preprocessing (train/val/test split, scaling)
    2. Cross-validation on the training set
    3. Training all models on the full training set
    4. Evaluation on the test set
    5. Saving the best model
"""
import argparse
import os
import joblib

from src.preprocessing import preprocess_data
from src.models import get_models
from src.train import train_models, cross_validate_models
from src.evaluate import evaluate_models

MODEL_DIR = "models"


def main(data_path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Preprocessing
    print("\n[1/4] Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = preprocess_data(data_path)

    # 2. Cross-validation
    print("\n[2/4] Cross-validation (3-fold) on the training set...")
    models = get_models()
    cv_results = cross_validate_models(models, X_train, y_train, n_splits=3)
    cv_results.to_csv("results/cv_results.csv", index=False)

    # 3. Training on the full training set
    print("\n[3/4] Training on the full training set...")
    models = get_models()   # fresh instances after CV
    trained_models = train_models(models, X_train, y_train)

    # 4. Evaluation on the test set
    print("\n[4/4] Evaluating on the test set...")
    metrics_df = evaluate_models(trained_models, X_test, y_test, results_dir="results")

    # 5. Save the best model
    best_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model,    os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"\n✓ Best model ({best_name}) saved → {MODEL_DIR}/best_model.pkl")
    print(f"✓ Scaler saved → {MODEL_DIR}/scaler.pkl")
    print(f"✓ Feature names saved → {MODEL_DIR}/feature_names.pkl\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BloodTest Classification Pipeline")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    args = parser.parse_args()
    main(args.data)
