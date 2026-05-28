import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


def train_models(models: dict, X_train, y_train) -> dict:
    """Trains all models on the training set."""
    trained_models = {}
    for name, model in models.items():
        print(f"Training: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  ✓ {name} done")
    return trained_models


def cross_validate_models(models: dict, X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """
    Performs stratified k-fold cross-validation on the training set.
    Results are returned as a DataFrame.

    Metrics: balanced accuracy, macro F1, macro Precision, macro Recall
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    rows = []
    for name, model in models.items():
        print(f"Cross-validation ({n_splits}-fold): {name}...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "Model":           name,
            "CV Balanced accuracy":     round(np.mean(scores["test_balanced_accuracy"]), 4),
            "CV Balanced accuracy std": round(np.std(scores["test_balanced_accuracy"]), 4),
            "CV F1 (macro)":   round(np.mean(scores["test_f1_macro"]), 4),
            "CV F1 std":       round(np.std(scores["test_f1_macro"]), 4),
            "CV Precision":    round(np.mean(scores["test_precision_macro"]), 4),
            "CV Recall":       round(np.mean(scores["test_recall_macro"]), 4),
        })

    df = pd.DataFrame(rows).sort_values("CV F1 (macro)", ascending=False)
    print("\n=== Cross-Validation Results ===")
    print(df.to_string(index=False))
    return df
