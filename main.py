from src.preprocessing import preprocess_data
from src.models import get_models
from src.train import train_models
from src.evaluate import evaluate_models

import joblib



# 1. preprocessing
X_train, X_test, y_train, y_test, scaler = preprocess_data("data/kantesti_global_health_insights_2025_2026.csv")

# 2. modele
models = get_models()

# 3. trening
trained_models = train_models(models, X_train, y_train)

# 4. ewaluacja
results = evaluate_models(trained_models, X_test, y_test)

# 5. zapis najlepszego modelu
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")

joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

