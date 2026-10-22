
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Chemins du projet
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"

# Charger les données
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["fraude"])
y = df["fraude"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp4-automl")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42,
    ),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        start = time.perf_counter()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        duration = time.perf_counter() - start
        score = f1_score(y_test, predictions)

        mlflow.log_param("model", name)
        mlflow.log_metric("f1", score)
        mlflow.log_metric("train_duration_seconds", duration)

        print(
            f"{name} | F1: {score:.4f} "
            f"| Durée: {duration:.2f} secondes"
        )

print("Comparaison des modèles de référence terminée.")
