
import argparse
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from flaml import AutoML
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
REPORTS_DIR = ROOT / "reports"


def main(budget):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger les données
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["fraude"])
    y = df["fraude"]

    # 2. Séparer les données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # 3. Préparer le journal FLAML
    log_file = REPORTS_DIR / f"flaml_{budget}s.log"

    # 4. Configurer MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tp4-automl")

    # 5. Configurer AutoML
    automl = AutoML()

    settings = {
        "time_budget": budget,
        "metric": "f1",
        "task": "classification",
        "seed": 42,
        "log_type": "all",
        "log_file_name": str(log_file),
        "estimator_list": [
            "lrl1",
            "lrl2",
            "rf",
            "extra_tree",
            "lgbm",
            "xgboost",
        ],
        "n_jobs": 1,
        "verbose": 1,
        "mlflow_logging": False,
    }

    # 6. Exécuter la recherche AutoML
    with mlflow.start_run(run_name=f"flaml-{budget}s"):
        start = time.perf_counter()

        automl.fit(
            X_train=X_train,
            y_train=y_train,
            **settings,
        )

        search_duration = time.perf_counter() - start

        # 7. Évaluer sur le jeu de test
        predictions = automl.predict(X_test)
        score = f1_score(y_test, predictions)

        # 8. Enregistrer les paramètres attendus
        mlflow.log_param("outil", "flaml")
        mlflow.log_param("budget_s", budget)
        mlflow.log_param("famille", automl.best_estimator)

        # 9. Enregistrer les métriques
        mlflow.log_metric("f1", score)
        mlflow.log_metric("search_duration_seconds", search_duration)
        mlflow.log_metric("f1_validation", 1 - automl.best_loss)

        # 10. Enregistrer le modèle
        mlflow.sklearn.log_model(
            automl.model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        print("\n" + "=" * 45)
        print(f"Budget demandé : {budget} secondes")
        print(f"Meilleur algorithme : {automl.best_estimator}")
        print(f"F1 sur le jeu de test : {score:.4f}")
        print(f"F1 de validation : {1 - automl.best_loss:.4f}")
        print(f"Durée de recherche : {search_duration:.2f} secondes")
        print(f"Journal : {log_file}")
        print("Modèle enregistré dans MLflow.")
        print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimisation automatique avec FLAML et MLflow"
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Budget de recherche en secondes",
    )

    args = parser.parse_args()
    main(args.budget)
