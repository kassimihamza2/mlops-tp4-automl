
import numpy as np
import pandas as pd

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Charger les données
df = pd.read_csv("data/transactions.csv")

# 2. Créer la colonne piège :
# le numéro du dossier n'existe qu'après la contestation d'une fraude.
rng = np.random.default_rng(0)

df["num_dossier_litige"] = np.where(
    df["fraude"] == 1,
    rng.integers(10000, 99999, len(df)),
    0,
)

# 3. Séparer les variables et la cible
X = df.drop(columns="fraude")
y = df["fraude"]

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

# 4. Entraîner FLAML
a = AutoML()

a.fit(
    Xtr,
    ytr,
    task="classification",
    metric="f1",
    time_budget=30,
    seed=42,
    verbose=0,
)

# 5. Afficher le modèle retenu
print("Famille retenue :", a.best_estimator)

# 6. Afficher l'importance des variables si disponible
modele = a.model.estimator

if hasattr(modele, "feature_importances_"):
    imp = sorted(
        zip(X.columns, modele.feature_importances_),
        key=lambda t: -t[1],
    )
    print(
        "Importance des variables :",
        [(c, round(float(v), 3)) for c, v in imp],
    )

# 7. Évaluer avec la colonne piège
score = f1_score(yte, a.predict(Xte))
print("F1 en test, avec la colonne piégée :", round(score, 4))

# 8. Simuler la production : aucun dossier n'existe encore
Xprod = Xte.copy()
Xprod["num_dossier_litige"] = 0

score_prod = f1_score(
    yte,
    a.predict(Xprod),
    zero_division=0,
)

print(
    'F1 "en production" (colonne vide) :',
    round(score_prod, 4),
)
