
import mlflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

runs = mlflow.search_runs(experiment_names=["tp4-automl"])

# Runs AutoML : garder le dernier run pour chaque budget
auto = runs[
    runs["params.outil"].isin(["flaml", "sklearn-maison"])
].copy()

auto["budget"] = auto["params.budget_s"].astype(int)
auto = auto.sort_values("start_time").drop_duplicates(
    ["params.outil", "budget"], keep="last"
).sort_values("budget")

# Modèles de référence scikit-learn
refs = runs[
    runs["params.model"].isin(["LogisticRegression", "RandomForest"])
].copy()

plt.figure(figsize=(7, 4))

for outil, g in auto.groupby("params.outil"):
    plt.plot(
        g["budget"],
        g["metrics.f1"],
        "o-",
        label=f"AutoML ({outil})",
    )

    for _, r in g.iterrows():
        print(
            f"{outil:15s} budget {int(r['budget']):4d} s"
            f" -> F1 {r['metrics.f1']:.4f}"
        )

for _, r in refs.iterrows():
    nom = r["params.model"]
    score = r["metrics.f1"]

    plt.axhline(score, linestyle="--", color="gray", linewidth=1)

    if not auto.empty:
        plt.text(
            auto["budget"].min(),
            score + 0.003,
            f"{nom} ({score:.4f})",
            fontsize=8,
            color="gray",
        )

    print(f"{nom:22s} F1 {score:.4f}")

plt.xscale("log")

if not auto.empty:
    budgets = sorted(auto["budget"].unique())
    plt.xticks(budgets, [str(b) for b in budgets])

plt.minorticks_off()
plt.xlabel("Budget de calcul (secondes, échelle log)")
plt.ylabel("F1 sur le jeu de test")
plt.title("Rendement du budget de calcul")
plt.legend(loc="lower right")
plt.tight_layout()

output = REPORTS_DIR / "rendement.png"
plt.savefig(output, dpi=120, bbox_inches="tight")
plt.close()

print(f"Graphique écrit : {output}")
