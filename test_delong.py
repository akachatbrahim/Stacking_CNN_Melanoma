import joblib
import pandas as pd
from pathlib import Path
from compare_auc_delong_xu import delong_roc_test
from modules.Config import path_project

# ============================================================
# Dossier des modèles
# ============================================================

BASE_DIR = Path(path_project) 

pipelines = {
    "Pipeline1": BASE_DIR / "Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e05_dw1e05.pkl",
    "Pipeline2": BASE_DIR / "Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr2e04_dw1e05.pkl",
    "Pipeline3": BASE_DIR / "Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr1e04_dw1e05.pkl",
    "Pipeline4": BASE_DIR / "Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e04_dw1e05.pkl",
}

# ============================================================
# Chargement des pipelines
# ============================================================

models = {}

for name, filename in pipelines.items():

    pipe = joblib.load(filename)

    stack_data = None
    for key in ("test_stack", "val_stack"):
        if key in pipe:
            stack_data = pipe[key]
            break

    if stack_data is None:
        raise KeyError(f"No stacking data found in {filename}")

    models[name] = {
        "y_true": pipe["y_true"],
        "pred": pipe["meta_learner"].predict_proba(stack_data)[:, 1],
        "auc": pipe["metrics"]["AUC"]
    }

# ============================================================
# Test de DeLong
# ============================================================

results = []

names = list(models.keys())

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        name1 = names[i]
        name2 = names[j]

        y_true = models[name1]["y_true"]

        pred1 = models[name1]["pred"]
        pred2 = models[name2]["pred"]

        log10_p = delong_roc_test(
            y_true,
            pred1,
            pred2
        )

        p_value = 10 ** log10_p

        results.append({
            "Model 1": name1,
            "Model 2": name2,
            "AUC 1": models[name1]["auc"],
            "AUC 2": models[name2]["auc"],
            "p-value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        })

        print(f"{name1} vs {name2}")
        print(f"AUC : {models[name1]['auc']:.2f}% vs {models[name2]['auc']:.2f}%")
        print(f"p-value = {p_value:.6g}")
        print("-" * 60)

# ============================================================
# Sauvegarde CSV
# ============================================================

df = pd.DataFrame(results)

csv_path = BASE_DIR / "test_delong_outputs/delong_results.csv"

df.to_csv(csv_path, index=False)

print(f"\nRésultats sauvegardés dans : {csv_path}")