"""
calibration_curve_brier.py
Calibration Curve + Brier Score
Compatible avec votre pipeline .pkl
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from modules.Config import path_project

# ==========================================================
# Configuration
# ==========================================================

from pathlib import Path

PIPELINE_PATH =   Path(rf"{path_project}/Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e05_dw1e05.pkl")

OUTPUT_DIR = Path(path_project)  / "Calibration_Results"

N_BINS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pipeline_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# ==========================================================
# Load pipeline
# ==========================================================

print("="*70)
print("Loading pipeline ...")
print("="*70)

pipeline = load_pipeline_artifact(PIPELINE_PATH)

print("Pipeline loaded successfully.\n")

# ==========================================================
# Extraction
# ==========================================================

ensemble_models = pipeline["ensemble_models"]

meta_model = pipeline["meta_learner"]

threshold = pipeline["threshold"]

base_model_predictions = pipeline["base_model_predictions"]

model_configs = pipeline["model_configs"]

y_true = np.asarray(
    pipeline["y_true"]
).ravel()

test_stack = pipeline["test_stack"]

print("Models :", len(base_model_predictions))
print("Samples:", len(y_true))
print()

# ==========================================================
# Stacking probability
# ==========================================================

if hasattr(meta_model, "predict_proba"):

    stacking_prob = meta_model.predict_proba(
        test_stack
    )[:, 1]

else:

    stacking_prob = meta_model.predict(
        test_stack
    )

# ==========================================================
# Brier score
# ==========================================================

results = []

print("="*70)
print("Brier Scores")
print("="*70)

for model_name in base_model_predictions.keys():

    probs = np.asarray(
        base_model_predictions[model_name]["test"]
    ).ravel()

    score = brier_score_loss(
        y_true,
        probs
    )

    print(f"{model_name:20s} : {score:.5f}")

    results.append({

        "Model": model_name,

        "Brier Score": score

    })

stack_score = brier_score_loss(
    y_true,
    stacking_prob
)

print(f"{'Stacking':20s} : {stack_score:.5f}")

results.append({

    "Model": "Stacking",

    "Brier Score": stack_score

})

print()

# ==========================================================
# Calibration curves
# ==========================================================

plt.figure(
    figsize=(7,7)
)

colors = [

    "tab:blue",

    "tab:green",

    "tab:red",

    "black"

]

# ----------------------------------------------------------

for color, model_name in zip(

    colors,

    base_model_predictions.keys()

):

    probs = np.asarray(

        base_model_predictions[model_name]["test"]

    ).ravel()

    prob_true, prob_pred = calibration_curve(

        y_true,

        probs,

        n_bins=N_BINS,

        strategy="uniform"

    )

    plt.plot(

        prob_pred,

        prob_true,

        marker="o",

        linewidth=2,

        label=model_name,

        color=color

    )

# ----------------------------------------------------------

prob_true, prob_pred = calibration_curve(

    y_true,

    stacking_prob,

    n_bins=N_BINS,

    strategy="uniform"

)

plt.plot(

    prob_pred,

    prob_true,

    marker="s",

    linewidth=3,

    color="orange",

    label="Stacking"

)

plt.plot(

    [0,1],

    [0,1],

    "--",

    color="gray",

    linewidth=2,

    label="Perfect calibration"

)

plt.xlabel(

    "Mean Predicted Probability",

    fontsize=12

)

plt.ylabel(

    "Observed Frequency",

    fontsize=12

)

plt.title(

    "Calibration Curves",

    fontsize=14

)

plt.grid(True)

plt.legend()

plt.tight_layout()

# ==========================================================
# Save figures
# ==========================================================

for ext in [

    "png",

    "pdf",

    "svg"

]:

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            f"CalibrationCurve.{ext}"

        ),

        dpi=300,

        bbox_inches="tight"

    )

plt.show()

# ==========================================================
# Save Brier table
# ==========================================================

df = pd.DataFrame(results)

df = df.sort_values(

    by="Brier Score"

)

print()

print("="*60)

print(df)

print("="*60)

df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "BrierScores.csv"

    ),

    index=False

)

print()

print("Results saved in")

print(OUTPUT_DIR)

print()

print("Calibration figure :")

print("   CalibrationCurve.png")

print("   CalibrationCurve.pdf")

print("   CalibrationCurve.svg")

print()

print("Brier table :")

print("   BrierScores.csv")
