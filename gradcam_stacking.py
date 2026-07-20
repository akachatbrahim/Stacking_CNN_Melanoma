# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:19:38 2026

@author: brahim
"""

"""
gradcam_stacking.py
Compatible avec votre pipeline sauvegardé (.pkl)
"""

import os
import pickle
import joblib
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.Config import path_project

# ============================================================
# Pipeline
# ============================================================

from pathlib import Path

PIPELINE_PATH =   Path(rf"{path_project}/Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e05_dw1e05.pkl")


def load_pipeline_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


pipeline = load_pipeline_artifact(PIPELINE_PATH)

print("="*60)
print("Pipeline loaded")
print("="*60)

# ============================================================
# Extraction automatique
# ============================================================

ensemble_models = pipeline["ensemble_models"]

meta_model = pipeline["meta_learner"]

threshold = pipeline["threshold"]

base_model_predictions = pipeline["base_model_predictions"]

model_configs = pipeline["model_configs"]

y_true = pipeline["y_true"]

test_stack = pipeline["test_stack"]

print()

print("Models :",len(ensemble_models))

print("Threshold :",threshold)

print()

# ============================================================
# Construction automatique
# ============================================================

models = {}

for model, config in zip(ensemble_models, model_configs):

    model_name = config[0]
    preprocess_fn = config[2]

    models[model_name] = {

        "model": model,

        "preprocess": preprocess_fn

    }

print()

print(models.keys())

# ============================================================
# Meta learner
# ============================================================

if hasattr(meta_model,"predict_proba"):

    stacking_probability = meta_model.predict_proba(
        test_stack
    )[:,1]

else:

    stacking_probability = meta_model.predict(
        test_stack
    )

# ============================================================
# Images
# ============================================================

IMAGE_DIR=Path(f"{path_project}/Images_test")

OUTPUT_DIR=Path(f"{path_project}/gradcam_outputs")

os.makedirs(OUTPUT_DIR,exist_ok=True)

images=[]

for f in os.listdir(IMAGE_DIR):

    if f.lower().endswith((".jpg",".jpeg",".png")):

        images.append(
            os.path.join(
                IMAGE_DIR,
                f
            )
        )

images=sorted(images)

print()

print("Images :",len(images))


def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        try:

            if len(layer.output.shape)==4:

                return layer.name

        except:

            continue

    raise ValueError("No convolution layer found.")

def make_gradcam_heatmap(
        img_array,
        model):

    layer_name=find_last_conv_layer(model)

    grad_model=tf.keras.Model(

        model.inputs,

        [

            model.get_layer(layer_name).output,

            model.output

        ]

    )

    with tf.GradientTape() as tape:

        conv_output,preds=grad_model(img_array)

        loss=preds[:,0]

    grads=tape.gradient(loss,conv_output)

    pooled=tf.reduce_mean(

        grads,

        axis=(0,1,2)

    )

    conv_output=conv_output[0]

    heatmap=tf.reduce_sum(

        pooled*conv_output,

        axis=-1

    )

    heatmap=tf.maximum(

        heatmap,

        0

    )

    heatmap/=tf.reduce_max(

        heatmap

    )+1e-8

    return heatmap.numpy()


def overlay(image,heatmap):

    heatmap=cv2.resize(

        heatmap,

        (image.shape[1],image.shape[0])

    )

    heatmap=np.uint8(255*heatmap)

    heatmap=cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )

    heatmap=cv2.cvtColor(

        heatmap,

        cv2.COLOR_BGR2RGB

    )

    return cv2.addWeighted(

        image,

        0.6,

        heatmap,

        0.4,

        0

    )

def process_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    return img


def infer_true_label(image_path):
    name = os.path.basename(image_path).lower()
    parent = os.path.basename(os.path.dirname(image_path)).lower()

    if any(k in name for k in ["melanoma", "malignant", "cancer"]):
        return "malignant"
    if any(k in name for k in ["benign", "healthy"]):
        return "benign"
    if any(k in parent for k in ["malignant", "melanoma", "cancer"]):
        return "malignant"
    if any(k in parent for k in ["benign", "healthy"]):
        return "benign"
    return "unknown"


def predict_stacking_probability(image, models, meta_model):
    features = []

    for _, info in models.items():
        model = info["model"]
        preprocess = info["preprocess"]

        x = preprocess(image.astype(np.float32))
        x = np.expand_dims(x, axis=0)
        prob = float(model.predict(x, verbose=0)[0][0])
        features.append(prob)

    X = np.array([features], dtype=np.float32)

    if hasattr(meta_model, "predict_proba"):
        return float(meta_model.predict_proba(X)[:, 1][0])

    return float(meta_model.predict(X)[0])


def run():

    # --------------------------------------------------
    # Lire les 6 premières images
    # --------------------------------------------------
    image_files = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < 6:
        raise ValueError(
            f"Le dossier '{IMAGE_DIR}' doit contenir au moins 6 images."
        )

    image_files = image_files[:6]

    print(f"{len(image_files)} images sélectionnées.")

    # ===========================================================
    # Boucle principale
    # ===========================================================

    for idx, image_path in enumerate(image_files):

        print(f"\nTraitement : {os.path.basename(image_path)}")

        image = cv2.imread(image_path)

        if image is None:
            print("Impossible de lire :", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        fig, axs = plt.subplots(
            1,
            5,
            figsize=(17, 4),
            constrained_layout=True
        )

        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[0].axis("off")

        cnn_probs = []

        col = 1

        for name, info in models.items():

            model = info["model"]
            preprocess = info["preprocess"]

            x = preprocess(image.astype(np.float32))
            x = np.expand_dims(x, axis=0)

            prob = float(
                model.predict(
                    x,
                    verbose=0
                )[0][0]
            )

            cnn_probs.append(prob)

            heatmap = make_gradcam_heatmap(
                x,
                model
            )

            cam = overlay(
                image,
                heatmap
            )

            axs[col].imshow(cam)
            axs[col].axis("off")
            axs[col].set_title(
                f"{name}\nP={prob:.3f}",
                fontsize=10
            )

            col += 1

        # --------------------------------------------
        # Probabilité finale du stacking
        # --------------------------------------------

        final_prob = predict_stacking_probability(image, models, meta_model)
        pred = int(final_prob >= threshold)
        pred_label = "malignant" if pred == 1 else "benign"

        true_label = infer_true_label(image_path)

        axs[4].text(
            0.5,
            0.65,
            f"Stack\n{final_prob:.3f}",
            ha="center",
            fontsize=12
        )

        axs[4].text(
            0.5,
            0.38,
            f"True: {true_label}",
            ha="center",
            fontsize=11,
            color="black",
            weight="bold"
        )

        axs[4].text(
            0.5,
            0.18,
            f"Pred: {pred_label}",
            ha="center",
            fontsize=11,
            color="red",
            weight="bold"
        )

        axs[4].axis("off")

        plt.suptitle(
            os.path.basename(image_path),
            fontsize=13
        )

        filename = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        for ext in ("png", "pdf", "svg"):

            plt.savefig(
                os.path.join(
                    OUTPUT_DIR,
                    f"{filename}_gradcam.{ext}"
                ),
                dpi=300,
                bbox_inches="tight"
            )

        plt.close()

    print("\nToutes les figures ont été sauvegardées.")


if __name__=="__main__":
    run()
