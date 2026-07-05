#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf

from modules.Config import img_size

# Make project root importable
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

# =========================================
# GPU MEMORY CONFIGURATION
# =========================================
try:
    # Enable GPU memory growth to prevent OOM on large models
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass  # Continue if GPU is not available


def clear_gpu_memory():
    """Clear TensorFlow GPU memory to prevent OOM errors."""
    try:
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.reset_memory_stats(gpu)
    except Exception:
        pass  # Silently ignore if GPU operations fail


def load_model_artifact(model_path: str):
    """Load model artifact with GPU memory management."""
    clear_gpu_memory()
    
    # Temporarily disable GPU to load model safely on CPU
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    try:
        return joblib.load(model_path)
    finally:
        # Restore GPU visibility
        if old_cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


DEFAULT_MODEL_INDEX = 43


def list_available_models(models_dir: Path):
    return sorted([p for p in models_dir.rglob("*.pkl") if p.is_file()])


def resolve_model_choice(model_files, choice):
    if not choice:
        raise ValueError("No model selection provided.")

    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(model_files):
            return model_files[index]
    else:
        candidate = Path(choice)
        if not candidate.is_absolute():
            candidate = (PROJECT_DIR / choice).resolve()

        if candidate.exists() and candidate.suffix.lower() == ".pkl":
            candidate = candidate.resolve()
            for model_file in model_files:
                if model_file.resolve() == candidate:
                    return model_file

            if candidate.name.endswith(".pkl"):
                for model_file in model_files:
                    if model_file.name == candidate.name:
                        return model_file

    raise ValueError("Invalid selection. Please choose a valid model number or a .pkl model path.")


def choose_model(models_dir: Path, selection=None):
    model_files = list_available_models(models_dir)
    if not model_files:
        raise FileNotFoundError("No .pkl model files were found under the Models folder.")

    # Group models by parent directory
    grouped_models = {}
    idx_map = {}  # Map from sequential index to model file
    seq_idx = 1

    for model_file in model_files:
        try:
            group_name = model_file.parent.name
        except Exception:
            group_name = "Other"

        if group_name not in grouped_models:
            grouped_models[group_name] = []

        grouped_models[group_name].append(model_file)
        idx_map[seq_idx] = model_file
        seq_idx += 1

    print("\n📊 Available models by group:")
    seq_idx = 1
    for group_name in sorted(grouped_models.keys()):
        print(f"\n[{group_name}]")
        for model_file in grouped_models[group_name]:
            try:
                display_path = model_file.relative_to(PROJECT_DIR)
            except ValueError:
                display_path = model_file
            print(f"  {seq_idx}. {display_path.name}")
            seq_idx += 1

    if selection is not None:
        return resolve_model_choice(model_files, str(selection))

    while True:
        print("\nSelect a model by number or enter a full path: ", end="")
        sys.stdout.flush()
        choice = input().strip()
        if not choice:
            print("No model selected.")
            continue

        try:
            return resolve_model_choice(model_files, choice)
        except ValueError as exc:
            print(exc)


def select_model_path(models_dir: Path, default_model=None, choice=None, model_choice=None):
    model_files = list_available_models(models_dir)
    if not model_files:
        raise FileNotFoundError("No .pkl model files were found under the Models folder.")

    if default_model is None:
        if len(model_files) > DEFAULT_MODEL_INDEX:
            default_model = model_files[DEFAULT_MODEL_INDEX]
        else:
            default_model = model_files[-1]

    if choice is None:
        print("\nPrediction mode:")
        print("1. Use the default model (model melanoma_stacking_safe_lr3e05_dw1e05)")
        print("2. Selected Model from the list")
        print("0. Return to the main menu")
        print("Choose an option (0-2): ", end="")
        choice = input().strip() or "1"

    if choice == "1":
        return default_model

    if choice == "2":
        print("\nSelect a model by number or enter a full path:", end="")
        return choose_model(models_dir, selection=model_choice)

    if choice == "0":
        print("\nReturning to the main menu.")
        raise SystemExit(0)

    raise ValueError("Invalid prediction mode selection. Please choose 0, 1, or 2.")


MODEL_PREPROCESSORS = {
    "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "DenseNet169": tf.keras.applications.densenet.preprocess_input,
}


def get_preprocess_fn(model_name):
    if model_name not in MODEL_PREPROCESSORS:
        raise ValueError(f"No preprocessing function found for model '{model_name}'")
    return MODEL_PREPROCESSORS[model_name]


def preprocess_image(image_path: Path, size=None, preprocess_fn=None):
    if size is None:
        size = img_size

    img_bytes = tf.io.read_file(str(image_path))
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32)

    if preprocess_fn is not None:
        img = preprocess_fn(img)

    return tf.expand_dims(img, axis=0)


def resolve_threshold(artifact, threshold=None):
    if threshold is not None:
        return float(threshold)

    artifact_threshold = artifact.get("threshold") if isinstance(artifact, dict) else None
    if artifact_threshold is not None:
        return float(artifact_threshold)

    return 0.5


def resolve_meta_learner(artifact):
    if not isinstance(artifact, dict):
        return None

    for key in ("meta_linear", "meta_learner"):
        learner = artifact.get(key)
        if learner is not None:
            return learner

    return None


def predict_image(image_path, model_path=None, threshold=None):
    if model_path is None:
        model_path = PROJECT_DIR / "Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e05_dw1e05.pkl"

    artifact = load_model_artifact(str(model_path))
    ensemble_models = artifact.get("ensemble_models", [])
    meta_learner = resolve_meta_learner(artifact)

    if not ensemble_models or meta_learner is None:
        raise ValueError("The saved stacking model artifact is incomplete.")

    preprocessors = [get_preprocess_fn(model_name) for model_name in ["EfficientNetV2B0", "MobileNetV2", "DenseNet169"][: len(ensemble_models)]]

    features = []
    for model, prep in zip(ensemble_models, preprocessors):
        x = preprocess_image(Path(image_path), size=img_size, preprocess_fn=prep)
        pred = model.predict(x, verbose=0)
        features.append(pred.reshape(-1))

    X = np.column_stack(features)
    prob_malignant = float(meta_learner.predict_proba(X)[0, 1])
    prob_benign = 1.0 - prob_malignant

    resolved_threshold = resolve_threshold(artifact, threshold)
    if prob_malignant >= resolved_threshold:
        label = "malignant"
    else:
        label = "benign"

    return {
        "label": label,
        "probability_malignant": prob_malignant * 100,
        "probability_benign": prob_benign * 100,
        "threshold": resolved_threshold,
    }


def get_image_path():
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if not candidate.is_absolute():
            candidate = (PROJECT_DIR / candidate).resolve()
        if candidate.exists():
            return candidate
        print("Image file not found. Please enter a valid path.")

    while True:
        print("\nEnter the image path (for example: data/test/malignant/melanoma_10128.jpg): ", end="")
        sys.stdout.flush()
        image_path = input().strip()
        if not image_path:
            print("No image path provided. Please try again.")
            continue

        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = (PROJECT_DIR / candidate).resolve()

        if candidate.exists():
            return candidate

        print("Image file not found. Please try again.")


if __name__ == "__main__":
    model_path = select_model_path(PROJECT_DIR / "Models")
    image_path = get_image_path()

    print(f"\nUsing model: {model_path.relative_to(PROJECT_DIR)}")
    result = predict_image(image_path, model_path=model_path)

    print("\n=== Prediction Result ===")
    print(f"Class: {result['label']}")
    print(f"Benign probability: {result['probability_benign']:.2f}%")
    print(f"Malignant probability: {result['probability_malignant']:.2f}%")
    print(f"Threshold: {result['threshold']}")
