# =========================================
# IMPORTS
# =========================================
import os
# Force-disable CUDA and reduce TF verbosity before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

# Now safe to import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.metrics import ConfusionMatrixDisplay

# IPython detection for notebook environments
try:
    from IPython import get_ipython
    from IPython.display import display
    if get_ipython() is None:
        display = None
except Exception:
    get_ipython = None
    display = None

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_INDEX = 43

def clear_gpu_memory():
    """Clear TensorFlow GPU memory to prevent OOM errors."""
    try:
        # Clear any existing TensorFlow resources
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        pass  # Silently ignore if GPU operations fail


# Create output directory for plots
PLOTS_OUTPUT_DIR = PROJECT_DIR / "plots_output"
PLOTS_OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_COUNTER = 0
CURRENT_MODEL_NAME = None


def show_plot(filename=None):
    """Display and save the current matplotlib figure.
    Saves to PNG file under a model-named subfolder inside plots_output.
    """
    global PLOT_COUNTER

    try:
        if CURRENT_MODEL_NAME:
            safe_model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in CURRENT_MODEL_NAME)
            model_output_dir = PLOTS_OUTPUT_DIR / safe_model_name
        else:
            model_output_dir = PLOTS_OUTPUT_DIR
        model_output_dir.mkdir(exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            PLOT_COUNTER += 1
            if CURRENT_MODEL_NAME:
                filename = f"{safe_model_name}_{PLOT_COUNTER:03d}.png"
            else:
                filename = f"plot_{PLOT_COUNTER:03d}.png"

        output_path = model_output_dir / filename

        # Save the figure
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"  📊 Saved plot: {output_path}")
        
    except Exception as e:
        print(f"  ⚠️ Could not save plot: {e}")
    finally:
        try:
            plt.close('all')
        except:
            pass

# =========================================
# CONFIG (SAFE)
# =========================================
from modules.Config import (
    patience, min_delta, epochs, learning_rate, weight_decay,path_project,batch_size,threshold,save_model_path,
    input_shape, num_classes, train_dir, test_dir, PLOTS_DIR
)



# =========================================
# 1. LOAD PIPELINE
# =========================================
def list_available_models(models_dir=None):
    if models_dir is None:
        models_dir = PROJECT_DIR / "Models"

    return sorted([p for p in Path(models_dir).rglob("*.pkl") if p.is_file()])


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
        if candidate.exists():
            return candidate

    raise ValueError("Invalid selection. Please choose a valid number or path.")


def select_model_path(models_dir=None, default_model=None, choice=None, model_choice=None):
    model_files = list_available_models(models_dir)
    if not model_files:
        raise FileNotFoundError("No .pkl model files were found under the Models folder.")

    if default_model is None:
        if len(model_files) > DEFAULT_MODEL_INDEX:
            default_model = model_files[DEFAULT_MODEL_INDEX]
        else:
            default_model = model_files[-1]

    if choice is None:
        print("\n📊 Plotting mode:")
        print("1. Display all metrics for the default model (model number 44)")
        print("2. Choose a model from the list and display all metrics")
        choice = input("Choose an option (1-2): ").strip() or "1"

    if choice == "1":
        return default_model

    if choice == "2":
        # Group models by parent directory
        grouped_models = {}
        seq_idx = 1

        for model_file in model_files:
            try:
                group_name = model_file.parent.name
            except Exception:
                group_name = "Other"

            if group_name not in grouped_models:
                grouped_models[group_name] = []

            grouped_models[group_name].append(model_file)
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

        if model_choice is None:
            model_choice = input("\nSelect a model by number or enter a full path: ").strip()

        return resolve_model_choice(model_files, model_choice)

    raise ValueError("Invalid plotting mode selection. Please choose 1 or 2.")


def load_pipeline(pkl_path):
    """
    Load saved stacking pipeline (.pkl)
    """
    print("\n🔄 Clearing GPU memory before loading model...")
    clear_gpu_memory()
    
    try:
        print("📦 Loading model from disk...")
        
        data = joblib.load(pkl_path)
        
        print("\n✅ Pipeline loaded successfully!")
        print(f"🔥 DEBUG - Top level keys: {list(data.keys())}")
        return {
            "ensemble_models": data.get("ensemble_models"),
            "meta_learner": data.get("meta_learner"),
            "scaler": data.get("scaler"),
            "metrics": data.get("metrics"),
            "threshold": data.get("threshold"),
            "model_configs": data.get("model_configs"),
            "histories": data.get("histories", {}),
            "base_model_metrics": data.get("base_model_metrics", {}),
            "base_model_predictions": data.get("base_model_predictions", {}),
            "y_true": data.get("y_true"),
            "y_pred": data.get("y_pred")
        }
    except Exception as e:
        if "out of memory" in str(e).lower() or "cudaSetDevice" in str(e):
            print("\n❌ ERROR: GPU out of memory!")
            print("💡 Suggestions:")
            print("   1. Close other GPU applications")
            print("   2. Try a different model from the list")
            print("   3. Restart the application to clear GPU memory")
            sys.exit(1)
        else:
            raise


# =========================================
# 2. DISPLAY METRICS
# =========================================
def show_metrics(metrics_dict):
    """
    Print formatted metrics
    """
    if metrics_dict is None:
        print("\n⚠️  WARNING: No metrics found in pipeline!")
        return
    
    print("\n" + "="*60)
    print("📊 FINAL MODEL METRICS")
    print("="*60)

    if isinstance(metrics_dict, dict):
        for k, v in metrics_dict.items():
            try:
                print(f"{k.upper():30s}: {float(v):.4f}")
            except:
                print(f"{k.upper():30s}: {v}")
    else:
        print(f"⚠️  Metrics is not a dict: {type(metrics_dict)}")
        print(f"   Value: {metrics_dict}")

    print("="*60)


# =========================================
# 3. BAR PLOT (NEW 🔥)
# =========================================
def plot_metrics_bar(metrics_dict):
    """
    Clean bar plot with colors (robust and simple)
    """
    from matplotlib import colormaps

    if not metrics_dict:
        print("⚠️ No metrics to plot")
        return

    names = []
    values = []

    for k, v in metrics_dict.items():
        try:
            val = float(v)  # 🔥 force conversion
            names.append(k)
            values.append(val)
        except Exception:
            print(f"⚠️ Skipping {k}: {v}")

    if not values:
        print("❌ No valid numeric metrics!")
        return

    # Normalize values for colormap (0 to 1 range)
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        normalized = [0.5] * len(values)
    else:
        normalized = [(v - min_val) / (max_val - min_val) for v in values]

    # Create colormap
    cmap = colormaps['viridis']
    colors = [cmap(norm_val) for norm_val in normalized]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)

    # Labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{val:.2f}%",
                 ha='center', va='bottom', fontweight='bold')

    plt.title("Model Performance Metrics", fontsize=14, fontweight='bold')
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score (%)", fontsize=12)
    plt.xticks(rotation=30)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    show_plot()

def plot_base_model_metrics_bar(base_model_metrics):
    """Plot one bar chart per base CNN using its saved metrics."""

    if not base_model_metrics:
        print("⚠️ No base-model metrics to plot")
        return

    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC"]

    for model_name, metrics_dict in base_model_metrics.items():
        values = []
        labels = []

        for metric_name in metric_names:
            metric_value = metrics_dict.get(metric_name)
            if metric_value is None:
                continue
            labels.append(metric_name)
            values.append(float(metric_value))

        if not values:
            print(f"⚠️ No numeric metrics found for {model_name}")
            continue

        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(values)))

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.0)

        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}%",
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        plt.title(f"Base CNN Metrics - {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Score (%)", fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        show_plot()


# =========================================
# 3. PLOT HISTORY
# =========================================
def plot_history(histories, max_plots=None):
    """
    Plot all training histories stored in dictionary
    """

    if not histories:
        print("⚠️ No histories found!")
        return

    count = 0

    for name, history in histories.items():

        if max_plots and count >= max_plots:
            break

        print(f"📈 Plotting: {name}")

        plt.figure(figsize=(12, 4))

        # =========================
        # ACCURACY
        # =========================
        plt.subplot(1, 2, 1)

        if "accuracy" in history:
            plt.plot(history["accuracy"], label="train accuracy")
        if "val_accuracy" in history:
            plt.plot(history["val_accuracy"], label="val accuracy")

        plt.title(f"{name} - Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # =========================
        # LOSS
        # =========================
        plt.subplot(1, 2, 2)

        if "loss" in history:
            plt.plot(history["loss"], label="train loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="val loss")

        plt.title(f"{name} - Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        show_plot()

        count += 1

# =========================================
# 5. CONFUSION MATRIX
# =========================================
def plot_confusion_matrix(y_true, y_pred, normalize=False, title="Confusion Matrix"):
    # Compute raw counts
    cm_counts = confusion_matrix(y_true, y_pred)

    # Percent of total for color scaling and annotation
    total = cm_counts.sum()
    if total == 0:
        cm_percent = cm_counts.astype(float)
    else:
        cm_percent = cm_counts.astype(float) / float(total)

    # If row-normalized view requested, compute row percentages for display header
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_row_norm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)
            cm_row_norm = np.nan_to_num(cm_row_norm)
    else:
        cm_row_norm = None

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Proportion of total')

    # Axis labels (binary default to Neg/Pos)
    classes = ["FALSE", "TRUE"] if cm_counts.shape == (2, 2) else [str(i) for i in range(cm_counts.shape[0])]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate cells with the total percentage and count
    thresh = cm_percent.max() / 2.0 if cm_percent.size else 0.5
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = int(cm_counts[i, j])
            pct_total = cm_percent[i, j]
            text = f"{pct_total:.2%}\n{count}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm_percent[i, j] > thresh else "black",
                    fontsize=10)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title + (" (row-normalized shown)" if normalize else ""))
    ax.grid(False)

    plt.tight_layout()
    show_plot()


def show_base_model_metrics(base_model_metrics):
    """Print metrics for each CNN saved in the pipeline."""
    if not base_model_metrics:
        print("\n⚠️ No per-CNN metrics found in pipeline!")
        return

    print("\n" + "=" * 60)
    print("📊 BASE CNN METRICS")
    print("=" * 60)
    for model_name, metrics_dict in base_model_metrics.items():
        print(f"\n[{model_name}]")
        for metric_name, metric_value in metrics_dict.items():
            try:
                print(f"{metric_name.upper():30s}: {float(metric_value):.4f}")
            except Exception:
                print(f"{metric_name.upper():30s}: {metric_value}")
    print("=" * 60)


def show_base_model_confusion_matrices(pipeline, normalize=True):
    """Plot one confusion matrix per saved CNN using its test predictions."""
    y_true = pipeline.get("y_true")
    base_model_predictions = pipeline.get("base_model_predictions", {})
    threshold_value = pipeline.get("threshold", 0.5)

    if y_true is None or not base_model_predictions:
        print("\n⚠️ No base-model predictions or y_true found in pipeline!")
        return

    for model_name, predictions in base_model_predictions.items():
        y_prob = predictions.get("test")
        if y_prob is None:
            print(f"⚠️ No test predictions found for {model_name}")
            continue

        y_pred = (np.asarray(y_prob) >= threshold_value).astype(int)
        title_prefix = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
        plot_confusion_matrix(
            y_true,
            y_pred,
            normalize=normalize,
            title=f"{title_prefix} (%) - {model_name}"
        )


# =========================================
# 4. MAIN EXECUTION
# =========================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate performance plots for a saved stacking model artifact")
    parser.add_argument("--mode", choices=["default", "select"], default=None,
                        help="Use the default model artifact or prompt for a selected one")
    parser.add_argument("--model-choice", default=None,
                        help="Model number or path to use when --mode select is chosen")
    args = parser.parse_args()
    
    try:
        if args.mode == "default":
            pkl_path = select_model_path(PROJECT_DIR / "Models", choice="1")
        elif args.mode == "select":
            pkl_path = select_model_path(PROJECT_DIR / "Models", choice="2", model_choice=args.model_choice)
        else:
            pkl_path = select_model_path(PROJECT_DIR / "Models")
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    
    #pkl_path = "/home/akachat/tf_env/Stacking_CNN_Melanoma/Models/Sens95_XGBoost_E40_T0.5/melanoma_stacking_safe_lr3e05_dw1e05.pkl"
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        exit(1)

    # Set current model name for plot filenames
    CURRENT_MODEL_NAME = pkl_path.stem

    print(f"\n🎯 Using model artifact: {pkl_path}")

    try:
        # Load pipeline
        pipeline = load_pipeline(str(pkl_path))

        threshold_value = pipeline.get("threshold")
        if threshold_value is None:
            threshold_value = 0.5

        print(f"\n🎯 Meta-learner threshold: {float(threshold_value):.4f}")

        print("\n🔥 DEBUG metrics content:")
        for k, v in pipeline["metrics"].items():
            print(k, type(v), v)
        
        #print(f"\n🔥 DEBUG - Pipeline keys: {list(pipeline.keys())}")
        #print(f"🔥 DEBUG - Metrics value: {pipeline.get('metrics')}")
        #print(f"🔥 DEBUG - Histories count: {len(pipeline.get('histories', {}))}")

        # Show metrics
        show_metrics(pipeline["metrics"])

    # 🔥 NEW BAR PLOT
        plot_metrics_bar(pipeline["metrics"])

        # Per-CNN metrics saved during training
        show_base_model_metrics(pipeline.get("base_model_metrics", {}))

        print("\n📊 Confusion matrices for each model")
        show_base_model_confusion_matrices(pipeline, normalize=True)

        print("\n📊 Bar chart for each base CNN")
        plot_base_model_metrics_bar(pipeline.get("base_model_metrics", {}))

        # Plot histories
        plot_history(pipeline["histories"])

        if pipeline["y_true"] is not None and pipeline["y_pred"] is not None:

            print("\n📊 Confusion Matrix")
            plot_confusion_matrix(
                pipeline["y_true"],
                pipeline["y_pred"],
                normalize=True,
                title="Confusion Matrix (%) - Final Stacked Model"
            )

            print("\n📊 Normalized Confusion Matrix")
            plot_confusion_matrix(
                pipeline["y_true"],
                pipeline["y_pred"],
                normalize=True,
                title="Normalized Confusion Matrix (%) - Final Stacked Model"
            )
        else:
            print("\n⚠️ No y_true/y_pred found in pipeline.")
            print("👉 Save them during training if needed.")
    finally:
        print("\n🧹 Clearing GPU memory after visualization...")
        clear_gpu_memory()