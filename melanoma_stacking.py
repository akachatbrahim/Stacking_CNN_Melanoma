# =========================================
# IMPORTS
# =========================================
import os
import gc
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import random

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_score, recall_score,
    f1_score, accuracy_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# =========================================
# CONFIG
# =========================================
from modules.Config import (
    patience, min_delta, epochs, learning_rate, weight_decay,
    path_project, batch_size, threshold, save_model_path,
    input_shape, num_classes, train_dir, test_dir, seed, youden_index_or_sens_sup_95
)

tf.keras.utils.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["PYTHONHASHSEED"] = str(seed)

from modules.Dataset import load_dataset
from modules.Model import model_configs

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# =========================================
# HYPERPARAMETERS (tunable)
# =========================================
UNFREEZE_LAYERS   = 10        # nombre de couches dégelées depuis la fin (-1 = tout)
DENSE_UNITS_1     = 512        # neurones première couche dense
DENSE_UNITS_2     = 128        # neurones deuxième couche dense
DROPOUT_1         = 0.5
DROPOUT_2         = 0.4
FOCAL_GAMMA       = 2.0        # Focal Loss gamma (0 = CrossEntropy classique)
LABEL_SMOOTHING   = 0.02
META_N_SPLITS     = 5          # folds pour OOF stacking

# =========================================
# XGBOOST (optionnel — commentez si absent)
# =========================================

try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("✅ XGBoost disponible — utilisé comme meta-learner")
except ImportError:
    USE_XGBOOST = False
    print("⚠️  XGBoost absent — fallback LogisticRegression")

# =========================================
# SAVE / LOAD PIPELINE
# =========================================
def save_pipeline(save_dir, name, ensemble_models, meta_learner, scaler,
                  metrics, threshold, model_configs, histories,
                  base_model_metrics=None, base_model_predictions=None,
                  y_true=None, y_pred=None, y_val=None, val_stack=None, test_stack=None):
    os.makedirs(save_dir, exist_ok=True)

    if metrics is None:
        metrics = {}
    if isinstance(metrics, dict):
        metrics = {k: v for k, v in metrics.items() if k not in ["y_true", "y_pred"]}

    pipeline_data = {
        "ensemble_models":        ensemble_models,
        "meta_learner":           meta_learner,
        "scaler":                 scaler,
        "metrics":                metrics,
        "threshold":              float(threshold) if threshold is not None else None,
        "model_configs":          model_configs,
        "histories":              histories,
        "base_model_metrics":     base_model_metrics,
        "base_model_predictions": base_model_predictions,
        "y_true":                 y_true,
        "y_pred":                 y_pred,
        "y_val":                  y_val,
        "val_stack":              val_stack,
        "test_stack":             test_stack,    
    }

    file_path = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(pipeline_data, file_path)

    print("\n✅ PIPELINE SAUVEGARDÉ")
    print(f"📦 Chemin : {file_path}")
    print(f"📊 Métriques : {list(metrics.keys()) if isinstance(metrics, dict) else None}")
    print(f"🧠 Modèles  : {len(ensemble_models) if ensemble_models else 0}")


def load_pipeline_results(file_path):
    return joblib.load(file_path)


def print_saved_base_model_results(file_path):
    pipeline_data = load_pipeline_results(file_path)
    base_model_metrics = pipeline_data.get("base_model_metrics", {})
    if not base_model_metrics:
        print("\nAucune métrique par CNN trouvée.")
        return
    for model_name, metrics in base_model_metrics.items():
        print_binary_metrics(model_name, metrics)

# =========================================
# METRICS
# =========================================
def compute_binary_metrics(y_true, y_prob, threshold_value=threshold):
    y_pred = (y_prob >= threshold_value).astype(int)
    cm = confusion_matrix(y_true,y_pred,labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy":    round(accuracy_score(y_true, y_pred) * 100, 2),
        "Sensitivity": round(recall_score(y_true, y_pred) * 100, 2),
        "Specificity": round((tn / (tn + fp) if (tn+fp) > 0 else 0) * 100, 2),
        "Precision":   round(precision_score(y_true, y_pred) * 100, 2),
        "F1":          round(f1_score(y_true, y_pred) * 100, 2),
        "AUC":         round(roc_auc_score(y_true, y_prob) * 100, 2),
    }


def print_binary_metrics(model_name, metrics):
    print(f"\n===== {model_name} =====")
    for k, v in metrics.items():
        print(f"{k:<14}: {v:.4f}")

# =========================================
# MEMORY
# =========================================
def clean():
    tf.keras.backend.clear_session()
    gc.collect()

# =========================================
# LOAD DATA
# =========================================
train_fit_ds, val_ds, test_ds = load_dataset(
    train_dir, test_dir,
    augment=False,
    deeplearning=False,
    preprocess_fn=None
)

train_fit_ds = train_fit_ds.cache()
val_ds = val_ds.cache()
test_ds = test_ds.cache()

def dataset_to_labels(ds):
    labels = []
    for _, batch_labels in ds:
        labels.append(batch_labels.numpy().ravel())
    return np.concatenate(labels).astype(int)

def preprocess_dataset(ds, preprocess_fn):
    return ds.map(
        lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def make_indexed_dataset(ds):
    return ds.unbatch().enumerate()


def make_fold_dataset(indexed_ds, selected_indices, preprocess_fn, shuffle=False):
    keys = tf.constant(selected_indices, dtype=tf.int64)
    values = tf.ones_like(keys, dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0,
    )

    ds = indexed_ds.filter(lambda idx, data: table.lookup(idx) > 0)
    ds = ds.map(lambda idx, data: data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (preprocess_fn(tf.cast(x, tf.float32)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=max(1, len(selected_indices)),seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True
        )
    ]

print("Chargement des labels...")
y       = dataset_to_labels(train_fit_ds)
y_val   = dataset_to_labels(val_ds)
y_test  = dataset_to_labels(test_ds)

# =========================================
# CLASS WEIGHTS
# =========================================
classes      = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = {
    cls: weight
    for cls, weight in zip(classes, class_weights)
}
#class_weights = dict(enumerate(class_weights))
print(f"\n🔥 CLASS WEIGHTS : {class_weights}")

# =========================================
# MODEL BUILDER — amélioré
# =========================================
def build_model(base_model_fn):
    base = base_model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # ── Dégel ──────────────────────────────────────────────
    if UNFREEZE_LAYERS == -1:
        # Dégeler tout le backbone
        for layer in base.layers:
            layer.trainable = True
    else:
        # Geler les couches avant UNFREEZE_LAYERS, dégeler le reste
        for layer in base.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False
        for layer in base.layers[-UNFREEZE_LAYERS:]:
            layer.trainable = True

    # ── Head ───────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(DENSE_UNITS_1, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_1)(x)
    x = tf.keras.layers.Dense(DENSE_UNITS_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_2)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base.input, outputs=out)

    # ── LR différentiel : backbone lent, head rapide ───────
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # ── Focal Loss + label smoothing ───────────────────────
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        from_logits=False
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# =========================================
# TRAIN LOOP — OOF stacking
# =========================================
train_preds_list = []
val_preds_list   = []
test_preds_list  = []
histories        = {}
all_models       = []
base_model_metrics     = {}
base_model_predictions = {}
indexed_train_fit_ds   = make_indexed_dataset(train_fit_ds)
skf = StratifiedKFold(n_splits=META_N_SPLITS, shuffle=True, random_state=seed)
sample_indices = np.arange(len(y))

for name, model_fn, preprocess_fn in model_configs:

    print(f"\n{'='*50}")
    print(f"🔥 Entraînement : {name}")
    print(f"{'='*50}")
    clean()

    oof_train_preds = np.zeros(len(y), dtype=np.float32)

    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(sample_indices, y), start=1):
        print(f"  → Fold {fold_idx}/{META_N_SPLITS}")
        clean()

        fold_train_ds = make_fold_dataset(indexed_train_fit_ds, train_idx, preprocess_fn, shuffle=True)
        fold_holdout_ds = make_fold_dataset(indexed_train_fit_ds, holdout_idx, preprocess_fn, shuffle=False)

        fold_model = build_model(model_fn)
        fold_model.fit(
            fold_train_ds,
            validation_data=fold_holdout_ds,
            callbacks=get_callbacks(),
            epochs=epochs,
            class_weight=class_weights,
            verbose=1
        )

        oof_train_preds[holdout_idx] = fold_model.predict(fold_holdout_ds, verbose=0).ravel()
        del fold_model, fold_train_ds, fold_holdout_ds
        clean()

    # ── Modèle final entraîné sur tout le jeu d'entraînement ─────────
    train_fit_ds_p  = preprocess_dataset(train_fit_ds,  preprocess_fn)
    val_ds_p        = preprocess_dataset(val_ds,        preprocess_fn)
    test_ds_p       = preprocess_dataset(test_ds,       preprocess_fn)

    model = build_model(model_fn)
    history = model.fit(
        train_fit_ds_p,
        validation_data=val_ds_p,
        callbacks=get_callbacks(),
        epochs=epochs,
        class_weight=class_weights,
        verbose=1
    )

    val_preds  = model.predict(val_ds_p,  verbose=0).ravel()
    test_preds = model.predict(test_ds_p, verbose=0).ravel()

    metrics = compute_binary_metrics(y_test, test_preds, threshold_value=threshold)
    base_model_metrics[name]     = metrics
    base_model_predictions[name] = {
        "train": oof_train_preds,
        "val":   val_preds,
        "test":  test_preds,
    }
    print_binary_metrics(name, metrics)

    #print(np.min(oof_train_preds), np.max(oof_train_preds))

    train_preds_list.append(oof_train_preds)
    val_preds_list.append(val_preds)
    test_preds_list.append(test_preds)
    all_models.append((model, preprocess_fn))
    histories[name] = history.history

    del train_fit_ds_p, val_ds_p, test_ds_p
    clean()

# =========================================
# STACKING — OOF propres + meta-learner
# =========================================
print("\n🔥 Construction du meta-learner (OOF stacking)...")

train_stack = np.column_stack(train_preds_list)
val_stack   = np.column_stack(val_preds_list)
test_stack  = np.column_stack(test_preds_list)

# ── Meta-learner ─────────────────────────────

if USE_XGBOOST:
    pos_weight = float(class_weights[0] / class_weights[1])
    meta_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        verbosity=0,
        early_stopping_rounds=50,
    )
    meta_model.fit(
        train_stack, y,
        eval_set=[(val_stack, y_val)],
        verbose=False
    )
else:
    base_meta  = LogisticRegression(max_iter=3000,random_state=seed)
    from sklearn.calibration import CalibratedClassifierCV
    meta_model = CalibratedClassifierCV(estimator=base_meta,method='sigmoid',cv=5)
    meta_model.fit(train_stack, y)

# ── Prédictions finales ──────────────────────────────
val_final_preds = meta_model.predict_proba(val_stack)[:, 1]
final_preds     = meta_model.predict_proba(test_stack)[:, 1]

# =========================================
# THRESHOLD (YOUden INDEX - ROBUST)
# =========================================
from sklearn.metrics import roc_curve
import numpy as np

if (youden_index_or_sens_sup_95 == "youden_index"):
    fpr, tpr, thresholds = roc_curve(y_val, val_final_preds)
    youden_scores = tpr - fpr
    youden_idx = np.argmax(youden_scores)
    best_thresh = thresholds[youden_idx]

    print(f"\n🔥 Meilleur seuil (Youden Index) : {best_thresh:.4f}")
    print(f"🔥 Validation Sensibilité : {tpr[youden_idx] * 100:.2f}%")
    print(f"🔥 Validation Spécificité : {100 - fpr[youden_idx] * 100:.2f}%")

elif (youden_index_or_sens_sup_95 == "sens_sup_95"):

    fpr, tpr, thresholds = roc_curve(y_val, val_final_preds)

    valid_idx = np.where(tpr >= 0.95)[0]

    if len(valid_idx) > 0:

        best_idx = valid_idx[np.argmin(fpr[valid_idx])]

    else:
        best_idx = np.argmax(tpr - fpr)

    best_thresh = thresholds[best_idx]

    print(f"\n🔥 Meilleur seuil : {best_thresh:.4f}")
    print(f"🔥 Validation Sensibilité : {tpr[best_idx] * 100:.2f}%")
    print(f"🔥 Validation Spécificité : {100 - fpr[best_idx] * 100:.2f}%")

# =========================================
# MÉTRIQUES FINALES
# =========================================
metrics_dict = compute_binary_metrics(y_test, final_preds, threshold_value=best_thresh)

print_binary_metrics("RÉSULTATS FINAUX", metrics_dict)

y_pred = (final_preds >= best_thresh).astype(int)

# =========================================
# SAUVEGARDE
# =========================================
ensemble_models = [model for model, _ in all_models]
lr_exp = f"{learning_rate:.0e}".replace("-", "")
dw_exp = f"{weight_decay:.0e}".replace("-", "")
name_model = f"melanoma_stacking_safe_lr{lr_exp}_dw{dw_exp}"
#name_model = f"{model_configs[0][0]}_{model_configs[1][0]}_{model_configs[2][0]}_lr{lr_exp}_dw{dw_exp}"

# =========================================
# COURBE ROC
# =========================================
fpr, tpr, _ = roc_curve(y_test, final_preds)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {metrics_dict['AUC']:.2f}%")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC — Stacking final")
plt.legend()
roc_path = os.path.join(save_model_path, name_model + "_roc_curve.png")
#os.makedirs(save_model_path, exist_ok=True)
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n📈 Courbe ROC sauvegardée : {roc_path}")

save_pipeline(
    save_dir=save_model_path,
    name=name_model,
    ensemble_models=ensemble_models,
    meta_learner=meta_model,
    scaler=None,
    metrics=metrics_dict,
    threshold=best_thresh,
    model_configs=model_configs,
    histories=histories,
    base_model_metrics=base_model_metrics,
    base_model_predictions=base_model_predictions,
    y_true=y_test,
    y_pred=y_pred,
    y_val=y_val,
    val_stack=val_stack,
    test_stack=test_stack
)

print("\n✅ Pipeline terminé")