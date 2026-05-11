import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc as sk_auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from .utils import ensure_dir, read_json


@dataclass
class EvaluateConfig:
    seed: int = 42
    dataset_csv: str = "data/processed/learner_grammar_dataset.csv"
    meta_json: str = "data/processed/learner_grammar_dataset_meta.json"
    model_dir: str = "models/classic"
    results_dir: str = "outputs/results"
    figures_dir: str = "outputs/figures"
    test_size: float = 0.2


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_dataset(cfg: EvaluateConfig) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(cfg.dataset_csv)
    meta = read_json(cfg.meta_json)
    return df, meta


def _class_labels(meta: Dict) -> List[str]:
    id2label = meta["id2label"]
    return [id2label[str(i)] for i in range(len(id2label))]


# ── individual plots ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: str, title: str
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=40, cmap="YlGn", values_format="d", ax=plt.gca())
    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    out_path: str,
    model_name: str,
) -> float:
    """One-vs-Rest ROC curves for all classes. Returns macro AUC."""
    ensure_dir(os.path.dirname(out_path) or ".")
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    colors = plt.cm.tab10.colors

    plt.figure(figsize=(9, 7))
    macro_auc_list = []
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = sk_auc(fpr, tpr)
        macro_auc_list.append(roc_auc)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random classifier")
    macro_auc = float(np.mean(macro_auc_list))
    plt.title(f"ROC Curves — {model_name}\nMacro AUC = {macro_auc:.3f}",
              fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return macro_auc


def _plot_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    out_path: str,
    model_name: str,
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    colors = plt.cm.tab10.colors

    plt.figure(figsize=(9, 7))
    for i, label in enumerate(labels):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        plt.plot(rec, prec, color=colors[i % len(colors)], lw=2,
                 label=f"{label} (AP = {ap:.2f})")

    plt.title(f"Precision-Recall Curves — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_learning_curve(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    model_name: str,
    seed: int,
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        n_jobs=None,
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mean, "o-", color="#58cc02", label="Training accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#58cc02")
    plt.plot(train_sizes, val_mean, "s--", color="#1CB0F6", label="Validation accuracy")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#1CB0F6")
    plt.title(f"Learning Curve — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_model_comparison(summary: Dict[str, Any], out_path: str) -> None:
    """Bar chart: accuracy, macro-F1, ROC-AUC side-by-side per model."""
    ensure_dir(os.path.dirname(out_path) or ".")
    models = list(summary.keys())
    metrics_names = ["Accuracy", "Macro F1", "ROC AUC (macro)"]
    values = {m: [] for m in models}

    for m in models:
        ms = summary[m]
        acc = ms["test_accuracy"]
        f1s = ms["classification"]["per_class"]["f1"]
        macro_f1 = float(np.mean(f1s))
        roc_auc = ms.get("roc_auc_macro", acc)
        values[m] = [acc, macro_f1, roc_auc]

    x = np.arange(len(metrics_names))
    width = 0.35
    _, ax = plt.subplots(figsize=(9, 6))
    colors = ["#58cc02", "#1CB0F6"]

    for idx, (model, vals) in enumerate(values.items()):
        bars = ax.bar(x + idx * width - width / 2, vals, width, label=model.upper(),
                      color=colors[idx % len(colors)], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — LogReg vs SVM", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_per_class_f1(summary: Dict[str, Any], labels: List[str], out_path: str) -> None:
    """Grouped bar: per-class F1 for each model."""
    ensure_dir(os.path.dirname(out_path) or ".")
    models = list(summary.keys())
    x = np.arange(len(labels))
    width = 0.35
    colors = ["#58cc02", "#1CB0F6"]

    _, ax = plt.subplots(figsize=(11, 6))
    for idx, model in enumerate(models):
        f1s = summary[model]["classification"]["per_class"]["f1"]
        ax.bar(x + idx * width - width / 2, f1s, width, label=model.upper(),
               color=colors[idx % len(colors)], edgecolor="white")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Score — LogReg vs SVM", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── main evaluation ───────────────────────────────────────────────────────────

def run_evaluate(cfg: EvaluateConfig) -> None:
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.figures_dir)

    df, meta = _load_dataset(cfg)
    labels = _class_labels(meta)
    n_classes = len(labels)

    X = df["text"].astype(str).values
    y = df["label_id"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    model_paths = {
        "logreg": os.path.join(cfg.model_dir, "logreg_pipeline.joblib"),
        "svm":    os.path.join(cfg.model_dir, "svm_pipeline.joblib"),
    }

    summary: Dict[str, Any] = {
        "seed": cfg.seed,
        "dataset_rows": int(len(df)),
        "num_classes": n_classes,
        "models": {},
    }

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing trained model: {model_path}. Run: python -m src.classifier"
            )

        print(f"\n--- Evaluating {model_name.upper()} ---")
        pipe: Pipeline = joblib.load(model_path)

        y_pred = pipe.predict(X_test)

        # ── basic metrics ──
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {np.mean(f1):.4f}")

        # ── confusion matrix ──
        cm_path = os.path.join(cfg.figures_dir, f"confusion_matrix_{model_name}.png")
        _plot_confusion_matrix(y_test, y_pred, labels, cm_path,
                               title=f"Confusion Matrix — {model_name.upper()}")
        print(f"  Saved confusion matrix -> {cm_path}")

        # ── score for ROC (proba or decision_function) ──
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)
        else:
            y_score = pipe.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.column_stack([-y_score, y_score])

        # ── ROC curves ──
        roc_path = os.path.join(cfg.figures_dir, f"roc_curves_{model_name}.png")
        macro_auc = _plot_roc_curves(y_test, y_score, labels, roc_path, model_name.upper())
        print(f"  ROC AUC  : {macro_auc:.4f}  -> {roc_path}")

        # ── Precision-Recall curves ──
        pr_path = os.path.join(cfg.figures_dir, f"precision_recall_{model_name}.png")
        _plot_precision_recall(y_test, y_score, labels, pr_path, model_name.upper())

        # ── Learning curve ──
        lc_path = os.path.join(cfg.figures_dir, f"learning_curve_{model_name}.png")
        _plot_learning_curve(pipe, X_train, y_train, lc_path, model_name.upper(), cfg.seed)
        print(f"  Learning curve -> {lc_path}")

        # ── Cross-validation ──
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
        print(f"  CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # ── Grid search ──
        if model_name == "logreg":
            param_grid = {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__solver": ["lbfgs"],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
            }
        else:
            param_grid = {
                "clf__C": [0.5, 1.0, 2.0],
                "tfidf__ngram_range": [(1, 1), (1, 2)],
            }

        grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=cv, verbose=0)
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}  CV={grid.best_score_:.4f}")

        summary["models"][model_name] = {
            "test_accuracy": float(acc),
            "roc_auc_macro": float(macro_auc),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "grid_search": {
                "best_params": grid.best_params_,
                "best_cv_accuracy": float(grid.best_score_),
            },
            "classification": {
                "accuracy": float(acc),
                "per_class": {
                    "labels": labels,
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "f1": f1.tolist(),
                    "support": support.tolist(),
                },
            },
            "figures": {
                "confusion_matrix": cm_path,
                "roc_curves": roc_path,
                "precision_recall": pr_path,
                "learning_curve": lc_path,
            },
        }

    # ── cross-model comparison plots ──────────────────────────────────────────
    comp_path = os.path.join(cfg.figures_dir, "model_comparison.png")
    _plot_model_comparison(summary["models"], comp_path)
    print(f"\n  Model comparison chart -> {comp_path}")

    f1_path = os.path.join(cfg.figures_dir, "per_class_f1.png")
    _plot_per_class_f1(summary["models"], labels, f1_path)
    print(f"  Per-class F1 chart     -> {f1_path}")

    # ── save summary JSON ─────────────────────────────────────────────────────
    out_json = os.path.join(cfg.results_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nOK Wrote evaluation summary -> {out_json}")


if __name__ == "__main__":
    run_evaluate(EvaluateConfig())
