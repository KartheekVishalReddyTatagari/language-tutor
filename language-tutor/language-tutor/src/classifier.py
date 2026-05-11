import os
import json
import joblib
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .utils import ensure_dir


@dataclass
class ClassifierConfig:
    seed: int = 42
    dataset_csv: str = "data/processed/learner_grammar_dataset.csv"
    meta_json: str = "data/processed/learner_grammar_dataset_meta.json"
    model_dir: str = "models/classic"
    test_size: float = 0.2

    # GridSearch later in evaluate.py, but we keep baseline here
    max_features: int = 20000


def _load_meta(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataset(cfg: ClassifierConfig) -> Tuple[pd.DataFrame, Dict]:
    if not os.path.exists(cfg.dataset_csv):
        raise FileNotFoundError(
            f"Dataset not found: {cfg.dataset_csv}. Run: python -m src.preprocess"
        )

    if not os.path.exists(cfg.meta_json):
        raise FileNotFoundError(
            f"Meta not found: {cfg.meta_json}. Run: python -m src.preprocess"
        )

    df = pd.read_csv(cfg.dataset_csv)
    meta = _load_meta(cfg.meta_json)

    # Ensure columns exist
    if "text" not in df.columns or "label_id" not in df.columns:
        raise ValueError("Dataset CSV must have columns: text, label_id")

    return df, meta


def build_logreg_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                ),
            ),
        ]
    )


def build_svm_pipeline(max_features: int) -> Pipeline:
    # LinearSVC supports multiclass but does not provide predict_proba reliably.
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LinearSVC(),
            ),
        ]
    )


def run_train(cfg: ClassifierConfig) -> None:
    ensure_dir(cfg.model_dir)

    df, meta = _load_dataset(cfg)

    X = df["text"].astype(str).values
    y = df["label_id"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    pipelines = {
        "logreg": build_logreg_pipeline(cfg.max_features),
        "svm": build_svm_pipeline(cfg.max_features),
    }

    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        # Quick sanity metric on held-out test
        acc = float(pipe.score(X_test, y_test))
        print(f"{name} test accuracy: {acc:.4f}")

        out_path = os.path.join(cfg.model_dir, f"{name}_pipeline.joblib")
        joblib.dump(pipe, out_path)

        results_path = os.path.join(cfg.model_dir, f"{name}_quick_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_accuracy": acc,
                    "seed": cfg.seed,
                    "dataset_rows": int(len(df)),
                    "num_classes": int(meta.get("num_classes", -1)),
                },
                f,
                indent=2,
            )

    print(f"Saved models to: {cfg.model_dir}")


if __name__ == "__main__":
    run_train(ClassifierConfig())

