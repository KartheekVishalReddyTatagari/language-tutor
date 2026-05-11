import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


@dataclass
class PreprocessConfig:
    seed: int = 42
    output_csv: str = "data/processed/learner_grammar_dataset.csv"


def _make_synthetic_dataset(n_per_class: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a large, diverse multilingual-ish learner dataset.

    This avoids external downloads (datasets/transformers), keeping the project runnable offline.

    We frame the task as: classify learner sentences by the dominant error type.
    """
    rng = random.Random(seed)

    templates: Dict[str, List[str]] = {
        "subject_verb_agreement": [
            "He {verb} to the store.",
            "She {verb} playing football.",
            "They {verb} a new car.",
            "My friend {verb} very fast.",
        ],
        "tense": [
            "Yesterday I {base} to school.",
            "Last week she {base} to Paris.",
            "In 2020 we {base} a house.",
            "Tomorrow I {base} to work.",
        ],
        "article_usage": [
            "I saw {article} dog in the park.",
            "She bought {article} apple.",
            "He is reading {article} book.",
            "We ate {article} pizza.",
        ],
        "plural_noun": [
            "There is two {noun} in the box.",
            "I have three {noun}.",
            "She wants two {noun}.",
            "They bought many {noun}.",
        ],
        "word_order": [
            "I am today happy.",
            "She always goes to school early.",
            "They to the museum went yesterday.",
            "We dinner at 7 eat.",
        ],
    }

    verbs_pres3 = ["go", "plays", "drives", "runs"]
    verbs_base = ["go", "visit", "build", "buy"]
    articles = ["a", "an", "the", ""]
    nouns = ["cats", "cars", "books", "phones", "students"]

    # To increase diversity, add spelling/noise + punctuation
    punct = [".", "!", "?", ""]
    noise = [
        "", "", "",  # mostly clean
        "gt", "the", "and",  # light word noise
    ]

    def inject_noise(s: str) -> str:
        if rng.random() < 0.08:
            token = rng.choice(noise).strip()
            if token:
                parts = s.split(" ")
                idx = rng.randint(1, max(1, len(parts) - 1))
                parts.insert(idx, token)
                return " ".join(parts)
        return s

    records: List[Dict[str, str]] = []
    classes = list(templates.keys())

    # Map to label id
    for label, tpl_list in templates.items():
        for _ in range(n_per_class):
            tpl = rng.choice(tpl_list)
            if label == "subject_verb_agreement":
                # Occasionally produce wrong agreement
                verb = rng.choice(["go", "play", "runs", "drives"])
                s = tpl.format(verb=verb + "s" if rng.random() < 0.5 else verb)
            elif label == "tense":
                base = rng.choice(verbs_base)
                # wrong tense: use base form instead of past for past markers
                s = tpl.format(base=base)
            elif label == "article_usage":
                art = rng.choice(articles)
                if art == "":
                    s = tpl.format(article="")
                    s = " ".join(s.split())
                else:
                    s = tpl.format(article=art)
            elif label == "plural_noun":
                noun = rng.choice(["cat", "car", "book", "phone", "student"])
                # wrong plurality: keep singular after numbers
                s = tpl.format(noun=noun)
            else:  # word_order
                # produce simple word order errors
                s = tpl

            s = s.replace("  ", " ").strip()
            s = inject_noise(s)
            s = s + rng.choice(punct)

            records.append({"text": s, "label": label})

    df = pd.DataFrame.from_records(records)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def run_preprocess(cfg: PreprocessConfig) -> Tuple[pd.DataFrame, Dict]:
    ensure_dir(os.path.dirname(cfg.output_csv) or ".")

    df = _make_synthetic_dataset(n_per_class=2000, seed=cfg.seed)

    label2id = {lab: i for i, lab in enumerate(sorted(df["label"].unique()))}
    id2label = {v: k for k, v in label2id.items()}

    df["label_id"] = df["label"].map(label2id).astype(int)

    meta = {
        "seed": cfg.seed,
        "num_rows": int(len(df)),
        "num_classes": int(df["label"].nunique()),
        "label2id": label2id,
        "id2label": id2label,
    }

    df.to_csv(cfg.output_csv, index=False, encoding="utf-8")
    meta_path = cfg.output_csv.replace(".csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return df, meta


if __name__ == "__main__":
    cfg = PreprocessConfig()
    df, meta = run_preprocess(cfg)
    print(f"Wrote dataset: {cfg.output_csv}")
    print(json.dumps(meta, indent=2))

