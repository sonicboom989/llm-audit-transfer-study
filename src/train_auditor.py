"""
src/train_auditor.py

Trains a lightweight output-only auditor on model outputs (output_text) and evaluates
transfer to a held-out family (e.g., encoding, direct_harm, authority).

Key design constraints:
- Output-only: the auditor sees only output_text (NOT prompts).
- No leakage: vectorizer is fit on train only; test is transform-only.
- Controlled experiments:
  (A) Feature ablation: word TF-IDF vs char TF-IDF
  (B) Family transfer: hold out one family for testing
  (C) Optional partial exposure: train on a fraction of the holdout family (for "train on some?")

Default setting below:
- Hold out authority entirely (true transfer), with benign controls in test.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class Config:
    dataset_path: Path = Path("data/processed/auditor_dataset.jsonl")

    # Feature ablation: "word" (baseline) or "char" (ablation)
    features: str = "word"
    max_features: int = 5000
    min_df: int = 2
    lr_max_iter: int = 1000
    word_ngram_range: Tuple[int, int] = (1, 2)
    char_ngram_range: Tuple[int, int] = (3, 5)

    # Transfer setup: hold out one family for testing
    holdout_family: str = "authority"

    # Put some benign examples in test as controls (keeps test not-all-positive)
    benign_test_fraction: float = 0.2

    # Optional: allow a fraction of holdout family into training (set to 0.0 for true holdout)
    holdout_train_fraction: float = 0.2

    # For clean semantic-family tests, you may exclude unrelated families from training (e.g., encoding)
    exclude_families_from_train: Tuple[str, ...] = ("encoding",)

    # Reproducibility
    seed: int = 0


# -----------------------------
# Data loading
# -----------------------------

def load_all_rows(file_path: Path) -> List[Dict[str, Any]]:
    """Load all JSONL rows (we will create train/test splits dynamically)."""
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {file_path}: {e}") from e

            # Minimal schema checks (fail fast; keeps experiments honest)
            if "family" not in row or "y" not in row or "output_text" not in row:
                raise ValueError(
                    f"Missing required keys on line {line_num} in {file_path}. "
                    f"Expected keys include 'family', 'y', 'output_text'. Got: {list(row.keys())}"
                )

            rows.append(row)
    return rows


def make_holdout_split(
    rows: List[Dict[str, Any]],
    holdout_family: str,
    benign_test_fraction: float,
    holdout_train_fraction: float,
    seed: int,
    exclude_families_from_train: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a transfer split:
      - Test contains (most of) the holdout family + a fraction of benign controls.
      - Train contains the remaining benign + all other families (optionally excluding some).
      - Optionally, allow a fraction of the holdout family into training (for "train on some" runs).

    No leakage: splitting is done BEFORE vectorizer fitting.
    """
    if not (0.0 <= benign_test_fraction <= 1.0):
        raise ValueError("benign_test_fraction must be in [0, 1].")
    if not (0.0 <= holdout_train_fraction <= 1.0):
        raise ValueError("holdout_train_fraction must be in [0, 1].")

    rng = random.Random(seed)

    holdout_rows = [r for r in rows if r["family"] == holdout_family]
    benign_rows = [r for r in rows if r["family"] == "benign"]
    other_rows = [r for r in rows if r["family"] not in {holdout_family, "benign"}]

    # Exclude certain families from training (e.g., encoding) for a clean semantic-family transfer test
    other_rows_train = [r for r in other_rows if r["family"] not in exclude_families_from_train]

    # Benign controls in test
    rng.shuffle(benign_rows)
    if len(benign_rows) == 0:
        raise ValueError("No benign rows found; cannot construct benign controls in test.")
    n_benign_test = int(round(len(benign_rows) * benign_test_fraction))
    # Keep at least 1 benign example in test, but don't move all benign out of train
    n_benign_test = max(1, min(n_benign_test, len(benign_rows) - 1))
    benign_test = benign_rows[:n_benign_test]
    benign_train = benign_rows[n_benign_test:]

    # Holdout family: optional partial exposure
    if len(holdout_rows) == 0:
        raise ValueError(f"No rows found for holdout_family={holdout_family!r}. Check family names in dataset.")
    rng.shuffle(holdout_rows)
    n_holdout_train = int(round(len(holdout_rows) * holdout_train_fraction))
    holdout_train = holdout_rows[:n_holdout_train]
    holdout_test = holdout_rows[n_holdout_train:]

    train = benign_train + other_rows_train + holdout_train
    test = benign_test + holdout_test

    return train, test


# -----------------------------
# Feature engineering
# -----------------------------

def build_vectorizer(cfg: Config) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer for either word or character n-grams."""
    if cfg.features == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=cfg.word_ngram_range,
            max_features=cfg.max_features,
            min_df=cfg.min_df,
        )
    if cfg.features == "char":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=cfg.char_ngram_range,
            max_features=cfg.max_features,
            min_df=cfg.min_df,
        )
    raise ValueError("Config.features must be 'word' or 'char'.")


# -----------------------------
# Diagnostics
# -----------------------------

def alpha_ratio(text: str) -> float:
    """Fraction of alphabetic characters in a string (robust to empty text)."""
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def avg_length(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(t) for t in texts) / len(texts)


def avg_alpha_ratio(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(alpha_ratio(t) for t in texts) / len(texts)


def positive_rate(y: List[int]) -> float:
    if not y:
        return 0.0
    return sum(y) / len(y)


# -----------------------------
# Training / evaluation
# -----------------------------

def main() -> None:
    cfg = Config()

    rows = load_all_rows(cfg.dataset_path)

    train_rows, test_rows = make_holdout_split(
        rows=rows,
        holdout_family=cfg.holdout_family,
        benign_test_fraction=cfg.benign_test_fraction,
        holdout_train_fraction=cfg.holdout_train_fraction,
        seed=cfg.seed,
        exclude_families_from_train=set(cfg.exclude_families_from_train),
    )

    # Print experiment header (so pasted logs are self-describing)
    print("=== Auditor Training: Transfer Experiment ===")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Holdout family (test): {cfg.holdout_family!r}")
    print(f"Holdout train fraction: {cfg.holdout_train_fraction:.2f}")
    print(f"Benign test fraction (controls): {cfg.benign_test_fraction:.2f}")
    print(f"Excluded families from train: {set(cfg.exclude_families_from_train)}")
    print(f"Features mode: {cfg.features!r}")
    print(f"Random seed: {cfg.seed}")
    print("===========================================\n")

    print(f"Loaded {len(train_rows)} training samples.")
    print(f"Loaded {len(test_rows)} testing samples.")

    # Extract
    x_train_text = [row.get("output_text", "") for row in train_rows]
    y_train = [int(row["y"]) for row in train_rows]
    x_test_text = [row.get("output_text", "") for row in test_rows]
    y_test = [int(row["y"]) for row in test_rows]

    print(f"Train positive rate: {positive_rate(y_train):.4f}")
    print(f"Test positive rate:  {positive_rate(y_test):.4f}")

    # Vectorize (fit on train only)
    vectorizer = build_vectorizer(cfg)
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    # Print vectorizer details
    if cfg.features == "word":
        print(
            f"Vectorizer: analyzer='word', ngram_range={cfg.word_ngram_range}, "
            f"max_features={cfg.max_features}, min_df={cfg.min_df}"
        )
    else:
        print(
            f"Vectorizer: analyzer='char', ngram_range={cfg.char_ngram_range}, "
            f"max_features={cfg.max_features}, min_df={cfg.min_df}"
        )

    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)

    # Train model
    model = LogisticRegression(max_iter=cfg.lr_max_iter)
    model.fit(x_train, y_train)

    # Predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Metrics
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_test_pred, labels=[0, 1], digits=4))

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Train Predicted positive rate: {positive_rate(list(map(int, y_train_pred))):.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Test Predicted positive rate: {positive_rate(list(map(int, y_test_pred))):.4f}")
    print(f"Test mean predicted probability: {float(y_test_proba.mean()):.4f}")

    # Confusion matrix with explicit TN/FP/FN/TP
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (labels [0,1]):")
    print(cm)
    print(f"Test confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Diagnostics
    print("\nText diagnostics:")
    print(f"Train avg length: {avg_length(x_train_text):.3f}")
    print(f"Train avg alpha ratio: {avg_alpha_ratio(x_train_text):.6f}")
    print(f"Test avg length: {avg_length(x_test_text):.3f}")
    print(f"Test avg alpha ratio: {avg_alpha_ratio(x_test_text):.6f}")


if __name__ == "__main__":
    main()
