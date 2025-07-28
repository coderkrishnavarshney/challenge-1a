
"""
train.py  –  Balanced XGBoost trainer for TITLE + H1‑H4 headings
================================================================

The script:
1. Loads the CSV produced by `extract_features.py`
2. Drops 'NONE' rows
3. **Up‑samples every minority class** so all labels have the same count
   as the current majority (capped to --max-samples if set)
4. One‑hot‑encodes categorical columns
5. Evaluates with StratifiedKFold (or Leave‑One‑Out if very small)
6. Saves:
      models/model_<timestamp>.pkl
      models/label_encoder.pkl
      models/feature_list.json
      models/metrics_<timestamp>.json
      models/importances_<timestamp>.json
"""

from __future__ import annotations
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
from typing import List

import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DROP_LABEL           = "NONE"
RANDOM_STATE         = 42
TINY_SET_THRESHOLD   = 40          # < rows → Leave‑One‑Out CV
MAX_UPSAMPLED_ROWS   = 500         # safety cap per class


# ───────────────────────── CLI & logging ──────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGB for TITLE + H‑levels")
    p.add_argument("csv", type=Path, help="train.csv from extract_features.py")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("models"),
                   help="folder to store artefacts")
    p.add_argument("--max-samples", type=int, default=MAX_UPSAMPLED_ROWS,
                   help="cap per class after up‑sampling (default 500)")
    p.add_argument("--n-estimators", type=int, default=500)
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def detect_categoricals(df: pd.DataFrame, max_unique=30) -> List[str]:
    return [
        col for col in df.columns
        if col != "label" and
           (df[col].dtype == "object" or df[col].nunique() <= max_unique)
    ]


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))


# ─────────────────────────── up‑sampling ───────────────────────────────
def upsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Simple bootstrap up‑sample so every class has `max_count` rows."""
    groups = [g for _, g in df.groupby("label")]
    max_count = min(max(len(g) for g in groups), max_rows)
    up_frames = []
    for g in groups:
        need = max_count - len(g)
        if need > 0:
            reps = g.sample(n=need, replace=True, random_state=RANDOM_STATE)
            up_frames.append(pd.concat([g, reps], ignore_index=True))
        else:
            up_frames.append(g.sample(n=max_count, random_state=RANDOM_STATE))
    balanced = pd.concat(up_frames, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle
    return balanced


# ───────────────────────── main routine ────────────────────────────────
def main():
    setup_logging()
    args = parse_args()

    if not args.csv.exists():
        logging.error("Input CSV not found: %s", args.csv)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # 1. Drop 'NONE'
    df = df[df["label"] != DROP_LABEL]
    logging.info("Rows after dropping '%s': %s", DROP_LABEL, len(df))
    if df.empty:
        logging.error("No labelled rows. Exiting."); sys.exit(1)

    # 2. Up‑sample
    before_counts = df["label"].value_counts().to_dict()
    df_bal = upsample(df, args.max_samples)
    after_counts = df_bal["label"].value_counts().to_dict()
    logging.info("Class distribution before: %s", before_counts)
    logging.info("Class distribution after : %s", after_counts)

    # 3. One‑hot
    cats = detect_categoricals(df_bal)
    X = pd.get_dummies(df_bal.drop(columns=["label"]), columns=cats)
    X = X.drop(columns=X.select_dtypes(include=["object"]).columns)
    feature_list = list(X.columns)

    # 4. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df_bal["label"].astype(str))
    classes = list(le.classes_)
    logging.info("Classes: %s", classes)

    # 5. Choose CV
    if len(df_bal) < TINY_SET_THRESHOLD:
        cv = LeaveOneOut()
        logging.warning("Tiny dataset (%s rows) → Leave‑One‑Out CV.", len(df_bal))
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 6. Cross‑validated predictions for metrics
    y_pred = np.empty_like(y)
    fold = 0
    for train_idx, test_idx in cv.split(X, y):
        fold += 1
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE + fold,
            n_jobs=8,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X.iloc[test_idx])

    acc      = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    report   = classification_report(y, y_pred, target_names=classes, zero_division=0)
    logging.info("CV Accuracy=%.3f  Macro‑F1=%.3f\n%s", acc, macro_f1, report)

    # 7. Final model on **full balanced set**
    final_clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    final_clf.fit(X, y)

    # 8. Save artefacts
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(final_clf, out / f"model_{ts}.pkl")
    joblib.dump(le,        out / "label_encoder.pkl")
    save_json(feature_list,               out / "feature_list.json")
    save_json(
        {"accuracy": float(acc), "macro_f1": float(macro_f1), "report": report},
        out / f"metrics_{ts}.json",
    )
    save_json(
        [{"feature": f, "importance": float(imp)}
         for f, imp in sorted(zip(feature_list, final_clf.feature_importances_),
                              key=lambda x: x[1], reverse=True)],
        out / f"importances_{ts}.json",
    )
    logging.info("✅  Training complete – model saved to %s/", out)


if __name__ == "__main__":
    main()
