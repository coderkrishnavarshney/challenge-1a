
"""predict.py
Load XGBoost model + encoder and generate outline JSON for PDFs.
Expects:
  /app/models/model_*.pkl
  /app/models/label_encoder.pkl
  /app/models/feature_list.json
"""
import json, joblib, logging, os, sys
from pathlib import Path
import pandas as pd
import numpy as np

from .extract_features import extract_pdf_features

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODELS_DIR = Path("/app/models")
INPUT_DIR  = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

def load_artifacts():
    model_path = max(MODELS_DIR.glob("model_*.pkl"), key=os.path.getmtime)
    model = joblib.load(model_path)
    le    = joblib.load(MODELS_DIR / "label_encoder.pkl")
    feature_list = json.loads((MODELS_DIR / "feature_list.json").read_text())
    return model, le, feature_list

def prepare_features(df: pd.DataFrame, feature_list):
    # 1. One‑hot encode booleans / categoricals
    df_proc = pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == "object" and c != "text"])
    # 2. Align columns
    for col in feature_list:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[feature_list]
    return df_proc

def build_outline(df: pd.DataFrame, preds, le):
    df = df.copy()
    df["label"] = le.inverse_transform(preds)
    # Title: pick first 'title' if exists, else empty
    title_row = df[df["label"] == "title"].head(1)
    title = title_row["text"].iloc[0] if not title_row.empty else ""
    outline_rows = df[df["label"].isin({"H1","H2","H3"})]
    outline_json = [
        {"level": row["label"], "text": row["text"], "page": int(row["page"])}
        for _, row in outline_rows.iterrows()
    ]
    return {"title": title, "outline": outline_json}

def process_pdf(pdf_path: Path, model, le, feature_list):
    df_raw = extract_pdf_features(pdf_path)
    if df_raw.empty:
        logging.warning("No text extracted from %s", pdf_path.name)
        return
    X = prepare_features(df_raw.drop(columns=["text"]), feature_list)
    preds = model.predict(X)
    outline = build_outline(df_raw, preds, le)
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
    out_path.write_text(json.dumps(outline, indent=2))
    logging.info("Saved outline to %s", out_path)

def main():
    model, le, feature_list = load_artifacts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        logging.error("No PDFs found in %s", INPUT_DIR); sys.exit(1)
    for pdf in pdfs:
        logging.info("Processing %s", pdf.name)
        process_pdf(pdf, model, le, feature_list)

if __name__ == "__main__":
    main()
