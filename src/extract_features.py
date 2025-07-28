
"""extract_features.py
Simple feature extractor for PDF outlines.
NOTE: Adapt feature engineering to match the features used in training.
"""
import re, math, pdfplumber, pandas as pd
from pathlib import Path

BULLETS = {"•", "-", "–", "—", "*", "·", "◦"}
URL_RE  = re.compile(r"https?://")
EMAIL_RE= re.compile(r"\b[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE= re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b")
TITLE_KW= {kw.lower() for kw in ["abstract","introduction","overview","conclusion","summary"]}

def _line_features(text:str):
    words = text.strip().split()
    word_count = len(words)
    len_chars  = len(text)
    num_digits = sum(c.isdigit() for c in text)
    num_upper  = sum(c.isupper() for c in text)
    all_caps_line = text.isupper()
    all_caps_words = sum(1 for w in words if w.isupper())
    starts_bullet = text.strip()[:1] in BULLETS
    has_url   = bool(URL_RE.search(text))
    has_email = bool(EMAIL_RE.search(text))
    has_phone = bool(PHONE_RE.search(text))
    title_kw_ratio = sum(1 for w in words if w.lower() in TITLE_KW) / word_count if word_count else 0
    return {
        "word_count": word_count,
        "len_chars": len_chars,
        "num_digits": num_digits,
        "num_upper": num_upper,
        "all_caps_line": all_caps_line,
        "all_caps_words": all_caps_words,
        "starts_bullet": starts_bullet,
        "has_url": has_url,
        "has_email": has_email,
        "has_phone": has_phone,
        "title_kw_ratio": title_kw_ratio
    }

def extract_pdf_features(pdf_path: Path):
    """Return DataFrame with one row per line (potential heading)."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            try:
                # Extract lines preserving layout; pdfplumber 0.10 has extract_text(layout=True)
                text = page.extract_text()
            except Exception:
                text = page.extract_text()
            if not text:
                continue
            for idx, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                feats = _line_features(line)
                feats.update({
                    "page": pnum,
                    "y": idx / max(len(text.splitlines()),1),  # relative vertical position
                    "text": line
                })
                rows.append(feats)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    df = extract_pdf_features(args.pdf)
    if args.out:
        df.to_csv(args.out, index=False)
    else:
        print(df.head())
