"""
convert_iam_to_ibm_format.py  (v2 — fixed paths)
=================================================

Converts the IAM dataset (Cheng et al., ACL 2022) into the two CSV files that
Notebook 05 (`EpiCite_05_TrainingPipeline.ipynb`) expects:

    data/ibm_evidence.csv   — sentence, is_evidence
    data/ibm_cdc.csv        — sentence, is_claim

Actual IAM repo layout (per the official README):

    IAM/
    ├── claims/
    │   ├── all_claims.txt        ← preferred source, full data
    │   ├── train.txt
    │   ├── dev.txt
    │   └── test.txt
    │   Columns (TAB-separated, no header):
    │     0: claim_label          'C' = is a claim, 'O' = not a claim
    │     1: topic_sentence
    │     2: claim_candidate_sentence   ← THE SENTENCE WE WANT
    │     3: article_id
    │     4: stance_label         (1 / -1 / 0)
    │
    ├── evidence/
    │   ├── evidences1.txt        ← preferred source, full data
    │   ├── train.txt
    │   ├── dev.txt
    │   └── test.txt
    │   Columns (TAB-separated, no header):
    │     0: evidence_label       'E' = is evidence, 'O' = not evidence
    │     1: claim_sentence
    │     2: evidence_candidate_sentence   ← THE SENTENCE WE WANT
    │     3: article_id
    │     4: full_label

Usage
-----
    cd <project_root>
    git clone https://github.com/LiyingCheng95/IAM.git data/IAM
    python convert_iam_to_ibm_format.py
    # Outputs: data/ibm_evidence.csv, data/ibm_cdc.csv

This v2 is defensive: it looks for the data in multiple plausible locations,
auto-detects the label column when row layout is unexpected, tolerates internal
tabs, and prints a warning rather than crashing on individual bad rows.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────── Configuration ──────────────────────────
HERE        = Path(__file__).parent
IAM_ROOT    = HERE / "data" / "IAM"
OUT_DIR     = HERE / "data"
OUT_DIR.mkdir(exist_ok=True)

OUT_EVIDENCE = OUT_DIR / "ibm_evidence.csv"
OUT_CLAIMS   = OUT_DIR / "ibm_cdc.csv"

MIN_TOKENS = 5
MAX_TOKENS = 80


# ─────────────────────────── Helpers ────────────────────────────────

def _word_count(s: str) -> int:
    return len(s.split())


def _is_clean_sentence(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    n = _word_count(s)
    if n < MIN_TOKENS or n > MAX_TOKENS:
        return False
    alpha_chars = sum(1 for c in s if c.isalpha())
    if alpha_chars < 0.5 * len(s):
        return False
    if re.match(r"^\s*\d+(\.\d+)*\s+[A-Z]", s):
        return False
    return True


def _read_tsv_robust(path: Path) -> list[list[str]]:
    """Read a tab-separated file from the IAM repo, tolerantly."""
    if not path.exists():
        return []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            rows.append(row)
    return rows


def _autodetect_label_col(rows: list[list[str]],
                          valid_labels: set[str]) -> int | None:
    """Return the column index that most likely holds the label.

    Defensive fallback used only when the documented column 0 doesn't
    contain valid labels — handles cases where IAM updates row layout.
    """
    if not rows:
        return None
    n_cols = max(len(r) for r in rows[:200])
    best_col, best_hit = None, -1
    for c in range(n_cols):
        hits = sum(1 for r in rows[:200] if c < len(r)
                   and r[c].strip().upper() in valid_labels)
        if hits > best_hit:
            best_hit, best_col = hits, c
    if best_hit < 0.6 * min(len(rows), 200):
        return None
    return best_col


def _autodetect_sentence_col(rows: list[list[str]],
                              skip_cols: set[int]) -> int | None:
    """Pick the column with the longest avg text content (excluding skip_cols)."""
    if not rows:
        return None
    n_cols = max(len(r) for r in rows[:200])
    best_col, best_avg = None, 0.0
    for c in range(n_cols):
        if c in skip_cols:
            continue
        lengths = []
        for r in rows[:200]:
            if c < len(r):
                lengths.append(len(r[c]))
        if not lengths:
            continue
        avg = sum(lengths) / len(lengths)
        if avg > best_avg:
            best_avg, best_col = avg, c
    return best_col


# ─────────────────────────── Claim conversion ───────────────────────

CLAIM_DIRS_TO_TRY = [
    IAM_ROOT / "claims",                # actual layout (correct)
    IAM_ROOT / "data" / "claim",         # fallback (older guess)
    IAM_ROOT / "claim",                  # alternative spelling
]

CLAIM_FILES_PRIORITY = [
    "all_claims.txt",
    "train.txt",
    "dev.txt",
    "test.txt",
]


def _find_claim_dir() -> Path:
    for d in CLAIM_DIRS_TO_TRY:
        if d.exists() and any((d / f).exists() for f in CLAIM_FILES_PRIORITY):
            return d
    raise FileNotFoundError(
        f"Could not find IAM claim files. Tried:\n  " +
        "\n  ".join(str(d) for d in CLAIM_DIRS_TO_TRY) +
        f"\nMake sure you ran: git clone https://github.com/LiyingCheng95/IAM.git {IAM_ROOT}"
    )


def convert_claims() -> pd.DataFrame:
    claim_dir = _find_claim_dir()
    print(f"[claims] using directory: {claim_dir}")

    rows: list[list[str]] = []
    if (claim_dir / "all_claims.txt").exists():
        rows = _read_tsv_robust(claim_dir / "all_claims.txt")
        print(f"[claims] all_claims.txt: {len(rows):,} raw rows")
    else:
        for name in ["train.txt", "dev.txt", "test.txt"]:
            chunk = _read_tsv_robust(claim_dir / name)
            if chunk:
                print(f"[claims] {name}: {len(chunk):,} raw rows")
                rows.extend(chunk)

    if not rows:
        raise FileNotFoundError(f"No claim rows readable in {claim_dir}")

    # Documented layout: col 0 = claim_label (C/O), col 2 = sentence
    LABEL_COL_DOC, SENT_COL_DOC = 0, 2
    valid_labels = {"C", "O"}

    sample = rows[:200]
    docu_ok = sum(1 for r in sample if len(r) > LABEL_COL_DOC
                  and r[LABEL_COL_DOC].strip().upper() in valid_labels)
    if docu_ok >= 0.6 * len(sample):
        label_col, sent_col = LABEL_COL_DOC, SENT_COL_DOC
        print(f"[claims] using documented layout: label=col{label_col}, sentence=col{sent_col}")
    else:
        label_col = _autodetect_label_col(rows, valid_labels)
        if label_col is None:
            raise ValueError(
                f"[claims] couldn't find C/O labels in any column. "
                f"First row was: {rows[0]}"
            )
        sent_col = _autodetect_sentence_col(rows, skip_cols={label_col})
        print(f"[claims] auto-detected: label=col{label_col}, sentence=col{sent_col}")

    sentences, is_claim, skipped = [], [], 0
    for row in rows:
        if len(row) <= max(label_col, sent_col):
            skipped += 1; continue
        sentence = row[sent_col].strip()
        label    = row[label_col].strip().upper()
        if label not in valid_labels:
            skipped += 1; continue
        if not _is_clean_sentence(sentence):
            skipped += 1; continue
        sentences.append(sentence)
        is_claim.append(1 if label == "C" else 0)

    df = pd.DataFrame({"sentence": sentences, "is_claim": is_claim})
    df = df.drop_duplicates("sentence").reset_index(drop=True)

    n_pos = int(df["is_claim"].sum())
    n_neg = len(df) - n_pos
    print(f"[claims] after filtering: {len(df):,} rows  "
          f"(positives={n_pos:,}, negatives={n_neg:,}, skipped={skipped:,})")
    return df


# ─────────────────────────── Evidence conversion ────────────────────

EVIDENCE_DIRS_TO_TRY = [
    IAM_ROOT / "evidence",
    IAM_ROOT / "data" / "evidence",
]

EVIDENCE_FILES_PRIORITY = [
    "evidences1.txt",
    "all_evidence.txt",
    "train.txt",
    "dev.txt",
    "test.txt",
]


def _find_evidence_dir() -> Path:
    for d in EVIDENCE_DIRS_TO_TRY:
        if d.exists() and any((d / f).exists() for f in EVIDENCE_FILES_PRIORITY):
            return d
    raise FileNotFoundError(
        f"Could not find IAM evidence files. Tried:\n  " +
        "\n  ".join(str(d) for d in EVIDENCE_DIRS_TO_TRY) +
        f"\nMake sure you ran: git clone https://github.com/LiyingCheng95/IAM.git {IAM_ROOT}"
    )


def convert_evidence() -> pd.DataFrame:
    evidence_dir = _find_evidence_dir()
    print(f"[evidence] using directory: {evidence_dir}")

    rows: list[list[str]] = []
    found_full = False
    for full_name in ["evidences1.txt", "all_evidence.txt"]:
        full = evidence_dir / full_name
        if full.exists():
            rows = _read_tsv_robust(full)
            print(f"[evidence] {full_name}: {len(rows):,} raw rows")
            found_full = True
            break

    if not found_full:
        for name in ["train.txt", "dev.txt", "test.txt"]:
            chunk = _read_tsv_robust(evidence_dir / name)
            if chunk:
                print(f"[evidence] {name}: {len(chunk):,} raw rows")
                rows.extend(chunk)

    if not rows:
        raise FileNotFoundError(f"No evidence rows readable in {evidence_dir}")

    # Documented layout: col 0 = evidence_label (E/O), col 2 = sentence
    LABEL_COL_DOC, SENT_COL_DOC = 0, 2
    valid_labels = {"E", "O"}

    sample = rows[:200]
    docu_ok = sum(1 for r in sample if len(r) > LABEL_COL_DOC
                  and r[LABEL_COL_DOC].strip().upper() in valid_labels)
    if docu_ok >= 0.6 * len(sample):
        label_col, sent_col = LABEL_COL_DOC, SENT_COL_DOC
        print(f"[evidence] using documented layout: label=col{label_col}, sentence=col{sent_col}")
    else:
        label_col = _autodetect_label_col(rows, valid_labels)
        if label_col is None:
            raise ValueError(
                f"[evidence] couldn't find E/O labels in any column. "
                f"First row was: {rows[0]}"
            )
        sent_col = _autodetect_sentence_col(rows, skip_cols={label_col})
        print(f"[evidence] auto-detected: label=col{label_col}, sentence=col{sent_col}")

    sentences, is_evidence, skipped = [], [], 0
    for row in rows:
        if len(row) <= max(label_col, sent_col):
            skipped += 1; continue
        sentence = row[sent_col].strip()
        label    = row[label_col].strip().upper()
        if label not in valid_labels:
            skipped += 1; continue
        if not _is_clean_sentence(sentence):
            skipped += 1; continue
        sentences.append(sentence)
        is_evidence.append(1 if label == "E" else 0)

    df = pd.DataFrame({"sentence": sentences, "is_evidence": is_evidence})
    df = df.drop_duplicates("sentence").reset_index(drop=True)

    n_pos = int(df["is_evidence"].sum())
    n_neg = len(df) - n_pos
    print(f"[evidence] after filtering: {len(df):,} rows  "
          f"(positives={n_pos:,}, negatives={n_neg:,}, skipped={skipped:,})")
    return df


# ─────────────────────────── Main ───────────────────────────────────

def main() -> None:
    if not IAM_ROOT.exists():
        print(f"ERROR: {IAM_ROOT} does not exist.")
        print("Run this first:")
        print(f"    git clone https://github.com/LiyingCheng95/IAM.git {IAM_ROOT}")
        sys.exit(1)

    print(f"IAM repo location: {IAM_ROOT}")
    print(f"Output directory : {OUT_DIR}")
    print()

    print("─── Converting claim-detection task ───")
    claims_df = convert_claims()
    claims_df.to_csv(OUT_CLAIMS, index=False)
    print(f"✓ wrote {OUT_CLAIMS}\n")

    print("─── Converting evidence-detection task ───")
    evidence_df = convert_evidence()
    evidence_df.to_csv(OUT_EVIDENCE, index=False)
    print(f"✓ wrote {OUT_EVIDENCE}\n")

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Claims file   : {OUT_CLAIMS}")
    print(f"     positives  : {int(claims_df['is_claim'].sum()):,}")
    print(f"     total rows : {len(claims_df):,}")
    print(f"  Evidence file : {OUT_EVIDENCE}")
    print(f"     positives  : {int(evidence_df['is_evidence'].sum()):,}")
    print(f"     total rows : {len(evidence_df):,}")
    print()
    print("Now you can run Notebook 05 with CONFIG['use_ibm']=True.")


if __name__ == "__main__":
    main()
