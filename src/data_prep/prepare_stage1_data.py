from __future__ import annotations

import argparse
import io
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.model_selection import train_test_split
from textblob import TextBlob

DetectorFactory.seed = 42

LABELS_5 = ["Claim", "Fact", "Evidence", "Opinion", "Background"]

TARGET_COUNTS = {
    "Claim": 6000,
    "Fact": 3400,
    "Evidence": 3400,
    "Opinion": 2400,
    "Background": 2400,
}

FACT_SOURCE_QUOTAS = {
    "fever": 3400,
}

EVIDENCE_SOURCE_QUOTAS = {
    "fever": 2200,
    "aaec_ukp": 1200,
}

CLAIM_SOURCE_QUOTAS = {
    "aaec_ukp": 2257,
    "fever": 3743,
}

OPINION_SOURCE_QUOTAS = {
    "imdb": 2200,
    "aaec_ukp": 200,
}

BACKGROUND_SOURCE_QUOTAS = {
    "aaec_ukp": 1326,
    "fever": 1074,
}

FIRST_PERSON_START_PATTERN = re.compile(r"^\s*(i|we)\b", re.IGNORECASE)
FIRST_PERSON_PATTERN = re.compile(r"\b(i|we|my|our|me|us|i'm|we're)\b", re.IGNORECASE)
HEDGING_PATTERN = re.compile(
    r"\b(may|might|could|possibly|perhaps|likely|seems?|appears?|probably|arguably)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
OPINION_LEXICON = {
    "think",
    "believe",
    "feel",
    "seems",
    "opinion",
    "view",
    "suppose",
    "love",
    "hate",
    "like",
    "dislike",
    "enjoy",
    "prefer",
    "amazing",
    "terrible",
    "awful",
    "fantastic",
    "boring",
    "brilliant",
    "stupid",
    "ridiculous",
    "disappointing",
}
OPINION_LEXICON_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(OPINION_LEXICON)) + r")\b", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def map_to_5class(row: Dict[str, object]) -> str:
    """Pure function: map one row to the unified Stage 1 label set."""
    source = normalize_text(str(row.get("source", ""))).lower()
    original = normalize_text(str(row.get("original_label", ""))).lower()

    if source == "aaec_ukp":
        if original == "premise":
            return "Evidence"
        if original in {"claim", "majorclaim"}:
            return "Claim"
        if original == "non-argument":
            if FIRST_PERSON_PATTERN.search(str(row.get("sentence", ""))):
                return "Opinion"
            return "Background"

    if source == "fever":
        if "refutes_claim" in original:
            return "Claim"
        if "supports_claim" in original:
            return "Fact"
        if "supports_evidence" in original:
            return "Evidence"
        if "nei_claim" in original:
            return "Background"

    if source == "imdb":
        return "Opinion"

    return "Background"


def map_with_reason(row: Dict[str, object]) -> Tuple[str, str]:
    source = normalize_text(str(row.get("source", ""))).lower()
    original = normalize_text(str(row.get("original_label", ""))).lower()
    sentence = normalize_text(str(row.get("sentence", "")))

    if source == "aaec_ukp":
        if original == "premise":
            return "Evidence", "aaec_premise_to_evidence"
        if original in {"claim", "majorclaim"}:
            return "Claim", "aaec_claim_to_claim"
        if original == "non-argument":
            if FIRST_PERSON_PATTERN.search(sentence):
                return "Opinion", "aaec_non_argument_first_person_to_opinion"
            return "Background", "aaec_non_argument_to_background"

    if source == "fever":
        if "refutes_claim" in original:
            return "Claim", "fever_refutes_claim_to_claim"
        if "supports_claim" in original:
            return "Fact", "fever_supports_claim_to_fact"
        if "supports_evidence" in original:
            return "Evidence", "fever_supports_evidence_to_evidence"
        if "nei_claim" in original:
            return "Background", "fever_nei_claim_to_background"

    if source == "imdb":
        return "Opinion", "imdb_subjective_sentence_to_opinion"

    return "Background", "fallback_background"


def sentence_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"[^.!?]+[.!?]|[^.!?]+$", text, flags=re.S):
        raw = m.group(0)
        sent = raw.strip()
        if not sent:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        start = m.start() + left_trim
        end = m.start() + right_trim
        spans.append((sent, start, end))
    return spans


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def parse_aaec_components(ann_text: str) -> List[Tuple[str, int, int, str]]:
    out: List[Tuple[str, int, int, str]] = []
    for line in ann_text.splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        label_info = parts[1].split()
        if len(label_info) < 3:
            continue
        comp_type = label_info[0]
        if comp_type not in {"MajorClaim", "Claim", "Premise"}:
            continue
        try:
            start = int(label_info[1])
            end = int(label_info[2])
        except ValueError:
            continue
        out.append((comp_type, start, end, normalize_text(parts[2])))
    return out


def load_aaec_candidates(raw_dir: Path) -> Tuple[pd.DataFrame, Counter]:
    zip_path = raw_dir / "UKP_sentential_argument_mining.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Please place ArgumentAnnotatedEssays zip there."
        )

    rows: List[Dict[str, str]] = []
    label_counter: Counter = Counter()

    with zipfile.ZipFile(zip_path, "r") as outer:
        nested = [
            n
            for n in outer.namelist()
            if n.lower().endswith("brat-project-final.zip") and "__macosx" not in n.lower()
        ]
        if not nested:
            raise RuntimeError("Could not find nested brat-project-final.zip in UKP archive.")

        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0])), "r") as inner:
            txt_files = [
                n for n in inner.namelist() if n.endswith(".txt") and "__macosx" not in n.lower()
            ]
            for txt_name in txt_files:
                ann_name = txt_name[:-4] + ".ann"
                if ann_name not in inner.namelist():
                    continue

                text = inner.read(txt_name).decode("utf-8", errors="ignore")
                ann = inner.read(ann_name).decode("utf-8", errors="ignore")
                comps = parse_aaec_components(ann)

                comp_spans = [(label, st, ed) for label, st, ed, _ in comps]
                for comp_label, _, _, comp_text in comps:
                    if not comp_text:
                        continue
                    rows.append(
                        {
                            "sentence": comp_text,
                            "source": "aaec_ukp",
                            "original_label": comp_label,
                        }
                    )
                    label_counter[comp_label] += 1

                for sent, s_start, s_end in sentence_spans(text):
                    if any(
                        spans_overlap(s_start, s_end, c_start, c_end)
                        for _, c_start, c_end in comp_spans
                    ):
                        continue
                    rows.append(
                        {
                            "sentence": sent,
                            "source": "aaec_ukp",
                            "original_label": "Non-Argument",
                        }
                    )
                    label_counter["Non-Argument"] += 1

    return pd.DataFrame(rows), label_counter


def load_fever_candidates() -> Tuple[pd.DataFrame, Counter]:
    ds: DatasetDict = load_dataset("copenlu/fever_gold_evidence")
    merged = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

    rows: List[Dict[str, str]] = []
    label_counter: Counter = Counter()

    for rec in merged:
        lbl = str(rec["label"])
        label_counter[lbl] += 1

        claim = normalize_text(rec["claim"])
        if claim:
            if lbl == "SUPPORTS":
                rows.append(
                    {
                        "sentence": claim,
                        "source": "fever",
                        "original_label": "SUPPORTS_claim",
                    }
                )
            elif lbl == "REFUTES":
                rows.append(
                    {
                        "sentence": claim,
                        "source": "fever",
                        "original_label": "REFUTES_claim",
                    }
                )
            elif lbl == "NOT ENOUGH INFO":
                rows.append(
                    {
                        "sentence": claim,
                        "source": "fever",
                        "original_label": "NEI_claim",
                    }
                )

        if lbl == "SUPPORTS":
            evidences = rec.get("evidence", [])
            for item in evidences:
                if len(item) < 3:
                    continue
                evidence_sentence = normalize_text(str(item[2]))
                if evidence_sentence:
                    rows.append(
                        {
                            "sentence": evidence_sentence,
                            "source": "fever",
                            "original_label": "SUPPORTS_evidence",
                        }
                    )

    return pd.DataFrame(rows), label_counter


def iter_review_sentences(text: str) -> Iterable[str]:
    clean = normalize_text(text.replace("<br />", " "))
    for s in SENTENCE_SPLIT_PATTERN.split(clean):
        sent = normalize_text(s)
        if sent:
            yield sent


def load_imdb_opinion_candidates(max_candidates: int = 8000) -> Tuple[pd.DataFrame, Counter]:
    ds: DatasetDict = load_dataset("stanfordnlp/imdb")
    merged = concatenate_datasets([ds["train"], ds["test"]])

    rows: List[Dict[str, str]] = []
    label_counter: Counter = Counter()

    for rec in merged:
        review_label = int(rec["label"])
        label_counter[str(review_label)] += 1

        for sent in iter_review_sentences(rec["text"]):
            n_words = word_count(sent)
            if n_words < 5 or n_words > 60:
                continue
            subj = TextBlob(sent).sentiment.subjectivity
            if subj <= 0.7:
                continue
            has_first_person = bool(FIRST_PERSON_PATTERN.search(sent))
            has_lexicon = bool(OPINION_LEXICON_PATTERN.search(sent))
            if not (has_first_person or has_lexicon):
                continue
            rows.append(
                {
                    "sentence": sent,
                    "source": "imdb",
                    "original_label": f"review_label_{review_label}",
                }
            )
            if len(rows) >= max_candidates:
                return pd.DataFrame(rows), label_counter

    return pd.DataFrame(rows), label_counter


def preprocess_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sentence"] = out["sentence"].astype(str).map(normalize_text)
    out = out[out["sentence"].str.len() > 0].copy()
    out["n_words"] = out["sentence"].map(word_count)
    out = out[(out["n_words"] >= 5) & (out["n_words"] <= 100)].copy()
    fever_fact_claim_mask = (
        (out["source"] == "fever") & (out["original_label"] == "SUPPORTS_claim")
    )
    fever_refutes_claim_mask = (
        (out["source"] == "fever") & (out["original_label"] == "REFUTES_claim")
    )
    out = out[
        (~fever_fact_claim_mask | ((out["n_words"] >= 5) & (out["n_words"] <= 40)))
        & (~fever_refutes_claim_mask | ((out["n_words"] >= 5) & (out["n_words"] <= 40)))
    ].copy()
    out = out.drop_duplicates(subset=["sentence", "source", "original_label"])
    return out.reset_index(drop=True)


def sample_with_quota(
    pool: pd.DataFrame,
    by_source: Dict[str, int],
    total_target: int,
    seed: int,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    for source, n in by_source.items():
        src_df = pool[pool["source"] == source]
        if len(src_df) == 0:
            continue
        take = min(n, len(src_df))
        parts.append(src_df.sample(n=take, random_state=seed))

    sampled = pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0].copy()

    if len(sampled) < total_target:
        remaining = pool.drop(index=sampled.index, errors="ignore")
        need = total_target - len(sampled)
        if len(remaining) > 0:
            topup = remaining.sample(n=min(need, len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, topup], ignore_index=True)

    if len(sampled) > total_target:
        sampled = sampled.sample(n=total_target, random_state=seed)

    return sampled.reset_index(drop=True)


def clean_final(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "input_rows": len(df),
        "dropped_duplicates": 0,
        "dropped_length": 0,
        "dropped_non_english": 0,
    }

    out = df.copy()
    out["sentence"] = out["sentence"].astype(str).map(normalize_text)

    before = len(out)
    out = out.drop_duplicates(subset=["sentence"]).copy()
    stats["dropped_duplicates"] = before - len(out)

    out["n_words"] = out["sentence"].map(word_count)
    before = len(out)
    out = out[(out["n_words"] >= 5) & (out["n_words"] <= 100)].copy()
    stats["dropped_length"] = before - len(out)

    before = len(out)
    out["is_english"] = out["sentence"].map(is_english)
    out = out[out["is_english"]].copy()
    stats["dropped_non_english"] = before - len(out)

    return out.reset_index(drop=True), stats


def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return (
        train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def backup_old_parquet(processed_dir: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ["stage1_train.parquet", "stage1_val.parquet", "stage1_test.parquet"]:
        src = processed_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)


def leakage_check_against_wikisqe(train_df: pd.DataFrame) -> Dict[str, int]:
    ds_train = load_dataset("ando55/WikiSQE_experiment", "all", split="train")
    ds_val = load_dataset("ando55/WikiSQE_experiment", "all", split="val")
    ds_test = load_dataset("ando55/WikiSQE_experiment", "all", split="test")
    merged = concatenate_datasets([ds_train, ds_val, ds_test])

    train_sentences = set(train_df["sentence"].astype(str).map(normalize_text).str.lower().tolist())
    overlaps = 0
    checked = 0
    for rec in merged:
        checked += 1
        text = normalize_text(str(rec["text"])).lower()
        if text in train_sentences:
            overlaps += 1

    return {
        "checked_wikisqe_rows": checked,
        "train_size": len(train_sentences),
        "overlap_count": overlaps,
    }


def render_report(
    report_path: Path,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_label_stats: Dict[str, Counter],
    clean_stats: Dict[str, int],
    leakage_stats: Dict[str, int],
    imdb_candidate_count: int,
    fever_refutes_len_mean: float,
    fever_refutes_len_median: float,
    seed: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Stage 1 Data Report (Repaired v2)")
    lines.append("")
    lines.append("## Source Notes")
    lines.append("- IBM primary candidate (Claim+Evidence+Context) was not publicly reachable in this environment.")
    lines.append("- IAM is removed by design: stance +/-1 is topic stance, not Claim/Evidence semantic type.")
    lines.append("- UKP/AAEC: ArgumentAnnotatedEssays-2.0 (MajorClaim/Claim/Premise + Non-Argument sentence extraction).")
    lines.append("- FEVER source used: copenlu/fever_gold_evidence (REFUTES claims + SUPPORTS claims + SUPPORTS evidence + NEI claims).")
    lines.append("- WikiSQE is removed from Stage 1 training data to avoid leakage into Stage 2.")
    lines.append("- Opinion source used: stanfordnlp/imdb with TextBlob subjectivity > 0.7.")
    lines.append("- Opinion dual filter: TextBlob > 0.7 AND (first-person OR opinion lexicon).")
    lines.append("")

    lines.append("## Source Original Label Distribution")
    for source, ctr in source_label_stats.items():
        lines.append("")
        lines.append(f"### {source}")
        for label, count in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {label}: {count}")

    lines.append("")
    lines.append("## Mapping Rules")
    lines.append("- AAEC MajorClaim/Claim -> Claim")
    lines.append("- FEVER REFUTES claim -> Claim")
    lines.append("- AAEC Premise -> Evidence")
    lines.append("- AAEC Non-Argument -> Background (or Opinion if first-person) ")
    lines.append("- FEVER SUPPORTS claim -> Fact")
    lines.append("- FEVER SUPPORTS evidence sentence -> Evidence")
    lines.append("- FEVER NOT ENOUGH INFO claim -> Background")
    lines.append("- IMDB subjective sentence (TextBlob > 0.7) -> Opinion")
    lines.append("")

    lines.append("## Cleaning Stats")
    for k, v in clean_stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    counts = full_df["label"].value_counts().reindex(LABELS_5, fill_value=0)
    lines.append("## Final 5-Class Distribution")
    for label in LABELS_5:
        c = int(counts[label])
        p = c / max(len(full_df), 1)
        lines.append(f"- {label}: {c} ({p:.4f})")
    lines.append("")

    lines.append("## Split Sizes")
    lines.append(f"- train: {len(train_df)}")
    lines.append(f"- val: {len(val_df)}")
    lines.append(f"- test: {len(test_df)}")
    lines.append("")

    lines.append("## Length Statistics")
    lines.append(f"- FEVER REFUTES sampled claim length mean: {fever_refutes_len_mean:.2f}")
    lines.append(f"- FEVER REFUTES sampled claim length median: {fever_refutes_len_median:.2f}")
    lines.append(f"- IMDB candidates after dual filter: {imdb_candidate_count}")
    lines.append("- IMDB baseline from previous run: 4190")
    lines.append("")

    lines.append("## Random Samples (15 per class)")
    for label in LABELS_5:
        lines.append("")
        lines.append(f"### {label}")
        subset = full_df[full_df["label"] == label]
        if subset.empty:
            lines.append("- (no samples)")
            continue
        n_samples = 20 if label in {"Claim", "Opinion"} else 15
        sample = subset.sample(n=min(n_samples, len(subset)), random_state=seed)
        for _, row in sample.iterrows():
            sentence = str(row["sentence"]).replace("\n", " ")
            lines.append(f"- [{row['source']}/{row['original_label']}] {sentence}")

    suspicious = full_df[
        full_df["mapping_reason"].isin(
            {
                "fever_refutes_claim_to_claim",
                "imdb_subjective_sentence_to_opinion",
                "aaec_non_argument_first_person_to_opinion",
            }
        )
    ]

    lines.append("")
    lines.append("## Potentially Weak Mapping Rules")
    lines.append("- fever_refutes_claim_to_claim: REFUTES claims can include noisy or ambiguous claims.")
    lines.append("- imdb_subjective_sentence_to_opinion: TextBlob subjectivity can mis-score rhetorical factual sentences.")
    lines.append("- aaec_non_argument_first_person_to_opinion: first-person heuristic may over-generate Opinion.")
    lines.append(f"- Suspicious sample count (rules above): {len(suspicious)}")
    for _, row in suspicious.head(50).iterrows():
        sentence = str(row["sentence"]).replace("\n", " ")
        lines.append(
            f"- [{row['source']}/{row['original_label']} -> {row['label']}] reason={row['mapping_reason']}: {sentence}"
        )

    lines.append("")
    lines.append("## Data Leakage Check")
    lines.append(
        "- Stage 1 train sentences were matched by exact normalized string against WikiSQE_experiment/all (train+val+test)."
    )
    lines.append(f"- WikiSQE rows checked: {leakage_stats['checked_wikisqe_rows']}")
    lines.append(f"- Stage 1 train unique sentences: {leakage_stats['train_size']}")
    lines.append(f"- Exact overlap count: {leakage_stats['overlap_count']}")
    if leakage_stats["overlap_count"] == 0:
        lines.append("- Result: No overlap detected.")
    else:
        lines.append("- Result: Overlap detected. Review required.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repaired Stage 1 data prep with balanced 5-class targets.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.project_root
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    archive_dir = processed_dir / "archive" / "v2"
    report_path = root / "reports" / "stage1_data_report.md"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    backup_old_parquet(processed_dir, archive_dir)

    aaec_df, aaec_labels = load_aaec_candidates(raw_dir)
    fever_df, fever_labels = load_fever_candidates()
    imdb_df, imdb_labels = load_imdb_opinion_candidates(max_candidates=9000)
    imdb_candidate_count = len(imdb_df)

    source_stats = {
        "aaec_ukp": aaec_labels,
        "fever": fever_labels,
        "imdb": imdb_labels,
    }

    merged = pd.concat([aaec_df, fever_df, imdb_df], ignore_index=True)
    merged = preprocess_rows(merged)

    mapped = merged.copy()
    mapping = mapped.apply(lambda r: map_with_reason(r.to_dict()), axis=1)
    mapped["label"] = [m[0] for m in mapping]
    mapped["mapping_reason"] = [m[1] for m in mapping]

    claim_pool = mapped[mapped["label"] == "Claim"].reset_index(drop=True)
    fact_pool = mapped[mapped["label"] == "Fact"].reset_index(drop=True)
    evidence_pool = mapped[mapped["label"] == "Evidence"].reset_index(drop=True)
    opinion_pool = mapped[mapped["label"] == "Opinion"].reset_index(drop=True)
    background_pool = mapped[mapped["label"] == "Background"].reset_index(drop=True)

    claim_sel = sample_with_quota(claim_pool, CLAIM_SOURCE_QUOTAS, TARGET_COUNTS["Claim"], args.seed)
    fact_sel = sample_with_quota(fact_pool, FACT_SOURCE_QUOTAS, TARGET_COUNTS["Fact"], args.seed)
    evidence_sel = sample_with_quota(
        evidence_pool, EVIDENCE_SOURCE_QUOTAS, TARGET_COUNTS["Evidence"], args.seed
    )
    opinion_sel = sample_with_quota(
        opinion_pool, OPINION_SOURCE_QUOTAS, TARGET_COUNTS["Opinion"], args.seed
    )
    background_sel = sample_with_quota(
        background_pool, BACKGROUND_SOURCE_QUOTAS, TARGET_COUNTS["Background"], args.seed
    )

    final_df = pd.concat(
        [claim_sel, fact_sel, evidence_sel, opinion_sel, background_sel],
        ignore_index=True,
    )

    final_df, clean_stats = clean_final(final_df)

    final_counts = final_df["label"].value_counts().reindex(LABELS_5, fill_value=0)
    for label in LABELS_5:
        need = TARGET_COUNTS[label] - int(final_counts[label])
        if need <= 0:
            continue

        pool = mapped[mapped["label"] == label].copy()
        used = set(final_df["sentence"].str.lower().tolist())
        pool = pool[~pool["sentence"].str.lower().isin(used)]
        if len(pool) == 0:
            continue

        refill = pool.sample(n=min(need, len(pool)), random_state=args.seed)
        final_df = pd.concat([final_df, refill], ignore_index=True)

    final_df, clean_stats_2 = clean_final(final_df)
    for k, v in clean_stats_2.items():
        clean_stats[k] = clean_stats.get(k, 0) + v

    train_df, val_df, test_df = stratified_split(final_df, seed=args.seed)

    fever_refutes_sample = final_df[
        (final_df["source"] == "fever") & (final_df["original_label"] == "REFUTES_claim")
    ].copy()
    if len(fever_refutes_sample) > 0:
        fever_refutes_len_mean = float(fever_refutes_sample["n_words"].mean())
        fever_refutes_len_median = float(fever_refutes_sample["n_words"].median())
    else:
        fever_refutes_len_mean = 0.0
        fever_refutes_len_median = 0.0

    leakage_stats = leakage_check_against_wikisqe(train_df)

    cols = ["sentence", "label", "source", "original_label"]
    train_df[cols].to_parquet(processed_dir / "stage1_train.parquet", index=False)
    val_df[cols].to_parquet(processed_dir / "stage1_val.parquet", index=False)
    test_df[cols].to_parquet(processed_dir / "stage1_test.parquet", index=False)

    render_report(
        report_path=report_path,
        full_df=final_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        source_label_stats=source_stats,
        clean_stats=clean_stats,
        leakage_stats=leakage_stats,
        imdb_candidate_count=imdb_candidate_count,
        fever_refutes_len_mean=fever_refutes_len_mean,
        fever_refutes_len_median=fever_refutes_len_median,
        seed=args.seed,
    )

    print("Stage 1 repaired data prepared.")
    print("Raw source label distributions:")
    for source, ctr in source_stats.items():
        print(f"- {source}: {dict(ctr)}")

    print("Final class counts:")
    final_counts = final_df["label"].value_counts().reindex(LABELS_5, fill_value=0)
    for label in LABELS_5:
        print(f"- {label}: {int(final_counts[label])}")
    print(f"FEVER REFUTES mean_len={fever_refutes_len_mean:.2f}, median_len={fever_refutes_len_median:.2f}")
    print(f"IMDB candidates after dual filter={imdb_candidate_count}")
    print(
        "Leakage check overlap="
        f"{leakage_stats['overlap_count']} / {leakage_stats['checked_wikisqe_rows']} checked WikiSQE rows"
    )

    print(f"Saved train: {processed_dir / 'stage1_train.parquet'}")
    print(f"Saved val: {processed_dir / 'stage1_val.parquet'}")
    print(f"Saved test: {processed_dir / 'stage1_test.parquet'}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
