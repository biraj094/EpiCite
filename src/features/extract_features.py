from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import spacy

FEATURE_NAMES = [
    "vague_quantifiers_ratio",
    "hedging_words_ratio",
    "superlative_adj_ratio",
    "passive_voice",
    "nominalization_rate",
    "pos_noun_ratio",
    "pos_verb_ratio",
    "pos_adj_ratio",
    "pos_adv_ratio",
    "sentence_position",
    "citation_density",
    "paragraph_relative_idx",
]

VAGUE_SINGLE = {
    "some",
    "many",
    "most",
    "several",
    "various",
    "numerous",
    "few",
}
VAGUE_MULTI = ["a lot of", "most of", "much of"]

HEDGING_WORDS = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "allegedly",
    "reportedly",
    "seemingly",
    "apparently",
    "suggest",
    "suggests",
    "appears",
    "likely",
    "probably",
}

NOMINALIZATION_SUFFIXES = ("tion", "ment", "ance", "ence", "ity", "ness")

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model en_core_web_sm is missing. Install via: python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_vague_quantifiers(text_lower: str) -> int:
    count = 0
    words = re.findall(r"\b\w+\b", text_lower)
    count += sum(1 for w in words if w in VAGUE_SINGLE)
    for phrase in VAGUE_MULTI:
        count += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
    return count


def _count_hedging_words(text_lower: str) -> int:
    words = re.findall(r"\b\w+\b", text_lower)
    return sum(1 for w in words if w in HEDGING_WORDS)


def extract_features(sentence: str, context: Optional[dict] = None) -> Dict[str, float]:
    nlp = _get_nlp()

    text = _normalize_space(str(sentence))
    n_words = _word_count(text)
    denom_words = max(n_words, 1)
    text_lower = text.lower()

    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    n_tokens = max(len(tokens), 1)

    vague_count = _count_vague_quantifiers(text_lower)
    hedge_count = _count_hedging_words(text_lower)
    superlative_count = sum(1 for t in tokens if t.tag_ == "JJS")

    passive_voice = 1.0 if any(t.dep_ == "nsubjpass" for t in doc) else 0.0

    noun_tokens = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"}]
    noun_count = len(noun_tokens)
    nominalization_count = sum(
        1
        for t in noun_tokens
        if t.lemma_.lower().endswith(NOMINALIZATION_SUFFIXES)
        or t.text.lower().endswith(NOMINALIZATION_SUFFIXES)
    )

    pos_noun = noun_count
    pos_verb = sum(1 for t in tokens if t.pos_ in {"VERB", "AUX"})
    pos_adj = sum(1 for t in tokens if t.pos_ == "ADJ")
    pos_adv = sum(1 for t in tokens if t.pos_ == "ADV")

    return {
        "vague_quantifiers_ratio": vague_count / denom_words,
        "hedging_words_ratio": hedge_count / denom_words,
        "superlative_adj_ratio": superlative_count / denom_words,
        "passive_voice": passive_voice,
        "nominalization_rate": (nominalization_count / noun_count) if noun_count > 0 else 0.0,
        "pos_noun_ratio": pos_noun / n_tokens,
        "pos_verb_ratio": pos_verb / n_tokens,
        "pos_adj_ratio": pos_adj / n_tokens,
        "pos_adv_ratio": pos_adv / n_tokens,
        "sentence_position": 0.5,
        "citation_density": 0.5,
        "paragraph_relative_idx": 0.5,
    }


def extract_features_batch(sentences: Iterable[str]) -> List[Dict[str, float]]:
    nlp = _get_nlp()
    texts = [_normalize_space(str(s)) for s in sentences]

    features: List[Dict[str, float]] = []
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=256), strict=False):
        n_words = _word_count(text)
        denom_words = max(n_words, 1)
        text_lower = text.lower()

        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        n_tokens = max(len(tokens), 1)

        vague_count = _count_vague_quantifiers(text_lower)
        hedge_count = _count_hedging_words(text_lower)
        superlative_count = sum(1 for t in tokens if t.tag_ == "JJS")

        passive_voice = 1.0 if any(t.dep_ == "nsubjpass" for t in doc) else 0.0

        noun_tokens = [t for t in tokens if t.pos_ in {"NOUN", "PROPN"}]
        noun_count = len(noun_tokens)
        nominalization_count = sum(
            1
            for t in noun_tokens
            if t.lemma_.lower().endswith(NOMINALIZATION_SUFFIXES)
            or t.text.lower().endswith(NOMINALIZATION_SUFFIXES)
        )

        pos_noun = noun_count
        pos_verb = sum(1 for t in tokens if t.pos_ in {"VERB", "AUX"})
        pos_adj = sum(1 for t in tokens if t.pos_ == "ADJ")
        pos_adv = sum(1 for t in tokens if t.pos_ == "ADV")

        features.append(
            {
                "vague_quantifiers_ratio": vague_count / denom_words,
                "hedging_words_ratio": hedge_count / denom_words,
                "superlative_adj_ratio": superlative_count / denom_words,
                "passive_voice": passive_voice,
                "nominalization_rate": (nominalization_count / noun_count) if noun_count > 0 else 0.0,
                "pos_noun_ratio": pos_noun / n_tokens,
                "pos_verb_ratio": pos_verb / n_tokens,
                "pos_adj_ratio": pos_adj / n_tokens,
                "pos_adv_ratio": pos_adv / n_tokens,
                "sentence_position": 0.5,
                "citation_density": 0.5,
                "paragraph_relative_idx": 0.5,
            }
        )

    return features


def add_features_to_parquet(input_path: Path, output_path: Path) -> Dict[str, float]:
    df = pd.read_parquet(input_path)
    start = time.perf_counter()
    feats = extract_features_batch(df["sentence"].astype(str).tolist())
    elapsed = time.perf_counter() - start
    feat_df = pd.DataFrame(feats)
    out = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    per_1k_sec = (elapsed / max(len(df), 1)) * 1000.0
    return {"rows": float(len(df)), "elapsed_sec": elapsed, "sec_per_1k": per_1k_sec}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add 12 linguistic features to a parquet file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = add_features_to_parquet(args.input, args.output)
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"rows={int(stats['rows'])}")
    print(f"elapsed_sec={stats['elapsed_sec']:.4f}")
    print(f"sec_per_1k={stats['sec_per_1k']:.4f}")


if __name__ == "__main__":
    main()
