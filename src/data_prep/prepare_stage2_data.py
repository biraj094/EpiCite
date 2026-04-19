from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from src.features.extract_features import FEATURE_NAMES, add_features_to_parquet

PTB_REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
}

POSITIVE_LABELS = [
    "citation needed",
    "dead link",
    "not in citation given",
    "verification needed",
    "original research",
    "better source needed",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def backup_stage1_to_v3(processed_dir: Path) -> None:
    archive = processed_dir / "archive" / "v3"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["stage1_train.parquet", "stage1_val.parquet", "stage1_test.parquet"]:
        src = processed_dir / name
        if src.exists():
            shutil.copy2(src, archive / name)


def clean_ptb_stage1(processed_dir: Path) -> int:
    affected = 0
    for name in ["stage1_train.parquet", "stage1_val.parquet", "stage1_test.parquet"]:
        path = processed_dir / name
        df = pd.read_parquet(path)
        if "sentence_raw" not in df.columns:
            df["sentence_raw"] = df["sentence"].astype(str)

        original = df["sentence"].astype(str)
        cleaned = original.copy()
        for old, new in PTB_REPLACEMENTS.items():
            cleaned = cleaned.str.replace(old, new, regex=False)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()

        affected += int((cleaned != original).sum())
        df["sentence"] = cleaned
        df.to_parquet(path, index=False)
    return affected


def _load_config_split(config: str, split: str):
    return load_dataset("ando55/WikiSQE_experiment", config, split=split)


def load_positive_negative_wikisqe() -> Tuple[Dict[str, pd.DataFrame], Dict[str, int], List[str]]:
    split_names = ["train", "val", "test"]
    pos_by_split = {s: [] for s in split_names}
    missing_configs: List[str] = []

    for cfg in POSITIVE_LABELS:
        try:
            for split in split_names:
                ds = _load_config_split(cfg, split)
                d = ds.to_pandas()[["text", "label"]]
                d = d[d["label"] == 1].copy()
                d["sentence"] = d["text"].astype(str).map(normalize_text)
                d["citation_needed"] = 1
                d["wikisqe_label"] = cfg
                pos_by_split[split].append(d[["sentence", "citation_needed", "wikisqe_label"]])
        except Exception:
            missing_configs.append(cfg)

    neg_by_split: Dict[str, pd.DataFrame] = {}
    for split in split_names:
        ds = _load_config_split("all", split)
        d = ds.to_pandas()[["text", "label"]]
        d = d[d["label"] == 0].copy()
        d["sentence"] = d["text"].astype(str).map(normalize_text)
        d["citation_needed"] = 0
        d["wikisqe_label"] = "none"
        neg_by_split[split] = d[["sentence", "citation_needed", "wikisqe_label"]]

    pos_frames = {
        split: (pd.concat(pos_by_split[split], ignore_index=True) if pos_by_split[split] else pd.DataFrame(columns=["sentence", "citation_needed", "wikisqe_label"]))
        for split in split_names
    }

    for split in split_names:
        pos_frames[split] = pos_frames[split].drop_duplicates(subset=["sentence", "wikisqe_label"])
        neg_by_split[split] = neg_by_split[split].drop_duplicates(subset=["sentence"])

    return (
        {
            "pos_train": pos_frames["train"],
            "pos_val": pos_frames["val"],
            "pos_test": pos_frames["test"],
            "neg_train": neg_by_split["train"],
            "neg_val": neg_by_split["val"],
            "neg_test": neg_by_split["test"],
        },
        {
            "pos_total": int(len(pos_frames["train"]) + len(pos_frames["val"]) + len(pos_frames["test"])),
            "neg_total": int(len(neg_by_split["train"]) + len(neg_by_split["val"]) + len(neg_by_split["test"])),
        },
        missing_configs,
    )


def _allocate_pos_targets(usable: Dict[str, int], pos_total_target: int) -> Dict[str, int]:
    keys = ["train", "val", "test"]
    total_usable = sum(usable[k] for k in keys)
    if total_usable == 0:
        return {k: 0 for k in keys}

    raw = {k: (usable[k] / total_usable) * pos_total_target for k in keys}
    base = {k: int(raw[k]) for k in keys}
    remain = pos_total_target - sum(base.values())

    frac = sorted(keys, key=lambda k: raw[k] - base[k], reverse=True)
    for k in frac:
        if remain <= 0:
            break
        if base[k] < usable[k]:
            base[k] += 1
            remain -= 1

    for k in keys:
        base[k] = min(base[k], usable[k])
    return base


def build_stage2_splits_official(
    data: Dict[str, pd.DataFrame],
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    pos = {
        "train": data["pos_train"],
        "val": data["pos_val"],
        "test": data["pos_test"],
    }
    neg = {
        "train": data["neg_train"],
        "val": data["neg_val"],
        "test": data["neg_test"],
    }

    usable_pos = {
        k: min(len(pos[k]), len(neg[k]) // 2) for k in ["train", "val", "test"]
    }

    max_total_rows = 3 * sum(usable_pos.values())
    target_total = min(30000, max_total_rows)
    if target_total < 15000:
        target_total = max_total_rows

    pos_target_total = target_total // 3
    pos_targets = _allocate_pos_targets(usable_pos, pos_target_total)

    split_frames = {}
    for split in ["train", "val", "test"]:
        p_n = pos_targets[split]
        n_n = p_n * 2
        p_df = pos[split].sample(n=p_n, random_state=seed) if p_n > 0 else pos[split].iloc[0:0]
        n_df = neg[split].sample(n=n_n, random_state=seed) if n_n > 0 else neg[split].iloc[0:0]

        out = pd.concat([p_df, n_df], ignore_index=True)
        out = out.drop_duplicates(subset=["sentence", "citation_needed", "wikisqe_label"])
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_frames[split] = out

    stats = {
        "target_total": int(target_total),
        "pos_target_total": int(sum(pos_targets.values())),
        "train": int(len(split_frames["train"])),
        "val": int(len(split_frames["val"])),
        "test": int(len(split_frames["test"])),
        "mode": "official",
    }
    return split_frames["train"], split_frames["val"], split_frames["test"], stats


def build_stage2_splits_stratified(
    data: Dict[str, pd.DataFrame],
    seed: int = 42,
    target_total: int = 30000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    pos_all = pd.concat([data["pos_train"], data["pos_val"], data["pos_test"]], ignore_index=True)
    neg_all = pd.concat([data["neg_train"], data["neg_val"], data["neg_test"]], ignore_index=True)

    max_pos = min(len(pos_all), len(neg_all) // 2)
    pos_target = min(target_total // 3, max_pos)
    neg_target = pos_target * 2

    pos_sample = pos_all.sample(n=pos_target, random_state=seed) if pos_target > 0 else pos_all.iloc[0:0]
    neg_sample = neg_all.sample(n=neg_target, random_state=seed) if neg_target > 0 else neg_all.iloc[0:0]

    merged = pd.concat([pos_sample, neg_sample], ignore_index=True)
    merged = merged.drop_duplicates(subset=["sentence", "citation_needed", "wikisqe_label"]).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        merged,
        test_size=0.2,
        random_state=seed,
        stratify=merged["citation_needed"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["citation_needed"],
    )

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stats = {
        "target_total": int(target_total),
        "pos_target_total": int(pos_target),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "mode": "stratified",
    }
    return train_df, val_df, test_df, stats


def remove_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage1_train_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    s1 = pd.read_parquet(stage1_train_path)
    s1_set = set(s1["sentence"].astype(str).map(normalize_text).str.lower().tolist())

    def _filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        norm = df["sentence"].astype(str).map(normalize_text).str.lower()
        mask = norm.isin(s1_set)
        removed = int(mask.sum())
        out = df.loc[~mask].reset_index(drop=True)
        return out, removed

    train_new, r1 = _filter(train_df)
    val_new, r2 = _filter(val_df)
    test_new, r3 = _filter(test_df)
    return train_new, val_new, test_new, (r1 + r2 + r3)


def save_stage2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
) -> None:
    keep_cols = ["sentence", "citation_needed", "wikisqe_label"]
    train_df[keep_cols].to_parquet(processed_dir / "stage2_train.parquet", index=False)
    val_df[keep_cols].to_parquet(processed_dir / "stage2_val.parquet", index=False)
    test_df[keep_cols].to_parquet(processed_dir / "stage2_test.parquet", index=False)


def add_features_all(processed_dir: Path) -> Dict[str, Dict[str, float]]:
    inputs = [
        "stage1_train.parquet",
        "stage1_val.parquet",
        "stage1_test.parquet",
        "stage2_train.parquet",
        "stage2_val.parquet",
        "stage2_test.parquet",
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for name in inputs:
        input_path = processed_dir / name
        output_path = processed_dir / name.replace(".parquet", "_with_features.parquet")
        stats[name] = add_features_to_parquet(input_path, output_path)
    return stats


def stage2_feature_stats(processed_dir: Path) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    df = pd.read_parquet(processed_dir / "stage2_train_with_features.parquet")
    rows = []
    high_zero = []

    for feat in FEATURE_NAMES:
        series = df[feat].astype(float)
        zero_rate = float((series == 0).mean())
        rows.append(
            {
                "feature": feat,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "zero_rate": zero_rate,
            }
        )
        if zero_rate > 0.8:
            high_zero.append(feat)

    sample = df.sample(n=min(3, len(df)), random_state=42).reset_index(drop=True)
    return pd.DataFrame(rows), high_zero, sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 data and feature engineering outputs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-official-split",
        action="store_true",
        help="Use official split structure; default is disabled and uses stratified 80/10/10.",
    )
    args = parser.parse_args()

    root = args.project_root
    processed_dir = root / "data" / "processed"

    backup_stage1_to_v3(processed_dir)
    affected = clean_ptb_stage1(processed_dir)

    wq_data, counts, missing_configs = load_positive_negative_wikisqe()
    if args.use_official_split:
        s2_train, s2_val, s2_test, split_stats = build_stage2_splits_official(
            wq_data, seed=args.seed
        )
    else:
        s2_train, s2_val, s2_test, split_stats = build_stage2_splits_stratified(
            wq_data, seed=args.seed, target_total=30000
        )

    s2_train, s2_val, s2_test, removed_overlap = remove_leakage(
        s2_train,
        s2_val,
        s2_test,
        processed_dir / "stage1_train.parquet",
    )

    save_stage2(s2_train, s2_val, s2_test, processed_dir)

    feat_timing = add_features_all(processed_dir)

    feat_stats, high_zero, sample_rows = stage2_feature_stats(processed_dir)

    final_all = pd.concat([s2_train, s2_val, s2_test], ignore_index=True)
    pos_n = int((final_all["citation_needed"] == 1).sum())
    neg_n = int((final_all["citation_needed"] == 0).sum())

    print(f"ptb_affected_sentences={affected}")
    print(f"missing_positive_label_configs={missing_configs}")
    print(f"wikisqe_available_counts={counts}")
    print(f"stage2_split_stats_before_leakage={split_stats}")
    print(f"stage2_overlap_removed={removed_overlap}")
    print(f"stage2_final_pos={pos_n}")
    print(f"stage2_final_neg={neg_n}")
    print(f"stage2_final_ratio_pos_to_neg=1:{(neg_n / max(pos_n, 1)):.4f}")
    print("feature_timing_sec_per_1k=")
    for k, v in feat_timing.items():
        print(f"  {k}: {v['sec_per_1k']:.4f}")

    print("stage2_train_feature_stats=")
    for _, row in feat_stats.iterrows():
        print(
            f"  {row['feature']}: mean={row['mean']:.6f}, std={row['std']:.6f}, min={row['min']:.6f}, max={row['max']:.6f}, zero_rate={row['zero_rate']:.6f}"
        )

    print(f"high_zero_features={high_zero}")
    print("sample_feature_rows=")
    for i, row in sample_rows.iterrows():
        payload = {feat: float(row[feat]) for feat in FEATURE_NAMES}
        print(f"  idx={i}, sentence={row['sentence']}")
        print(f"  features={payload}")


if __name__ == "__main__":
    main()
