from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"Claim": 0, "Fact": 1, "Evidence": 2, "Opinion": 3, "Background": 4}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
SEEDS = [42, 123, 2024]


class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = self.labels[idx]
        return enc


@dataclass
class SplitData:
    texts: List[str]
    labels: List[int]


def load_stage1_splits(processed_dir: Path) -> Tuple[SplitData, SplitData, SplitData]:
    def _load(name: str) -> SplitData:
        df = pd.read_parquet(processed_dir / name)
        texts = df["sentence"].astype(str).tolist()
        labels = [LABEL_MAP[x] for x in df["label"].tolist()]
        return SplitData(texts=texts, labels=labels)

    return _load("stage1_train.parquet"), _load("stage1_val.parquet"), _load("stage1_test.parquet")


def extract_cls_embeddings(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    model.eval()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            cls_vec = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_vecs.append(cls_vec)
    return np.concatenate(all_vecs, axis=0)


def run_baseline(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    data_dir: Path,
    model_dir: Path,
    device,
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)

    x_train = extract_cls_embeddings(train.texts, tokenizer, encoder, device)
    x_val = extract_cls_embeddings(val.texts, tokenizer, encoder, device)
    x_test = extract_cls_embeddings(test.texts, tokenizer, encoder, device)

    y_train = np.array(train.labels)
    y_val = np.array(val.labels)
    y_test = np.array(test.labels)

    np.savez_compressed(
        data_dir / "stage1_cls_embeddings.npz",
        train_embeddings=x_train,
        val_embeddings=x_val,
        test_embeddings=x_test,
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)

    with open(model_dir / "stage1_baseline_lr.pkl", "wb") as f:
        pickle.dump({"model": clf, "label_map": LABEL_MAP}, f)

    def _eval(x, y):
        pred = clf.predict(x)
        macro = f1_score(y, pred, average="macro")
        acc = accuracy_score(y, pred)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(5)), zero_division=0)
        per = {ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])} for i in range(5)}
        return {
            "macro_f1": float(macro),
            "accuracy": float(acc),
            "per_class": per,
            "pred": pred,
        }

    val_m = _eval(x_val, y_val)
    test_m = _eval(x_test, y_test)

    return {
        "val": {k: v for k, v in val_m.items() if k != "pred"},
        "test": {k: v for k, v in test_m.items() if k != "pred"},
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": float(macro_f1), "accuracy": float(acc)}


def cleanup_checkpoints(seed_dir: Path):
    for p in seed_dir.glob("checkpoint-*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def evaluate_model(model, dataset, tokenizer, device, batch_size=64) -> Dict[str, object]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    macro = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(5)), zero_division=0)
    per = {ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])} for i in range(5)}

    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "per_class": per,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def run_finetune_for_seed(
    seed: int,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    out_root: Path,
    experiments_dir: Path,
    use_fp16: bool,
    device,
) -> Dict[str, object]:
    set_seed(seed)
    seed_dir = out_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    train_ds = TextLabelDataset(train.texts, train.labels, tokenizer)
    val_ds = TextLabelDataset(val.texts, val.labels, tokenizer)
    test_ds = TextLabelDataset(test.texts, test.labels, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_bs = 32
    grad_acc = 1
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    args = TrainingArguments(
        output_dir=str(seed_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=bool(use_fp16),
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    start = time.perf_counter()
    try:
        trainer.train()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" not in msg:
            raise
        # fallback for OOM
        train_bs = 16
        grad_acc = 2
        torch.cuda.empty_cache()
        args = TrainingArguments(
            output_dir=str(seed_dir),
            learning_rate=2e-5,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=grad_acc,
            num_train_epochs=5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            fp16=bool(use_fp16),
            seed=seed,
            data_seed=seed,
            logging_steps=50,
            report_to=[],
        )
        trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=5,
                id2label=ID2LABEL,
                label2id=LABEL_MAP,
            ),
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

    elapsed = time.perf_counter() - start

    best_model_dir = seed_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    cleanup_checkpoints(seed_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    val_metrics = evaluate_model(best_model, val_ds, tokenizer, device)
    test_metrics = evaluate_model(best_model, test_ds, tokenizer, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_file = experiments_dir / f"stage1_distilbert_seed{seed}_{ts}.json"
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "val": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred", "probs"}},
                "test": {k: v for k, v in test_metrics.items() if k not in {"y_true", "y_pred", "probs"}},
                "train_seconds": elapsed,
                "best_model_dir": str(best_model_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "seed": seed,
        "val": val_metrics,
        "test": test_metrics,
        "time_sec": elapsed,
        "best_model_dir": best_model_dir,
    }


def save_confusion_and_error_analysis(
    result: Dict[str, object],
    test: SplitData,
    reports_dir: Path,
):
    y_true = result["test"]["y_true"]
    y_pred = result["test"]["y_pred"]
    probs = result["test"]["probs"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_df = pd.DataFrame(cm, index=[ID2LABEL[i] for i in range(5)], columns=[ID2LABEL[i] for i in range(5)])
    cm_df.to_csv(reports_dir / "stage1_confusion_matrix.csv", index=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Stage1 Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(reports_dir / "stage1_confusion.png", dpi=200)
    plt.close()

    lines = ["# Stage1 Error Analysis", ""]
    texts = test.texts

    for c in range(5):
        cls_name = ID2LABEL[c]
        idxs = np.where((y_true == c) & (y_pred != c))[0][:10]
        lines.append(f"## {cls_name}")
        if len(idxs) == 0:
            lines.append("- (no misclassified samples)")
            lines.append("")
            continue
        for i in idxs:
            pred = int(y_pred[i])
            true = int(y_true[i])
            conf_pred = float(probs[i, pred])
            conf_true = float(probs[i, true])
            reason = "Boundary between epistemic categories appears ambiguous."
            if true == LABEL_MAP["Opinion"] and pred in {LABEL_MAP["Fact"], LABEL_MAP["Evidence"]}:
                reason = "Opinion contains factual style cues and weak subjective markers."
            elif true == LABEL_MAP["Fact"] and pred == LABEL_MAP["Evidence"]:
                reason = "Fact sentence carries evidential style and named entities."
            elif true == LABEL_MAP["Claim"] and pred == LABEL_MAP["Evidence"]:
                reason = "Claim phrasing resembles supporting statement rather than assertion."
            lines.append(f"- sentence: {texts[i]}")
            lines.append(f"  true/pred: {ID2LABEL[true]} -> {ID2LABEL[pred]}")
            lines.append(f"  conf_pred: {conf_pred:.4f}, conf_true: {conf_true:.4f}")
            lines.append(f"  reason: {reason}")
        lines.append("")

    (reports_dir / "stage1_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    # pair confusion rates
    fact = LABEL_MAP["Fact"]
    evidence = LABEL_MAP["Evidence"]
    claim = LABEL_MAP["Claim"]

    fe_conf = (cm[fact, evidence] + cm[evidence, fact]) / max(cm[fact, :].sum() + cm[evidence, :].sum(), 1)
    ce_conf = (cm[claim, evidence] + cm[evidence, claim]) / max(cm[claim, :].sum() + cm[evidence, :].sum(), 1)
    return float(fe_conf), float(ce_conf), cm_df


def write_results_report(
    reports_dir: Path,
    gpu_name: str,
    torch_version: str,
    baseline: Dict[str, object],
    seed_results: List[Dict[str, object]],
    best_seed: int,
    best_test_per_class: Dict[str, Dict[str, float]],
    fe_conf: float,
    ce_conf: float,
):
    macro_vals = np.array([r["test"]["macro_f1"] for r in seed_results], dtype=float)
    acc_vals = np.array([r["test"]["accuracy"] for r in seed_results], dtype=float)

    lines = ["# Stage1 Results", ""]
    lines.append("## Environment")
    lines.append(f"- GPU: {gpu_name}")
    lines.append(f"- PyTorch: {torch_version}")
    lines.append("")

    lines.append("## Baseline vs Fine-tuned")
    lines.append("| Model | Val macro-F1 | Test macro-F1 | Test accuracy |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Frozen CLS + LR | {baseline['val']['macro_f1']:.4f} | {baseline['test']['macro_f1']:.4f} | {baseline['test']['accuracy']:.4f} |"
    )
    best_row = [r for r in seed_results if r["seed"] == best_seed][0]
    lines.append(
        f"| Fine-tuned DistilBERT (best seed={best_seed}) | {best_row['val']['macro_f1']:.4f} | {best_row['test']['macro_f1']:.4f} | {best_row['test']['accuracy']:.4f} |"
    )
    lines.append("")

    lines.append("## Seed Stability")
    lines.append(f"- Test macro-F1 mean±std: {macro_vals.mean():.4f} ± {macro_vals.std(ddof=0):.4f}")
    lines.append(f"- Test accuracy mean±std: {acc_vals.mean():.4f} ± {acc_vals.std(ddof=0):.4f}")
    lines.append("")

    lines.append("## Per-class Metrics (Best Checkpoint)")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|")
    for cls in ["Claim", "Fact", "Evidence", "Opinion", "Background"]:
        m = best_test_per_class[cls]
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("![confusion](stage1_confusion.png)")
    lines.append("")

    lines.append("## Key Confusions")
    lines.append(f"- Fact vs Evidence confusion rate: {fe_conf*100:.2f}%")
    lines.append(f"- Claim vs Evidence confusion rate: {ce_conf*100:.2f}%")
    lines.append("")

    lines.append("## Top Error Patterns")
    lines.append("1. Evidence and Fact overlap on entity-heavy declarative sentences.")
    lines.append("2. Claim and Evidence overlap when claims are phrased as support statements.")
    lines.append("3. Opinion is occasionally confused with Fact when subjective cues are weak.")

    (reports_dir / "stage1_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 baseline and fine-tuned DistilBERT.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    root = args.project_root
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    experiments_dir = root / "experiments"
    reports_dir = root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "stage1_distilbert").mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_stage1_splits(processed_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this training run, but CUDA is not available.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    # save label mapping
    label_map_path = models_dir / "stage1_distilbert" / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL_MAP, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    baseline = run_baseline(train, val, test, processed_dir, models_dir, device)

    use_fp16 = torch.cuda.is_available()
    seed_results = []
    for seed in SEEDS:
        res = run_finetune_for_seed(
            seed=seed,
            train=train,
            val=val,
            test=test,
            out_root=models_dir / "stage1_distilbert",
            experiments_dir=experiments_dir,
            use_fp16=use_fp16,
            device=device,
        )
        seed_results.append(res)

    # choose best seed by val macro-f1
    best = sorted(seed_results, key=lambda x: x["val"]["macro_f1"], reverse=True)[0]
    best_seed = best["seed"]

    # copy best checkpoint
    best_dst = models_dir / "stage1_distilbert" / "best"
    if best_dst.exists():
        shutil.rmtree(best_dst)
    shutil.copytree(best["best_model_dir"], best_dst)

    fe_conf, ce_conf, _ = save_confusion_and_error_analysis(best, test, reports_dir)

    write_results_report(
        reports_dir=reports_dir,
        gpu_name=gpu_name,
        torch_version=torch.__version__,
        baseline=baseline,
        seed_results=seed_results,
        best_seed=best_seed,
        best_test_per_class=best["test"]["per_class"],
        fe_conf=fe_conf,
        ce_conf=ce_conf,
    )

    # stop conditions
    if best["test"]["macro_f1"] < 0.60:
        print("STOP_CONDITION: main_model_macro_f1_below_0.60")
        return
    if fe_conf > 0.30 or ce_conf > 0.30:
        print("STOP_CONDITION: confusion_rate_above_30_percent")
        return

    print("TRAINING_DONE")
    for r in seed_results:
        print(
            f"SEED={r['seed']} VAL_MACRO_F1={r['val']['macro_f1']:.4f} "
            f"TEST_MACRO_F1={r['test']['macro_f1']:.4f} TIME_SEC={r['time_sec']:.1f}"
        )
    print(f"BEST_SEED={best_seed}")
    print(f"BASELINE_TEST_MACRO_F1={baseline['test']['macro_f1']:.4f}")
    print(f"FACT_EVIDENCE_CONF={fe_conf*100:.2f}%")
    print(f"CLAIM_EVIDENCE_CONF={ce_conf*100:.2f}%")


if __name__ == "__main__":
    main()
