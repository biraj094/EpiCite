"""
EpiCite — Streamlit demo
========================
Two-stage citation-worthiness analyzer:
  Stage 1 — DistilBERT 4-class epistemic role classifier
            (Claim / Evidence / Opinion / Background)
  Stage 2 — Late-Fusion model: DistilBERT [CLS] + 34 engineered features
            -> sigmoid citation-worthiness score

Features:
  - Mobile-friendly layout (single-column, tabs)
  - Pre-selected example sentences
  - User-configurable model directory (sidebar)
  - Two-stage inference (Stage 1 -> features -> Stage 2)
  - Interpretability:
        * Token-level Integrated Gradients (Captum, optional)
        * Last-layer attention heatmap (CLS -> tokens)
        * Engineered feature values vs. dataset distribution
        * Stage 1 epistemic probability bar chart

Run:
    streamlit run app.py

Author: Streamlit demo for Group 12 (AIT NLP) — EpiCite
"""

from __future__ import annotations

import json
import os
import re
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page config — must be first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EpiCite Demo",
    page_icon="📚",
    layout="centered",  # centered = mobile-friendly
    initial_sidebar_state="auto",
)

# -----------------------------------------------------------------------------
# Constants — kept identical to Notebook 05 so checkpoints load cleanly
# -----------------------------------------------------------------------------
CLASSES = ["Claim", "Evidence", "Opinion", "Background"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
NUM_CLASSES = len(CLASSES)
MODEL_NAME = "distilbert-base-uncased"

CLASS_COLORS = {
    "Claim": "#4C72B0",
    "Evidence": "#55A868",
    "Opinion": "#C44E52",
    "Background": "#8172B2",
}

BASELINE_COLS = [
    "n_hedge_verbs", "n_hedge_aux", "n_hedge_adv", "n_vague_quant",
    "n_boosters", "n_subjective_pron", "n_negations", "n_passive",
    "n_proper_nouns", "n_numbers", "n_modals", "sentence_length",
]
V2_NEW_COLS = [
    "rate_percent", "rate_dollar", "rate_year", "rate_digit", "rate_inline_cite",
    "rate_ent_person", "rate_ent_org", "rate_ent_gpe", "rate_ent_date", "rate_ent_total",
    "rate_universal_quant", "rate_superlative",
    "rate_reporting_verbs", "rate_causal_markers",
    "dep_tree_depth", "n_clauses", "noun_verb_ratio",
]
EPI_COLS = [f"epi_prob_{c}" for c in CLASSES] + ["epi_confidence"]
DEFAULT_FULL_COLS = BASELINE_COLS + V2_NEW_COLS + EPI_COLS  # 34-dim default

# Lexicons (must match Notebook 05 exactly)
HEDGE_VERBS = ["seem", "seems", "seemed", "appear", "appears", "appeared",
               "suggest", "suggests", "suggested", "indicate", "indicates", "indicated",
               "propose", "tend", "tends", "tended", "believe", "believes", "believed",
               "assume", "assumes", "assumed", "speculate", "speculates", "speculated",
               "hypothesize", "estimate", "estimates", "estimated", "report", "reports", "reported"]
HEDGE_AUX = ["may", "might", "could", "would", "should", "can"]
HEDGE_ADV = ["possibly", "probably", "perhaps", "allegedly", "reportedly", "presumably",
             "arguably", "apparently", "supposedly", "ostensibly", "likely", "unlikely",
             "generally", "typically", "usually", "often", "sometimes", "occasionally",
             "roughly", "approximately", "potentially", "conceivably", "purportedly"]
VAGUE_QUANT = ["some", "many", "several", "a lot", "most", "much", "few", "various",
               "numerous", "plenty", "considerable", "significant"]
BOOSTERS = ["clearly", "obviously", "certainly", "definitely", "undoubtedly", "surely"]
SUBJECTIVE = ["i", "we", "my", "our", "myself", "ourselves", "me", "us"]
UNIVERSAL_QUANT = ["all", "every", "any", "always", "never", "none", "no one",
                   "everyone", "everywhere", "everything", "nothing"]
SUPERLATIVE_LEX = ["first", "largest", "smallest", "best", "worst", "most", "least",
                   "highest", "lowest", "biggest", "greatest", "oldest", "newest",
                   "fastest", "slowest", "strongest", "weakest", "only", "unique"]
REPORTING_VERBS = ["according", "reports", "reported", "stated", "states", "said",
                   "claims", "claimed", "argues", "argued", "wrote", "writes", "cites",
                   "cited", "notes", "noted", "observed", "observes"]
CAUSAL_MARKERS = ["causes", "caused", "leads to", "results in", "because", "therefore",
                  "thus", "consequently", "hence", "due to", "triggers", "triggered",
                  "produces", "produced", "contributes to"]

LEX = {
    "hedge_verbs": {w.lower() for w in HEDGE_VERBS},
    "hedge_aux": {w.lower() for w in HEDGE_AUX},
    "hedge_adv": {w.lower() for w in HEDGE_ADV},
    "vague_quant": {w.lower() for w in VAGUE_QUANT},
    "boosters": {w.lower() for w in BOOSTERS},
    "subjective": {w.lower() for w in SUBJECTIVE},
    "universal_quant": {w.lower() for w in UNIVERSAL_QUANT},
    "superlative": {w.lower() for w in SUPERLATIVE_LEX},
    "reporting_verbs": {w.lower() for w in REPORTING_VERBS},
    "causal_markers": {w.lower() for w in CAUSAL_MARKERS},
}

RE_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
RE_DOLLAR = re.compile(r"\$\s*\d")
RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
RE_DIGIT = re.compile(r"\b\d+(?:[.,]\d+)*\b")
RE_INLINE_CITE = re.compile(r"\[\s*\d+\s*\]")

# -----------------------------------------------------------------------------
# Pre-selected demo sentences
# -----------------------------------------------------------------------------
EXAMPLES: List[Tuple[str, str]] = [
    (
        "📊 Statistic with citation hint",
        "Studies show that approximately 23% of patients with diabetes "
        "improved after the new treatment in 2023.",
    ),
    (
        "🌍 WHO claim with year & percentage",
        "According to the WHO, malaria deaths fell by 30% between 2000 and 2015.",
    ),
    (
        "🏛️ Famous painting fact",
        "The Mona Lisa was painted by Leonardo da Vinci.",
    ),
    (
        "🎨 Subjective opinion",
        "It is widely considered the most famous painting in the world.",
    ),
    (
        "🗽 Background description",
        "It is currently housed in the Louvre.",
    ),
    (
        "🗼 Numerical claim with year",
        "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    ),
    (
        "💬 Hedged opinion",
        "Many visitors find it beautiful at night.",
    ),
    (
        "🧪 Causal scientific claim",
        "Smoking causes lung cancer in approximately 80% of cases reported in 2019.",
    ),
    (
        "📰 Reporting verb (low cite need)",
        "The president said he was happy with the results.",
    ),
    (
        "🎯 Strong assertion (high cite need)",
        "Climate change has caused sea levels to rise by 21cm since 1880.",
    ),
]

# -----------------------------------------------------------------------------
# Heavy imports — done once on app start so failures are diagnosable
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


@st.cache_resource(show_spinner=False)
def _import_transformers():
    from transformers import (
        DistilBertTokenizerFast, DistilBertModel, DistilBertForSequenceClassification,
    )
    return DistilBertTokenizerFast, DistilBertModel, DistilBertForSequenceClassification


@st.cache_resource(show_spinner=False)
def _import_spacy():
    """Try to import & load spaCy. Return None if unavailable so we can fall back."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None
        return nlp
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _import_captum():
    try:
        from captum.attr import LayerIntegratedGradients
        return LayerIntegratedGradients
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Late-Fusion Model (must match Notebook 05 exactly)
# -----------------------------------------------------------------------------
def make_late_fusion_class():
    torch, nn, F = _import_torch()

    class LateFusionCitationScorer(nn.Module):
        def __init__(self, distilbert, num_eng_features: int, dropout: float = 0.3):
            super().__init__()
            self.distilbert = distilbert
            self.cls_dim = distilbert.config.hidden_size
            self.eng_dim = num_eng_features
            fused_dim = self.cls_dim + self.eng_dim
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, 256), nn.GELU(), nn.Dropout(dropout),
                nn.LayerNorm(256),
                nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, input_ids, attention_mask, eng_features):
            out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            fused = torch.cat([cls, eng_features], dim=-1)
            return torch.sigmoid(self.head(fused)).squeeze(-1)

    return LateFusionCitationScorer


# -----------------------------------------------------------------------------
# Feature extraction (mirrors Notebook 05 extract_features_full)
# -----------------------------------------------------------------------------
def _dep_depth(token) -> int:
    if not list(token.children):
        return 1
    return 1 + max(_dep_depth(c) for c in token.children)


def extract_features(text: str, nlp) -> Dict[str, float]:
    """Full v2 feature extractor. If spaCy unavailable, returns degraded approximation."""
    out: Dict[str, float] = {c: 0.0 for c in DEFAULT_FULL_COLS}

    if nlp is None:
        # Degraded fallback — only regex & whitespace features
        toks = text.split()
        n = max(len(toks), 1)
        lower = [t.lower() for t in toks]
        out["sentence_length"] = float(n)
        out["n_hedge_verbs"] = float(sum(1 for w in lower if w in LEX["hedge_verbs"]))
        out["n_hedge_aux"] = float(sum(1 for w in lower if w in LEX["hedge_aux"]))
        out["n_hedge_adv"] = float(sum(1 for w in lower if w in LEX["hedge_adv"]))
        out["n_vague_quant"] = float(sum(1 for w in lower if w in LEX["vague_quant"]))
        out["n_boosters"] = float(sum(1 for w in lower if w in LEX["boosters"]))
        out["n_subjective_pron"] = float(sum(1 for w in lower if w in LEX["subjective"]))
        out["rate_percent"] = len(RE_PERCENT.findall(text)) / n
        out["rate_dollar"] = len(RE_DOLLAR.findall(text)) / n
        out["rate_year"] = len(RE_YEAR.findall(text)) / n
        out["rate_digit"] = len(RE_DIGIT.findall(text)) / n
        out["rate_inline_cite"] = len(RE_INLINE_CITE.findall(text)) / n
        text_lower = text.lower()
        causal_count = sum(text_lower.count(m) for m in LEX["causal_markers"])
        out["rate_causal_markers"] = causal_count / n
        return out

    # Full spaCy path
    doc = nlp(str(text))
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    lower = [t.text.lower() for t in toks]

    out["n_hedge_verbs"] = sum(1 for w in lower if w in LEX["hedge_verbs"])
    out["n_hedge_aux"] = sum(1 for w in lower if w in LEX["hedge_aux"])
    out["n_hedge_adv"] = sum(1 for w in lower if w in LEX["hedge_adv"])
    out["n_vague_quant"] = sum(1 for w in lower if w in LEX["vague_quant"])
    out["n_boosters"] = sum(1 for w in lower if w in LEX["boosters"])
    out["n_subjective_pron"] = sum(1 for w in lower if w in LEX["subjective"])
    out["n_negations"] = sum(1 for t in doc if t.dep_ == "neg")
    out["n_passive"] = sum(1 for t in doc if t.dep_ == "auxpass")
    out["n_proper_nouns"] = sum(1 for t in doc if t.pos_ == "PROPN")
    out["n_numbers"] = sum(1 for t in doc if t.like_num)
    out["n_modals"] = sum(1 for t in doc if t.tag_ == "MD")
    out["sentence_length"] = float(n)

    out["rate_percent"] = len(RE_PERCENT.findall(text)) / n
    out["rate_dollar"] = len(RE_DOLLAR.findall(text)) / n
    out["rate_year"] = len(RE_YEAR.findall(text)) / n
    out["rate_digit"] = len(RE_DIGIT.findall(text)) / n
    out["rate_inline_cite"] = len(RE_INLINE_CITE.findall(text)) / n

    ent_labels = [e.label_ for e in doc.ents]
    out["rate_ent_person"] = ent_labels.count("PERSON") / n
    out["rate_ent_org"] = ent_labels.count("ORG") / n
    out["rate_ent_gpe"] = ent_labels.count("GPE") / n
    out["rate_ent_date"] = ent_labels.count("DATE") / n
    out["rate_ent_total"] = len(ent_labels) / n

    out["rate_universal_quant"] = sum(1 for w in lower if w in LEX["universal_quant"]) / n
    out["rate_superlative"] = sum(1 for w in lower if w in LEX["superlative"]) / n
    out["rate_reporting_verbs"] = sum(1 for w in lower if w in LEX["reporting_verbs"]) / n

    text_lower = text.lower()
    causal_count = sum(text_lower.count(m) for m in LEX["causal_markers"])
    out["rate_causal_markers"] = causal_count / n

    roots = [t for t in doc if t.dep_ == "ROOT"]
    out["dep_tree_depth"] = float(max((_dep_depth(r) for r in roots), default=1))
    out["n_clauses"] = float(sum(1 for t in doc if t.dep_ in {"ccomp", "xcomp", "advcl", "relcl"}) + 1)
    n_noun = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    n_verb = sum(1 for t in doc if t.pos_ == "VERB")
    out["noun_verb_ratio"] = n_noun / max(n_verb, 1)

    return out


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _resolve_paths(out_dir: str) -> Dict[str, Path]:
    out = Path(out_dir).expanduser().resolve()
    ckpt_dir = Path("best-pt")
    return {
        "out_dir": out,
        "stage1_ckpt":  ckpt_dir / "stage1v3_focal_best.pt",
        "stage2_ckpt": ckpt_dir / "stage2v3_late_fusion_best.pt",
        "stage2_no_epi_ckpt": ckpt_dir / "stage2v3_no_epi_best.pt",
        "metadata": out / "stage2v3_metadata.json",
        "feat_csv": out / "stage2v3_engineered_features.csv",
    }


@st.cache_resource(show_spinner="Loading models (one-time)...")
def load_artifacts(out_dir: str) -> Dict[str, Any]:
    """Load Stage 1, Stage 2 (full), tokenizer, and metadata. Cached on out_dir."""
    torch, nn, F = _import_torch()
    DistilBertTokenizerFast, DistilBertModel, DistilBertForSequenceClassification = _import_transformers()
    LateFusionCitationScorer = make_late_fusion_class()

    paths = _resolve_paths(out_dir)
    artifacts: Dict[str, Any] = {"paths": paths, "errors": []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts["device"] = device

    # ---- Tokenizer (downloads from HF if needed) ----
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    except Exception as e:
        artifacts["errors"].append(f"Tokenizer load failed: {e}")
        return artifacts
    artifacts["tokenizer"] = tokenizer

    # ---- Metadata (optional) ----
    eng_cols_full = DEFAULT_FULL_COLS
    if paths["metadata"].exists():
        try:
            with open(paths["metadata"]) as f:
                meta = json.load(f)
            eng_cols_full = meta.get("stage2", {}).get("feature_cols_full", DEFAULT_FULL_COLS)
            artifacts["metadata"] = meta
        except Exception as e:
            artifacts["errors"].append(f"metadata.json present but unreadable: {e}")
    artifacts["eng_cols_full"] = eng_cols_full

    # ---- Stage 1: 4-class DistilBERT classifier ----
    if paths["stage1_ckpt"].exists():
        try:
            s1_model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_CLASSES,
                id2label=ID2LABEL, label2id=LABEL2ID,
                output_attentions=True,  # for the attention heatmap
            ).to(device)
            state = torch.load(paths["stage1_ckpt"], map_location=device)
            s1_model.load_state_dict(state, strict=False)
            s1_model.eval()
            artifacts["s1_model"] = s1_model
        except Exception as e:
            artifacts["errors"].append(f"Stage 1 checkpoint load failed: {e}")
    else:
        artifacts["errors"].append(f"Stage 1 checkpoint not found: {paths['stage1_ckpt']}")

    # ---- Stage 2: Late-fusion model ----
    if paths["stage2_ckpt"].exists():
        try:
            distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
            s2_model = LateFusionCitationScorer(
                distilbert, num_eng_features=len(eng_cols_full),
            ).to(device)
            state = torch.load(paths["stage2_ckpt"], map_location=device)
            s2_model.load_state_dict(state, strict=False)
            s2_model.eval()
            artifacts["s2_model"] = s2_model
        except Exception as e:
            artifacts["errors"].append(f"Stage 2 checkpoint load failed: {e}")
    else:
        artifacts["errors"].append(f"Stage 2 checkpoint not found: {paths['stage2_ckpt']}")

    # ---- Engineered features dataset stats (for normalization in viz) ----
    if paths["feat_csv"].exists():
        try:
            feat_df = pd.read_csv(paths["feat_csv"])
            # Robust z-score reference: median + IQR
            cols = [c for c in eng_cols_full if c in feat_df.columns]
            artifacts["feat_stats"] = {
                "mean": feat_df[cols].mean().to_dict(),
                "std": feat_df[cols].std().to_dict() if len(feat_df) > 1 else {c: 1.0 for c in cols},
                "max": feat_df[cols].max().to_dict(),
            }
        except Exception as e:
            artifacts["errors"].append(f"Feature CSV present but unreadable: {e}")

    return artifacts


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_pipeline(
    sentence: str,
    artifacts: Dict[str, Any],
    nlp,
    decision_threshold: float = 0.5,
    max_len: int = 128,
) -> Dict[str, Any]:
    torch, nn, F = _import_torch()
    tokenizer = artifacts["tokenizer"]
    device = artifacts["device"]
    eng_cols_full = artifacts["eng_cols_full"]
    s1_model = artifacts.get("s1_model")
    s2_model = artifacts.get("s2_model")

    enc = tokenizer(
        sentence, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)

    result: Dict[str, Any] = {"sentence": sentence}

    # ---- Stage 1 inference ----
    if s1_model is None:
        result["error"] = "Stage 1 model not loaded"
        return result

    with torch.no_grad():
        s1_out = s1_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        s1_probs = F.softmax(s1_out.logits, dim=-1).squeeze(0).cpu().numpy()

    epi_probs = {c: float(s1_probs[i]) for i, c in enumerate(CLASSES)}
    epi_class = ID2LABEL[int(s1_probs.argmax())]
    epi_confidence = float(s1_probs.max())

    result["epi_probs"] = epi_probs
    result["epi_class"] = epi_class
    result["epi_confidence"] = epi_confidence

    # Stage 1 attentions (last-layer, mean over heads, CLS row)
    if hasattr(s1_out, "attentions") and s1_out.attentions:
        # tuple of (B, H, T, T) per layer; take last layer
        last_attn = s1_out.attentions[-1].squeeze(0)  # (H, T, T)
        cls_attn = last_attn.mean(dim=0)[0].cpu().numpy()  # CLS row averaged over heads
        result["cls_attention"] = cls_attn

    # ---- Engineered features ----
    feat_dict = extract_features(sentence, nlp)
    # Inject Stage 1 epi probs into feature vector
    for i, c in enumerate(CLASSES):
        feat_dict[f"epi_prob_{c}"] = float(s1_probs[i])
    feat_dict["epi_confidence"] = epi_confidence
    eng_vec = np.array([float(feat_dict.get(c, 0.0)) for c in eng_cols_full], dtype=np.float32)
    result["feat_dict"] = feat_dict
    result["eng_vec"] = eng_vec

    # ---- Stage 2 inference ----
    if s2_model is not None:
        with torch.no_grad():
            feat_t = torch.tensor(eng_vec, dtype=torch.float32, device=device).unsqueeze(0)
            score = float(s2_model(enc["input_ids"], enc["attention_mask"], feat_t).item())
        result["citation_score"] = score
        result["needs_citation"] = bool(score >= decision_threshold)
    else:
        result["citation_score"] = None
        result["needs_citation"] = None

    # ---- Token list (for visualizations) ----
    ids = enc["input_ids"].squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    pad_id = tokenizer.pad_token_id
    keep_mask = ids != pad_id
    result["tokens"] = [t for t, k in zip(tokens, keep_mask) if k]
    result["enc"] = enc
    return result


# -----------------------------------------------------------------------------
# Interpretability — Integrated Gradients (Captum)
# -----------------------------------------------------------------------------
def compute_ig_attribution(
    sentence: str,
    eng_vec: np.ndarray,
    artifacts: Dict[str, Any],
    n_steps: int = 30,
    max_len: int = 128,
) -> Optional[List[Tuple[str, float]]]:
    LayerIntegratedGradients = _import_captum()
    if LayerIntegratedGradients is None:
        return None
    s2_model = artifacts.get("s2_model")
    if s2_model is None:
        return None

    torch, nn, F = _import_torch()
    tokenizer = artifacts["tokenizer"]
    device = artifacts["device"]

    class _ScoreOnly(nn.Module):
        def __init__(self, lf):
            super().__init__()
            self.lf = lf

        def forward(self, input_ids, attention_mask, eng_features):
            return self.lf(input_ids, attention_mask, eng_features)

    score_only = _ScoreOnly(s2_model).to(device).eval()
    lig = LayerIntegratedGradients(score_only, score_only.lf.distilbert.embeddings)

    enc = tokenizer(
        sentence, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    feat_t = torch.tensor(eng_vec, dtype=torch.float32, device=device).unsqueeze(0)
    ref_ids = torch.zeros_like(enc["input_ids"]).fill_(tokenizer.pad_token_id)

    try:
        attrs = lig.attribute(
            inputs=enc["input_ids"], baselines=ref_ids,
            additional_forward_args=(enc["attention_mask"], feat_t),
            n_steps=n_steps, return_convergence_delta=False,
        )
    except Exception as e:
        st.warning(f"Integrated Gradients failed: {e}")
        return None

    attrs_token = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    if np.abs(attrs_token).max() > 0:
        attrs_token = attrs_token / np.abs(attrs_token).max()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).cpu().numpy())
    return [(t, float(a)) for t, a in zip(tokens, attrs_token) if t != tokenizer.pad_token]


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def _color_for_weight(w: float, vmax: float = 1.0) -> str:
    """Map [-1, 1] weight to a CSS rgba background. Red=positive, blue=negative."""
    if vmax <= 0:
        vmax = 1.0
    intensity = min(abs(w) / vmax, 1.0)
    alpha = 0.15 + 0.65 * intensity
    if w >= 0:
        return f"rgba(220, 60, 60, {alpha:.3f})"  # red = pushes toward citation-needed
    return f"rgba(60, 110, 220, {alpha:.3f})"  # blue = pushes away


def render_token_attribution(tokens_with_weights: List[Tuple[str, float]]) -> str:
    """Return HTML with each token highlighted by its attribution weight."""
    if not tokens_with_weights:
        return "<i>No attributions to display.</i>"
    weights = np.array([w for _, w in tokens_with_weights])
    vmax = float(np.abs(weights).max()) if len(weights) else 1.0
    parts: List[str] = []
    for tok, w in tokens_with_weights:
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        # WordPiece — preserve "##" continuation visually
        display = tok[2:] if tok.startswith("##") else " " + tok
        bg = _color_for_weight(w, vmax)
        parts.append(
            f'<span style="background:{bg}; padding:2px 3px; border-radius:4px;'
            f' margin:1px; display:inline-block;" title="{w:+.3f}">'
            f'{escape(display)}</span>'
        )
    legend = (
        '<div style="margin-top:10px; font-size:0.8em; color:#666;">'
        '<span style="background:rgba(220,60,60,0.65); padding:2px 6px; border-radius:4px;">redder</span>'
        ' = pushes toward "needs citation" &nbsp;·&nbsp; '
        '<span style="background:rgba(60,110,220,0.65); padding:2px 6px; border-radius:4px;">bluer</span>'
        ' = pushes away'
        '</div>'
    )
    return (
        '<div style="line-height:2; padding:8px; background:#fafafa;'
        ' border-radius:8px; border:1px solid #eee;">'
        + "".join(parts)
        + "</div>"
        + legend
    )


def render_attention_strip(tokens: List[str], attn: np.ndarray) -> str:
    """Visualize CLS-attention weights over tokens as colored strip."""
    if attn is None or len(tokens) == 0:
        return "<i>No attention data.</i>"
    keep = [(t, a) for t, a in zip(tokens, attn[: len(tokens)])
            if t not in ("[CLS]", "[SEP]", "[PAD]")]
    if not keep:
        return "<i>(empty)</i>"
    weights = np.array([a for _, a in keep])
    vmax = float(weights.max()) if len(weights) else 1.0
    parts: List[str] = []
    for tok, w in keep:
        display = tok[2:] if tok.startswith("##") else " " + tok
        intensity = min(w / vmax, 1.0) if vmax > 0 else 0.0
        alpha = 0.10 + 0.7 * intensity
        bg = f"rgba(85, 168, 104, {alpha:.3f})"
        parts.append(
            f'<span style="background:{bg}; padding:2px 3px; border-radius:4px;'
            f' margin:1px; display:inline-block;" title="attn={w:.3f}">'
            f'{escape(display)}</span>'
        )
    legend = (
        '<div style="margin-top:10px; font-size:0.8em; color:#666;">'
        'darker green = higher attention from [CLS]'
        '</div>'
    )
    return (
        '<div style="line-height:2; padding:8px; background:#fafafa;'
        ' border-radius:8px; border:1px solid #eee;">'
        + "".join(parts)
        + "</div>"
        + legend
    )


def render_score_bar(score: float, threshold: float = 0.5) -> str:
    pct = max(0.0, min(1.0, score))
    bar_color = "#C44E52" if pct >= threshold else "#55A868"
    label = "NEEDS CITATION" if pct >= threshold else "OK as-is"
    pos = pct * 100
    th_pos = threshold * 100
    return f"""
<div style="margin: 8px 0;">
  <div style="display:flex; justify-content:space-between; font-size:0.9em; color:#666; margin-bottom:4px;">
    <span><b style="color:{bar_color}; font-size:1.1em;">{label}</b></span>
    <span><b>{pct:.1%}</b> citation worthiness</span>
  </div>
  <div style="position:relative; height:18px; background:#eee; border-radius:9px; overflow:hidden;">
    <div style="width:{pos:.2f}%; height:100%; background:{bar_color}; transition:width .3s ease;"></div>
    <div style="position:absolute; left:{th_pos:.2f}%; top:0; bottom:0; width:2px; background:#444;"></div>
  </div>
  <div style="font-size:0.75em; color:#888; margin-top:2px;">
    Vertical line = decision threshold ({threshold:.2f})
  </div>
</div>
"""


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main() -> None:
    # Custom CSS for compact, mobile-friendly layout
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 720px; }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stTabs [data-baseweb="tab"] { padding: 6px 10px; }
        .stTextArea textarea { font-size: 0.95rem; }
        @media (max-width: 600px) {
            .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
            h1 { font-size: 1.4rem !important; }
            .stTabs [data-baseweb="tab"] { padding: 4px 6px; font-size: 0.85em; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Header ---
    st.title("📚 EpiCite")
    st.caption(
        "Two-stage citation-worthiness analyzer · Group 12, AIT NLP · "
        "DistilBERT-based epistemic role classifier + Late-Fusion citation scorer"
    )

    # --- Sidebar: model paths & settings ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.markdown("**Model directory**")
        st.caption(
            "Folder containing `stage1v3_focal_best.pt`, `stage2v3_late_fusion_best.pt`, "
            "and (optionally) `stage2v3_metadata.json`. "
            "Either an absolute path or one relative to the working dir."
        )
        default_dir = st.session_state.get("out_dir", "output")
        out_dir = st.text_input(
            "Path",
            value=default_dir,
            help="e.g. /home/user/EpiCite/output  or  ./output",
            key="out_dir_input",
        )

        if st.button("🔄 Reload models", use_container_width=True):
            load_artifacts.clear()
            st.session_state["out_dir"] = out_dir
            st.rerun()

        st.session_state["out_dir"] = out_dir

        st.divider()
        st.markdown("**Inference settings**")
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
        max_len = st.slider("Max token length", 32, 256, 128, 16)
        show_ig = st.checkbox(
            "Token attribution (Integrated Gradients)",
            value=True,
            help="Slower (~2-5s per sentence). Requires `captum`.",
        )
        ig_steps = st.slider("IG steps", 10, 80, 30, 10, disabled=not show_ig)
        show_attn = st.checkbox("Stage 1 attention heatmap", value=True)
        st.divider()
        st.caption(
            "💡 If the app says checkpoints are missing, edit the path above and "
            "click *Reload models*."
        )

    # --- Load artifacts ---
    artifacts = load_artifacts(out_dir)
    nlp = _import_spacy()

    # --- Status banner ---
    has_s1 = "s1_model" in artifacts
    has_s2 = "s2_model" in artifacts
    if not has_s1 or not has_s2:
        with st.expander("⚠️ Model loading status — click to expand", expanded=True):
            st.markdown(f"**Resolved model directory:** `{artifacts['paths']['out_dir']}`")
            st.markdown(
                f"- Stage 1 checkpoint: {'✅ loaded' if has_s1 else '❌ missing'}\n"
                f"- Stage 2 checkpoint: {'✅ loaded' if has_s2 else '❌ missing'}\n"
                f"- Metadata: {'✅' if 'metadata' in artifacts else '⚠️ missing (using defaults)'}\n"
                f"- spaCy en_core_web_sm: {'✅' if nlp is not None else '⚠️ not installed (degraded features)'}\n"
            )
            for err in artifacts.get("errors", []):
                st.error(err)
            st.info(
                "Make sure the directory you typed into the sidebar contains the "
                "`stage1v3_focal_best.pt` and `stage2v3_late_fusion_best.pt` files "
                "produced by Notebook 05."
            )

    if not has_s1:
        st.stop()

    # --- Sentence input ---
    st.subheader("Sentence to analyze")
    example_labels = ["— pick an example —"] + [lbl for lbl, _ in EXAMPLES]
    chosen_label = st.selectbox("Pre-selected examples", example_labels, index=1)
    if chosen_label != "— pick an example —":
        for lbl, txt in EXAMPLES:
            if lbl == chosen_label:
                st.session_state.setdefault("sentence_text", txt)
                # If user just changed selection, override
                if st.session_state.get("last_chosen_label") != chosen_label:
                    st.session_state["sentence_text"] = txt
                    st.session_state["last_chosen_label"] = chosen_label
                break

    sentence = st.text_area(
        "Or type / paste your own:",
        key="sentence_text",
        height=110,
        placeholder="e.g. Studies show that approximately 23% of patients improved after treatment.",
    )

    analyze = st.button("🔍 Analyze sentence", type="primary", use_container_width=True)

    if not analyze:
        st.info("Pick an example above (or type your own) and hit **Analyze sentence**.")
        return

    if not sentence or not sentence.strip():
        st.warning("Please enter a sentence first.")
        return

    # --- Run pipeline ---
    with st.spinner("Running Stage 1 → Stage 2 inference..."):
        result = run_pipeline(
            sentence.strip(), artifacts, nlp,
            decision_threshold=threshold, max_len=max_len,
        )

    if "error" in result:
        st.error(result["error"])
        return

    # --- Top-level summary card ---
    epi = result["epi_class"]
    epi_color = CLASS_COLORS.get(epi, "#666")
    score = result.get("citation_score")
    if score is None:
        st.warning(
            "Stage 2 model not loaded — only epistemic-role prediction is available."
        )

    # Quick summary using two columns (collapses on mobile)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
            <div style="padding:14px; border-radius:10px; background:#f7f7fa;
                        border-left:5px solid {epi_color}; margin-bottom:6px;">
                <div style="font-size:0.8em; color:#666;">EPISTEMIC ROLE</div>
                <div style="font-size:1.4em; font-weight:bold; color:{epi_color};">{epi}</div>
                <div style="font-size:0.85em; color:#666;">
                  confidence {result['epi_confidence']:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        if score is not None:
            decision_color = "#C44E52" if result["needs_citation"] else "#55A868"
            decision_text = "NEEDS CITATION" if result["needs_citation"] else "OK"
            st.markdown(
                f"""
                <div style="padding:14px; border-radius:10px; background:#f7f7fa;
                            border-left:5px solid {decision_color}; margin-bottom:6px;">
                    <div style="font-size:0.8em; color:#666;">CITATION SCORE</div>
                    <div style="font-size:1.4em; font-weight:bold; color:{decision_color};">
                        {score:.1%}
                    </div>
                    <div style="font-size:0.85em; color:#666;">{decision_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="padding:14px; border-radius:10px; background:#f7f7fa;">
                    <div style="font-size:0.8em; color:#666;">CITATION SCORE</div>
                    <div style="color:#999;">— Stage 2 unavailable —</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Tabs (mobile-friendly grouping) ---
    tabs = st.tabs(["📊 Probabilities", "🔬 Interpretability", "📋 Features"])

    # ============== Tab 1: Probabilities ==============
    with tabs[0]:
        st.markdown("##### Stage 1 — Epistemic role probabilities")
        prob_df = pd.DataFrame(
            {"class": CLASSES,
             "probability": [result["epi_probs"][c] for c in CLASSES]}
        )
        st.bar_chart(prob_df.set_index("class"), height=260)

        if score is not None:
            st.markdown("##### Stage 2 — Citation worthiness")
            st.markdown(render_score_bar(score, threshold), unsafe_allow_html=True)

    # ============== Tab 2: Interpretability ==============
    with tabs[1]:
        if score is not None and show_ig:
            st.markdown("##### 🎯 Why this score? — token attribution (Integrated Gradients)")
            with st.spinner("Computing Integrated Gradients..."):
                ig = compute_ig_attribution(
                    sentence.strip(), result["eng_vec"],
                    artifacts, n_steps=ig_steps, max_len=max_len,
                )
            if ig is None:
                st.info(
                    "Captum isn't installed (or IG failed). "
                    "Install with `pip install captum` to enable token-level attribution."
                )
            else:
                st.markdown(render_token_attribution(ig), unsafe_allow_html=True)

        if show_attn and "cls_attention" in result:
            st.markdown(
                "##### 👀 What did Stage 1 attend to? — last-layer [CLS] attention"
            )
            st.markdown(
                render_attention_strip(result["tokens"], result["cls_attention"]),
                unsafe_allow_html=True,
            )

        if not show_ig and not show_attn:
            st.info("Enable interpretability options in the sidebar.")

    # ============== Tab 3: Features ==============
    with tabs[2]:
        st.markdown("##### 🧰 Engineered features (Stage 2 inputs)")
        feat_dict = result["feat_dict"]
        feat_df = pd.DataFrame(
            [(c, feat_dict.get(c, 0.0)) for c in artifacts["eng_cols_full"]],
            columns=["feature", "value"],
        )
        # Add feature-family annotation
        def _family(name: str) -> str:
            if name.startswith("epi_"):
                return "epistemic"
            if name in V2_NEW_COLS:
                if any(k in name for k in ("percent", "dollar", "year", "digit", "inline_cite")):
                    return "v2-numerical"
                if "ent_" in name:
                    return "v2-entity"
                if "universal" in name or "superlative" in name:
                    return "v2-quantifier"
                if "reporting" in name or "causal" in name:
                    return "v2-verb"
                return "v2-syntactic"
            return "baseline"

        feat_df["family"] = feat_df["feature"].apply(_family)
        # Highlight non-zero rows
        non_zero = feat_df[feat_df["value"] != 0.0].sort_values("value", ascending=False)
        st.markdown(f"**{len(non_zero)} of {len(feat_df)} features are non-zero**")
        st.dataframe(
            non_zero,
            use_container_width=True,
            hide_index=True,
            column_config={
                "value": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        with st.expander("Show all features (including zeros)"):
            st.dataframe(
                feat_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "value": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    # --- Footer ---
    st.divider()
    st.caption(
        "Architecture · Stage 1 = DistilBERT-base 4-class classifier (focal loss, IBM-augmented). "
        "Stage 2 = Late-Fusion of [CLS] embedding + 34 engineered features. "
        "Trained on Stage 1 corpus + WikiSQE (citation-needed config)."
    )


if __name__ == "__main__":
    main()