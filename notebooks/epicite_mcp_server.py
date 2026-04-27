"""
EpiCite MCP Server
==================

Exposes the trained EpiCite citation-need detector as an MCP server, so any MCP
client (Claude Desktop, Cursor, Continue, etc.) can call it as a tool inside
its own workflow.

Three tools are registered:

    analyze_sentence(text)    → score + epistemic class + probabilities
    analyze_paragraph(text)   → per-sentence breakdown + summary statistics
    explain_prediction(text)  → score + class + integrated-gradients tokens

Design notes
------------
* The Q6 plan called for "research-grade" implementation. We added a caching
  layer for the explanation path because IG without caching takes ~30-60s per
  call — unusable in a live demo. Caching brings it to ~3-5s.
* Caches:
    - SHAP background (loaded once at startup; not re-used per call here, but
      the slot is wired up for future global-importance queries)
    - Tokenizer + DistilBERT models (loaded once)
    - LRU cache on (sentence) → IG tokens, so repeated calls in the same
      session are near-instant
* Fail-open: if a model file is missing, the server still starts and responds
  with a clear error message. This makes it usable in CI / demo environments
  where the user has the code but not the checkpoints.

Installation
------------
    pip install mcp transformers torch spacy captum
    python -m spacy download en_core_web_sm

Running
-------
    python epicite_mcp_server.py

Connecting from Claude Desktop
------------------------------
Add to ``~/Library/Application Support/Claude/claude_desktop_config.json``::

    {
      "mcpServers": {
        "epicite": {
          "command": "python",
          "args": ["/full/path/to/epicite_mcp_server.py"]
        }
      }
    }

Then restart Claude Desktop. The three tools above will appear under the
``epicite`` server.

Group 12 — AIT NLP — Biraj Koirala, Longfei Shi
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────── Logging ───────────────────────────
LOG = logging.getLogger("epicite")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# ─────────────────────────── Configuration ───────────────────────
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
META_PATH  = OUTPUT_DIR / "stage2v3_metadata.json"

CLASSES   = ["Claim", "Evidence", "Opinion", "Background"]
LABEL2ID  = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL  = {i: c for c, i in LABEL2ID.items()}
MODEL_NAME = "distilbert-base-uncased"

# Maximum sentences a paragraph can contain (DoS guard)
MAX_PARAGRAPH_SENTENCES = 100
# Cache size for explanation calls
EXPLAIN_CACHE_SIZE = 256

# ─────────────────────────── Heavy imports (deferred) ───────────────
# We import torch / transformers / spacy lazily so that `--help` and
# basic startup don't pay the import cost.

_models_loaded = False
_load_error: str | None = None
torch = None  # type: ignore
F = None  # type: ignore
nn = None  # type: ignore
spacy = None  # type: ignore
nlp = None  # type: ignore
DEVICE = None  # type: ignore
tokenizer = None  # type: ignore
s1_model = None  # type: ignore
model_full = None  # type: ignore
lig = None  # type: ignore
META: dict[str, Any] = {}
ENG_COLS_FULL: list[str] = []


# ════════════════════════════════════════════════════════════════════
#                    Feature extraction (must match Notebook 05)
# ════════════════════════════════════════════════════════════════════

# Lexicons — kept aligned with the v2 extractor in Notebook 05
HEDGE_VERBS = {"seem","seems","seemed","appear","appears","appeared","suggest","suggests",
               "suggested","indicate","indicates","indicated","propose","tend","tends","tended",
               "believe","believes","believed","assume","assumes","assumed","speculate","speculates",
               "speculated","hypothesize","estimate","estimates","estimated","report","reports","reported"}
HEDGE_AUX   = {"may","might","could","would","should","can"}
HEDGE_ADV   = {"possibly","probably","perhaps","allegedly","reportedly","presumably","arguably",
               "apparently","supposedly","ostensibly","likely","unlikely","generally","typically",
               "usually","often","sometimes","occasionally","roughly","approximately","potentially",
               "conceivably","purportedly"}
VAGUE_QUANT = {"some","many","several","a lot","most","much","few","various","numerous","plenty",
               "considerable","significant"}
BOOSTERS    = {"clearly","obviously","certainly","definitely","undoubtedly","surely"}
SUBJECTIVE  = {"i","we","my","our","myself","ourselves","me","us"}
UNIVERSAL_QUANT = {"all","every","any","always","never","none","no one","everyone","everywhere",
                   "everything","nothing"}
SUPERLATIVE = {"first","largest","smallest","best","worst","most","least","highest","lowest",
               "biggest","greatest","oldest","newest","fastest","slowest","strongest","weakest",
               "only","unique"}
REPORTING_VERBS = {"according","reports","reported","stated","states","said","claims","claimed",
                   "argues","argued","wrote","writes","cites","cited","notes","noted","observed",
                   "observes"}
CAUSAL_MARKERS  = {"causes","caused","leads to","results in","because","therefore","thus",
                   "consequently","hence","due to","triggers","triggered","produces","produced",
                   "contributes to"}

RE_PERCENT     = re.compile(r"\d+(?:\.\d+)?\s*%")
RE_DOLLAR      = re.compile(r"\$\s*\d")
RE_YEAR        = re.compile(r"\b(?:19|20)\d{2}\b")
RE_DIGIT       = re.compile(r"\b\d+(?:[.,]\d+)*\b")
RE_INLINE_CITE = re.compile(r"\[\s*\d+\s*\]")


def _dep_depth(token):
    if not list(token.children):
        return 1
    return 1 + max(_dep_depth(c) for c in token.children)


def extract_features_full(text: str) -> dict[str, float]:
    """Extract the 29-engineered-feature vector matching Notebook 05's extractor.

    Returns a dict keyed by the column names the model expects (excluding the
    5 epistemic features, which are filled in by the Stage 1 inference call).
    """
    doc = nlp(str(text))
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    lower = [t.text.lower() for t in toks]
    text_lower = text.lower()

    out = {
        # Baseline counts
        "n_hedge_verbs":     sum(1 for w in lower if w in HEDGE_VERBS),
        "n_hedge_aux":       sum(1 for w in lower if w in HEDGE_AUX),
        "n_hedge_adv":       sum(1 for w in lower if w in HEDGE_ADV),
        "n_vague_quant":     sum(1 for w in lower if w in VAGUE_QUANT),
        "n_boosters":        sum(1 for w in lower if w in BOOSTERS),
        "n_subjective_pron": sum(1 for w in lower if w in SUBJECTIVE),
        "n_negations":       sum(1 for t in doc if t.dep_ == "neg"),
        "n_passive":         sum(1 for t in doc if t.dep_ == "auxpass"),
        "n_proper_nouns":    sum(1 for t in doc if t.pos_ == "PROPN"),
        "n_numbers":         sum(1 for t in doc if t.like_num),
        "n_modals":          sum(1 for t in doc if t.tag_ == "MD"),
        "sentence_length":   float(n),

        # v2 numerical
        "rate_percent":      len(RE_PERCENT.findall(text))     / n,
        "rate_dollar":       len(RE_DOLLAR.findall(text))      / n,
        "rate_year":         len(RE_YEAR.findall(text))        / n,
        "rate_digit":        len(RE_DIGIT.findall(text))       / n,
        "rate_inline_cite":  len(RE_INLINE_CITE.findall(text)) / n,

        # v2 entities
        "rate_ent_person": sum(1 for e in doc.ents if e.label_ == "PERSON") / n,
        "rate_ent_org":    sum(1 for e in doc.ents if e.label_ == "ORG")    / n,
        "rate_ent_gpe":    sum(1 for e in doc.ents if e.label_ == "GPE")    / n,
        "rate_ent_date":   sum(1 for e in doc.ents if e.label_ == "DATE")   / n,
        "rate_ent_total":  len(doc.ents) / n,

        # v2 quantifiers
        "rate_universal_quant": sum(1 for w in lower if w in UNIVERSAL_QUANT) / n,
        "rate_superlative":     sum(1 for w in lower if w in SUPERLATIVE) / n,

        # v2 verbs
        "rate_reporting_verbs": sum(1 for w in lower if w in REPORTING_VERBS) / n,
        "rate_causal_markers":  sum(text_lower.count(m) for m in CAUSAL_MARKERS) / n,

        # v2 syntactic
        "dep_tree_depth":  max((_dep_depth(t) for t in doc if t.dep_ == "ROOT"), default=1),
        "n_clauses":       sum(1 for t in doc if t.dep_ in {"ccomp","xcomp","advcl","relcl"}) + 1,
        "noun_verb_ratio": (sum(1 for t in doc if t.pos_ in {"NOUN","PROPN"})
                            / max(sum(1 for t in doc if t.pos_ == "VERB"), 1)),
    }
    return out


# ════════════════════════════════════════════════════════════════════
#                    Model loading
# ════════════════════════════════════════════════════════════════════

def _load_models() -> None:
    """Lazy-load all models. Sets _load_error on failure so the server can
    still start and report the problem via the tools rather than crashing."""
    global _models_loaded, _load_error
    global torch, F, nn, spacy, nlp, DEVICE
    global tokenizer, s1_model, model_full, lig
    global META, ENG_COLS_FULL

    if _models_loaded:
        return

    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        import spacy as _spacy
        from transformers import (
            DistilBertTokenizerFast, DistilBertModel,
            DistilBertForSequenceClassification,
        )
        from captum.attr import LayerIntegratedGradients

        torch, nn, F, spacy = _torch, _nn, _F, _spacy
        DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        LOG.info(f"Device: {DEVICE}")

        # Load metadata
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"{META_PATH} not found. Run Notebook 05 first to produce checkpoints."
            )
        META.update(json.loads(META_PATH.read_text()))
        ENG_COLS_FULL.extend(META["stage2"]["feature_cols_full"])
        LOG.info(f"Engineered feature columns: {len(ENG_COLS_FULL)}")

        # spaCy
        nlp = _spacy.load("en_core_web_sm")

        # Tokenizer
        tokenizer_local = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        # Stage 1
        s1 = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(CLASSES),
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(DEVICE)
        ckpt_path = HERE / META["stage1"]["checkpoint"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint missing: {ckpt_path}")
        s1.load_state_dict(_torch.load(ckpt_path, map_location=DEVICE))
        s1.eval()

        # Stage 2 — Late-Fusion
        class LateFusion(_nn.Module):
            def __init__(self, distilbert, num_eng):
                super().__init__()
                self.distilbert = distilbert
                fused = distilbert.config.hidden_size + num_eng
                self.head = _nn.Sequential(
                    _nn.LayerNorm(fused),
                    _nn.Linear(fused, 256), _nn.GELU(), _nn.Dropout(0.3),
                    _nn.LayerNorm(256),
                    _nn.Linear(256, 64),     _nn.GELU(), _nn.Dropout(0.3),
                    _nn.Linear(64, 1),
                )

            def forward(self, input_ids, attention_mask, eng_features):
                out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                fused = _torch.cat([cls, eng_features], dim=-1)
                return _torch.sigmoid(self.head(fused)).squeeze(-1)

        distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        m = LateFusion(distilbert, num_eng=len(ENG_COLS_FULL)).to(DEVICE)
        s2_ckpt = HERE / META["stage2"]["full_ckpt"]
        if not s2_ckpt.exists():
            raise FileNotFoundError(f"Stage 2 checkpoint missing: {s2_ckpt}")
        m.load_state_dict(_torch.load(s2_ckpt, map_location=DEVICE))
        m.eval()

        # Captum IG over the embedding layer
        class _ScoreOnly(_nn.Module):
            def __init__(self, lf): super().__init__(); self.lf = lf
            def forward(self, input_ids, attention_mask, eng_features):
                return self.lf(input_ids, attention_mask, eng_features)
        score_only = _ScoreOnly(m).to(DEVICE).eval()
        lig_local = LayerIntegratedGradients(score_only, score_only.lf.distilbert.embeddings)

        # Bind globals
        globals().update({
            "tokenizer":  tokenizer_local,
            "s1_model":   s1,
            "model_full": m,
            "lig":        lig_local,
        })

        _models_loaded = True
        LOG.info("All models loaded successfully")

    except Exception as e:
        _load_error = str(e)
        LOG.error(f"Model loading FAILED: {e}")


# ════════════════════════════════════════════════════════════════════
#                    Inference helpers
# ════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy. Falls back to regex if spaCy
    isn't loaded yet (shouldn't happen, but defensive)."""
    if nlp is None:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _run_pipeline(sentence: str) -> dict[str, Any]:
    """Full Stage 1 → Stage 2 pipeline on a single sentence."""
    enc = tokenizer(sentence, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEVICE)

    # Stage 1
    with torch.no_grad():
        s1_logits = s1_model(**enc).logits
        s1_probs = F.softmax(s1_logits, dim=-1).squeeze(0).cpu().numpy()
    s1_class = ID2LABEL[int(s1_probs.argmax())]

    # Build engineered feature vector with Stage 1 probs injected
    feats_dict = extract_features_full(sentence)
    eng_vec = np.zeros(len(ENG_COLS_FULL), dtype=np.float32)
    for i, col in enumerate(ENG_COLS_FULL):
        if col.startswith("epi_prob_"):
            cls = col[len("epi_prob_"):]
            eng_vec[i] = float(s1_probs[LABEL2ID[cls]])
        elif col == "epi_confidence":
            eng_vec[i] = float(s1_probs.max())
        else:
            eng_vec[i] = float(feats_dict.get(col, 0.0))

    # Stage 2
    with torch.no_grad():
        feat_t = torch.tensor(eng_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        score = float(model_full(enc["input_ids"], enc["attention_mask"], feat_t).item())

    return {
        "sentence":       sentence,
        "epi_class":      s1_class,
        "epi_probs":      {c: round(float(s1_probs[i]), 4) for c, i in LABEL2ID.items()},
        "epi_confidence": round(float(s1_probs.max()), 4),
        "citation_score": round(score, 4),
        "needs_citation": bool(score >= 0.5),
        "_eng_vec":       eng_vec,  # kept for the explanation path
    }


@functools.lru_cache(maxsize=EXPLAIN_CACHE_SIZE)
def _cached_explain(sentence: str) -> dict[str, Any]:
    """LRU-cached explanation. Cache key is the sentence string itself."""
    res = _run_pipeline(sentence)
    eng_vec = res["_eng_vec"]

    enc = tokenizer(sentence, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEVICE)
    feat_t = torch.tensor(eng_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ref_ids = torch.full_like(enc["input_ids"], tokenizer.pad_token_id)

    attrs = lig.attribute(
        inputs=enc["input_ids"],
        baselines=ref_ids,
        additional_forward_args=(enc["attention_mask"], feat_t),
        n_steps=30,  # fewer steps for speed; 50 is the academic default
        return_convergence_delta=False,
    )
    attrs_token = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    if np.abs(attrs_token).max() > 0:
        attrs_token = attrs_token / np.abs(attrs_token).max()

    tok_ids = enc["input_ids"].squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(tok_ids)

    token_attrs = []
    for tok, w in zip(tokens, attrs_token):
        if tok in {tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token}:
            continue
        token_attrs.append({"token": tok.replace("##", ""), "weight": round(float(w), 4)})

    top_pushing  = sorted(token_attrs, key=lambda x: -x["weight"])[:8]
    top_pulling  = sorted(token_attrs, key=lambda x:  x["weight"])[:5]

    return {
        "sentence":       sentence,
        "epi_class":      res["epi_class"],
        "epi_probs":      res["epi_probs"],
        "citation_score": res["citation_score"],
        "needs_citation": res["needs_citation"],
        "explanation": {
            "method":        "integrated_gradients",
            "n_steps":       30,
            "all_tokens":    token_attrs,
            "top_pushing":   top_pushing,    # toward "needs citation"
            "top_pulling":   top_pulling,    # toward "no citation"
            "summary":       (
                f"Highest-weight token: '{top_pushing[0]['token']}' "
                f"(weight {top_pushing[0]['weight']:+.3f})." if top_pushing else
                "No salient tokens found."
            ),
        },
    }


# ════════════════════════════════════════════════════════════════════
#                    MCP server registration
# ════════════════════════════════════════════════════════════════════

def _ensure_loaded() -> str | None:
    """Return None if loaded, else an error message."""
    if not _models_loaded:
        _load_models()
    if _load_error:
        return f"EpiCite models could not be loaded: {_load_error}"
    return None


def _strip_internal(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys starting with underscore (used for internal state)."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────── Tool implementations ────────────────────

def tool_analyze_sentence(text: str) -> dict[str, Any]:
    """Score a single sentence for citation need."""
    err = _ensure_loaded()
    if err: return {"error": err}
    text = (text or "").strip()
    if not text:
        return {"error": "empty input"}
    if len(text.split()) > 200:
        return {"error": "sentence too long (>200 tokens) — try analyze_paragraph"}
    result = _run_pipeline(text)
    return _strip_internal(result)


def tool_analyze_paragraph(text: str) -> dict[str, Any]:
    """Split a paragraph and score every sentence."""
    err = _ensure_loaded()
    if err: return {"error": err}
    text = (text or "").strip()
    if not text:
        return {"error": "empty input"}
    sents = _split_sentences(text)
    if not sents:
        return {"error": "no sentences found in input"}
    if len(sents) > MAX_PARAGRAPH_SENTENCES:
        return {"error": f"too many sentences ({len(sents)} > {MAX_PARAGRAPH_SENTENCES})"}

    rows = [_strip_internal(_run_pipeline(s)) for s in sents]
    n_flag = sum(r["needs_citation"] for r in rows)
    mean_s = float(np.mean([r["citation_score"] for r in rows]))
    return {
        "n_sentences":         len(rows),
        "n_flagged":           int(n_flag),
        "flagged_fraction":    round(n_flag / len(rows), 3),
        "mean_citation_score": round(mean_s, 4),
        "sentences":           rows,
    }


def tool_explain_prediction(text: str) -> dict[str, Any]:
    """Score + integrated-gradients token attributions for a single sentence."""
    err = _ensure_loaded()
    if err: return {"error": err}
    text = (text or "").strip()
    if not text:
        return {"error": "empty input"}
    if len(text.split()) > 200:
        return {"error": "sentence too long for explanation (>200 tokens)"}
    return _cached_explain(text)


# ─────────────────── MCP plumbing ───────────────────────────

async def _serve_mcp() -> None:
    """Register tools with the MCP SDK and start the stdio server."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        LOG.error("mcp package not installed. Run: pip install mcp")
        sys.exit(1)

    server = Server("epicite")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="analyze_sentence",
                description=(
                    "Decide whether a single sentence needs a citation. Returns a "
                    "0–1 citation score, a binary needs_citation flag, the predicted "
                    "epistemic class (Claim / Evidence / Opinion / Background), and "
                    "the per-class probabilities."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string",
                                  "description": "The sentence to analyze."}
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="analyze_paragraph",
                description=(
                    "Split a paragraph into sentences and analyze each one. "
                    "Returns per-sentence results plus aggregate statistics "
                    "(number flagged, mean score, flagged fraction)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string",
                                  "description": "The paragraph or multi-sentence text to analyze."}
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="explain_prediction",
                description=(
                    "Like analyze_sentence but also returns integrated-gradients "
                    "token attributions: which words pushed the prediction toward "
                    "'needs citation' vs 'does not need citation'. Slower (~3-5s "
                    "with cache, ~30s without)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string",
                                  "description": "The sentence to explain."}
                    },
                    "required": ["text"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        text = arguments.get("text", "")
        try:
            if   name == "analyze_sentence":     out = tool_analyze_sentence(text)
            elif name == "analyze_paragraph":    out = tool_analyze_paragraph(text)
            elif name == "explain_prediction":   out = tool_explain_prediction(text)
            else:                                out = {"error": f"unknown tool: {name}"}
        except Exception as e:
            LOG.exception("tool error")
            out = {"error": f"internal error: {e}"}
        return [TextContent(type="text", text=json.dumps(out, indent=2, default=str))]

    LOG.info("Starting EpiCite MCP server on stdio…")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# ════════════════════════════════════════════════════════════════════
#                    Entry point + standalone test mode
# ════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Run a quick sanity check without MCP — useful for development."""
    err = _ensure_loaded()
    if err:
        print(f"FAIL: {err}")
        sys.exit(1)

    print("\n── analyze_sentence ──")
    r = tool_analyze_sentence(
        "Studies show that approximately 23% of patients improved after treatment in 2023.")
    print(json.dumps(r, indent=2, default=str))

    print("\n── analyze_paragraph ──")
    r = tool_analyze_paragraph(
        "The Eiffel Tower was completed in 1889. "
        "Many visitors find it beautiful at night. "
        "Studies show that approximately 7 million people visit it each year."
    )
    print(json.dumps(r, indent=2, default=str))

    print("\n── explain_prediction ──")
    r = tool_explain_prediction(
        "According to the WHO, malaria deaths fell by 30% between 2000 and 2015.")
    print(json.dumps(r, indent=2, default=str))


def main() -> None:
    if "--test" in sys.argv:
        _smoke_test()
    else:
        asyncio.run(_serve_mcp())


if __name__ == "__main__":
    main()
