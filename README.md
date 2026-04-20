# EpiCite — Epistemic Sentence Classifier & Citation Need Scorer

> **Two-stage NLP pipeline** that identifies the epistemic weight of sentences to predict citation necessity with explainable reasoning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface&logoColor=white)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AIT](https://img.shields.io/badge/AIT-NLP%20Group%2012-navy)](https://ait.ac.th)

**Authors:** Biraj Koirala · Longfei Shi
**Department of Computer Science, Asian Institute of Technology**

---

<details>
<summary><h2>📖 Project Description</h2></summary>

### Overview

Current citation detection models rely on surface-level features — sentence length, document position, surrounding citation density — and treat all flagged sentences as functionally identical. They can identify *that* a citation might be missing but cannot explain *why*.

EpiCite addresses this by introducing an intermediate **epistemic layer**: a classifier that determines the linguistic purpose of a sentence before scoring its verifiability. This makes every prediction explainable in human terms: *"This sentence was flagged because it is classified as a Claim, and Claims are the strongest predictor of citation need."*

---

### Pipeline Architecture

```
Raw Sentence Text
       │
       ▼
┌─────────────────────────────┐
│  Feature Extraction (spaCy) │  ← 12 linguistic features
│  Lexical · Syntactic · Ctx  │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   Stage 1 — DistilBERT      │  ← Fine-tuned epistemic classifier
│   Epistemic Classification  │
│                             │
│  Claim / Evidence / Opinion │
│  Background / Fact          │
└─────────────────────────────┘
       │  class probabilities
       ▼
┌─────────────────────────────┐
│   Stage 2 — Citation Scorer │  ← Calibrated probability output
│   XGBoost + Isotonic Reg.   │
│                             │
│   Citation Score ∈ [0, 1]   │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│   SHAP Explainability       │  ← Per-prediction feature attribution
│   Feature Importance        │
└─────────────────────────────┘
```

---

### Methodology

**1 — Feature Extraction**

Derives 12 interpretable linguistic features per sentence using spaCy and NLTK, grouped into three categories:

| Category | Features |
|---|---|
| Lexical | Vague quantifiers, hedging words, superlative adjectives |
| Syntactic | Passive voice, nominalization rate, POS ratios (NOUN/VERB/ADJ/ADV) |
| Contextual | Sentence position, citation density, paragraph-relative index |

**2 — Stage 1: Epistemic Classification**

A fine-tuned `distilbert-base-uncased` model classifies each sentence into one of five epistemic categories:

| Label | Definition |
|---|---|
| **Claim** | An assertion that takes a verifiable position on a topic |
| **Evidence** | Supports or refutes a claim, typically citing data or a source |
| **Opinion** | A subjective judgment, preference, or personal belief |
| **Background** | Neutral, contextual, or off-topic information |
| **Fact** | An objective, verifiable statement (future work) |

Three model tiers are evaluated: Logistic Regression baseline, BiLSTM intermediate, and DistilBERT proposed.

**3 — Stage 2: Citation Scoring**

The Stage 1 class probability distribution (5 values) is combined with the 12 linguistic features to form an 18-dimensional input vector fed into a Gradient Boosting classifier. Isotonic Regression calibrates the raw scores into well-calibrated probabilities on [0, 1].

**4 — Explainability**

SHAP (SHapley Additive exPlanations) values are computed for every prediction, identifying which features drove each citation flag. This directly addresses the interpretability gap in existing citation detection models.

---

### Research Questions & Hypotheses

| ID | Statement |
|---|---|
| **RQ1** | Can a two-stage pipeline match end-to-end models while providing superior interpretability? |
| **RQ2** | Is epistemic category a stronger predictor of citation need than surface features like sentence length? |
| **RQ3** | How significantly does a Wikipedia-trained model degrade on news articles and student essays? |
| **H1** | Epistemic type outperforms surface-level features as a citation predictor |
| **H2** | Two-stage pipeline provides higher interpretability via SHAP than end-to-end DistilBERT |
| **H3** | Models trained on Wikipedia show measurable F1 degradation on out-of-domain text |

---

### Data Sources

| Dataset | Purpose | Size |
|---|---|---|
| IBM Claim Detection | Stage 1 training | ~2,500 sentences |
| UKP Argument Mining | Stage 1 training | ~25,000 sentences |
| WikiSQE (`citation needed`) | Stage 2 training | ~150,000 sentences |
| All The News (Kaggle) | Out-of-domain test | 500 sentences |
| Persuasive Essays (UKP) | Out-of-domain test | ~5,000 sentences |

---

### Technical Stack

| Component | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP | spaCy, NLTK, Hugging Face Transformers |
| ML / Training | PyTorch, Scikit-learn, XGBoost |
| Explainability | SHAP, Captum (Integrated Gradients) |
| Calibration | Isotonic Regression |
| Data | Hugging Face Datasets |

</details>

---

<details>
<summary><h2>🔄 Updates & Progress</h2></summary>

> Last updated — Stage 1 v2 + Stage 2 Late-Fusion
> Full notebook: [`./notebook/EpiCite_04_Stage1v2_Stage2_LateFusion-STRIP.ipynb`](./notebook/EpiCite_04_Stage1v2_Stage2_LateFusion-STRIP.ipynb)

---

### What Changed & Why

The original implementation had two critical problems discovered during error analysis:

1. **Stage 1 class imbalance** — The training corpus had 13,225 Claim sentences and only 204 Evidence sentences. With equal loss weights, Evidence contributed 0.7% of the total training signal. The model had no meaningful incentive to learn what Evidence looks like.

2. **Stage 2 feature bottleneck** — The original XGBoost Stage 2 only received 18 summary features — it never saw the actual sentence. The rich semantic context encoded in DistilBERT was completely discarded after Stage 1.

---

### Fix 1 — Stage 1 Weighted Retraining

**Problem:** Equal loss weights made the model ignore minority classes.

**Solution:** Inverse-frequency class weights applied to `CrossEntropyLoss`.

```
weight_c = N_total / (4 × N_c)
```

| Class | Samples | Raw Weight | Applied Weight |
|---|---|---|---|
| Claim | 13,225 | 0.52 | 0.52 |
| Background | 11,536 | 0.59 | 0.59 |
| Opinion | 2,433 | 2.82 | 2.82 |
| Evidence | 204 | 33.6 | **10.0** ← capped |

Evidence's weight is capped at 10× to prevent loss explosion. The model is retrained from the original checkpoint (not from scratch) at a conservative LR of 1e-5 for 4 epochs.

**Results (Test Set):**

| Metric | Before | After |
|---|---|---|
| Macro F1 | ~0.72 | **0.80** ✓ |
| Evidence Recall | ~0.00 | **0.85** |
| Opinion F1 | ~0.76 | **0.84** |
| Accuracy | — | **80.7%** |

> ⚠️ Evidence test support = 20 samples. The 0.85 recall is directionally valid but statistically fragile. A dedicated Evidence corpus (FEVER, SciCite) is needed for robust evaluation.

---

### Fix 2 — Stage 2 Late-Fusion Architecture

**Problem:** XGBoost only saw 18 summary features, never the sentence itself.

**Solution:** A 3-layer MLP attached directly to DistilBERT's `[CLS]` token, fusing semantic embeddings with engineered features at the decision layer.

```
Sentence → DistilBERT → [CLS] (768-dim)
                                 │
                    concat with 18-dim feature vector
                                 │
                          786-dim combined
                                 │
              Linear(786→256) + LayerNorm + ReLU + Dropout(0.3)
                                 │
              Linear(256→64)  + LayerNorm + ReLU + Dropout(0.2)
                                 │
               Linear(64→1)   + Sigmoid
                                 │
                    Citation score ∈ [0, 1]
```

**Why LayerNorm instead of BatchNorm:** DistilBERT uses LayerNorm internally. BatchNorm behaves differently at train vs inference time and creates instabilities when mixed with Transformer layers. LayerNorm normalises per sample — consistent at any batch size.

**Training — 3-Phase Gradual Unfreezing:**

The MLP head starts from random weights. Immediately training all 66M DistilBERT parameters would corrupt the transformer before the head stabilises.

| Phase | What Trains | LR | Epochs | Val AUC |
|---|---|---|---|---|
| 1 — Head only | MLP (218k params) | 1e-3 | 2 | 0.704 |
| 2 — Top 2 layers | MLP + last 2 transformer layers (14.4M) | 2e-5 | 3 | 0.755 |
| 3 — Full model | All 66.6M parameters | 5e-6 | 2 | 0.759 |

**Results (Test Set, WikiSQE `citation needed`, 1,000 sentences):**

| Metric | Value |
|---|---|
| ROC-AUC | **0.746** |
| Binary F1 | **0.684** |
| Accuracy | **68.4%** |
| Avg. Precision (PR-AUC) | **0.730** |
| Brier Score | **0.209** |

---

### SHAP Analysis — H1 Finding

**H1:** *Epistemic type is a stronger predictor of citation need than surface features.*

**Verdict: NOT SUPPORTED in current configuration.**

SHAP feature rankings (mean |SHAP| value):

| Rank | Feature | Value | Type |
|---|---|---|---|
| 1 | `sentence_length` | 0.04598 | Linguistic |
| 2 | `passive_voice` | 0.01324 | Linguistic |
| 3 | `hedge_count` | 0.00647 | Linguistic |
| ... | ... | ... | ... |
| 11 | `epi_prob_Claim` | 0.00115 | **Epistemic** |
| 13 | `epi_confidence` | 0.00089 | **Epistemic** |

**Why H1 failed — domain mismatch:**

Stage 1 was trained on IBM/UKP debate text. When applied to Wikipedia prose, it labels **90.5% of sentences as Background** — nearly zero epistemic variance across 150k sentences. With constant output, SHAP correctly assigns low importance to epistemic features.

This is not evidence that epistemic type is unimportant — it is evidence that the epistemic classifier does not transfer to encyclopedic writing. H1 should be re-evaluated after retraining Stage 1 on Wikipedia-sourced epistemic annotations.

Mean epistemic probabilities by citation label (early H1 check):

| Label | `epi_prob_Claim` | `epi_prob_Background` |
|---|---|---|
| No citation needed (0) | 0.073 | 0.892 |
| Citation needed (1) | 0.102 | 0.863 |

The direction is correct (citation-needed sentences do have higher Claim probability) but the magnitude is too small to be a dominant signal.

---

### Known Limitations

| # | Issue | Impact | Proposed Fix |
|---|---|---|---|
| 1 | Stage 1 domain mismatch | Epistemic features near-useless for Wikipedia | Retrain Stage 1 on Wikipedia-sourced labels |
| 2 | Evidence data scarcity (204 samples) | Evidence F1 statistically fragile | Add FEVER / SciCite Evidence sentences |
| 3 | `sentence_length` dominates SHAP | Model exploits surface correlation | Length-normalised features or penalty term |
| 4 | Factual sentences over-flagged | "Water boils at 100°C" scores 0.66 | Length exploitation drives false positives |

---

### Roadmap

- [x] Data acquisition and taxonomy definition
- [x] 12-feature spaCy extraction pipeline
- [x] Stage 1 DistilBERT fine-tuning (equal weights)
- [x] Stage 1 v2 — class-weighted retraining
- [x] Stage 2 Late-Fusion architecture + gradual unfreezing
- [x] SHAP Track 1 — engineered feature importance
- [x] Integrated Gradients Track 2 — token attribution
- [ ] Stage 1 retraining on Wikipedia-sourced epistemic labels
- [ ] Evidence class augmentation (FEVER / SciCite)
- [ ] Out-of-domain evaluation — news articles (RQ3 / H3)
- [ ] Out-of-domain evaluation — student essays (RQ3 / H3)
- [ ] Streamlit interactive demo

</details>

---

<details>
<summary><h2>📁 Repository Structure</h2></summary>

```
EpiCite/
├── notebook/
│   ├── EpiCite_01_Dataset_Construction.ipynb       # Stage 1 corpus: IBM + UKP → unified taxonomy
│   ├── EpiCite_02_Features_Training.ipynb          # spaCy feature extraction + Stage 1 DistilBERT training
│   ├── EpiCite_03_Stage2_CitationScorer.ipynb      # Original XGBoost Stage 2
│   ├── EpiCite_04_Stage1v2_Stage2_LateFusion-STRIP.ipynb  # ← LATEST: weighted S1 + Late-Fusion S2
│   └── EpiCite_EDA_Stage1_Predictions.ipynb        # Error analysis on Stage 1 predictions
│
├── output/
│   ├── stage1_combined.csv                         # 27k labeled sentences (Notebook 01)
│   ├── stage1_features.csv                         # 12 spaCy features per sentence
│   ├── stage1_predictions.csv                      # DistilBERT class probabilities
│   ├── stage1v2_weighted_best.pt                   # Class-weighted Stage 1 checkpoint
│   ├── stage2v2_best_final.pt                      # Late-Fusion Stage 2 checkpoint
│   ├── stage2v2_wiki_features.csv                  # Cached WikiSQE spaCy features
│   ├── stage2v2_wiki_epiprobs.csv                  # Cached WikiSQE epistemic probabilities
│   ├── stage2v2_shap_bar.png                       # SHAP feature importance plot
│   ├── stage2v2_ig_heatmap.png                     # Token attribution heatmaps
│   └── stage2v2_results.json                       # All evaluation metrics
│
└── README.md
```

</details>

---

<details>
<summary><h2>📊 Results Summary</h2></summary>

### Stage 1 — Epistemic Classifier (v2, class-weighted)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Claim | 0.78 | 0.85 | 0.81 | 1,323 |
| Evidence | 0.71 | 0.85 | 0.77 | 20 ⚠️ |
| Opinion | 0.78 | 0.92 | 0.84 | 244 |
| Background | 0.85 | 0.73 | 0.79 | 1,153 |
| **Macro avg** | **0.78** | **0.84** | **0.80** | **2,740** |

### Stage 2 — Late-Fusion Citation Scorer

| Metric | Value |
|---|---|
| ROC-AUC | 0.746 |
| Binary F1 | 0.684 |
| Avg. Precision | 0.730 |
| Brier Score | 0.209 |

</details>

---

<details>
<summary><h2>⚙️ Setup & Reproduction</h2></summary>

### Requirements

```bash
pip install transformers datasets torch scikit-learn xgboost shap spacy tqdm captum
python -m spacy download en_core_web_sm
```

### Run Order

```
Notebook 01 → builds stage1_combined.csv
Notebook 02 → trains Stage 1 DistilBERT, extracts features
Notebook 04 → retrains Stage 1 with class weights + trains Late-Fusion Stage 2
```

### Hardware

Tested on NVIDIA RTX 2080 Ti (11.5 GB VRAM). Estimated runtimes on this hardware:

| Step | Time |
|---|---|
| Stage 1 v2 retraining (4 epochs) | ~8 min |
| WikiSQE spaCy feature extraction (152k sentences) | ~25 min |
| WikiSQE DistilBERT inference (152k sentences) | ~15 min |
| Stage 2 training (7 epochs total, 3 phases) | ~60 min |

Both the WikiSQE feature extraction and inference results are cached to disk. Re-runs skip these steps automatically.

</details>

---

*EpiCite · Group 12 · Department of Computer Science · Asian Institute of Technology*
