# EpiCite: Epistemic Sentence Classifier & Citation Need Scorer

EpiCite is a two-stage NLP pipeline designed to identify the epistemic weight of sentences to predict citation necessity with high interpretability.

## Overview

Current models often rely on surface-level features like sentence length and position to predict citation needs. EpiCite introduces an intermediate layer that classifies the linguistic purpose of a sentence before scoring its verifiability, allowing users to understand the "why" behind a citation flag.

## Methodology

The system uses a sequential architecture to move from raw text to an explainable score:

1.  **Feature Extraction**: Derives 12 linguistic features including lexical (vague quantifiers, hedging), syntactic (nominalization, POS distribution), and contextual markers (sentence position, citation density).
2.  **Stage 1 (Epistemic Classification)**: A fine-tuned DistilBERT model classifies sentences into five categories: Claim, Fact, Evidence, Opinion, and Background.
3.  **Stage 2 (Citation Scoring)**: An XGBoost classifier uses the Stage 1 labels and linguistic features to produce a calibrated probability score (0-1).
4.  **Explainability**: SHAP analysis identifies which specific features or epistemic types drove each prediction.

## Data Sources

The project utilizes several specialized datasets for training and cross-domain evaluation:

| Dataset | Purpose | Size |
| :--- | :--- | :--- |
| IBM Claim Detection | Stage 1 Training | 2,500 sentences |
| UKP Argument Mining | Stage 1 Training | 25,000 sentences |
| WikiSQE | Stage 2 Training | 150,000 sentences |
| All The News | Out-of-Domain Test | 500 sentences |
| Persuasive Essays | Out-of-Domain Test | 5,000 sentences |

## Research Goals

* Evaluate if a two-stage pipeline matches end-to-end models while providing better interpretability.
* Determine if epistemic categories (e.g., "Claim") are stronger predictors of citation need than surface features like length.
* Test how well models trained on Wikipedia generalize to news and student essays.

## Technical Stack

* **Language**: Python
* **NLP Libraries**: spaCy, NLTK, Hugging Face Transformers
* **Machine Learning**: Scikit-learn, XGBoost, SHAP
* **Frameworks**: DistilBERT for classification, Isotonic Regression for calibration

## Status and Roadmap

* **Completed**: Data acquisition, taxonomy definition, and baseline feature extraction scripts.
* **In Progress**: Fine-tuning DistilBERT, building the Stage 2 architecture, and designing a Streamlit dashboard.
* **Upcoming**: Cross-domain evaluation and SHAP-based error analysis.

## Authors

* Biraj Koirala
* Longfei Shi
* Department of Computer Science, Asian Institute of Technology