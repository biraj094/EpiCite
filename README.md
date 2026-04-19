# EpiCite

EpiCite is an NLP course project for interpretable citation-need scoring using a two-stage pipeline.

## Goal
- Stage 1: Fine-tune DistilBERT to classify sentence epistemic type:
  Claim / Fact / Evidence / Opinion / Background
- Stage 2: Train XGBoost + Isotonic Regression to predict citation_needed probability in [0, 1]
  using Stage 1 outputs and handcrafted linguistic features.

## Repository Layout
- data/raw: downloaded source data
- data/processed: cleaned and model-ready data
- data/annotations: manually annotated OOD samples
- src/data_prep: data processing scripts
- src/features: handcrafted feature extraction
- src/stage1: DistilBERT training/inference
- src/stage2: XGBoost training/calibration
- src/evaluation: metrics, SHAP, OOD tests
- src/utils: shared utility code
- experiments: experiment configs and JSON results
- models: checkpoints and fitted artifacts
- reports: figures and final report outputs

## Reproducibility (Minimal)
1. Create environment and install dependencies:
   - python -m venv .venv
   - .venv\\Scripts\\activate  (Windows)
   - pip install -r requirements.txt

2. Download data sources:
   - Stage 1: IBM Claim Detection + UKP Argument Mining
   - Stage 2: ando55/WikiSQE_experiment (citation config)
   - OOD: All The News + Persuasive Essays (manual annotation subset)

3. Run training pipeline:
   - Stage 1 training script
   - Stage 2 feature build + training script

4. Run evaluation:
   - Stage 1: macro-F1/accuracy/per-class metrics
   - Stage 2: ROC-AUC/PR-AUC/Brier/ECE
   - SHAP for feature attribution

## Notes
- Project assumptions and label definitions are documented in project_spec.md.
- WikiSQE authorship in this project is Ando, Sekine, Komachi (AAAI 2024).
