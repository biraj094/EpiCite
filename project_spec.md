# EpiCite Project Specification

## Project Overview
EpiCite is a two-stage NLP pipeline for citation-need prediction with interpretability:
- Stage 1: Fine-tuned DistilBERT predicts epistemic sentence type.
- Stage 2: XGBoost + Isotonic Regression predicts citation_needed probability in [0, 1] using Stage 1 outputs and handcrafted linguistic features.

## Research Questions
1. RQ1: Can a two-stage architecture (epistemic classification then citation scoring) outperform or match an end-to-end DistilBERT baseline while providing better interpretability?
2. RQ2: Is epistemic type a stronger predictor of citation need than surface-level linguistic features?
3. RQ3: How much does performance degrade when a model trained on Wikipedia-style data is evaluated on out-of-domain News and Essays data?

## Hypotheses
- H1: Predicted epistemic type is a stronger predictor of citation need than sentence-level surface features.
- H2: The two-stage pipeline provides better interpretability (via SHAP and decomposed reasoning) than an end-to-end DistilBERT classifier with comparable predictive performance.
- H3: Models trained mainly on Wikipedia editorial patterns will show statistically meaningful generalization drop on News and Essays.

## Evaluation Metrics
### Stage 1 (Epistemic Classification)
- Primary: Macro-F1
- Secondary: Accuracy
- Diagnostics: Per-class Precision, Per-class Recall

### Stage 2 (Citation Need Scoring)
- Primary: ROC-AUC
- Secondary: PR-AUC
- Calibration: Brier Score, Expected Calibration Error (ECE)

## Label Operational Definitions
### Claim
- Operational definition: A sentence asserting a disputable proposition, conclusion, or stance that is not self-justified inside the sentence.
- Positive example: "Open primaries improve democratic participation."
- Negative example: "The paper was published in 2021."

### Fact
- Operational definition: A declarative statement presented as true but without explicit supporting source, citation marker, or concrete evidence in the sentence itself.
- Positive example: "The city has a population of over one million."
- Negative example: "According to the 2020 census, the city has 1,043,210 residents."

### Evidence
- Operational definition: A statement that explicitly provides support for a proposition, typically by citing a source, statistic, experiment, quote, or dataset.
- Positive example: "According to World Bank data, GDP grew by 3.2% in 2023."
- Negative example: "GDP grew quickly in recent years."

### Opinion
- Operational definition: A subjective judgment, preference, or evaluative statement often tied to personal or normative language.
- Positive example: "This policy is unfair to low-income families."
- Negative example: "The policy was enacted in 2018."

### Background
- Operational definition: Context-setting content that introduces topic, definitions, or narrative setup without making a key disputable assertion.
- Positive example: "Climate negotiations have occurred annually since the 1990s."
- Negative example: "This treaty reduced emissions by 30%."

### Fact vs Evidence Boundary (Draft)
- Draft rule: Evidence = Fact with explicit support signal.
- Support signal examples: citation marker (e.g., [12], (Author, 2020)), named source attribution ("according to WHO"), concrete measured result tied to source.
- If no explicit support signal exists, default to Fact.

## WikiSQE citation_needed Mapping (Draft)
Dataset: ando55/WikiSQE_experiment (config: citation)
- label = 1 -> citation_needed = 1 (positive)
- label = 0 -> citation_needed = 0 (negative)

Draft filtering notes for training:
- Use official splits: train / val / test.
- Remove empty or non-string text rows.
- Keep only English sentence rows as provided.

Potential refinement for later review:
- Optionally drop extremely short sentences (e.g., < 5 tokens) as low-information noise.
- Optionally run duplicate removal by normalized text hash within split.

## Risk Log
1. Risk: Label noise and domain mismatch between Wikipedia supervision and OOD data.
- Mitigation: Keep News/Essays strictly held out; report OOD delta and confidence intervals.

2. Risk: Error propagation from Stage 1 to Stage 2.
- Mitigation: Run ablation with oracle Stage 1 labels vs predicted labels to quantify propagation loss.

3. Risk: Calibration drift across domains.
- Mitigation: Evaluate Brier/ECE per domain; recalibrate on small validation sets when allowed.

## Timeline (4 Weeks Approx.)
1. Phase 1 (3 days): Data acquisition, schema checks, and split verification.
2. Phase 2 (5 days): Stage 1 label mapping, training, and class-wise diagnostics.
3. Phase 3 (5 days): 12-feature extractor implementation and validation.
4. Phase 4 (5 days): Stage 2 training, calibration, and in-domain metrics.
5. Phase 5 (5 days): OOD evaluation on News/Essays and error taxonomy.
6. Phase 6 (5 days): SHAP analysis, report finalization, and demo polishing.
