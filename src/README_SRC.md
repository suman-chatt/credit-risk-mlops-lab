# Credit Risk Modeling & MLOps Pipeline (SRC)

## Overview

This project builds a **production-grade credit risk modeling pipeline** using the Kaggle *Give Me Some Credit* dataset.

The objective is not just predictive performance — it is to demonstrate a **complete, real-world machine learning system**, including:

- End-to-end data → model → decision pipeline
- Multiple modeling approaches (interpretable + ML + NN + ensembles)
- Champion / challenger framework
- Production-style scoring
- Monitoring and drift detection
- Explainability and fairness diagnostics

---

## Business Problem

The objective of this system is to estimate **Probability of Default (PD)** for consumer credit applicants.

This supports key real-world decisions:

- Credit approval / decline
- Risk-based pricing
- Credit limit assignment
- Portfolio risk monitoring

The model outputs are structured to align with how banks actually operate:

- PD score (`production_pd`)
- Risk bands (Low → Very High Risk)
- Percentile ranking

This allows downstream systems (pricing, underwriting, collections) to act on model outputs.

---

## Project Structure

```text
src/
├── models/           # Model training (logistic, NB, NN, trees)
├── ensemble/         # Stacking + weighted ensembles
├── scoring/          # Scoring pipeline + risk bands
├── monitoring/       # Drift + stability monitoring
├── explainability/   # Feature importance, SHAP, fairness
├── pipeline/         # Entry points (run_*.py)
└── config.py         # Central configuration
```
---

## Outputs

All outputs are written to:

```text
outputs_src/
├── registry/         # Model registry + artifacts
├── ensemble/         # Ensemble registry
├── scoring/          # Predictions + summaries
├── monitoring/       # Drift outputs
├── explainability/   # Model explanations
```
## Pipeline Execution Order

Run the full pipeline in sequence:

```bash
python -m src.pipeline.run_preprocessing
python -m src.pipeline.run_binning
python -m src.pipeline.run_scaling
python -m src.pipeline.run_feature_engineering
python -m src.pipeline.run_training
python -m src.pipeline.run_ensemble
python -m src.pipeline.run_scoring
python -m src.pipeline.run_monitoring
python -m src.pipeline.run_explainability
```
## Data Representations

Multiple dataset versions are created to support different model families:

| Dataset Type   | Description                                      |
|----------------|--------------------------------------------------|
| `tree`         | Raw / minimally processed features               |
| `scaled`       | Standardized continuous variables                |
| `woe`          | Weight of Evidence encoded features              |
| `binned_logit` | Manually binned variables for interpretable logit|
| `binned_ohe`   | One-hot encoded binned variables                 |

---

## Model Families

The system trains and compares a broad spectrum of models to balance performance, stability, and interpretability:

### Interpretable Models
- WOE Logistic Regression
- Binned Logistic Regression

### Linear / Baseline Models
- Regularized Logistic Regression (L1 / L2)
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- WOE Gaussian Naive Bayes

### Machine Learning Models
- Decision Trees
- Random Forests
- Extra Trees
- HistGradientBoosting

### Gradient Boosting Models
- XGBoost
- LightGBM
- CatBoost

### Neural Networks
- Multi-layer Perceptron (MLPClassifier variants)

---

## Ensemble Models

Ensemble methods are built using base model predictions:

### Stacking Models
- Logistic Regression (L1 / L2)
- Ridge / Lasso stacking
- Neural Network stacking
- Random Forest stacking
- XGBoost stacking

### Weighted Ensembles
- Equal-weight averaging
- AUC-weighted averaging
- KS-weighted averaging
- Log-loss inverse weighting
- Optimized weight search

---

## Champion Selection Framework

Models are not selected purely on AUC. Instead, a structured framework is used:

- **Performance Champion** → Best overall predictive power (typically ensemble)
- **Practical Champion** → Strong performance with simpler deployment (e.g., CatBoost)
- **Neural Network Challenger**
- **Simple Model Challenger** (e.g., Naive Bayes)
- **Interpretable Challengers**
  - WOE Logistic
  - Binned Logistic

This ensures:
- Redundancy
- Explainability coverage
- Production flexibility

---

## Scoring Pipeline

The scoring layer simulates a production decision system:

- Loads champion models from registry
- Scores full applicant population
- Produces:
  - Probability of Default (`production_pd`)
  - Score percentiles
  - Risk bands

### Outputs
- `scored_applicants.csv`
- `all_model_scores.csv`
- `score_summary.csv`
- `score_correlations.csv`
- `risk_band_summary.csv`

---

## Monitoring

Monitoring ensures model reliability over time.

### Drift Detection
- Feature distribution drift (PSI-style)
- Score distribution drift

### Outputs
- `feature_drift_summary.csv`
- `feature_drift_buckets.csv`
- `score_drift_summary.csv`
- `score_drift_buckets.csv`

### Recommendations
- Automated monitoring recommendations based on drift severity

---

## Stability Tracking

(Framework ready — extendable with time-based data)

- Score stability over time
- Performance stability (AUC, KS)

---

## Explainability

Provides transparency into model behavior.

### Global Explainability
- Feature importance
- SHAP global importance

### Local Explainability
- Applicant-level reason codes
- SHAP-based explanations

### Fairness Analysis
- Group-based score comparisons:
  - Age
  - Income
  - Dependents

### Outputs
- `global_feature_importance.csv`
- `shap_global_importance.csv`
- `applicant_shap_reasons.csv`
- `fairness_*_group.csv`
- `champion_score_comparison.csv`

---

## Key Design Principles

- **Modular architecture** — each step independently testable
- **Model-agnostic pipeline** — supports multiple families
- **Production realism** — scoring, monitoring, explainability included
- **Separation of concerns** — training vs scoring vs monitoring
- **Registry-driven system** — models and artifacts tracked centrally

---

## Model Governance & Risk Management

This pipeline is designed to align with Model Risk Management (MRM) principles:

- **Model registry** tracks all trained models and artifacts
- **Champion / challenger framework** enables controlled model updates
- **Monitoring layer** detects data and score drift
- **Explainability layer** ensures transparency for regulatory review
- **Fairness checks** support responsible AI practices

This mirrors real-world regulatory expectations (e.g., SR 11-7, internal model governance frameworks).

---

## Production Roadmap

The current system is designed with production-readiness in mind. The following enhancements would be implemented in a real deployment:

- **MLflow integration**  
  For experiment tracking, model versioning, and reproducibility

- **Time-based backtesting datasets**  
  To properly evaluate model stability and temporal drift

- **Production data ingestion pipeline**  
  Integration with real-time or batch data sources

- **Interactive dashboard (Streamlit / BI tool)**  
  - Model comparison and champion tracking  
  - Risk segmentation and portfolio views  
  - Monitoring alerts and drift visualization  
  - Explainability and fairness insights  

These extensions are aligned with how credit risk systems are deployed and governed in production environments.

---

## Summary

This project demonstrates a **full-stack credit risk ML system**, covering:

- Data engineering
- Feature engineering
- Model development
- Model selection
- Production scoring
- Monitoring and governance
- Explainability and fairness

It is designed to reflect **real-world banking / fintech modeling workflows**, not just isolated modeling performance.