# Credit Risk Modeling & Model Risk Framework

An end-to-end credit risk modeling system designed to simulate a real-world banking workflow:

- Predict Probability of Default (PD)
- Compare multiple model families
- Evaluate performance, stability, and interpretability
- Select a production-ready champion model
- Support model governance and monitoring

## Key Highlights

- 10+ model families (Logistic, Boosting, NN, Ensembles)
- Full champion / challenger framework
- Interpretability review (coefficients, feature importance, ensemble logic)
- Risk-band and scoring analysis
- Synthetic monitoring and drift framework
- Modular, production-style Python pipeline

## Tech Stack

- Python (pandas, sklearn, xgboost, lightgbm, catboost)
- Streamlit (interactive dashboards)
- Plotly (visualization)
- MLflow (experiment tracking)

## Data Contracts

The system assumes consistent input/output structures:

### Inputs

- borrower-level dataset
- numeric and categorical features
- binary target (default / no default)

### Outputs

- model predictions (PD)
- risk bands
- evaluation metrics
- monitoring indicators

All intermediate datasets are saved under `/data` or `/artifacts` for traceability.

## Pipeline Execution

The `pipeline/` folder contains orchestration scripts for each stage of the workflow.

Examples:

- `run_preprocessing.py` → prepares clean datasets  
- `run_feature_engineering.py` → generates model-ready features  
- `run_training.py` → trains all models and logs outputs  
- `run_scoring.py` → generates predictions and risk bands  
- `run_monitoring.py` → evaluates stability and drift  

These scripts allow the workflow to be executed step-by-step or end-to-end.

## Modeling Design

The modeling layer supports multiple model families with a consistent interface:

- Each model is trained using standardized inputs
- Outputs are stored in a central model registry
- Predictions are generated using reusable scoring logic
- Ensembles operate on base model predictions rather than raw features

This ensures comparability and reproducibility across all models.

## Monitoring Framework

The monitoring module simulates production monitoring using synthetic cohorts.

It evaluates:

- population drift (PSI, JS, Wasserstein)
- score stability
- risk distribution changes

Note:
This is a **framework demonstration**, not a live production monitoring system.

[Raw Data] → [Preprocessing] → [Feature Engineering] → [Models] → [Evaluation]
                                             ↓
                                      [Ensemble]
                                             ↓
                                     [Scoring]
                                             ↓
                                   [Monitoring]
                                             ↓
                                 [Streamlit App]