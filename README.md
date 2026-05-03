# Credit Risk Modeling & Model Risk Framework

🔗 **Live Interactive Dashboard:** [ADD_YOUR_STREAMLIT_LINK_HERE]

End-to-end credit risk system designed to mirror real-world banking workflows — from raw borrower data to production-style decision outputs.

This project goes beyond model performance to answer the real question:

> **Which model is not just accurate, but usable in a real lending environment?**

It integrates modeling, scoring, risk segmentation, and monitoring into a single, consistent decision pipeline.

---

## What This Project Does

- Estimates **Probability of Default (PD)** at borrower level  
- Compares multiple model families under a unified framework  
- Translates predictions into **risk bands and decision signals**  
- Evaluates models on **performance, stability, and interpretability**  
- Simulates **model monitoring and drift detection**  
- Supports **champion vs challenger model governance**

---

## Why This Matters

In real-world credit risk:

- Small improvements in risk ranking can drive large financial impact  
- The best-performing model is not always the best production model  
- Models must be explainable, stable, and operationally feasible  

This project demonstrates how to move from:

> “Which model scores highest?”  
to  
> “Which model should actually be deployed?”

---

## Key Highlights

- 10+ model families (Logistic, Boosting, Neural Networks, Ensembles)
- Full **champion / challenger framework**
- Risk-band and scoring system aligned to business decisions
- Interpretability review tailored by model type
- Synthetic monitoring and drift detection framework
- Modular, production-style Python pipeline
- Interactive **Streamlit dashboard for model evaluation**

---

## Tech Stack

- Python (pandas, scikit-learn, xgboost, lightgbm, catboost)
- Streamlit (interactive app)
- Plotly (visualization)
- MLflow (experiment tracking)

---

## Pipeline Overview

[Raw Data]
↓
[Preprocessing]
↓
[Feature Engineering]
↓
[Model Training]
↓
[Evaluation]
↓
[Ensemble]
↓
[Scoring & Risk Bands]
↓
[Monitoring]
↓
[Streamlit App]

## Data Contracts

### Inputs

- Borrower-level dataset  

- Numeric and categorical features  

- Binary target (default / no default)  

### Outputs

- Model predictions (PD)  

- Risk bands  

- Evaluation metrics  

- Monitoring indicators  

All intermediate datasets are saved under `/data` or `/artifacts` for traceability.

---

## Pipeline Execution

The `pipeline/` folder contains orchestration scripts for each stage.

Examples:

- `run_preprocessing.py` → prepares clean datasets  

- `run_feature_engineering.py` → generates model-ready features  

- `run_training.py` → trains models and logs outputs  

- `run_scoring.py` → generates predictions and risk bands  

- `run_monitoring.py` → evaluates stability and drift  

The workflow can be executed step-by-step or end-to-end.

---

## Modeling Design

- Standardized input/output structure across all models  

- Centralized model registry for tracking outputs  

- Reusable scoring logic across models  

- Ensembles built on base model predictions  

This ensures **comparability, reproducibility, and auditability**.

---

## Monitoring Framework

The monitoring module simulates production monitoring using synthetic cohorts.

It evaluates:

- Population drift (PSI, JS, Wasserstein)

- Score stability

- Risk distribution changes

> Note: This is a framework demonstration, not a live production system.

---

## Deployment Context

In a real system:

1. Application received  

2. Data validated and transformed  

3. Features generated  

4. Model predicts PD  

5. Risk band assigned  

6. Business rules applied  

7. Decision made  

8. Model monitored over time  

The model is one component of a controlled decision system.

---

## Limitations

- Based on historical dataset (not real-time production data)  

- No true time-series monitoring  

- No protected attributes → fairness testing not included  

- Some complex models reduce interpretability  

These are explicitly acknowledged to reflect real-world model risk considerations.

---

## How to Run

```bash

# Create virtual environment

python -m venv .venv

source .venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Run Streamlit app

streamlit run streamlit_app/Home.py