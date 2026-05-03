# Credit Risk Modeling System — src Overview

## Purpose

The `src/` directory contains the **production-style implementation** of the credit risk modeling system.

Unlike the notebooks, which are exploratory and iterative, the `src/` code is:

- deterministic  
- modular  
- reusable  
- structured for deployment  

---

## Design Philosophy

This system follows a few key principles:

### 1. Single Responsibility per Module
Each file has one clear purpose:
- data loading
- preprocessing
- feature engineering
- modeling
- monitoring
- explainability

---

### 2. No Hidden Logic
All transformations are explicit.

Example:
- No silent imputation
- No implicit column changes
- No global variables

---

### 3. Fit vs Transform Separation

Any step that learns from data is split into:

- `fit` → learns parameters (e.g., medians, bins, WOE)
- `transform` → applies them

This prevents:
- data leakage
- inconsistent scoring behavior

---

### 4. Artifacts are First-Class

All learned components are saved and reused:

- preprocessing parameters (medians, caps)
- binning rules
- WOE mappings
- trained models

This ensures:
- reproducibility
- auditability
- production consistency

---

## Folder Structure

src/
├── config.py                  # global paths and constants
│
├── data/
│   └── load_data.py           # raw data loading
│
├── preprocessing/
│   ├── preprocessing.py       # cleaning + transformations
│   └── preprocessing_config.py
│
├── features/
│   ├── binning.py
│   ├── woe.py
│   └── feature_engineering.py
│
├── models/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── ensemble.py
│
├── monitoring/
│   ├── drift.py
│   └── stability.py
│
├── explainability/
│   ├── shap_utils.py
│   └── reason_codes.py
│
├── pipeline/
│   └── run_pipeline.py        # orchestration layer
│
└── utils/
├── io.py                 # saving/loading artifacts
└── validation.py         # input validation

---

## Data Flow (End-to-End)

Raw Data
↓
load_data.py
↓
preprocessing.py
↓
feature_engineering.py
↓
train.py
↓
ensemble.py
↓
predict.py
↓
monitoring / explainability

---

## Key Components

### 1. Data Loading

`data/load_data.py`

- Reads raw files from `/data`
- No transformations applied

---

### 2. Preprocessing

`preprocessing/preprocessing.py`

Responsible for:
- duplicate removal
- invalid value filtering
- missing value handling
- feature transformations (caps, logs, flags)

Outputs:
- clean, model-ready base dataset

---

### 3. Feature Engineering

`features/`

Includes:
- binning logic
- WOE transformations
- dataset preparation for different model types

---

### 4. Modeling

`models/`

Includes:
- model training
- evaluation
- ensemble construction
- prediction generation

---

### 5. Monitoring

`monitoring/`

Tracks:
- PSI
- CSI
- distribution drift
- stability metrics

---

### 6. Explainability

`explainability/`

Provides:
- SHAP values
- reason codes
- driver analysis

---

### 7. Pipeline

`pipeline/run_pipeline.py`

This is the entry point.

It orchestrates:
- data loading
- preprocessing
- feature engineering
- model training
- evaluation
- artifact saving

---

## Deployment North Star

The codebase is designed to support three execution environments:

1. Local development in VSCode  
2. Streamlit application deployment  
3. Databricks workflow execution with MLflow tracking  

Core business logic lives in `src/`.

Environment-specific orchestration should live outside the core modules:

- `apps/` for Streamlit  
- `databricks/` for Databricks notebooks/jobs  
- `src/pipeline/` for local orchestration  

MLflow should be used primarily in the training and evaluation workflow.

The following modules should remain framework-independent:

- preprocessing  
- feature engineering  
- scoring  
- monitoring  
- explainability  

---

## Important Notes

- `src/` code contains **no EDA logic**
- notebooks remain the source of experimentation
- all transformations in `src/` must be reproducible
- no print-heavy debugging or display logic is used

---
## Notebook vs src Differences

The notebooks are treated as exploratory validation work, not the final production source of truth.

The `src/` implementation is allowed to intentionally differ from notebooks when the difference is documented and justified.

Current intentional difference:

- `DebtRatio_explicit_bin` is included in the `src` WOE/logistic feature set.
- The notebook final WOE set excluded raw `DebtRatio`, but later review suggested this variable may carry useful incremental signal.
- Because `DebtRatio_explicit_bin` enters the IV ranking at rank 6, lower-ranked variables shift down by one position.
- This is expected and acceptable.
- The project goal is not to exactly reproduce notebook-selected champions, but to convert validated notebook logic into a cleaner, more complete production-style modeling pipeline.

In a real production workflow, notebooks may be run on samples for exploration and validation, while modular Python code performs the full repeatable pipeline on larger datasets.

---

## How to Use

Eventually, the system will be run via:

```bash
python src/pipeline/run_pipeline.py