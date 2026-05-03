import streamlit as st
import plotly.express as px

from utils.data_loader import load_full_model_registry


st.set_page_config(
    page_title="Project Summary",
    layout="wide",
)

st.title("Project Summary")
st.caption("Recruiter-ready overview of the credit risk modeling and model-risk management project.")

st.info("Use the left sidebar to navigate through the full Streamlit application.")


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = load_full_model_registry()

if df.empty:
    st.error("Model registry could not be loaded. Please run the model training/export pipeline first.")
    st.stop()


# --------------------------------------------------
# Sidebar TOC
# --------------------------------------------------

st.sidebar.title("Navigation")

st.sidebar.markdown(
    """
- Executive Summary
- Business Question
- Economic Importance
- Data Overview
- Modeling Workflow
- Model Families
- Evaluation Framework
- Interpretability
- Scoring and Risk Bands
- Monitoring
- Champion Selection
- Limitations
- Deployment Context
- Recruiter Takeaway
"""
)


# --------------------------------------------------
# 1. Executive Summary
# --------------------------------------------------

st.markdown("## 1. Executive Summary")

st.markdown(
    """
This project builds an end-to-end **credit risk modeling and model-risk management framework** using borrower-level financial data.

The objective is straightforward:

> **Estimate borrower default risk and translate it into consistent, defensible lending decisions.**

What makes this project valuable is not just model performance, but the full decision pipeline:

- data preparation and feature engineering
- multi-model training across diverse model families
- structured model comparison (leaderboard, diagnostics)
- interpretability and governance review
- score behavior and risk segmentation
- synthetic monitoring and stress testing
- champion vs challenger selection

This project answers two questions:

> “Which model performs best?”  
> “Which model is actually usable in a real credit-risk environment?”

The second question drives the final outcome.
"""
)

# --------------------------------------------------
# 2. Business Question
# --------------------------------------------------

st.markdown("## 2. Business Question")

st.markdown(
    """
The core business question is:

> **Can borrower financial and credit behavior data be used to reliably rank default risk?**

In a lending environment, this ranking supports decisions such as:

- application approval or review
- risk-based pricing
- credit limit strategy
- portfolio risk segmentation
- monitoring of changing borrower quality

The model does **not** replace business judgment.  
It provides a consistent, data-driven risk signal that can support human and policy-driven decision-making.
"""
)


# --------------------------------------------------
# 3. Economic Importance
# --------------------------------------------------

st.markdown("## 3. Economic Importance")

st.markdown(
    """
Credit risk models matter because small improvements in risk ranking can have large financial impact at portfolio scale.

A better model can help a lender:

- identify high-risk borrowers earlier
- reduce unexpected credit losses
- approve more low-risk borrowers with confidence
- improve pricing discipline
- monitor portfolio quality over time

The economic value comes from the tradeoff between:

- **risk control**: avoiding borrowers likely to default
- **growth**: approving profitable borrowers who are likely to repay

A strong model improves this tradeoff.
"""
)


# --------------------------------------------------
# 4. Data Overview
# --------------------------------------------------

st.markdown("## 4. Data Overview")

st.markdown(
    """
The project uses borrower-level credit and financial variables.

Examples of input variables include:

- revolving credit utilization
- age
- debt ratio
- monthly income
- number of open credit lines
- number of real estate loans
- number of dependents
- past-due payment history

The target variable represents whether the borrower experiences serious delinquency.

In business language:

> The model estimates **Probability of Default (PD)** or serious delinquency risk.
"""
)


# --------------------------------------------------
# 5. Modeling Workflow
# --------------------------------------------------

st.markdown("## 5. Modeling Workflow")

st.markdown(
    """
The workflow was built to resemble a practical credit-risk modeling process.

### Step 1: Data preparation

The raw dataset was cleaned and transformed.

Key preparation steps included:

- missing value treatment
- outlier handling
- capped variables
- missingness flags
- transformed variables
- model-ready datasets for different model families

### Step 2: Feature engineering

Different model families require different feature formats.

The project created multiple model-ready views of the same borrower data:

- raw/capped features for tree-based models
- scaled features for neural networks
- binned features for bin-based models
- Weight-of-Evidence style features for logistic regression

### Step 3: Model training

Multiple model families were trained and compared.

### Step 4: Validation review

Models were evaluated on validation performance, score behavior, calibration, lift, gains, risk bands, and stability.

### Step 5: Champion selection

The final model decision is based on both statistical performance and practical model-risk considerations.
"""
)


# --------------------------------------------------
# 6. Model Families
# --------------------------------------------------

st.markdown("## 6. Model Families Trained")

st.markdown(
    """
This project compares a broad set of model families.

| Model Family | What it represents | Why it matters |
|---|---|---|
| Logistic Regression | Transparent linear model | Traditional credit-risk benchmark |
| WOE Logistic Regression | Logistic model using Weight-of-Evidence variables | Common in banking scorecards |
| Regularized Logistic Regression | Penalized logistic model | Controls complexity and overfitting |
| Naive Bayes | Probabilistic classifier | Simple challenger baseline |
| Decision Trees / Random Forests | Rule-based tree models | Captures non-linear patterns |
| Gradient Boosting | XGBoost, LightGBM, CatBoost, sklearn boosting | Strong tabular-data performance |
| Neural Networks | Multilayer perceptron models | Tests flexible non-linear modeling |
| Ensembles / Stacking | Combines multiple models | Tests whether combinations improve performance |

The app consistently compares:

- **CAT002** as the modeler-recommended practical champion
- **XGBSTACK001** as the best validation-AUC benchmark
- up to five user-selected challenger models
"""
)


# --------------------------------------------------
# 7. Key Results
# --------------------------------------------------

st.markdown("## 7. Key Results")

st.markdown(
    """
The project produced several practical findings:

- Boosting models and ensembles delivered the strongest validation performance.
- `XGBSTACK001` achieved the highest AUC and represents the performance ceiling.
- `CAT002` was selected as the practical champion due to strong performance and lower complexity.
- Logistic models remained valuable as interpretable and stable benchmarks.
- Neural networks and ensembles increased complexity without consistently improving decision outcomes.

### Most important takeaway

> Higher AUC alone does not justify model selection.

Model choice must also consider:

- stability under population changes  
- interpretability for model review  
- operational complexity  
- impact on real decision thresholds  

This is what separates a strong model from a usable one.
"""
)


# --------------------------------------------------
# 8. Performance Chart
# --------------------------------------------------

st.markdown("## 8. Model Performance Snapshot")

st.markdown(
    """
The chart below compares top models using validation AUC and validation log loss.

- Higher AUC means better ranking power.
- Lower log loss means better probability quality.
- A good candidate should perform well on both dimensions.
"""
)

top_models = df.sort_values("validation_auc", ascending=False).head(20)

fig = px.scatter(
    top_models,
    x="validation_log_loss",
    y="validation_auc",
    color="model_family",
    hover_data=["model_id", "validation_ks"],
    title="Top Models: Validation AUC vs Log Loss",
)

fig.update_xaxes(title="Validation Log Loss")
fig.update_yaxes(title="Validation AUC")
fig.update_layout(height=520)

st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# 9. Evaluation Framework
# --------------------------------------------------

st.markdown("## 9. Evaluation Framework")

st.markdown(
    """
Models were evaluated using multiple diagnostics because no single metric is enough.

| Metric | What it answers |
|---|---|
| AUC | Can the model rank risky borrowers above safer borrowers? |
| KS | How strongly does the model separate defaults from non-defaults? |
| Log Loss | Are predicted probabilities numerically useful? |
| Brier Score | How large is average probability error? |
| Gains / Lift | How much default risk is captured in the riskiest population segments? |
| Calibration | Do predicted probabilities match observed default rates? |
| Threshold Analysis | What happens when PD is converted into a business decision cutoff? |
| Risk Bands | Do score bands create meaningful borrower risk segments? |

This makes the analysis more realistic than simply sorting models by AUC.
"""
)


# --------------------------------------------------
# 10. Interpretability
# --------------------------------------------------

st.markdown("## 10. Interpretability Review")

st.markdown(
    """
The interpretability review evaluates whether model outputs can be explained in a way that satisfies model-risk governance requirements.

Rather than forcing a single explanation method, the project uses model-appropriate approaches:

- logistic models → coefficient and directional analysis  
- tree/boosting models → feature importance  
- ensembles → base-model contribution analysis  

This reflects how real model validation works in practice.
"""
)

st.warning(
    """
SHAP/local explanation outputs are not included in this version.

That is intentional. The project focuses on consistent comparison across many model families.  
In a production model approval process, local explainability such as SHAP would be applied to shortlisted final models, not blindly applied across every experimental model.
"""
)


# --------------------------------------------------
# 11. Scoring and Risk Bands
# --------------------------------------------------

st.markdown("## 11. Scoring and Risk Bands")

st.markdown(
    """
The scoring page translates model predictions into business-facing outputs.

It reviews:

- predicted probability of default
- score distributions
- score percentiles
- risk-band assignment
- actual default rates by risk band
- risk-band transitions between models
- top-risk overlap
- top-risk capture
- threshold-based predicted default flags

This matters because model performance only becomes useful when predictions can support decisions.

The scoring section answers:

> If this model were used operationally, how would it segment borrowers and affect decisions?
"""
)


# --------------------------------------------------
# 12. Monitoring
# --------------------------------------------------

st.markdown("## 12. Monitoring and Stability")

st.markdown(
    """
The dataset does not contain real time-series production data, so this project does not claim live monitoring.

Instead, it implements **synthetic monitoring** to simulate population shifts.

### What this tests

- how predicted PD changes as borrower risk mix increases  
- whether model ranking (AUC) remains stable  
- how approval / rejection behavior shifts under stress  
- whether monitoring signals remain interpretable  

### Why this matters

In production, models fail not because AUC drops immediately, but because:

- population shifts  
- score distributions drift  
- decision thresholds behave unpredictably  

This section demonstrates how those risks would be identified.

It is a **monitoring framework**, not just a metric report.
"""
)


# --------------------------------------------------
# 13. Champion Selection
# --------------------------------------------------

st.markdown("## 13. Champion Selection")

st.markdown(
    """
The Champion Selection page consolidates all prior analysis into a final decision.

It compares:

- **Recommended Champion:** `CAT002`
- **Benchmark Challenger:** `XGBSTACK001`
- **User-Selected Candidate**

The decision is based on:

- performance (AUC, KS, calibration)
- score behavior (risk bands, thresholds)
- stability under synthetic stress
- interpretability and explainability
- operational complexity
- governance feasibility

### Final position

> **CAT002 is selected as the production-style champion.**

It provides:

- strong performance close to the best benchmark  
- simpler implementation than stacked ensembles  
- more interpretable behavior  
- more stable decision outcomes  

`XGBSTACK001` remains a critical challenger as the performance benchmark.

This reflects real-world model governance:

- one model for production  
- one model to continuously challenge it  
"""
)

# --------------------------------------------------
# 14. Limitations
# --------------------------------------------------

st.markdown("## 14. Limitations")

st.markdown(
    """
This project is intentionally transparent about its limitations.

### Data limitations

- The dataset is historical and does not represent live production data.
- No true time-period field is available for real temporal monitoring.
- Protected attributes are not available, so true fairness testing is not performed.
- External macroeconomic variables are not included.

### Modeling limitations

- Validation performance is not the same as live production performance.
- Some complex models are harder to explain.
- Ensembles may improve metrics but increase governance burden.
- Thresholds used in the app are decision simulations, not final business policy.

### Explainability limitations

- SHAP/local explanations are not implemented in this version.
- Fairness metrics are not implemented because protected or approved segmentation attributes are unavailable.

These limitations are not hidden. They are documented because realistic model-risk work requires clarity about what has and has not been proven.
"""
)


# --------------------------------------------------
# 15. Deployment Context
# --------------------------------------------------

st.markdown("## 15. Deployment Context")

st.markdown(
    """
A production deployment would place the model inside a larger credit decision system.

A typical workflow would be:

1. Borrower application is received.
2. Data is validated and cleaned.
3. Features are created using approved transformations.
4. The champion model generates predicted PD.
5. The applicant is assigned to a risk band.
6. Business policy applies approval, pricing, or review rules.
7. Outcomes are monitored after deployment.
8. Model performance, drift, and stability are reviewed periodically.

The model is one component of a controlled decision process.
"""
)


# --------------------------------------------------
# 16. What This Project Demonstrates
# --------------------------------------------------

st.markdown("## 16. What This Project Demonstrates")

st.markdown(
    """
This project demonstrates end-to-end capability across modeling and decision systems.

### Technical

- multi-model training and comparison
- feature engineering across model families
- structured model outputs and diagnostics
- Streamlit-based analytics interface

### Data Science

- model evaluation beyond AUC
- calibration and threshold thinking
- risk segmentation and score behavior
- stability and stress testing

### Model Risk / Banking

- interpretability review aligned to model type
- governance-aware model selection
- monitoring framework design
- champion / challenger structure

The output is not just a model.

> It is a **decision framework that could be used in a real credit-risk environment.**
"""
)