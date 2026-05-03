# Credit Risk Modeling: Economic Rationale and Business Value

## 1. The Real-World Problem

Lenders face a fundamental economic question:

> Which borrowers are likely to default, and how should that risk influence lending decisions?

Every loan issued carries uncertainty.

- Approve too many high-risk borrowers → losses increase  
- Reject too many low-risk borrowers → revenue is lost  

This creates a core trade-off:

- **Risk** (credit losses)  
- **Revenue** (loan growth and interest income)  

The objective of this project is to quantify borrower risk in a **consistent, data-driven way** so that this trade-off can be optimized.

---

## 2. What This Project Is Testing

This project evaluates multiple model families to answer:

> How well can we predict serious delinquency using borrower-level data, and which model is most suitable for real-world deployment?

We test fundamentally different approaches:

- Logistic Regression → interpretable, industry-standard baseline  
- Tree-Based Models (LightGBM, XGBoost, CatBoost) → nonlinear, high-performance models  
- Naive Bayes → simplified probabilistic baseline  
- Neural Networks → flexible nonlinear models  
- Ensembles / Stacking → combined model predictions  

The goal is not just to find the most accurate model, but to understand **how different modeling assumptions impact risk estimation**.

---

## 3. Why Multiple Models Matter

Each model type reflects a different view of how risk behaves:

- Logistic Regression → assumes smooth, linear relationships  
- Tree models → capture threshold effects (risk jumps at certain levels)  
- Naive Bayes → tests simplified independence assumptions  
- Neural networks → capture complex nonlinear interactions  

By comparing them, we are effectively asking:

> Is borrower risk primarily linear, threshold-driven, or interaction-driven?

This is not just a modeling decision — it is an **economic structure decision**.

---

## 4. What the Model Predicts

Target variable:

- **SeriousDlqin2yrs = 1 → borrower defaults**  
- **SeriousDlqin2yrs = 0 → borrower does not default**  

Model output:

> **Probability of Default (PD)**

This is the core quantity used in real-world credit decision systems.

---

## 5. How This Translates to Business Decisions

### 5.1 Credit Approval

Instead of static rules:

- Approve if PD < threshold  
- Reject if PD ≥ threshold  

This creates **consistent, explainable decision boundaries**.

---

### 5.2 Risk-Based Pricing

Higher-risk borrowers can still be approved, but priced differently:

- Higher PD → higher interest rate  
- Lower PD → lower interest rate  

This directly links model output to revenue generation.

---

### 5.3 Portfolio Risk Management

Aggregated PD estimates allow firms to answer:

- What is the expected default rate of the portfolio?  
- Are we overexposed to high-risk segments?  
- How does risk shift under changing borrower mix?  

---

### 5.4 Capital Allocation

Banks must hold capital against expected losses:

> Expected Loss ≈ PD × Exposure × Loss Given Default  

Better PD estimates → more efficient capital usage.

---

## 6. Why Evaluation Metrics Matter Economically

This project uses multiple metrics because each captures a different business concern:

- **AUC / KS** → ranking ability (who is riskier than whom)  
- **Log Loss / Brier Score** → probability accuracy (critical for pricing and capital)  
- **Top-risk capture / lift** → ability to isolate high-risk borrowers  
- **Threshold behavior** → impact on approvals and declines  

This ensures models are evaluated based on **decision impact**, not just statistical performance.

---

## 7. Key Insight on Model Complexity

A critical result from this project:

> The best ensemble model did not materially outperform the best single boosting model.

This has direct economic implications:

- Higher complexity increases governance and operational cost  
- Explainability becomes harder  
- Monitoring becomes more difficult  

If performance gains are marginal, complexity is **not justified**.

---

## 8. Champion Selection Logic

The project does not select a model based on AUC alone.

Instead, it evaluates:

- performance (AUC, KS, calibration)  
- score behavior (risk bands, thresholds)  
- stability (synthetic monitoring)  
- interpretability (coefficients, feature importance)  
- governance feasibility  

This leads to a realistic decision:

- **Champion:** CAT002 (balanced, practical model)  
- **Benchmark Challenger:** XGBSTACK001 (highest AUC, higher complexity)  

This mirrors real-world credit-risk model governance.

---

## 9. Why This Project Matters

This project demonstrates:

- understanding of credit risk economics  
- ability to evaluate models beyond accuracy  
- awareness of governance and deployment constraints  
- translation of model outputs into business decisions  

It is not just a modeling exercise.

> It is a **decision system design exercise**.

---

## 10. Final Takeaway

This project answers a simple but critical question:

> How can borrower risk be quantified in a way that improves lending decisions?

The output is not just a model.

It is a framework that can:

- reduce losses  
- improve pricing  
- optimize approvals  
- support portfolio-level decisions  

That is where the real economic value lies.