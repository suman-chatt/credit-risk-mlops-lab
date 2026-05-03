## Model Selection Narrative

The model tournament evaluated a broad set of approaches, including:

- interpretable models (logistic / WOE / binned logistic)  
- classical machine learning models (Naive Bayes, tree-based models)  
- neural networks  
- gradient boosting models  
- ensemble and stacking models  

### What the results showed

Tree-based boosting models — particularly **CatBoost, XGBoost, and LightGBM** — consistently delivered the strongest validation performance.

- These models achieved the highest **AUC** and **KS**, indicating strong ranking power  
- They captured **nonlinear relationships and interactions** more effectively than linear or probabilistic models  
- Simpler models (logistic, Naive Bayes) remained competitive but did not match boosting performance  

### Key insight on ensembles

Although stacking and ensemble models were tested, the results showed:

> The best ensemble model did **not materially outperform** the best individual boosting model.

This is a critical finding.

While ensembles can improve performance in some cases, they introduce:

- higher complexity  
- greater governance burden  
- more difficult explainability  
- more operational overhead  

If performance gains are marginal, that complexity is often **not justified in production**.

---

## Recommended Champion Model

**CAT002 — CatBoostClassifier**

### Why this model was selected

CAT002 represents the best balance across all evaluation dimensions:

- Strong validation AUC and KS  
- Competitive log loss and Brier score  
- Stable scoring behavior  
- Consistent performance across synthetic monitoring cohorts  
- Good top-risk capture and decision behavior  
- Well-suited for tabular credit-risk data  

### Practical advantages

Compared to stacked ensembles:

- simpler deployment architecture  
- fewer moving parts in production  
- easier to monitor and troubleshoot  
- clearer feature-level explanations (via importance)  
- lower model-risk governance burden  

> In a real banking environment, these factors matter just as much as raw performance.

---

## Role of the Benchmark Model

**XGBSTACK001 — Stacking Ensemble**

This model remains important, but in a different role:

- Represents the **upper bound of validation performance**  
- Serves as a **benchmark challenger**  
- Useful for ongoing comparison and monitoring  

However:

- higher complexity  
- harder to explain end-to-end  
- greater documentation and governance requirements  

> It is better positioned as a challenger than a production champion.

---

## Recommended Champion–Challenger Framework

To reflect a realistic credit-risk modeling setup:

- **Production Champion:** CAT002  
- **Benchmark Challenger:** XGBSTACK001  
- **Boosting Challenger:** XGB002  
- **Neural Network Challenger:** NN008  
- **Interpretable Challenger:** WLOG005 or BLOG005  
- **Simple Baseline Challenger:** BNB018  

---

## Final Takeaway

The final model choice is not driven by AUC alone.

It reflects a broader decision:

> **Choose the model that delivers strong performance while remaining stable, explainable, and operationally manageable.**

CAT002 satisfies that balance and is therefore the most appropriate **production-style champion** in this project.