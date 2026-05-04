from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(
    page_title="Model Interpretability Review",
    layout="wide",
)

st.title("Model Interpretability Review")
st.caption(
    "Review whether candidate models are explainable, directionally sensible, and suitable for model governance."
)

# Paths

PROJECT_ROOT = Path.cwd()
INTERP_DIR = PROJECT_ROOT / "outputs_src" / "interpretability"

SUMMARY_PATH = INTERP_DIR / "summary" / "interpretability_model_summary.csv"
LOGISTIC_PATH = INTERP_DIR / "logistic" / "logistic_coefficients_summary.csv"
FEATURE_IMPORTANCE_PATH = INTERP_DIR / "feature_importance" / "feature_importance_summary.csv"
ENSEMBLE_WEIGHTS_PATH = INTERP_DIR / "ensemble" / "ensemble_base_model_weights.csv"
STACKING_IMPORTANCE_PATH = INTERP_DIR / "ensemble" / "stacking_meta_model_importance.csv"
GOVERNANCE_PATH = INTERP_DIR / "governance" / "model_governance_checks.csv"
SHAP_STATUS_PATH = INTERP_DIR / "shap" / "shap_export_status.csv"

MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"

# Load data

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


summary_df = load_csv(SUMMARY_PATH)
logistic_df = load_csv(LOGISTIC_PATH)
importance_df = load_csv(FEATURE_IMPORTANCE_PATH)
ensemble_weights_df = load_csv(ENSEMBLE_WEIGHTS_PATH)
stacking_df = load_csv(STACKING_IMPORTANCE_PATH)
governance_df = load_csv(GOVERNANCE_PATH)
shap_status_df = load_csv(SHAP_STATUS_PATH)

if summary_df.empty:
    st.error(
        """
Interpretability outputs were not found.

Run:

`python -m src.interpretability.export_interpretability_outputs`
"""
    )
    st.stop()

# Helpers

def build_table(data: pd.DataFrame, height: int = 350):
    if data.empty:
        st.info("No data available for this section.")
        return

    gb = GridOptionsBuilder.from_dataframe(data)

    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
    )

    if "model_id" in data.columns:
        gb.configure_column("model_id", pinned="left", width=150)
    if "feature" in data.columns:
        gb.configure_column("feature", pinned="left", width=280)
    if "governance_checks" in data.columns:
        gb.configure_column("governance_checks", width=560)
    if "interpretability_comment" in data.columns:
        gb.configure_column("interpretability_comment", width=520)
    if "direction_check" in data.columns:
        gb.configure_column("direction_check", width=180)

    AgGrid(
        data,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        theme="streamlit",
    )


def safe_round_cols(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def model_label(row: pd.Series) -> str:
    desc = model_description(
        row.get("model_id"),
        row.get("model_family"),
        row.get("model_name"),
    )

    return (
        f"{row['model_id']} | {row['model_family']} | "
        f"AUC={row.get('validation_auc', 0):.4f} | {desc}"
    )


def get_reference_models(selected_models: list[str]) -> list[str]:
    refs = [MODELERS_DEFAULT_MODEL_ID, BEST_AUC_MODEL_ID]
    refs.extend(selected_models)
    return list(dict.fromkeys([m for m in refs if m in summary_df["model_id"].values]))


def interpretability_score(row: pd.Series) -> str:
    flag = str(row.get("governance_flag", ""))

    if "missing" in flag:
        return "Not review-ready"
    if "high_complexity" in flag:
        return "Review required"
    if "possible_overfit" in flag:
        return "Review required"
    if "governance_friendly" in flag:
        return "Strong"
    return "Acceptable with support"


def clean_feature_name(feature: str) -> str:
    return str(feature).lower().replace("_woe", "").replace("_scaled", "")


def expected_direction_for_feature(feature: str) -> str:
    """
    Expected direction is expressed relative to default risk.

    positive = higher value / riskier bin should generally increase default risk
    negative = higher value / safer bin should generally reduce default risk
    mixed = business direction depends on bin definition or transformation
    neutral = no fixed business direction expected
    """
    f = clean_feature_name(feature)

    if f in {"const", "intercept"}:
        return "neutral"

    positive_patterns = [
        "90dayslate",
        "90_days_late",
        "numberoftimes90",
        "30-59",
        "30_59",
        "pastduenotworse",
        "60-89",
        "60_89",
        "revolvingutilization",
        "revolving_utilization",
        "debt ratio",
        "debtratio",
        "high_flag",
        "missing_flag",
        "100%+",
        "75-100",
        "50-75",
        "3+",
    ]

    negative_patterns = [
        "monthlyincome",
        "monthly_income",
        "income",
        "age",
        "70+",
        "60-69",
    ]

    mixed_patterns = [
        "numberofopencreditlines",
        "opencreditlines",
        "realestateloans",
        "realestateloansorlines",
        "numberofdependents",
        "dependents",
    ]

    if any(p in f for p in positive_patterns):
        return "positive"

    if any(p in f for p in negative_patterns):
        return "negative"

    if any(p in f for p in mixed_patterns):
        return "mixed"

    return "neutral"


def direction_check(expected: str, actual: str) -> str:
    if expected in {"neutral", "mixed", ""}:
        return "not_directional"

    if actual == expected:
        return "aligned"

    if actual in {"positive", "negative"} and actual != expected:
        return "review"

    return "not_assessed"


def add_direction_logic(logit_data: pd.DataFrame) -> pd.DataFrame:
    out = logit_data.copy()

    if out.empty or "feature" not in out.columns:
        return out

    if "actual_direction" not in out.columns and "coefficient" in out.columns:
        out["actual_direction"] = out["coefficient"].apply(
            lambda x: "positive" if x > 0 else ("negative" if x < 0 else "zero")
        )

    out["expected_direction"] = out["feature"].apply(expected_direction_for_feature)
    out["direction_check"] = out.apply(
        lambda r: direction_check(
            str(r.get("expected_direction", "")),
            str(r.get("actual_direction", "")),
        ),
        axis=1,
    )

    def comment(row):
        feature = str(row.get("feature", ""))
        expected = row.get("expected_direction", "")
        actual = row.get("actual_direction", "")
        check = row.get("direction_check", "")

        if feature.lower() in {"const", "intercept"}:
            return "Intercept term; not interpreted as a borrower risk driver."

        if check == "aligned":
            return f"Direction is aligned: expected {expected}, observed {actual}."

        if check == "review":
            return f"Direction requires review: expected {expected}, observed {actual}."

        if check == "not_directional":
            return "No single fixed direction expected for this feature; review bin-level context."

        return "Direction could not be assessed from available coefficient output."

    out["interpretability_comment"] = out.apply(comment, axis=1)
    return out


def summarize_direction_checks(logit_data: pd.DataFrame) -> pd.DataFrame:
    if logit_data.empty or "direction_check" not in logit_data.columns:
        return pd.DataFrame()

    filtered = logit_data[
        ~logit_data["feature"].astype(str).str.lower().isin(["const", "intercept"])
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    summary = (
        filtered.groupby(["model_id", "direction_check"], as_index=False)
        .size()
        .pivot(index="model_id", columns="direction_check", values="size")
        .fillna(0)
        .reset_index()
    )

    for col in ["aligned", "review", "not_directional", "not_assessed"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["directional_review_flag"] = summary["review"].apply(
        lambda x: "Review required" if x > 0 else "No directional conflicts flagged"
    )

    return summary


def add_reference_role(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    def role(model_id):
        roles = []
        if model_id == MODELERS_DEFAULT_MODEL_ID:
            roles.append("Modeler Default")
        if model_id == BEST_AUC_MODEL_ID:
            roles.append("Best Validation AUC")
        if not roles:
            roles.append("Selected Challenger")
        return " + ".join(roles)

    out["reference_role"] = out["model_id"].apply(role)
    return out

def model_description(model_id: str, model_family: str, model_name: str = "") -> str:
    model_id = str(model_id).upper()
    model_family = str(model_family).lower()
    model_name = str(model_name)

    if model_id.startswith("WLOG"):
        return "WOE Logistic Regression: traditional credit-risk logistic model using Weight-of-Evidence transformed variables."
    if model_id.startswith("BLOG"):
        return "Binned Logistic Regression: logistic regression using discretized/binned borrower features."
    if model_id.startswith("RLOG"):
        return "Regularized Logistic Regression: logistic regression with L1/L2 penalty to control complexity."
    if model_id.startswith("CAT"):
        return "CatBoost model: gradient boosting model designed for strong tabular-data performance."
    if model_id.startswith("XGB") and "STACK" not in model_id:
        return "XGBoost model: regularized gradient boosting tree model."
    if model_id.startswith("LGB"):
        return "LightGBM model: fast gradient boosting model using leaf-wise tree growth."
    if model_id.startswith("GB"):
        return "Histogram Gradient Boosting model: sklearn boosting model for tabular prediction."
    if model_id.startswith("BNB"):
        return "Bernoulli Naive Bayes: probabilistic model for binary / one-hot encoded features."
    if model_id.startswith("GNB") or model_id.startswith("WGNB"):
        return "Gaussian Naive Bayes: probabilistic model using class-conditional feature distributions."
    if model_id.startswith("NN") and "STACK" not in model_id:
        return "Neural Network: multilayer perceptron model trained on scaled features."
    if "STACK" in model_id:
        return "Stacking ensemble: second-level model that combines predictions from base models."
    if model_family == "ensemble":
        return "Ensemble model: combines multiple base model predictions."
    return model_name or "Model description unavailable."

# Intro

st.markdown(
    """
This page checks whether candidate models are not only accurate, but also explainable enough for a credit-risk model review.

In banking, a model can perform well and still fail review if the drivers are unreasonable, unstable, too concentrated, or unsupported by clear evidence.
"""
)

st.info(
    f"""
This page always includes two benchmark models:

- **Modeler's default recommendation:** `{MODELERS_DEFAULT_MODEL_ID}`
- **Best validation-AUC benchmark:** `{BEST_AUC_MODEL_ID}`

You can add up to **five additional models** for comparison.
"""
)

with st.expander("Model ID guide: what do WLOG, BLOG, CAT, XGB, etc. mean?", expanded=False):
    st.markdown(
        """
| Prefix | Meaning | Plain-English explanation |
|---|---|---|
| `WLOG` | WOE Logistic Regression | Traditional credit-risk logistic model using Weight-of-Evidence variables |
| `BLOG` | Binned Logistic Regression | Logistic regression using discretized borrower feature bins |
| `RLOG` | Regularized Logistic Regression | Logistic regression with L1/L2 penalty to control complexity |
| `BNB` | Bernoulli Naive Bayes | Probabilistic model for binary / one-hot encoded features |
| `GNB` / `WGNB` | Gaussian Naive Bayes | Probabilistic model using class-based feature distributions |
| `NN` | Neural Network | Multilayer perceptron trained on scaled borrower features |
| `GB` | Histogram Gradient Boosting | sklearn gradient boosting model |
| `XGB` | XGBoost | Regularized gradient boosting tree model |
| `LGB` | LightGBM | Fast gradient boosting model using leaf-wise tree growth |
| `CAT` | CatBoost | Gradient boosting model designed for strong tabular-data performance |
| `STACK` | Stacking Ensemble | Second-level model combining predictions from base models |
| `WA` / `EA` / `RA` | Weighted / Equal / Rank Average Ensemble | Ensemble averaging methods over base model predictions |
"""
    )

with st.expander("Model form guide: how these models are built", expanded=False):
    st.markdown(
        """
### Logistic model forms

- **WLOG**: logistic regression using Weight-of-Evidence transformed variables.
- **BLOG**: logistic regression using binned/discretized variables.
- **RLOG**: regularized logistic regression using penalty terms to control complexity.

### Tree and boosting model forms

- **CAT / XGB / LGB / GB**: gradient-boosted tree models. These learn non-linear splits and interactions automatically.
- **RF / ET**: tree ensemble models that average many decision trees.

### Probabilistic model forms

- **BNB**: Bernoulli Naive Bayes, usually suited to binary or one-hot encoded variables.
- **GNB / WGNB**: Gaussian Naive Bayes variants using distributional assumptions.

### Neural network model forms

- **NN**: multilayer perceptron trained on scaled model-ready features.

### Ensemble model forms

- **STACK**: stacking model that combines base model predictions through a second-level model.
- **WA / EA / RA**: weighted, equal, or rank-average ensemble methods.

The important point is that not all model families explain themselves the same way.  
That is why this page separates coefficients, feature importance, and ensemble contributions instead of forcing one interpretation method across every model.
"""
    )

# Sidebar

st.sidebar.header("Interpretability Controls")

families = sorted(summary_df["model_family"].dropna().unique().tolist())

selected_model_ids = []

for i in range(1, 6):
    st.sidebar.markdown(f"### Optional Challenger {i}")

    fam = st.sidebar.selectbox(
        f"Challenger {i} model family",
        ["None"] + families,
        index=0,
        key=f"interp_family_{i}",
    )

    if fam == "None":
        continue

    models = (
        summary_df[summary_df["model_family"] == fam]
        .sort_values("validation_auc", ascending=False)
        .copy()
    )

    models["label"] = models.apply(model_label, axis=1)

    label = st.sidebar.selectbox(
        f"Challenger {i} model",
        models["label"].tolist(),
        key=f"interp_model_{i}",
    )

    selected_model_ids.append(label.split(" | ")[0])

comparison_model_ids = get_reference_models(selected_model_ids)

# Build comparison summary 

comparison_summary = summary_df[
    summary_df["model_id"].isin(comparison_model_ids)
].copy()

# Add interpretability review status
comparison_summary["interpretability_review_status"] = comparison_summary.apply(
    interpretability_score,
    axis=1,
)

# Add reference role labels (Modeler default / Best AUC / Selected)
comparison_summary = add_reference_role(comparison_summary)

comparison_summary["model_description"] = comparison_summary.apply(
    lambda r: model_description(
        r.get("model_id"),
        r.get("model_family"),
        r.get("model_name"),
    ),
    axis=1,
)

# Navigation

section = st.radio(
    "Jump to section",
    [
        "Comparison Overview",
        "Logistic Coefficients",
        "Feature Importance",
        "Ensemble Contributions",
        "Governance Checks",
        "Interpretability Decision",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# Comparison Overview

if section == "Comparison Overview":
    st.subheader("Comparison Overview")

    overview_cols = [
        "reference_role",
        "model_id",
        "model_family",
        "model_name",
        "model_description",
        "dataset_type",
        "feature_count",
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "selection_score",
        "interpretability_type",
        "governance_flag",
        "interpretability_review_status",
        "model_artifact_available",
    ]

    overview_cols = [c for c in overview_cols if c in comparison_summary.columns]

    build_table(
        safe_round_cols(comparison_summary[overview_cols]),
        height=330,
    )

    metric_cols = [
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "selection_score",
    ]

    available_metric_cols = [c for c in metric_cols if c in comparison_summary.columns]

    metric_df = comparison_summary[
        ["model_id", "model_family"] + available_metric_cols
    ].copy()

    metric_long = metric_df.melt(
        id_vars=["model_id", "model_family"],
        var_name="metric",
        value_name="value",
    )

    fig = px.bar(
        metric_long,
        x="model_id",
        y="value",
        color="metric",
        barmode="group",
        title="Reference Model Metric Comparison",
    )

    st.plotly_chart(fig, width="stretch")

# Logistic Coefficients

elif section == "Logistic Coefficients":
    st.subheader("Logistic Coefficients, P-values, Odds Ratios, and Direction Checks")

    st.markdown(
        """
This section reviews logistic model interpretability using coefficient magnitude, statistical significance, odds ratios, and expected business direction.

For statsmodels logistic models, p-values and significance stars are available. For sklearn regularized logistic models, coefficients and odds ratios are available, but p-values are not part of the fitted estimator.
"""
    )

    logit_subset = logistic_df[
        logistic_df["model_id"].isin(comparison_model_ids)
    ].copy()

    if logit_subset.empty:
        st.info("None of the selected comparison models are logistic models.")
    else:
        logit_subset = add_direction_logic(logit_subset)

        direction_summary = summarize_direction_checks(logit_subset)

        if not direction_summary.empty:
            st.markdown("### Direction Check Summary")
            build_table(safe_round_cols(direction_summary), height=180)

        display_cols = [
            "model_id",
            "feature",
            "coefficient",
            "std_error",
            "z_value",
            "p_value",
            "significance",
            "odds_ratio",
            "expected_direction",
            "actual_direction",
            "direction_check",
            "interpretability_comment",
            "source",
            "status",
        ]

        display_cols = [c for c in display_cols if c in logit_subset.columns]

        sort_cols = ["model_id"]
        if "p_value" in logit_subset.columns:
            sort_cols.append("p_value")

        for model_id in sorted(logit_subset["model_id"].unique()):

            st.markdown(f"### {model_id} Coefficient Table")

            model_df = logit_subset[
                logit_subset["model_id"] == model_id
            ].copy()

            build_table(
                safe_round_cols(
                    model_df[display_cols].sort_values(sort_cols, na_position="last")
                ),
                height=350,
            )

        plot_df = logit_subset[
            (logit_subset["feature"].astype(str).str.lower() != "const")
            & (logit_subset["feature"].astype(str).str.lower() != "intercept")
            & (logit_subset["coefficient"].notna())
        ].copy()

        if not plot_df.empty:
            for model_id in sorted(plot_df["model_id"].unique()):
                model_plot = plot_df[plot_df["model_id"] == model_id].copy()

                st.markdown(f"### {model_id} Coefficient Direction")

                fig = px.bar(
                    model_plot.sort_values("coefficient"),
                    x="coefficient",
                    y="feature",
                    color="direction_check",
                    orientation="h",
                    title=f"{model_id}: Coefficient Direction Check",
                )

                fig.update_xaxes(title="Coefficient")
                fig.update_yaxes(title="Feature")
                fig.update_layout(height=max(500, 26 * len(model_plot)))

                st.plotly_chart(fig, width="stretch")

        review_count = 0
        if "direction_check" in logit_subset.columns:
            review_count = int((logit_subset["direction_check"] == "review").sum())

        if review_count == 0:
            st.success(
                "No directional conflicts were flagged for the selected logistic model outputs."
            )
        else:
            st.error(
                f"{review_count} coefficient direction item(s) require review before champion approval."
            )

# Feature Importance

elif section == "Feature Importance":
    st.subheader("Feature Importance Review")

    st.markdown(
        """
This section reviews which features drive model predictions.

Key interpretation rule:

- We use **importance_normalized**.
- This shows relative importance **within each model only**.
- Do **not** compare raw importance values across model libraries.
- Ensemble model weights are excluded here and reviewed separately under **Ensemble Contributions**.

Each model is shown separately to avoid mixing interpretation types.
"""
    )

    non_ensemble_model_ids = comparison_summary[
        comparison_summary["model_family"].str.lower() != "ensemble"
    ]["model_id"].tolist()

    imp_subset = importance_df[
        importance_df["model_id"].isin(non_ensemble_model_ids)
    ].copy()

    if imp_subset.empty:
        st.info("No feature importance outputs found for the selected non-ensemble models.")
    else:
        imp_subset = imp_subset[imp_subset["status"] == "available"].copy()

        if imp_subset.empty:
            st.info("Feature importance is not available for the selected non-ensemble models.")
        else:
            for model_id in sorted(imp_subset["model_id"].unique()):
                model_imp = imp_subset[imp_subset["model_id"] == model_id].copy()

                st.markdown(f"### {model_id} Feature Importance")

                display_cols = [
                    "model_id",
                    "model_family",
                    "feature",
                    "importance_normalized",
                    "importance_type",
                    "rank",
                ]

                display_cols = [c for c in display_cols if c in model_imp.columns]

                build_table(
                    safe_round_cols(
                        model_imp.sort_values("rank")[display_cols]
                    ),
                    height=350,
                )

                fig = px.bar(
                    model_imp.sort_values("importance_normalized", ascending=True),
                    x="importance_normalized",
                    y="feature",
                    orientation="h",
                    title=f"{model_id}: Normalized Feature Importance",
                )

                fig.update_xaxes(title="Normalized importance within this model")
                fig.update_yaxes(title="Feature")
                fig.update_layout(height=max(500, 25 * len(model_imp)))

                st.plotly_chart(fig, width="stretch")

                top3 = (
                    model_imp.sort_values("importance_normalized", ascending=False)
                    .head(3)["importance_normalized"]
                    .sum()
                )

                if top3 >= 0.75:
                    st.error(f"High concentration: top 3 features = {top3:.2f}")
                else:
                    st.success(f"Acceptable concentration: top 3 features = {top3:.2f}")

# Ensemble Contributions

elif section == "Ensemble Contributions":
    st.subheader("Ensemble Contributions and Stacking Meta-Model Review")

    st.markdown(
        """
For weighted ensembles, this section shows base-model weights.

For stacking models, this section shows meta-model coefficients or feature importances where available.
"""
    )

    weight_subset = (
        ensemble_weights_df[
            ensemble_weights_df["ensemble_model_id"].isin(comparison_model_ids)
        ].copy()
        if not ensemble_weights_df.empty
        and "ensemble_model_id" in ensemble_weights_df.columns
        else pd.DataFrame()
    )

    stack_subset = (
        stacking_df[
            stacking_df["stack_model_id"].isin(comparison_model_ids)
        ].copy()
        if not stacking_df.empty and "stack_model_id" in stacking_df.columns
        else pd.DataFrame()
    )

    if weight_subset.empty and stack_subset.empty:
        st.info("No ensemble contribution outputs found for the selected models.")
    else:
        if not weight_subset.empty:
            st.markdown("### Weighted / Equal Ensemble Weights")

            build_table(
                safe_round_cols(
                    weight_subset.sort_values(
                        ["ensemble_model_id", "weight"],
                        ascending=[True, False],
                    )
                ),
                height=300,
            )

            fig = px.bar(
                weight_subset,
                x="base_model_id",
                y="weight",
                color="ensemble_model_id",
                barmode="group",
                title="Base Model Weights",
            )

            st.plotly_chart(fig, width="stretch")

        if not stack_subset.empty:
            st.markdown("### Stacking Meta-Model Importance")

            build_table(
                safe_round_cols(
                    stack_subset.sort_values(
                        ["stack_model_id", "meta_importance_abs"],
                        ascending=[True, False],
                    )
                ),
                height=320,
            )

            fig = px.bar(
                stack_subset,
                x="base_model_id",
                y="meta_importance_abs",
                color="stack_model_id",
                barmode="group",
                title="Stacking Meta-Model Importance",
            )

            st.plotly_chart(fig, width="stretch")

            st.info(
                """
            All ensemble models are built using the same base model set.
            Differences across ensembles reflect how predictions are combined,
            not differences in underlying signals.
            """
            )
# Governance Checks

elif section == "Governance Checks":
    st.subheader("Governance Checks")

    st.markdown(
        """
This section flags practical model-risk concerns that should be reviewed before champion selection.
"""
    )

    gov_subset = governance_df[
        governance_df["model_id"].isin(comparison_model_ids)
    ].copy()

    if gov_subset.empty:
        st.info("No governance checks found for the selected models.")
    else:
        display_cols = [
            "model_id",
            "model_family",
            "validation_auc",
            "validation_ks",
            "validation_log_loss",
            "validation_brier_score",
            "auc_gap",
            "feature_count",
            "selection_score",
            "model_artifact_available",
            "governance_flag",
            "governance_checks",
        ]

        display_cols = [c for c in display_cols if c in gov_subset.columns]

        build_table(
            safe_round_cols(gov_subset[display_cols]),
            height=390,
        )

# Interpretability Decision

elif section == "Interpretability Decision":
    st.subheader("Interpretability Decision")

    decision_df = comparison_summary[
        [
            "reference_role",
            "model_id",
            "model_family",
            "model_name",
            "model_description",
            "validation_auc",
            "validation_ks",
            "validation_brier_score",
            "auc_gap",
            "interpretability_type",
            "governance_flag",
            "interpretability_review_status",
            "model_artifact_available",
        ]
    ].copy()

    build_table(safe_round_cols(decision_df), height=300)

    st.markdown(
        """
### Review interpretation

- **Strong**: suitable for governance review when supported by coefficients or clear importance outputs.
- **Acceptable with support**: can proceed with supporting documentation and monitoring.
- **Review required**: can proceed only with explicit justification.
- **Not review-ready**: missing required artifacts or interpretability evidence.

### Decision rule

Only models that pass this interpretability review should move into **Champion Selection**.
"""
    )