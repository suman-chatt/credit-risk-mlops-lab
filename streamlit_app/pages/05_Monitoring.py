from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(page_title="Monitoring", layout="wide")

st.title("Model Monitoring")
st.caption(
    "Review model stability, synthetic drift response, and decision consistency."
)



# Paths


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

MONITORING_DIR = OUTPUTS_DIR / "monitoring"
SCORING_DIR = OUTPUTS_DIR / "scoring"

REGISTRY_PATH = OUTPUTS_DIR / "registry" / "model_summary.csv"
ENSEMBLE_REGISTRY_PATH = OUTPUTS_DIR / "ensemble_registry.xlsx"

SYNTHETIC_STRESS_PATH = MONITORING_DIR / "synthetic_stress_summary_app.csv"
SYNTHETIC_COHORT_DESIGN_PATH = MONITORING_DIR / "synthetic_stress_cohort_design_app.csv"
THRESHOLD_SUMMARY_APP_PATH = SCORING_DIR / "model_threshold_summary_app.csv"

FEATURE_DRIFT_PATH = MONITORING_DIR / "feature_drift_summary.csv"
SCORE_DRIFT_PATH = MONITORING_DIR / "score_drift_summary.csv"
RECOMMENDATIONS_PATH = MONITORING_DIR / "monitoring_recommendations.csv"

MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"



# Loaders


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_registry() -> pd.DataFrame:
    frames = []

    if REGISTRY_PATH.exists():
        frames.append(pd.read_csv(REGISTRY_PATH))

    if ENSEMBLE_REGISTRY_PATH.exists():
        frames.append(pd.read_excel(ENSEMBLE_REGISTRY_PATH))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model_id"], keep="last").reset_index(drop=True)
    return df

stress_all_df = load_csv(SYNTHETIC_STRESS_PATH)
stress_cohort_design_df = load_csv(SYNTHETIC_COHORT_DESIGN_PATH)
threshold_summary_df = load_csv(THRESHOLD_SUMMARY_APP_PATH)

registry_df = load_registry()
feature_drift_df = load_csv(FEATURE_DRIFT_PATH)
score_drift_df = load_csv(SCORE_DRIFT_PATH)
recommendations_df = load_csv(RECOMMENDATIONS_PATH)

if stress_all_df.empty:
    st.error(
        """
Streamlit-ready monitoring summaries not found.

Run:

`python scripts/build_streamlit_app_summaries.py`
"""
    )
    st.stop()



# Helpers


def build_table(data: pd.DataFrame, height: int = 350):
    if data.empty:
        st.info("No data available.")
        return

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

    if "model_id" in data.columns:
        gb.configure_column("model_id", pinned="left", width=150)
    if "feature" in data.columns:
        gb.configure_column("feature", pinned="left", width=260)

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


def get_available_models() -> list[str]:
    return sorted(stress_all_df["model_id"].dropna().astype(str).unique().tolist())


def get_comparison_models(selected_models: list[str]) -> list[str]:
    refs = [MODELERS_DEFAULT_MODEL_ID, BEST_AUC_MODEL_ID]
    refs.extend(selected_models)

    available = set(get_available_models())
    return list(dict.fromkeys([m for m in refs if m in available]))


def model_description(model_id: str, model_family: str = "", model_name: str = "") -> str:
    model_id = str(model_id).upper()
    model_family = str(model_family).lower()
    model_name = str(model_name)

    if model_id.startswith("WLOG"):
        return "WOE Logistic Regression"
    if model_id.startswith("BLOG"):
        return "Binned Logistic Regression"
    if model_id.startswith("RLOG"):
        return "Regularized Logistic Regression"
    if model_id.startswith("CAT"):
        return "CatBoost gradient boosting model"
    if model_id.startswith("XGB") and "STACK" not in model_id:
        return "XGBoost model"
    if model_id.startswith("LGB"):
        return "LightGBM model"
    if model_id.startswith("GB"):
        return "Histogram Gradient Boosting model"
    if model_id.startswith("RF"):
        return "Random Forest model"
    if model_id.startswith("ET"):
        return "Extra Trees model"
    if model_id.startswith("BNB"):
        return "Bernoulli Naive Bayes model"
    if model_id.startswith("GNB") or model_id.startswith("WGNB"):
        return "Gaussian Naive Bayes model"
    if model_id.startswith("NN") and "STACK" not in model_id:
        return "Neural Network model"
    if "STACK" in model_id:
        return "Stacking ensemble model"
    return model_name or model_family or "Model"


def model_label(row: pd.Series) -> str:
    return (
        f"{row['model_id']} | {row.get('model_family', '')} | "
        f"AUC={row.get('validation_auc', 0):.4f} | "
        f"{model_description(row.get('model_id'), row.get('model_family'), row.get('model_name'))}"
    )


def flag_psi(value: float) -> str:
    if value < 0.10:
        return "green"
    if value < 0.25:
        return "yellow"
    return "red"


def flag_js(value: float) -> str:
    if value < 0.05:
        return "green"
    if value < 0.10:
        return "yellow"
    return "red"


def flag_wasserstein(value: float) -> str:
    if value < 0.05:
        return "green"
    if value < 0.15:
        return "yellow"
    return "red"


def status_label(flag: str) -> str:
    flag = str(flag).lower()
    if flag == "green":
        return "Stable"
    if flag == "yellow":
        return "Monitor"
    if flag == "red":
        return "Investigate"
    return "Review"





def build_governance_action_table(
    stress_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    if not stress_df.empty:
        model_summary = (
            stress_df.groupby("model_id", as_index=False)
            .agg(
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
                min_approval_rate=("approval_rate", "min"),
                max_approval_rate=("approval_rate", "max"),
                min_predicted_default_rate=("predicted_default_rate", "min"),
                max_predicted_default_rate=("predicted_default_rate", "max"),
            )
        )

        model_summary["auc_range"] = model_summary["max_auc"] - model_summary["min_auc"]
        model_summary["approval_rate_range"] = (
            model_summary["max_approval_rate"] - model_summary["min_approval_rate"]
        )
        model_summary["predicted_default_rate_range"] = (
            model_summary["max_predicted_default_rate"]
            - model_summary["min_predicted_default_rate"]
        )

        for _, row in model_summary.iterrows():
            if row["auc_range"] >= 0.05:
                action = "Investigate performance sensitivity"
                severity = "Investigate"
            elif row["approval_rate_range"] >= 0.10:
                action = "Review decision stability"
                severity = "Monitor"
            elif row["predicted_default_rate_range"] >= 0.10:
                action = "Review threshold sensitivity"
                severity = "Monitor"
            else:
                action = "No action required"
                severity = "Stable"

            records.append(
                {
                    "monitoring_area": "Model stress response",
                    "model_id": row["model_id"],
                    "severity": severity,
                    "recommended_action": action,
                    "auc_range": row["auc_range"],
                    "approval_rate_range": row["approval_rate_range"],
                    "predicted_default_rate_range": row[
                        "predicted_default_rate_range"
                    ],
                }
            )

    if not feature_drift_df.empty and "psi" in feature_drift_df.columns:
        max_psi = feature_drift_df["psi"].max()

        if max_psi >= 0.25:
            severity = "Investigate"
            action = "Investigate feature drift and data pipeline"
        elif max_psi >= 0.10:
            severity = "Monitor"
            action = "Monitor feature drift closely"
        else:
            severity = "Stable"
            action = "No action required"

        records.append(
            {
                "monitoring_area": "Feature drift",
                "model_id": "All monitored models",
                "severity": severity,
                "recommended_action": action,
                "auc_range": None,
                "approval_rate_range": None,
                "predicted_default_rate_range": None,
            }
        )

    return pd.DataFrame(records)



# Sidebar


st.sidebar.header("Monitoring Controls")

available_models = get_available_models()

if registry_df.empty:
    selectable_registry = pd.DataFrame({"model_id": available_models})
else:
    selectable_registry = registry_df[
        registry_df["model_id"].astype(str).isin(available_models)
    ].copy()

families = sorted(selectable_registry["model_family"].dropna().unique().tolist())

selected_challengers = []

for i in range(1, 6):
    st.sidebar.markdown(f"### Optional Challenger {i}")

    fam = st.sidebar.selectbox(
        f"Challenger {i} model family",
        ["None"] + families,
        index=0,
        key=f"monitor_family_{i}",
    )

    if fam == "None":
        continue

    models = (
        selectable_registry[selectable_registry["model_family"] == fam]
        .sort_values("validation_auc", ascending=False)
        .copy()
    )

    models["label"] = models.apply(model_label, axis=1)

    label = st.sidebar.selectbox(
        f"Challenger {i} model",
        models["label"].tolist(),
        key=f"monitor_model_{i}",
    )

    selected_challengers.append(label.split(" | ")[0])

comparison_models = get_comparison_models(selected_challengers)


# Shared app-ready monitoring data


monitoring_threshold = st.sidebar.slider(
    "Monitoring PD threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.25,
    step=0.01,
)

stress_df = stress_all_df[
    stress_all_df["model_id"].astype(str).isin(comparison_models)
].copy()

stress_cohort_summary = stress_cohort_design_df.copy()

governance_action_df = build_governance_action_table(
    stress_df=stress_df,
    feature_drift_df=feature_drift_df,
)

# Navigation


section = st.radio(
    "Monitoring Sections",
    [
        "Overview",
        "Monitoring Health",
        "Score Stability",
        "Performance Stability",
        "Decision Stability",
        "Stress Testing Summary",
        "Decision Sensitivity",
        "Feature Drift",
        "Governance Actions",
        "Monitoring Recommendations",
    ],
    horizontal=True,
    label_visibility="collapsed",
)



# Overview


if section == "Overview":
    st.subheader("Monitoring Overview")

    st.markdown(
        """
This page evaluates whether model behavior remains stable under changing population conditions.

### Important context

This dataset does **not** contain real time periods.  
So this page does **not** claim true production time-series monitoring.

Instead, it uses **synthetic monitoring cohorts**.

### What are synthetic cohorts?

Synthetic cohorts are controlled validation subsets created to mimic changing applicant populations.

For this page:

- The validation dataset is used as the base population.
- Borrowers are ranked using the benchmark model score.
- Cohorts are created with gradually larger shares of high-risk borrowers.
- The actual target value is still the real validation target.
- Only the population mix is changed.

### Why this is useful

This tests whether models behave consistently when the applicant mix becomes riskier.

It helps answer:

- Do model scores shift smoothly?
- Does model performance remain stable?
- Do approval/default-flag rates change too aggressively?
- Would monitoring alerts be needed if this happened in production?

This is a **production-monitoring proxy**, not a substitute for real post-deployment monitoring.
"""
    )

    st.markdown("### Selected Monitoring Models")

    selected_table = pd.DataFrame(
        {
            "model_id": comparison_models,
            "role": [
                "Modeler Default"
                if m == MODELERS_DEFAULT_MODEL_ID
                else "Best Validation AUC Benchmark"
                if m == BEST_AUC_MODEL_ID
                else "Selected Challenger"
                for m in comparison_models
            ],
        }
    )

    build_table(selected_table, height=240)

    st.markdown("### Synthetic Cohort Design")

    build_table(safe_round_cols(stress_cohort_summary), height=240)



# Monitoring Health


elif section == "Monitoring Health":
    st.subheader("Monitoring Health Summary")

    st.markdown(
        """
This section provides a compact monitoring status view.

It uses drift metrics to classify whether monitored features appear stable, need monitoring, or require investigation.

### Flag rules used here

- **PSI < 0.10** → green / stable
- **0.10 ≤ PSI < 0.25** → yellow / monitor
- **PSI ≥ 0.25** → red / investigate

For JS distance and Wasserstein distance, simple project thresholds are used as supporting indicators.  
They should be treated as review aids, not formal regulatory thresholds.
"""
    )

    if feature_drift_df.empty:
        st.info("No monitoring outputs available.")
    else:
        health = feature_drift_df.copy()

        health["psi_status"] = health["psi"].apply(flag_psi).apply(status_label)

        if "js_distance" in health.columns:
            health["js_status"] = health["js_distance"].apply(flag_js).apply(status_label)

        if "wasserstein_distance" in health.columns:
            health["wasserstein_status"] = (
                health["wasserstein_distance"]
                .apply(flag_wasserstein)
                .apply(status_label)
            )

        display_cols = [
            "feature",
            "feature_type",
            "psi",
            "psi_status",
            "js_distance",
            "js_status",
            "wasserstein_distance",
            "wasserstein_status",
        ]

        display_cols = [c for c in display_cols if c in health.columns]

        build_table(safe_round_cols(health[display_cols]), height=420)

        st.markdown(
            """
### How to interpret this table

A stable monitoring table means the simulated population has not materially shifted.

In your current baseline outputs, many PSI values are zero. That means the original monitoring export did not inject meaningful feature drift.

That is not a model failure. It means:

- the baseline distribution and comparison distribution are effectively identical
- the table is useful as a monitoring framework
- future production data or stronger synthetic drift will make this section more informative
"""
        )



# Score Stability


elif section == "Score Stability":
    st.subheader("Score Stability Under Synthetic Population Drift")

    st.markdown(
        """
This section checks whether model scores move smoothly as the applicant mix becomes riskier.

The synthetic cohorts increase the share of high-risk borrowers.  
A stable model should show controlled, explainable movement in average predicted PD.
"""
    )

    if stress_df.empty:
        st.info("No synthetic stress monitoring data available.")
    else:
        fig = px.line(
            stress_df,
            x="cohort",
            y="mean_pd",
            color="model_id",
            markers=True,
            title="Mean Predicted PD Across Synthetic Cohorts",
        )

        fig.update_xaxes(title="Synthetic cohort")
        fig.update_yaxes(title="Mean predicted PD")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

- A rising line means the model assigns higher risk as the cohort becomes riskier.
- A flat line may mean the model is not sensitive enough to changing population risk.
- A very steep line may mean the model reacts aggressively to population changes.

The goal is not a perfectly flat line.  
The goal is stable and explainable score movement.
"""
        )

        display_cols = [
            "cohort",
            "model_id",
            "high_risk_mix",
            "mean_pd",
            "p50_pd",
            "p90_pd",
        ]

        build_table(safe_round_cols(stress_df[display_cols]), height=360)



# Performance Stability


elif section == "Performance Stability":
    st.subheader("Performance Stability Under Synthetic Cohorts")

    st.markdown(
        """
This section checks whether validation performance remains stable as population risk mix changes.

Because the dataset has no real time periods, this is **not true time monitoring**.  
It is a controlled stress test using validation records.
"""
    )

    if stress_df.empty:
        st.info("No synthetic performance monitoring data available.")
    else:
        fig = px.line(
            stress_df,
            x="cohort",
            y="auc",
            color="model_id",
            markers=True,
            title="AUC Across Synthetic Cohorts",
        )

        fig.update_xaxes(title="Synthetic cohort")
        fig.update_yaxes(title="AUC")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

- Stable AUC means the model keeps ranking borrowers effectively across different population mixes.
- Falling AUC suggests the model may be less robust when applicant composition changes.
- Similar AUC across models means business decisioning, calibration, and governance may matter more than raw discrimination.

AUC alone should not decide the champion model.
"""
        )

        cohort_perf = stress_df[
            ["cohort", "cohort_size", "high_risk_mix", "actual_default_rate"]
        ].drop_duplicates()

        fig = px.line(
            cohort_perf,
            x="cohort",
            y="actual_default_rate",
            markers=True,
            title="Observed Default Rate Across Synthetic Cohorts",
        )

        fig.update_xaxes(title="Synthetic cohort")
        fig.update_yaxes(title="Observed default rate", tickformat=".0%")
        fig.update_layout(height=480)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

The observed default rate increases when the synthetic cohort contains more high-risk applicants.

This confirms the synthetic stress design is working:  
we are not changing labels, but we are changing the population mix.
"""
        )



# Decision Stability


elif section == "Decision Stability":
    st.subheader("Decision Stability")

    st.markdown(
        f"""
This section converts predicted PD into decision flags using the monitoring threshold selected in the sidebar.

Current threshold: **{monitoring_threshold:.0%} PD**

Decision rule:

- Predicted PD >= threshold → predicted default flag = 1
- Predicted PD < threshold → predicted default flag = 0

This helps test whether model-driven decisions remain stable when the applicant mix changes.
"""
    )

    if stress_df.empty:
        st.info("No decision stability data available.")
    else:
        display_cols = [
            "cohort",
            "model_id",
            "high_risk_mix",
            "approval_rate",
            "predicted_default_rate",
            "actual_default_rate",
        ]

        build_table(safe_round_cols(stress_df[display_cols]), height=360)

        fig = px.line(
            stress_df,
            x="cohort",
            y="predicted_default_rate",
            color="model_id",
            markers=True,
            title="Predicted Default Flag Rate Across Synthetic Cohorts",
        )

        fig.update_xaxes(title="Synthetic cohort")
        fig.update_yaxes(title="Predicted default flag rate", tickformat=".0%")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

- Higher predicted default flag rate means the model is becoming stricter.
- Lower approval rate means fewer applicants would pass the decision cutoff.
- If one model reacts much more strongly than others, it may create unstable business outcomes.

This is where monitoring becomes operational:  
score drift matters because it changes decisions.
"""
        )


# Stress Testing Summary


elif section == "Stress Testing Summary":
    st.subheader("Stress Testing Summary")

    st.markdown(
        """
This section summarizes how selected models behave under synthetic stress.

The stress test gradually increases the share of high-risk borrowers in each cohort.

This does not create artificial defaults.  
It resamples real validation records so the target values remain real, while the applicant mix becomes riskier.

What this tests:

- whether model scores increase smoothly as risk mix increases
- whether AUC remains stable
- whether approval/default-flag rates change in a controlled way
"""
    )

    if stress_df.empty:
        st.info("No stress testing data available.")
    else:
        stress_summary = (
            stress_df.groupby("model_id", as_index=False)
            .agg(
                min_auc=("auc", "min"),
                max_auc=("auc", "max"),
                min_mean_pd=("mean_pd", "min"),
                max_mean_pd=("mean_pd", "max"),
                min_approval_rate=("approval_rate", "min"),
                max_approval_rate=("approval_rate", "max"),
                min_predicted_default_rate=("predicted_default_rate", "min"),
                max_predicted_default_rate=("predicted_default_rate", "max"),
            )
        )

        stress_summary["auc_range"] = (
            stress_summary["max_auc"] - stress_summary["min_auc"]
        )
        stress_summary["mean_pd_range"] = (
            stress_summary["max_mean_pd"] - stress_summary["min_mean_pd"]
        )
        stress_summary["approval_rate_range"] = (
            stress_summary["max_approval_rate"]
            - stress_summary["min_approval_rate"]
        )

        build_table(safe_round_cols(stress_summary), height=360)

        fig = px.bar(
            stress_summary.sort_values("approval_rate_range", ascending=False),
            x="model_id",
            y="approval_rate_range",
            title="Approval Rate Sensitivity Under Synthetic Stress",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title="Approval rate range")
        fig.update_layout(height=460)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

This chart shows how much each model's approval rate changes from the mildest to the most severe synthetic cohort.

- Larger movement means the model is more sensitive to applicant mix changes.
- Smaller movement means the model is more stable under stress.
- Very high sensitivity may require tighter monitoring or threshold review.
"""
        )



# Decision Sensitivity


elif section == "Decision Sensitivity":
    st.subheader("Decision Sensitivity")

    st.markdown(
        """
This section tests how model decisions change across different PD thresholds.

Unlike the Decision Stability section, which uses one selected threshold, this section scans multiple thresholds.

This helps answer:

- Is the model highly sensitive to small cutoff changes?
- Do some models flag far more borrowers than others?
- Is the current threshold reasonable compared with nearby alternatives?
"""
    )

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    sensitivity_df = threshold_summary_df[
        (threshold_summary_df["model_id"].astype(str).isin(comparison_models))
        & (threshold_summary_df["split"].astype(str).str.lower() == "validation")
        & (threshold_summary_df["threshold"].round(2).isin(thresholds))
    ].copy()

    if sensitivity_df.empty:
        st.info("No decision sensitivity summary available.")
    else:
        sensitivity_df["approval_rate"] = 1 - sensitivity_df["predicted_1_share"]
        sensitivity_df["predicted_default_rate"] = sensitivity_df["predicted_1_share"]

        build_table(safe_round_cols(sensitivity_df), height=360)

        fig = px.line(
            sensitivity_df,
            x="threshold",
            y="approval_rate",
            color="model_id",
            markers=True,
            title="Approval Rate by PD Threshold",
        )

        fig.update_xaxes(title="PD threshold", tickformat=".0%")
        fig.update_yaxes(title="Approval rate", tickformat=".0%")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**Chart interpretation:**

As the PD threshold increases, approval rates usually increase because fewer borrowers exceed the cutoff.

A model with a steep line is more sensitive to threshold choice.  
That means small policy changes can materially change approvals and risk exposure.

This section helps governance teams understand whether a proposed cutoff is stable or fragile.
"""
        )


# Feature Drift


elif section == "Feature Drift":
    st.subheader("Feature Drift Summary")

    st.markdown(
        """
This section reviews input-feature drift using the existing monitoring export.

The main metric is **PSI**, or Population Stability Index.

### PSI interpretation

- PSI < 0.10 → stable
- 0.10 to 0.25 → moderate drift
- PSI above 0.25 → significant drift

Charts are intentionally not shown here because the current PSI values are mostly zero.  
A chart of zeros adds noise without adding insight.
"""
    )

    if feature_drift_df.empty:
        st.info("No feature drift data.")
    else:
        drift = feature_drift_df.copy()

        drift["psi_status"] = drift["psi"].apply(flag_psi).apply(status_label)

        if "js_distance" in drift.columns:
            drift["js_status"] = drift["js_distance"].apply(flag_js).apply(status_label)

        if "wasserstein_distance" in drift.columns:
            drift["wasserstein_status"] = (
                drift["wasserstein_distance"]
                .apply(flag_wasserstein)
                .apply(status_label)
            )

        display_cols = [
            "feature",
            "feature_type",
            "psi",
            "psi_status",
            "js_distance",
            "js_status",
            "wasserstein_distance",
            "wasserstein_status",
        ]

        display_cols = [c for c in display_cols if c in drift.columns]

        build_table(safe_round_cols(drift[display_cols]), height=460)

        

        st.markdown(
            """
### About Wasserstein Distance

Wasserstein distance measures how much one distribution would need to “move” to match another.

- 0 → identical distributions  
- Higher values → larger distribution shift  

Unlike PSI, it does not depend on binning and captures full distribution movement.

In this dataset:
- Values are near zero → no measurable shift  
- Therefore the model inputs are stable across cohorts

### How to use this section

In production, this table should be reviewed regularly.

A feature with rising PSI may indicate:

- borrower population changes
- data pipeline changes
- policy changes
- new missingness patterns
- macroeconomic shifts

Feature drift does not automatically mean the model is wrong.  
It means the model is operating on a population that may differ from the one it was approved on.
"""
        )



# Governance Actions


elif section == "Governance Actions":
    st.subheader("Governance Actions")

    st.markdown(
        """
This section turns monitoring signals into governance actions.

It is not meant to replace human review.  
It provides a structured first-pass control layer for model owners and reviewers.

Action logic used here:

- Large AUC movement → investigate performance sensitivity
- Large approval-rate movement → review decision stability
- Large predicted-default movement → review threshold sensitivity
- Large feature PSI → investigate data or population drift
"""
    )

    if governance_action_df.empty:
        st.info("No governance actions generated.")
    else:
        build_table(safe_round_cols(governance_action_df), height=420)

        severity_counts = (
            governance_action_df.groupby("severity", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        fig = px.bar(
            severity_counts,
            x="severity",
            y="count",
            title="Governance Action Counts by Severity",
        )

        fig.update_xaxes(title="Severity")
        fig.update_yaxes(title="Count")
        fig.update_layout(height=420)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**How to interpret this section:**

- Stable means no immediate action is suggested.
- Monitor means the item should be tracked but does not require urgent remediation.
- Investigate means the model owner should document the issue and determine whether recalibration, policy adjustment, or retraining is needed.

This is the bridge between monitoring analytics and model risk management.
"""
        )



# Monitoring Recommendations


elif section == "Monitoring Recommendations":
    st.subheader("Monitoring Recommendations")

    st.markdown(
        """
This section translates monitoring signals into practical action.

### Recommendation logic

- **Green** → no action required
- **Yellow** → monitor closely
- **Red** → investigate, document, or consider recalibration/retraining

These recommendations are rule-based and should support model review, not replace judgment.
"""
    )

    if recommendations_df.empty:
        st.info("No recommendations found.")
    else:
        recs = recommendations_df.copy()

        if "feature" in recs.columns:
            selected_recs = recs[
                recs["feature"].astype(str).isin(comparison_models)
                | ~recs["feature"].astype(str).isin(get_available_models())
            ].copy()
        else:
            selected_recs = recs.copy()

        build_table(safe_round_cols(selected_recs), height=440)

        st.markdown(
            """
### How to interpret this table

A “No action required” result means the monitoring indicators did not breach the current project thresholds.

In a real MRM process, this table would feed into:

- monitoring sign-off
- model owner review
- challenger evaluation
- retraining decisioning
- governance documentation

### Important limitation

Because this project uses synthetic cohorts rather than true production months, recommendations should be interpreted as a monitoring framework demonstration.

Once true production data exists, the same structure can be reused with actual monthly or quarterly monitoring periods.
"""
        )