import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
from utils.data_loader import load_full_model_registry


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="Model Leaderboard",
    layout="wide",
)

st.title("Model Leaderboard")

df = load_full_model_registry()

st.markdown(
    """
This page compares all trained models and highlights the key tradeoffs required for real-world model selection.

The objective is not to pick the model with the highest AUC in isolation. Instead, this view helps evaluate:

- **Performance**: how well the model ranks higher-risk vs lower-risk borrowers  
- **Stability**: whether performance holds between training and validation (overfitting risk)  
- **Calibration**: how reliable the predicted probabilities are  
- **Complexity**: how difficult the model is to implement, monitor, and maintain  
- **Interpretability**: how easily results can be explained to business and risk stakeholders  

This is a **model comparison and evidence-building step**, not the final decision.
The final champion model is selected later using both quantitative results and practical deployment considerations.
"""
)

st.info(
    """
This page is a candidate comparison view, not the final model approval page.
Models are ranked by validation performance here, but the final champion is selected later using
performance, stability, calibration, interpretability, complexity, and governance suitability.

In practice, a slightly lower-performing model may be preferred if it is more stable,
easier to govern, and better aligned with production constraints.
"""
)

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def add_auc_gap(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "auc_gap" not in data.columns and {"train_auc", "validation_auc"}.issubset(data.columns):
        data["auc_gap"] = data["train_auc"] - data["validation_auc"]
    return data


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.max() == s.min():
        return pd.Series(1.0, index=s.index)
    score = (s - s.min()) / (s.max() - s.min())
    return score if higher_is_better else 1 - score


def add_selection_score(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    required = [
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "feature_count",
    ]

    if not all(c in data.columns for c in required):
        data["selection_score"] = np.nan
        return data

    auc_score = minmax_score(data["validation_auc"], higher_is_better=True)
    ks_score = minmax_score(data["validation_ks"], higher_is_better=True)
    logloss_score = minmax_score(data["validation_log_loss"], higher_is_better=False)
    brier_score = minmax_score(data["validation_brier_score"], higher_is_better=False)
    gap_score = minmax_score(data["auc_gap"].abs(), higher_is_better=False)
    simplicity_score = minmax_score(data["feature_count"], higher_is_better=False)

    data["selection_score"] = (
        0.30 * auc_score
        + 0.20 * ks_score
        + 0.20 * logloss_score
        + 0.15 * brier_score
        + 0.10 * gap_score
        + 0.05 * simplicity_score
    )

    return data


def add_champion_candidate_flag(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    required = [
        "validation_auc",
        "validation_log_loss",
        "auc_gap",
        "feature_count",
    ]

    if not all(c in data.columns for c in required):
        data["champion_candidate_flag"] = False
        return data

    values = data[required].copy()
    values["auc_gap_abs"] = data["auc_gap"].abs()

    pareto_flags = []

    for i, row in values.iterrows():
        better_or_equal = (
            (values["validation_auc"] >= row["validation_auc"])
            & (values["validation_log_loss"] <= row["validation_log_loss"])
            & (values["auc_gap_abs"] <= row["auc_gap_abs"])
            & (values["feature_count"] <= row["feature_count"])
        )

        strictly_better = (
            (values["validation_auc"] > row["validation_auc"])
            | (values["validation_log_loss"] < row["validation_log_loss"])
            | (values["auc_gap_abs"] < row["auc_gap_abs"])
            | (values["feature_count"] < row["feature_count"])
        )

        dominated = bool((better_or_equal & strictly_better).any())
        pareto_flags.append(not dominated)

    data["champion_candidate_flag"] = pareto_flags
    return data


def build_aggrid_table(data: pd.DataFrame, height: int = 430):
    COLUMN_HELP = {
        "model_id": "Unique model identifier used across registry, MLflow, scoring, and monitoring outputs.",
        "model_family": "Broad model class: boosting, logistic, neural network, naive_bayes, tree, or ensemble.",
        "model_name": "Specific algorithm or model implementation.",
        "dataset_type": "Input representation used by the model, such as tree, scaled, woe, binned_ohe, or base_model_predictions.",
        "feature_count": "Number of input features used. Lower can mean simpler, but not always better.",
        "validation_auc": "Ranking power on validation data. Higher is better. 0.5 is random; closer to 1.0 is stronger.",
        "validation_ks": "Separation between low-risk and high-risk borrowers. Higher is better.",
        "validation_log_loss": "Probability calibration error. Lower is better.",
        "validation_brier_score": "Average probability prediction error. Lower is better.",
        "train_auc": "Training AUC. Compare with validation AUC to check possible overfitting.",
        "auc_gap": "Train AUC minus validation AUC. Smaller absolute values are usually better.",
        "selection_score": "Composite model selection score combining AUC, KS, calibration, stability, and simplicity. Higher is better.",
        "champion_candidate_flag": "True if the model remains a strong candidate after considering AUC, calibration, stability, and simplicity. This is not final approval.",
        "ensemble_method_type": "How the ensemble combines base model predictions.",
        "dominant_base_model": "For weighted ensembles, the base model with the largest contribution. For stacking, contribution is learned by the meta-model.",
        "max_weight_share": "Largest contribution share assigned to any base model in a weighted ensemble.",
        "ensemble_diversity_score": "For weighted ensembles, calculated as 1 - max_weight_share. Higher means more diversified.",
        "ensemble_note": "Plain-English interpretation of ensemble behavior.",
    }

    gb = GridOptionsBuilder.from_dataframe(data)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
    )

    for col in data.columns:
        gb.configure_column(
            col,
            headerTooltip=COLUMN_HELP.get(col, "No description available."),
        )

    if "model_id" in data.columns:
        gb.configure_column("model_id", pinned="left", width=150)
    if "model_family" in data.columns:
        gb.configure_column("model_family", pinned="left", width=140)
    if "model_name" in data.columns:
        gb.configure_column("model_name", width=240)
    if "ensemble_note" in data.columns:
        gb.configure_column("ensemble_note", width=360)

    gb.configure_grid_options(
        domLayout="normal",
        enableBrowserTooltips=True,
    )

    AgGrid(
        data,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        theme="streamlit",
    )


# ---------------------------------------
# PREP DATA
# ---------------------------------------

df = add_auc_gap(df)
df = add_selection_score(df)
df = add_champion_candidate_flag(df)

# ---------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------

st.sidebar.header("Filters")

model_families = sorted(df["model_family"].dropna().unique().tolist())
selected_families = st.sidebar.multiselect(
    "Model family",
    model_families,
    default=model_families,
)

dataset_types = sorted(df["dataset_type"].dropna().unique().tolist())
selected_dataset_types = st.sidebar.multiselect(
    "Dataset type",
    dataset_types,
    default=dataset_types,
)

top_n = st.sidebar.slider(
    "Top N models by validation AUC",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)

include_ensembles = st.sidebar.checkbox(
    "Include ensemble models",
    value=True,
)

filtered = df[
    df["model_family"].isin(selected_families)
    & df["dataset_type"].isin(selected_dataset_types)
].copy()

if not include_ensembles:
    filtered = filtered[filtered["model_family"] != "ensemble"]

leaderboard = (
    filtered.sort_values("validation_auc", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)

# ---------------------------------------
# SECTION NAVIGATION
# ---------------------------------------

st.markdown("## Page Sections")

section = st.radio(
    "Jump to section",
    [
        "Leaderboard Table",
        "Charts / Visualizations",
        "Best Model by Family",
        "Initial Interpretation",
        "Ensemble Diversity Score",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------------------------------------
# KPI CARDS
# ---------------------------------------

st.subheader("Highest Validation AUC Candidate")

st.caption(
    "This is the highest model by validation AUC under the current filters. "
    "It is not automatically the final champion model."
)

if leaderboard.empty:
    st.warning("No models match the current filter selection.")
    st.stop()

best_row = leaderboard.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Highest-AUC Candidate", best_row["model_id"])
col2.metric("Validation AUC", f"{best_row['validation_auc']:.4f}")
col3.metric("Validation KS", f"{best_row['validation_ks']:.4f}")
col4.metric("Log Loss", f"{best_row['validation_log_loss']:.4f}")
col5.metric("AUC Gap", f"{best_row.get('auc_gap', np.nan):.4f}")

# ---------------------------------------
# LEADERBOARD TABLE
# ---------------------------------------

if section == "Leaderboard Table":
    st.subheader("Leaderboard Table")

    st.info(
        """
    Use this table as the primary model comparison view.

    Hover over column headers for definitions, interpretation guidance, and whether higher or lower is better.
    The table is intentionally more important than plain AUC bar charts because model performance is close and sorting is more useful than decoration.
    """
    )

    display_cols = [
        "model_id",
        "model_family",
        "model_name",
        "dataset_type",
        "feature_count",
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "train_auc",
        "auc_gap",
        "selection_score",
        "champion_candidate_flag",
        "ensemble_method_type",
        "dominant_base_model",
        "max_weight_share",
        "ensemble_diversity_score",
        "ensemble_note",
    ]

    available_cols = [c for c in display_cols if c in leaderboard.columns]
    leaderboard_display = leaderboard[available_cols].copy()

    build_aggrid_table(leaderboard_display, height=450)

# ---------------------------------------
# CHARTS / VISUALIZATIONS
# ---------------------------------------

elif section == "Charts / Visualizations":
    st.subheader("Charts / Visualizations")

    st.markdown(
        """
    These visuals focus on decision-useful comparisons.

    Plain AUC, KS, and log-loss bar charts were intentionally avoided because those metrics can be sorted directly in the table.
    The charts below answer better questions: which models are stable, which models are simple, which models are calibrated, and which models survive tradeoff analysis?
    """
    )

    chart_df = leaderboard.copy()

    # -----------------------------
    # Stability vs Performance
    # -----------------------------

    st.markdown("### 1. Stability vs Performance")

    st.caption(
        """
    This chart compares validation AUC against the train-validation AUC gap.
    The best region is high on the chart and close to zero on the x-axis.
    """
    )

    required_cols = ["validation_auc", "auc_gap", "model_family", "model_id"]
    if all(c in chart_df.columns for c in required_cols):
        fig = px.scatter(
            chart_df,
            x="auc_gap",
            y="validation_auc",
            color="model_family",
            hover_name="model_id",
            hover_data=[
                "model_name",
                "dataset_type",
                "feature_count",
                "validation_ks",
                "validation_log_loss",
                "selection_score",
                "champion_candidate_flag",
            ],
            title="Stability vs Performance",
        )

        fig.update_xaxes(title="Train AUC - Validation AUC | lower absolute value is better")
        fig.update_yaxes(title="Validation AUC | higher is better")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This chart requires validation_auc, train_auc, and auc_gap.")


    # -----------------------------
    # Calibration vs Ranking
    # -----------------------------

    st.markdown("### 2. Calibration vs Ranking")

    st.caption(
        """
    This chart compares ranking strength against probability quality.
    Best region: upper-left, meaning high AUC and low log loss.
    """
    )

    calibration_cols = ["validation_log_loss", "validation_auc", "model_family", "model_id"]
    if all(c in chart_df.columns for c in calibration_cols):
        fig = px.scatter(
            chart_df,
            x="validation_log_loss",
            y="validation_auc",
            color="model_family",
            hover_name="model_id",
            hover_data=[
                "model_name",
                "dataset_type",
                "feature_count",
                "validation_ks",
                "validation_brier_score",
                "auc_gap",
                "selection_score",
                "champion_candidate_flag",
            ],
            title="Calibration vs Ranking",
        )

        fig.update_xaxes(title="Validation Log Loss | lower is better")
        fig.update_yaxes(title="Validation AUC | higher is better")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This chart requires validation_log_loss and validation_auc.")

    # -----------------------------
    # Pareto Frontier
    # -----------------------------

    st.markdown("### 3. Champion Candidate Shortlist")

    st.caption(
    """
    This shortlist highlights models that remain strong after considering more than AUC alone.
    A model is flagged when it is not clearly beaten by another model across validation AUC,
    log loss, AUC gap, and feature count.

    This is a screening view, not final model approval.
    """
    )

    candidate_cols = [
        "model_id",
        "model_family",
        "model_name",
        "validation_auc",
        "validation_log_loss",
        "auc_gap",
        "feature_count",
        "selection_score",
        "champion_candidate_flag",
    ]

    candidate_cols = [c for c in candidate_cols if c in chart_df.columns]
    candidate_table = chart_df[candidate_cols].sort_values(
        ["champion_candidate_flag", "selection_score", "validation_auc"],
        ascending=[False, False, False],
    )

    build_aggrid_table(candidate_table, height=320)

    # -----------------------------
    # Composite score
    # -----------------------------

    st.markdown("### 4. Composite Selection Score")

    st.caption(
    """
    The **selection score is a composite decision-support metric**, not the final model decision.

    It combines performance, calibration, stability, and simplicity into one normalized score.

    The chart below focuses on one practical question:

    **Which models score well overall while also avoiding overfitting?**
    """
    )

    st.info(
    """
    Important: A higher selection score does not automatically mean the model should be chosen.
    Final model selection also considers interpretability, governance, deployment constraints,
    and business usability.
    """
    )

    with st.expander("How is the selection score calculated?"):
        st.markdown(
            """
    The selection score is calculated as a weighted combination of normalized metrics:

    - **30% Validation AUC** — ranking power
    - **20% Validation KS** — risk separation
    - **20% Log Loss** — inverted, lower is better
    - **15% Brier Score** — inverted, lower is better
    - **10% AUC Gap** — penalizes overfitting
    - **5% Feature Count** — penalizes complexity

    All metrics are scaled to a 0–1 range before weighting.

    The weights reflect practical credit risk priorities: performance matters most, but calibration, stability, and simplicity matter for production use.
    """
        )

    if "selection_score" in chart_df.columns and "auc_gap" in chart_df.columns:
        plot_df = chart_df.dropna(subset=["selection_score", "auc_gap"]).copy()

        score_cutoff = plot_df["selection_score"].median()
        gap_cutoff = plot_df["auc_gap"].abs().median()

        fig = px.scatter(
            plot_df,
            x="auc_gap",
            y="selection_score",
            color="model_family",
            hover_name="model_id",
            hover_data=[
                "model_name",
                "dataset_type",
                "feature_count",
                "validation_auc",
                "validation_ks",
                "validation_log_loss",
                "validation_brier_score",
            ],
            title="Selection Score vs Stability",
        )

        y_min = plot_df["selection_score"].min()
        y_max = plot_df["selection_score"].max()
        x_min = plot_df["auc_gap"].min()
        x_max = plot_df["auc_gap"].max()

        fig.add_vrect(
            x0=x_min,
            x1=gap_cutoff,
            fillcolor="green",
            opacity=0.08,
            line_width=0,
            annotation_text="Stable region",
            annotation_position="top left",
        )

        fig.add_hrect(
            y0=score_cutoff,
            y1=y_max,
            fillcolor="green",
            opacity=0.06,
            line_width=0,
            annotation_text="Higher selection score",
            annotation_position="top right",
        )

        fig.add_vline(
            x=gap_cutoff,
            line_dash="dash",
            line_width=1,
            annotation_text="Median AUC gap",
            annotation_position="top",
        )

        fig.add_hline(
            y=score_cutoff,
            line_dash="dash",
            line_width=1,
            annotation_text="Median selection score",
            annotation_position="right",
        )

        fig.update_xaxes(title="AUC Gap (Train - Validation | lower is better)")
        fig.update_yaxes(title="Selection Score (higher is better)")

        fig.update_layout(
            height=520,
            legend_title_text="Model Family",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            """
    The most attractive region is the **upper-left**: high selection score and low overfitting risk.

    This chart should be used as a screening tool, not as final model approval.
    """
        )

    else:
        st.info("Selection score chart requires selection_score and auc_gap.")

# ---------------------------------------
# BEST MODEL BY FAMILY
# ---------------------------------------

elif section == "Best Model by Family":
    st.subheader("Best Model by Family")

    st.markdown(
    """
    This view prevents the leaderboard from becoming only a boosting or ensemble ranking.
    It shows the strongest representative from each model family.
    """
    )

    family_summary = (
        filtered.sort_values("validation_auc", ascending=False)
        .groupby("model_family", as_index=False)
        .first()
    )

    family_cols = [
        "model_family",
        "model_id",
        "model_name",
        "dataset_type",
        "feature_count",
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "validation_brier_score",
        "auc_gap",
        "selection_score",
        "champion_candidate_flag",
    ]

    family_cols = [c for c in family_cols if c in family_summary.columns]
    family_display = family_summary[family_cols].sort_values("validation_auc", ascending=False)

    build_aggrid_table(family_display, height=330)

# ---------------------------------------
# INITIAL INTERPRETATION
# ---------------------------------------

elif section == "Initial Interpretation":
    st.subheader("Initial Interpretation")

    st.markdown(
    """
    ### What the leaderboard is telling us

    The strongest models are concentrated around boosting and ensemble methods. That is expected for credit-risk data because tree-based boosting models are good at capturing nonlinear patterns and borrower-risk interactions.

    However, the final model should not be chosen by validation AUC alone.

    A practical credit-risk model should consider:

    - **Performance:** Does the model rank risky borrowers effectively?
    - **Stability:** Does validation performance remain close to training performance?
    - **Calibration:** Are predicted probabilities usable for business decisions?
    - **Complexity:** Is the additional performance worth the extra maintenance burden?
    - **Explainability:** Can risk, audit, and governance teams understand the model?
    - **Deployment readiness:** Can the model be scored, monitored, and refreshed reliably?

    ### Practical interpretation

    If an ensemble barely beats a simpler model but is harder to explain and maintain, the simpler model may be the better production champion.

    If an ensemble materially improves performance and remains stable, it can be selected as the performance champion, while a simpler model can remain a governance-friendly challenger.
    """
    )

# ---------------------------------------
# ENSEMBLE DIVERSITY SCORE
# ---------------------------------------

elif section == "Ensemble Diversity Score":
    st.subheader("Ensemble Diversity Score")

    st.markdown(
    """
    ### Why this matters

    An ensemble is useful only if it combines genuinely different model signals.

    If a weighted ensemble gives almost all its weight to one base model, then it is not really acting like a diversified ensemble. It is effectively behaving like that dominant model.

    ### Definition

    For weighted ensembles:

    `ensemble_diversity_score = 1 - max_weight_share`

    Interpretation:

    - Near `0`: ensemble is dominated by one model
    - Near `1`: ensemble is more evenly distributed across base models
    - Not directly applicable: stacking models use a meta-learner rather than fixed contribution weights

    ### Business interpretation

    This helps explain why a weighted ensemble may show nearly identical performance to a base model such as `CAT002`.

    In that case, the ensemble is not adding meaningful diversification and should not automatically be treated as a better production choice.
    """
    )

    ensemble_cols = [
        "model_id",
        "model_name",
        "validation_auc",
        "validation_ks",
        "validation_log_loss",
        "dominant_base_model",
        "max_weight_share",
        "ensemble_diversity_score",
        "ensemble_note",
    ]

    ensemble_cols = [c for c in ensemble_cols if c in filtered.columns]
    ensemble_df = filtered[filtered["model_family"] == "ensemble"].copy()

    if not ensemble_df.empty:
        build_aggrid_table(
            ensemble_df[ensemble_cols].sort_values("validation_auc", ascending=False),
            height=360,
        )
    else:
        st.info("No ensemble models are currently selected.")