from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(
    page_title="Scoring & Risk Bands",
    layout="wide",
)

st.title("Scoring & Risk Bands")

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

SCORING_DIR = OUTPUTS_DIR / "scoring"
REGISTRY_PATH = OUTPUTS_DIR / "registry" / "model_summary.csv"
ENSEMBLE_REGISTRY_PATH = OUTPUTS_DIR / "ensemble_registry.xlsx"


MODEL_SCORE_SUMMARY_PATH = SCORING_DIR / "model_score_summary.csv"
MODEL_SCORE_CORRELATIONS_PATH = SCORING_DIR / "model_score_correlations.csv"
MODEL_RISK_BAND_SUMMARY_PATH = SCORING_DIR / "model_risk_band_summary.csv"
MODEL_SCORE_DISTRIBUTION_SAMPLE_PATH = SCORING_DIR / "model_score_distribution_sample.csv"
ACTUAL_OUTCOME_SUMMARY_PATH = SCORING_DIR / "actual_outcome_summary.csv"
MODEL_THRESHOLD_SUMMARY_APP_PATH = SCORING_DIR / "model_threshold_summary_app.csv"
MODEL_TOP_RISK_CAPTURE_APP_PATH = SCORING_DIR / "model_top_risk_capture_app.csv"
MODEL_TOPK_OVERLAP_APP_PATH = SCORING_DIR / "model_topk_overlap_app.csv"
MODEL_RISK_BAND_TRANSITIONS_APP_PATH = SCORING_DIR / "model_risk_band_transitions_app.csv"


MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"


# --------------------------------------------------
# Loaders
# --------------------------------------------------

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


registry_df = load_registry()

score_summary_df = load_csv(MODEL_SCORE_SUMMARY_PATH)
score_corr_df = load_csv(MODEL_SCORE_CORRELATIONS_PATH)
risk_band_summary_df = load_csv(MODEL_RISK_BAND_SUMMARY_PATH)
score_distribution_sample_df = load_csv(MODEL_SCORE_DISTRIBUTION_SAMPLE_PATH)
actual_outcome_summary_df = load_csv(ACTUAL_OUTCOME_SUMMARY_PATH)
threshold_summary_df = load_csv(MODEL_THRESHOLD_SUMMARY_APP_PATH)
top_risk_capture_app_df = load_csv(MODEL_TOP_RISK_CAPTURE_APP_PATH)
topk_overlap_app_df = load_csv(MODEL_TOPK_OVERLAP_APP_PATH)
risk_band_transitions_df = load_csv(MODEL_RISK_BAND_TRANSITIONS_APP_PATH)


if score_distribution_sample_df.empty:
    st.error(
        """
Streamlit-ready scoring outputs not found.

Run:

`python -m src.scoring.export_model_scoring_outputs`
"""
    )
    st.stop()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_table(data: pd.DataFrame, height: int = 350):
    if data.empty:
        st.info("No data available.")
        return

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

    if "model_id" in data.columns:
        gb.configure_column("model_id", pinned="left", width=150)
    if "row_id" in data.columns:
        gb.configure_column("row_id", pinned="left", width=100)

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


def get_available_score_models() -> list[str]:
    return sorted(score_distribution_sample_df["model_id"].dropna().astype(str).unique().tolist())


def get_comparison_model_ids(selected_models: list[str]) -> list[str]:
    refs = [MODELERS_DEFAULT_MODEL_ID, BEST_AUC_MODEL_ID]
    refs.extend(selected_models)

    available = set(get_available_score_models())
    return list(dict.fromkeys([m for m in refs if m in available]))


def add_reference_role(model_id: str) -> str:
    roles = []
    if model_id == MODELERS_DEFAULT_MODEL_ID:
        roles.append("Modeler Default")
    if model_id == BEST_AUC_MODEL_ID:
        roles.append("Benchmark (Best Validation AUC)")
    if not roles:
        roles.append("Selected Comparator")
    return " + ".join(roles)

required_reference_models = {MODELERS_DEFAULT_MODEL_ID, BEST_AUC_MODEL_ID}
missing_reference_models = required_reference_models - set(get_available_score_models())

if missing_reference_models:
    st.error(
        f"""
Required reference models are missing from scoring outputs: {sorted(missing_reference_models)}

Re-run:

`python -m src.performance.export_model_performance_outputs`
`python -m src.scoring.export_model_scoring_outputs`
"""
    )
    st.stop()

# --------------------------------------------------
# Intro
# --------------------------------------------------

st.caption(
    "Review validation-based scoring outputs, score distributions, risk bands, and model score agreement."
    )

st.markdown(
    """
    This page reviews how model predictions are converted into usable scoring outputs and risk bands.

    Important distinction:

    - This page focuses on **validation-based scoring behavior**, score distributions, risk bands, and model agreement.
    - Actual default outcomes are included here because the scoring export is built from validation predictions.
    - Final production scoring on unseen applicants should be reviewed separately when a true production scoring file is available.

    Each selected model receives its own predicted PD, score percentile, and risk band assignment.
    """
    )

st.info(
    f"""
    Fixed reference models included on this page:

    - **Modeler's default recommendation:** `{MODELERS_DEFAULT_MODEL_ID}`
    - **Best validation-AUC benchmark:** `{BEST_AUC_MODEL_ID}`

    You can add up to **five challenger models** for comparison.
    """
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Scoring Controls")

available_score_models = get_available_score_models()

if registry_df.empty:
    selectable_registry = pd.DataFrame({"model_id": available_score_models})
else:
    selectable_registry = registry_df[
        registry_df["model_id"].astype(str).isin(available_score_models)
    ].copy()

families = sorted(selectable_registry["model_family"].dropna().unique().tolist())

selected_challengers = []

for i in range(1, 6):
    st.sidebar.markdown(f"### Optional Challenger {i}")

    fam = st.sidebar.selectbox(
        f"Challenger {i} model family",
        ["None"] + families,
        index=0,
        key=f"score_family_{i}",
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
        key=f"score_model_{i}",
    )

    selected_challengers.append(label.split(" | ")[0])

comparison_model_ids = get_comparison_model_ids(selected_challengers)


# --------------------------------------------------
# Build comparison frames
# --------------------------------------------------

score_long = score_distribution_sample_df[
    score_distribution_sample_df["model_id"].astype(str).isin(comparison_model_ids)
].copy()

score_long["reference_role"] = score_long["model_id"].apply(add_reference_role)

summary = score_summary_df[
    score_summary_df["model_id"].astype(str).isin(comparison_model_ids)
].copy()

if not summary.empty:
    summary["reference_role"] = summary["model_id"].apply(add_reference_role)

# --------------------------------------------------
# Navigation
# --------------------------------------------------

section = st.radio(
    "Jump to section",
    [
        "Score Overview",
        "Score Distribution",
        "Score Summary",
        "Risk Band Summary",
        "Actual Default Rate by Band",
        "Model Agreement",
        "Risk Band Transitions",
        "Top-K Risk Overlap",
        "Top Risk Capture",
        "Score Disagreement",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# --------------------------------------------------
# Score Overview
# --------------------------------------------------

if section == "Score Overview":
    st.subheader("Score Overview")

    st.markdown(
        """
This page treats model scores as **predicted probability of default (PD)**.

How to read the score:

- Higher PD = higher estimated default risk.
- Lower PD = lower estimated default risk.
- Risk bands convert continuous PDs into business-friendly categories.

The main question here is not whether a model is accurate. That was handled in validation diagnostics.  
The main question is whether the scoring output behaves sensibly and is usable for business decisioning.
"""
    )

    overview = pd.DataFrame(
        {
            "model_id": comparison_model_ids,
            "reference_role": [add_reference_role(m) for m in comparison_model_ids],
            "description": [
                model_description(
                    m,
                    registry_df.loc[registry_df["model_id"] == m, "model_family"].iloc[0]
                    if not registry_df[registry_df["model_id"] == m].empty
                    and "model_family" in registry_df.columns
                    else "",
                    registry_df.loc[registry_df["model_id"] == m, "model_name"].iloc[0]
                    if not registry_df[registry_df["model_id"] == m].empty
                    and "model_name" in registry_df.columns
                    else "",
                )
                for m in comparison_model_ids
            ],
        }
    )

    build_table(overview, height=250)

    st.markdown(
    """
    ### Current scoring setup

    The Streamlit scoring export includes:

    - model-level predicted PD
    - actual validation target
    - score percentile
    - risk band
    - model-specific risk band summaries

    This allows direct comparison of risk-band behavior across `CAT002`, `XGBSTACK001`, and selected challenger models.
    """
    )


# --------------------------------------------------
# Score Distribution
# --------------------------------------------------

elif section == "Score Distribution":
    st.subheader("Score Distribution")

    st.markdown(
        """
This section compares score distributions across selected models.

How to read it:

- The x-axis is predicted PD.
- The y-axis shows borrower count.
- Models with more mass on the right assign more borrowers to higher risk.

Use the selector to avoid overcrowding when many models are chosen.
"""
    )

    plot_models = st.multiselect(
        "Select models to display",
        options=comparison_model_ids,
        default=comparison_model_ids[:3],
    )

    plot_df = score_long[
        (score_long["model_id"].isin(plot_models))
        & (score_long["split"].astype(str).str.lower() == "validation")
    ]

    if plot_df.empty:
        st.info("No data available for selected models.")
    else:
        fig = px.box(
            plot_df,
            x="model_id",
            y="predicted_pd",
            color="model_id",
            title="Predicted PD Distribution (Box Plot)",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title="Predicted probability of default")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Models with wider spread have stronger differentiation across borrowers.

Very compressed distributions may limit usefulness for segmentation, pricing, or cutoff strategies.
"""
    )


# --------------------------------------------------
# Score Summary
# --------------------------------------------------

elif section == "Score Summary":
    st.subheader("Score Summary")

    st.markdown(
        """
This table summarizes each selected model's score distribution.

Useful fields:

- `p50`: median PD
- `p95` / `p99`: high-risk tail behavior
- `std`: how spread out the model scores are
- `max`: highest assigned PD

This helps identify models that are too compressed, too extreme, or materially different from the benchmark models.
"""
    )

    if summary.empty:
        st.info("No score summary found for selected models.")
    else:
        display_cols = [
            "reference_role",
            "model_id",
            "split",
            "count",
            "mean",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]

        build_table(safe_round_cols(summary[display_cols]), height=320)

        plot_summary = summary[
            summary["split"].astype(str).str.lower() == "validation"
        ].copy()

        plot_df = plot_summary[["model_id", "p50", "p75", "p95", "p99"]].melt(
            id_vars="model_id",
            var_name="percentile",
            value_name="pd_value",
        )

        fig = px.line(
            plot_df,
            x="percentile",
            y="pd_value",
            color="model_id",
            markers=True,
            title="Selected Score Percentiles by Model",
        )

        fig.update_xaxes(title="Percentile")
        fig.update_yaxes(title="Predicted PD")
        fig.update_layout(height=500)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
        """
        **How to interpret this chart:**

        - `p50` shows the typical borrower risk.
        - `p75` and above show how quickly risk increases.
        - `p95` / `p99` show tail risk — the highest-risk borrowers.

        A model with a steeper increase toward `p95`/`p99` is more aggressive at identifying high-risk borrowers.
        """
        )

# --------------------------------------------------
# Risk Band Summary
# --------------------------------------------------

elif section == "Risk Band Summary":
    st.subheader("Risk Band Summary")

    st.markdown(
        """
This section compares model-specific risk band assignments.

How risk bands are created:

- Each model ranks borrowers by predicted PD.
- Score percentile is calculated within each model and split.
- Percentile bands are mapped into business-friendly risk bands.
- Because this is validation-based, each band also includes observed default outcomes.

How to read it:

- Higher-risk bands should have higher mean PD.
- Higher-risk bands should also show higher actual default rates.
- Applicant share shows how much of the population falls into each band.
"""
    )

    band = risk_band_summary_df[
        (risk_band_summary_df["model_id"].astype(str).isin(comparison_model_ids))
        & (risk_band_summary_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if band.empty:
        st.info("No model risk band summary found.")
    else:
        band["reference_role"] = band["model_id"].apply(add_reference_role)

        build_table(safe_round_cols(band), height=360)

        fig = px.bar(
            band,
            x="risk_band",
            y="applicant_pct",
            color="model_id",
            barmode="group",
            title="Applicant Share by Risk Band",
        )

        fig.update_xaxes(title="Risk band")
        fig.update_yaxes(title="Applicant share", tickformat=".0%")
        fig.update_layout(height=500)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
        """
        **How to interpret this chart:**

        This shows how each model distributes applicants across risk bands.

        - More weight in higher-risk bands → stricter model
        - More weight in lower-risk bands → more lenient model

        Different distributions can materially impact approvals and portfolio risk.
        """
        )

        fig = px.line(
            band,
            x="risk_band",
            y="mean_pd",
            color="model_id",
            markers=True,
            title="Average Predicted PD by Risk Band",
        )

        fig.update_xaxes(title="Risk band")
        fig.update_yaxes(title="Average predicted PD")
        fig.update_layout(height=500)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
        """
        **How to interpret this chart:**

        Average predicted PD should increase as risk bands move from Low to Extreme.

        If this relationship is not monotonic, the banding logic may not be consistent with the model’s scoring.
        """
        )

# --------------------------------------------------
# Actual Default Rate by Band
# --------------------------------------------------

elif section == "Actual Default Rate by Band":
    st.subheader("Actual Default Rate by Risk Band")

    st.markdown(
    """
    This section compares model-assigned risk bands against **observed validation outcomes**.

    Important clarification:

    - The model assigns borrowers to risk bands based on predicted PD.
    - The **actual default rate** comes from real observed outcomes (`y_true`) in the validation dataset.
    - The model does NOT generate the actual default rate — it only groups borrowers.

    How to read it:

    - Each band shows the observed default rate for borrowers assigned to that band.
    - A well-performing model should show increasing default rates from Low Risk to Extreme Risk.

    This is a key validation that the risk bands are meaningful and aligned with real outcomes.
    """
    )

    band = risk_band_summary_df[
        (risk_band_summary_df["model_id"].astype(str).isin(comparison_model_ids))
        & (risk_band_summary_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if band.empty:
        st.info("No model risk band summary found.")
    else:
        fig = px.line(
            band,
            x="risk_band",
            y="actual_default_rate",
            color="model_id",
            markers=True,
            title="Actual Default Rate by Risk Band",
        )

        fig.update_xaxes(title="Risk band")
        fig.update_yaxes(title="Actual default rate", tickformat=".0%")
        fig.update_layout(height=520)

        st.plotly_chart(fig, width="stretch")

        display_cols = [
            "model_id",
            "risk_band",
            "applicant_count",
            "applicant_pct",
            "mean_pd",
            "actual_default_rate",
            "default_count",
        ]

        display_cols = [c for c in display_cols if c in band.columns]

        build_table(
            safe_round_cols(band[display_cols]),
            height=360,
        )

# --------------------------------------------------
# Model Agreement
# --------------------------------------------------

elif section == "Model Agreement":
    st.subheader("Model Agreement")

    st.markdown(
        """
This section checks whether selected models rank applicants similarly.

How to read it:

- Correlation close to 1 means two models produce very similar score ordering.
- Lower correlation means the models disagree more.
- High correlation between `CAT002` and `XGBSTACK001` means the practical scoring behavior may be very similar even if the algorithms differ.

This is useful for champion/challenger review because a complex model may not add much business value if its scores are almost identical to a simpler model.
"""
    )

    corr_subset = score_corr_df[
        (score_corr_df["split"].astype(str).str.lower() == "validation")
        & score_corr_df["score_col_1"].astype(str).isin(comparison_model_ids)
        & score_corr_df["score_col_2"].astype(str).isin(comparison_model_ids)
    ].copy()

    if corr_subset.empty:
        st.info("No score correlation output found.")
    else:
        corr_matrix = corr_subset.pivot(
            index="score_col_1",
            columns="score_col_2",
            values="correlation",
        )

        fig = px.imshow(
            corr_matrix,
            text_auto=".3f",
            aspect="auto",
            title="Score Correlation Heatmap",
        )

        fig.update_layout(height=600)

        st.plotly_chart(fig, width="stretch")

        build_table(safe_round_cols(corr_subset), height=330)


# --------------------------------------------------
# Risk Band Transitions
# --------------------------------------------------

elif section == "Risk Band Transitions":
    st.subheader("Risk Band Transitions")

    st.markdown(
        """
This section compares risk band assignments between two selected models.

How to read it:

- Rows = reference model risk bands
- Columns = comparison model risk bands
- Diagonal cells = both models assign applicants to the same band
- Off-diagonal cells = the models assign applicants to different bands

Large off-diagonal values indicate that two models may create materially different business decisions.
"""
    )

    risk_band_order = [
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Very High Risk",
        "Extreme Risk",
    ]

    model_a = st.selectbox(
        "Reference Model",
        comparison_model_ids,
        index=0,
        key="transition_model_a",
    )

    model_b = st.selectbox(
        "Comparison Model",
        comparison_model_ids,
        index=1 if len(comparison_model_ids) > 1 else 0,
        key="transition_model_b",
    )

    view = risk_band_transitions_df[
        (risk_band_transitions_df["reference_model"].astype(str) == str(model_a))
        & (risk_band_transitions_df["comparison_model"].astype(str) == str(model_b))
        & (risk_band_transitions_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if view.empty:
        st.info("No transition data available for selected models.")
    else:
        matrix = view.pivot_table(
            index="reference_risk_band",
            columns="comparison_risk_band",
            values="applicant_pct",
            aggfunc="sum",
            fill_value=0,
        )

        matrix = matrix.reindex(
            index=risk_band_order,
            columns=risk_band_order,
            fill_value=0,
        )

        fig = px.imshow(
            matrix,
            text_auto=".1%",
            aspect="auto",
            title=f"Risk Band Transition Matrix: {model_a} vs {model_b}",
        )

        fig.update_xaxes(title=f"{model_b} risk band")
        fig.update_yaxes(title=f"{model_a} risk band")
        fig.update_layout(height=560)

        st.plotly_chart(fig, width="stretch")

        build_table(safe_round_cols(view), height=360)


# --------------------------------------------------
# Top-K Risk Overlap
# --------------------------------------------------

elif section == "Top-K Risk Overlap":
    st.subheader("Top-K Risk Overlap")

    st.markdown(
        """
This section checks whether selected models identify the same highest-risk applicants.

Why this matters:

- Models may have similar AUC or correlation.
- But they may still disagree on **which applicants** fall into the highest-risk group.
- This matters because real business actions are taken on specific applicants, not just aggregate metrics.

How to read it:

- Select one reference model.
- Choose a top-risk population share, such as top 10%.
- The table shows how much each selected model's top-risk group overlaps with the reference model's top-risk group.

Higher overlap means the models are targeting similar applicants.
"""
    )

    reference_model = st.selectbox(
        "Reference model",
        comparison_model_ids,
        index=0,
        key="topk_reference_model",
    )

    top_k_pct = st.slider(
        "Top-risk population share",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        key="topk_pct",
    )

    overlap_df = topk_overlap_app_df[
        (topk_overlap_app_df["reference_model"].astype(str) == str(reference_model))
        & (topk_overlap_app_df["comparison_model"].astype(str).isin(comparison_model_ids))
        & (topk_overlap_app_df["split"].astype(str).str.lower() == "validation")
        & (topk_overlap_app_df["top_population_share"].round(2) == round(top_k_pct, 2))
    ].copy()

    if overlap_df.empty:
        st.info("No Top-K overlap summary available for selected models.")
    else:
        overlap_df = overlap_df.rename(
            columns={
                "reference_model": "Reference Model",
                "comparison_model": "Comparison Model",
                "top_population_share": "Top % of Applicants",
                "top_population_count": "Top Segment Size",
                "shared_high_risk_applicants": "Shared High-Risk Applicants",
                "overlap_share": "Overlap (%)",
                "different_applicants": "Different Applicants",
            }
        )

        overlap_df["Top % of Applicants"] = (
            overlap_df["Top % of Applicants"] * 100
        ).round(0)
        overlap_df["Overlap (%)"] = (overlap_df["Overlap (%)"] * 100).round(2)

        build_table(safe_round_cols(overlap_df), height=320)

        fig = px.bar(
            overlap_df.sort_values("Overlap (%)", ascending=False),
            x="Comparison Model",
            y="Overlap (%)",
            title=f"Top {top_k_pct:.0%} High-Risk Overlap vs {reference_model}",
        )

        fig.update_xaxes(title="Comparison model")
        fig.update_yaxes(title="Overlap (%)", range=[0, 100])
        fig.update_layout(height=460)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**How to interpret this:**

- Overlap (%) shows how similar two models are in identifying the highest-risk applicants.
- 100% overlap means both models flag the exact same borrowers as high risk.
- Lower overlap means the models would lead to different approval or pricing decisions.

This is one of the most practical ways to compare models beyond AUC.
"""
        )

# --------------------------------------------------
# Top Risk Capture
# --------------------------------------------------

elif section == "Top Risk Capture":
    st.subheader("Top Risk Capture")

    st.markdown(
        """
This section measures how much actual default risk each model captures in the highest-risk slice of the validation population.

How it works:

- Each model ranks applicants from highest predicted PD to lowest.
- The top-risk group is selected, such as the top 10%.
- The section then calculates:
    - observed default rate in that top-risk group
    - share of all validation defaults captured by that group
    - lift versus the overall validation default rate

Why this matters:

A strong risk model should concentrate actual defaults in the highest predicted-risk segment.
"""
    )

    top_capture_pct = st.slider(
        "Top-risk population share",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        key="top_capture_pct",
    )

    capture_df = top_risk_capture_app_df[
        (top_risk_capture_app_df["model_id"].astype(str).isin(comparison_model_ids))
        & (top_risk_capture_app_df["split"].astype(str).str.lower() == "validation")
        & (top_risk_capture_app_df["top_population_share"].round(2) == round(top_capture_pct, 2))
    ].copy()

    if capture_df.empty:
        st.info("No top-risk capture summary available.")
    else:
        capture_df = capture_df.rename(
            columns={
                "model_id": "Model",
                "top_population_share": "Top % of Applicants",
                "top_population_count": "Top Segment Size",
                "overall_default_rate": "Overall Default Rate",
                "top_group_default_rate": "Default Rate (Top Segment)",
                "top_group_default_count": "Defaults Captured (Count)",
                "default_capture_share": "Defaults Captured (%)",
                "lift_vs_overall": "Lift vs Overall",
            }
        )

        capture_df["Top % of Applicants"] = (
            capture_df["Top % of Applicants"] * 100
        ).round(0)
        capture_df["Defaults Captured (%)"] = (
            capture_df["Defaults Captured (%)"] * 100
        ).round(2)
        capture_df["Overall Default Rate"] = (
            capture_df["Overall Default Rate"] * 100
        ).round(2)
        capture_df["Default Rate (Top Segment)"] = (
            capture_df["Default Rate (Top Segment)"] * 100
        ).round(2)
        capture_df["Lift vs Overall"] = capture_df["Lift vs Overall"].round(2)

        build_table(safe_round_cols(capture_df), height=330)

        fig = px.bar(
            capture_df.sort_values("Defaults Captured (%)", ascending=False),
            x="Model",
            y="Defaults Captured (%)",
            title=f"Defaults Captured in Top {top_capture_pct:.0%} Highest-Risk Applicants",
        )

        fig.update_yaxes(
            title="Defaults Captured (%)",
            range=[
                capture_df["Defaults Captured (%)"].min() * 0.95,
                capture_df["Defaults Captured (%)"].max() * 1.05,
            ],
        )

        fig.update_layout(height=460)

        st.plotly_chart(fig, width="stretch")

        fig = px.bar(
            capture_df.sort_values("Lift vs Overall", ascending=False),
            x="Model",
            y="Lift vs Overall",
            title=f"Lift in Top {top_capture_pct:.0%} Highest-Risk Applicants",
        )

        fig.update_yaxes(
            title="Lift vs Overall",
            range=[
                capture_df["Lift vs Overall"].min() * 0.95,
                capture_df["Lift vs Overall"].max() * 1.05,
            ],
        )

        fig.update_xaxes(title="Model")
        fig.update_layout(height=460)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**How to interpret this:**

- Defaults Captured (%) shows how much of total portfolio risk is concentrated in the top-risk segment.
- Higher values mean the model is better at prioritizing risky applicants.
- Lift vs Overall shows how much riskier the top segment is compared to the average borrower.

Even small differences matter at scale — a 1–2% improvement can translate into significant financial impact.
"""
        )
    

# --------------------------------------------------
# Score Disagreement
# --------------------------------------------------

elif section == "Score Disagreement":
    st.subheader("Threshold-Based Score Disagreement")

    st.markdown(
        """
This section converts predicted PDs into simple predicted default flags using a selected threshold.

Decision rule:

- Predicted PD >= threshold → predicted default flag = 1
- Predicted PD < threshold → predicted default flag = 0

The actual default count is shown first because it is the fixed validation outcome baseline.  
The model table then shows how each selected model behaves under the chosen PD cutoff.
"""
    )

    if actual_outcome_summary_df.empty:
        st.info("No actual validation outcome summary available.")
    else:
        st.markdown("### Actual Validation Outcome Counts")
        build_table(safe_round_cols(actual_outcome_summary_df), height=180)

    st.markdown("### Model Predicted Default Flags")

    threshold = st.slider(
        "PD Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.25,
        step=0.01,
    )

    st.markdown(
        """
**Threshold selection context:**

A default threshold of **25% probability of default** is used as a starting point.

This threshold provides a practical separation point for converting continuous predicted PDs into binary predicted default flags in the validation sample.

In practice:

- Lower thresholds flag more borrowers as risky
- Higher thresholds flag fewer borrowers as risky

Use the slider to compare how each selected model behaves under different decision cutoffs.
"""
    )

    model_flag_summary = threshold_summary_df[
        (threshold_summary_df["model_id"].astype(str).isin(comparison_model_ids))
        & (threshold_summary_df["split"].astype(str).str.lower() == "validation")
        & (threshold_summary_df["threshold"].round(2) == round(threshold, 2))
    ].copy()

    if model_flag_summary.empty:
        st.info("No selected model threshold summary available.")
    else:
        build_table(safe_round_cols(model_flag_summary), height=360)

        fig = px.bar(
            model_flag_summary,
            x="model_id",
            y="predicted_1_share",
            title=f"Share of Applicants Flagged as Predicted Default at {threshold:.0%} PD Threshold",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title="Predicted default share", tickformat=".0%")
        fig.update_layout(height=460)

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
**How to interpret this table:**

The table shows how aggressively each selected model flags applicants as predicted defaults under the chosen PD threshold.

- More predicted 1s = stricter model behavior
- Fewer predicted 1s = more lenient model behavior
- True positives are actual defaults correctly flagged
- False positives are non-defaults flagged as risky
- False negatives are actual defaults missed by the model
- True negatives are non-defaults correctly not flagged

This is a threshold-based decision simulation, not the final production cutoff.
"""
        )