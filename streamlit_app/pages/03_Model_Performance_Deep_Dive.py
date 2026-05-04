from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(
    page_title="Model Performance Deep Dive",
    layout="wide",
)

st.title("Model Performance Deep Dive")
st.caption(
    "Validation-only performance review for CAT002, XGBSTACK001, and selected challenger models."
)



# Paths


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

BASE_REGISTRY_PATH = OUTPUTS_DIR / "registry" / "model_summary.csv"
ENSEMBLE_REGISTRY_PATH = OUTPUTS_DIR / "ensemble_registry.xlsx"

BASE_DIAG_DIR = OUTPUTS_DIR / "diagnostics"
ENSEMBLE_DIAG_DIR = OUTPUTS_DIR / "ensemble" / "diagnostics"

PERF_DIR = OUTPUTS_DIR / "performance"

PVA_DECILE_PATH = PERF_DIR / "predicted_actual_decile_summary.csv"
PD_BAND_ERROR_PATH = PERF_DIR / "pd_band_error_summary.csv"
THRESHOLD_PATH = PERF_DIR / "threshold_analysis.csv"
TOP_BUCKET_PATH = PERF_DIR / "top_bucket_summary.csv"
CALIBRATION_ERROR_PATH = PERF_DIR / "calibration_error_summary.csv"
METRIC_GAPS_PATH = PERF_DIR / "train_validation_metric_gaps.csv"

MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"



# Loaders


@st.cache_data
def load_registry() -> pd.DataFrame:
    frames = []

    if BASE_REGISTRY_PATH.exists():
        frames.append(pd.read_csv(BASE_REGISTRY_PATH))

    if ENSEMBLE_REGISTRY_PATH.exists():
        frames.append(pd.read_excel(ENSEMBLE_REGISTRY_PATH))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model_id"], keep="last").reset_index(drop=True)

    if "auc_gap" not in df.columns and {"train_auc", "validation_auc"}.issubset(df.columns):
        df["auc_gap"] = df["train_auc"] - df["validation_auc"]

    return df


@st.cache_data
def load_diag_file(filename: str) -> pd.DataFrame:
    frames = []

    base_path = BASE_DIAG_DIR / filename
    ensemble_path = ENSEMBLE_DIAG_DIR / filename

    if base_path.exists():
        frames.append(pd.read_csv(base_path))

    if ensemble_path.exists():
        frames.append(pd.read_csv(ensemble_path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_performance_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

registry_df = load_registry()
lift_df = load_diag_file("lift_tables.csv")
ks_df = load_diag_file("ks_curves_sample.csv")
calibration_df = load_diag_file("calibration_tables.csv")

if registry_df.empty:
    st.error("Model registry not found. Run the training and ensemble pipelines first.")
    st.stop()

pva_decile_df = load_performance_file(PVA_DECILE_PATH)
pd_band_error_df = load_performance_file(PD_BAND_ERROR_PATH)
threshold_df = load_performance_file(THRESHOLD_PATH)
top_bucket_df = load_performance_file(TOP_BUCKET_PATH)
calibration_error_df = load_performance_file(CALIBRATION_ERROR_PATH)
metric_gaps_df = load_performance_file(METRIC_GAPS_PATH)



# Helpers


def build_table(data: pd.DataFrame, height: int = 350, key: str | None = None):
    if data.empty:
        st.info("No data available.")
        return

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

    if "model_id" in data.columns:
        gb.configure_column("model_id", pinned="left", width=150)

    if "model_description" in data.columns:
        gb.configure_column("model_description", width=520)

    AgGrid(
        data,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        theme="streamlit",
        key=key,
    )


def safe_round_cols(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def model_description(model_id: str, model_family: str, model_name: str = "") -> str:
    model_id = str(model_id).upper()
    model_family = str(model_family).lower()
    model_name = str(model_name)

    if model_id.startswith("WLOG"):
        return "WOE Logistic Regression: traditional credit-risk logistic model using Weight-of-Evidence variables."
    if model_id.startswith("BLOG"):
        return "Binned Logistic Regression: logistic regression using discretized borrower feature bins."
    if model_id.startswith("RLOG"):
        return "Regularized Logistic Regression: logistic regression with L1/L2 penalty."
    if model_id.startswith("CAT"):
        return "CatBoost: gradient boosting model designed for strong tabular-data performance."
    if model_id.startswith("XGB") and "STACK" not in model_id:
        return "XGBoost: regularized gradient boosting tree model."
    if model_id.startswith("LGB"):
        return "LightGBM: fast gradient boosting model using leaf-wise tree growth."
    if model_id.startswith("GB"):
        return "Histogram Gradient Boosting: sklearn gradient boosting model."
    if model_id.startswith("RF"):
        return "Random Forest: bagged tree ensemble model."
    if model_id.startswith("ET"):
        return "Extra Trees: randomized tree ensemble model."
    if model_id.startswith("DT"):
        return "Decision Tree: single tree model."
    if model_id.startswith("BNB"):
        return "Bernoulli Naive Bayes: probabilistic model for binary / one-hot encoded features."
    if model_id.startswith("GNB") or model_id.startswith("WGNB"):
        return "Gaussian Naive Bayes: probabilistic model using class-conditional feature distributions."
    if model_id.startswith("NN") and "STACK" not in model_id:
        return "Neural Network: multilayer perceptron trained on scaled borrower features."
    if "STACK" in model_id:
        return "Stacking ensemble: second-level model combining base model predictions."
    if model_family == "ensemble":
        return "Ensemble model: combines multiple base model predictions."
    return model_name or "Model description unavailable."


def model_label(row: pd.Series) -> str:
    desc = model_description(
        row.get("model_id"),
        row.get("model_family"),
        row.get("model_name"),
    )
    return (
        f"{row['model_id']} | {row.get('model_family', '')} | "
        f"AUC={row.get('validation_auc', 0):.4f} | {desc}"
    )


def get_comparison_model_ids(selected_models: list[str]) -> list[str]:
    refs = [MODELERS_DEFAULT_MODEL_ID, BEST_AUC_MODEL_ID]
    refs.extend(selected_models)
    available = set(registry_df["model_id"].astype(str))
    return list(dict.fromkeys([m for m in refs if m in available]))


def add_reference_role(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    def role(model_id):
        roles = []
        if model_id == MODELERS_DEFAULT_MODEL_ID:
            roles.append("Modeler Default")
        if model_id == BEST_AUC_MODEL_ID:
            roles.append("Best Validation AUC")
        if not roles:
            roles.append("Selected Comparator")
        return " + ".join(roles)

    out["reference_role"] = out["model_id"].apply(role)
    return out


def validation_only(data: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    if data.empty:
        return data

    out = data[
        (data["model_id"].astype(str).isin(model_ids))
        & (data["split"].astype(str).str.lower() == "validation")
    ].copy()

    return out



# Intro


st.markdown(
    """
This page reviews **validation-set performance only**. It does not use the `cs-test.csv` scoring file and does not claim an out-of-time test.

The purpose is to answer:

**Which models perform well on validation data, and how do they behave across ranking, separation, calibration, lift, gains, and KS diagnostics?**
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



# Sidebar selectors


st.sidebar.header("Performance Controls")

families = sorted(registry_df["model_family"].dropna().unique().tolist())

selected_challengers = []

for i in range(1, 6):
    st.sidebar.markdown(f"### Optional Challenger {i}")

    fam = st.sidebar.selectbox(
        f"Challenger {i} model family",
        ["None"] + families,
        index=0,
        key=f"perf_family_{i}",
    )

    if fam == "None":
        continue

    models = (
        registry_df[registry_df["model_family"] == fam]
        .sort_values("validation_auc", ascending=False)
        .copy()
    )

    models["label"] = models.apply(model_label, axis=1)

    label = st.sidebar.selectbox(
        f"Challenger {i} model",
        models["label"].tolist(),
        key=f"perf_model_{i}",
    )

    selected_challengers.append(label.split(" | ")[0])

comparison_model_ids = get_comparison_model_ids(selected_challengers)

comparison_df = registry_df[
    registry_df["model_id"].astype(str).isin(comparison_model_ids)
].copy()

comparison_df = add_reference_role(comparison_df)
comparison_df["model_description"] = comparison_df.apply(
    lambda r: model_description(
        r.get("model_id"),
        r.get("model_family"),
        r.get("model_name"),
    ),
    axis=1,
)

# stable order: CAT002, XGBSTACK001, selected challengers
order_map = {m: i for i, m in enumerate(comparison_model_ids)}
comparison_df["sort_order"] = comparison_df["model_id"].map(order_map)
comparison_df = comparison_df.sort_values("sort_order")



# Navigation


section = st.radio(
    "Jump to section",
    [
        "Metric Overview",
        "Validation Metric Charts",
        "Train vs Validation Gaps",
        "Gains Chart",
        "Lift Chart",
        "Top-of-Book Summary",
        "KS Curve",
        "Calibration Curve",
        "Calibration Error",
        "Threshold Analysis",
        "Predicted vs Actual (Deciles)",
        "Error by PD Band",
        "Ranking Consistency",
    ],
    horizontal=True,
    label_visibility="collapsed",
)



# Metric Overview


if section == "Metric Overview":
    st.subheader("Metric Overview")

    st.markdown(
        """
This table summarizes validation performance for the fixed reference models and selected challengers.
"""
    )

    display_cols = [
        "reference_role",
        "model_id",
        "model_family",
        "model_name",
        "model_description",
        "feature_count",
        "validation_auc",
        "validation_ks",
        "validation_brier_score",
        "validation_log_loss",
        "train_auc",
        "auc_gap",
    ]

    display_cols = [c for c in display_cols if c in comparison_df.columns]

    build_table(
        safe_round_cols(comparison_df[display_cols]),
        height=380,
    )



# Validation Metric Charts


elif section == "Validation Metric Charts":
    st.subheader("Validation Metric Charts")

    st.markdown(
        """
This section compares the selected models on four validation metrics.

Each metric is shown separately because the scales are different. AUC and KS are shown as percentages because they are easier to read that way.

- **AUC:** ranking power. Higher is better.
- **KS:** separation between higher-risk and lower-risk borrowers. Higher is better.
- **Brier Score:** average probability error. Lower is better.
- **Log Loss:** probability penalty. Lower is better, and bad probability estimates are penalized heavily.

A model should not be selected from one metric alone. The goal is to find a model that ranks well, separates risk well, and produces usable probabilities.
"""
    )

    metric_info = [
        ("validation_auc", "Validation AUC (%)", "higher", True, [60, 90]),
        ("validation_ks", "Validation KS (%)", "higher", True, [50, 70]),
        ("validation_brier_score", "Validation Brier Score", "lower", False, None),
        ("validation_log_loss", "Validation Log Loss", "lower", False, None),
    ]

    for metric, title, direction, as_pct, y_range in metric_info:
        if metric not in comparison_df.columns:
            continue

        plot_df = comparison_df[["model_id", "model_family", metric]].dropna().copy()
        plot_df["plot_value"] = plot_df[metric] * 100 if as_pct else plot_df[metric]
        plot_df = plot_df.sort_values("plot_value", ascending=(direction == "lower"))

        fig = px.bar(
            plot_df,
            x="model_id",
            y="plot_value",
            color="model_family",
            title=f"{title} ({'higher is better' if direction == 'higher' else 'lower is better'})",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title=title)

        if y_range is not None:
            fig.update_yaxes(range=y_range)

        fig.update_layout(height=420)

        st.plotly_chart(fig, use_container_width=True)


# Train vs Validation Gaps


elif section == "Train vs Validation Gaps":
    st.subheader("Train vs Validation Gaps")

    st.markdown(
        """
This section checks whether model performance drops materially from training to validation.

How to read it:

- **AUC Gap** = Train AUC - Validation AUC.
- **KS Gap** = Train KS - Validation KS.
- Smaller gaps are better.
- Large positive gaps suggest the model may be learning training-specific patterns that do not generalize as well to validation data.

This is not a time-drift test. It is a train-validation generalization check.
"""
    )

    gaps = metric_gaps_df[
        metric_gaps_df["model_id"].astype(str).isin(comparison_model_ids)
    ].copy()

    if gaps.empty:
        st.info("No train-validation gap output found. Run the performance exporter first.")
    else:
        build_table(safe_round_cols(gaps), height=300)

        gap_cols = [c for c in ["auc_gap", "ks_gap"] if c in gaps.columns]

        if gap_cols:
            gap_long = gaps[["model_id"] + gap_cols].melt(
                id_vars="model_id",
                var_name="gap_metric",
                value_name="gap_value",
            )

            fig = px.bar(
                gap_long,
                x="model_id",
                y="gap_value",
                color="gap_metric",
                barmode="group",
                title="Train vs Validation Performance Gaps",
            )

            fig.update_xaxes(title="Model")
            fig.update_yaxes(title="Train - Validation gap")
            fig.update_layout(height=450)

            st.plotly_chart(fig, use_container_width=True)


# Gains Chart


elif section == "Gains Chart":
    st.subheader("Gains Chart")

    st.markdown(
        """
    The gains chart shows how quickly a model captures actual defaults as we move from the highest predicted-risk borrowers to the lowest predicted-risk borrowers.

    How to read it:

    - The validation population is sorted from highest predicted default probability to lowest.
    - The sorted population is split into risk buckets.
    - The x-axis shows the cumulative share of the population reviewed.
    - The y-axis shows the cumulative share of actual defaults captured.
    - The dashed diagonal line is random selection.

    A stronger model rises above the random baseline early. For example, if the first 20% of the population captures 70% of defaults, the model is concentrating risk well.
    """
    )

    gains = validation_only(lift_df, comparison_model_ids)

    if gains.empty:
        st.info("No validation gains data found for the selected models.")
    else:
        fig = px.line(
            gains.sort_values(["model_id", "cum_total_pct"]),
            x="cum_total_pct",
            y="cum_bad_pct",
            color="model_id",
            markers=True,
            title="Validation Gains Chart",
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random baseline",
                line=dict(dash="dash"),
            )
        )

        fig.update_xaxes(title="Cumulative population share")
        fig.update_yaxes(title="Cumulative default capture share")
        fig.update_layout(height=620)

        st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "model_id",
            "bucket",
            "total",
            "bad",
            "good",
            "bad_rate",
            "cum_bad_pct",
            "cum_total_pct",
            "lift",
        ]
        display_cols = [c for c in display_cols if c in gains.columns]

        build_table(
            safe_round_cols(gains[display_cols].sort_values(["model_id", "cum_total_pct"])),
            height=350,
        )



# Lift Chart


elif section == "Lift Chart":
    st.subheader("Lift Chart")

    st.markdown(
        """
    The lift chart shows how much better each risk bucket is than random selection.

    How to read it:

    - Bucket 1 is the highest predicted-risk group.
    - Later buckets represent progressively lower predicted-risk groups.
    - The y-axis shows lift. A lift of 5 means that bucket has about 5 times the default concentration of random selection.
    - The dashed line at 1 represents random performance.

    A good credit-risk model should show high lift in the first few buckets, then gradually decline toward 1 as we move into safer borrowers.
    """
    )

    lift = validation_only(lift_df, comparison_model_ids)

    if lift.empty:
        st.info("No validation lift data found for the selected models.")
    else:
        lift = lift.copy()
        lift["bucket_rank"] = lift.groupby("model_id").cumcount() + 1

        fig = px.line(
            lift,
            x="bucket_rank",
            y="lift",
            color="model_id",
            markers=True,
            title="Validation Lift by Risk Bucket",
        )

        fig.add_hline(y=1.0, line_dash="dash", annotation_text="Random baseline")

        fig.update_xaxes(title="Risk bucket rank: 1 = highest predicted risk")
        fig.update_yaxes(title="Lift")
        fig.update_layout(height=620)

        st.plotly_chart(fig, use_container_width=True)

        top_bucket = (
            lift.sort_values(["model_id", "bucket_rank"])
            .groupby("model_id", as_index=False)
            .first()
            [["model_id", "bad_rate", "cum_bad_pct", "lift"]]
            .rename(
                columns={
                    "bad_rate": "top_bucket_bad_rate",
                    "cum_bad_pct": "top_bucket_cumulative_default_capture",
                    "lift": "top_bucket_lift",
                }
            )
        )

        st.markdown("### Top-Risk Bucket Summary")
        build_table(safe_round_cols(top_bucket), height=220)



# Top-of-Book Summary 


elif section == "Top-of-Book Summary":
    st.subheader("Top-of-Book Summary")

    st.markdown(
        """
This section summarizes how much risk each model concentrates in the highest-risk slice of the validation population.

How to read it:

- The model scores borrowers by predicted probability of default.
- Borrowers are sorted from highest predicted PD to lowest predicted PD.
- The top 5%, 10%, and 20% groups are reviewed.
- **Top bucket bad rate** shows the observed default rate in that high-risk slice.
- **Top bucket default capture** shows how much of all validation defaults are captured in that slice.

This is a practical version of the lift/gains view. It answers:  
**If we only focused on the highest-risk borrowers, how much default risk would each model capture?**
"""
    )

    top_book = top_bucket_df[
        (top_bucket_df["model_id"].astype(str).isin(comparison_model_ids))
        & (top_bucket_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if top_book.empty:
        st.info("No top-of-book summary found. Run the performance exporter first.")
    else:
        top_book["top_population_pct_display"] = (
            top_book["top_population_pct"] * 100
        ).round(0).astype(int).astype(str) + "%"

        build_table(safe_round_cols(top_book), height=330)

        fig = px.bar(
            top_book,
            x="model_id",
            y="top_bucket_default_capture",
            color="top_population_pct_display",
            barmode="group",
            title="Default Capture in Highest-Risk Population Segments",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(
            title="Share of all defaults captured",
            tickformat=".0%",
            range=[0.30, 0.80],
        )
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)



# KS Curve


elif section == "KS Curve":
    st.subheader("KS Curve")

    st.markdown(
        """
    The KS curve measures how well the model separates defaults from non-defaults across the ranked population.

    How to read it:

    - The population is sorted from highest predicted risk to lowest predicted risk.
    - The x-axis shows cumulative population share.
    - The y-axis shows the gap between cumulative defaults captured and cumulative non-defaults captured.
    - The highest point of the curve is the KS statistic.

    A higher peak means better separation. If models have very similar KS curves, they are separating risk in broadly similar ways, even if one has a slightly higher validation AUC.
    """
    )

    ks = validation_only(ks_df, comparison_model_ids)

    if ks.empty:
        st.info("No validation KS curve data found for the selected models.")
    else:
        fig = px.line(
            ks.sort_values(["model_id", "population_pct"]),
            x="population_pct",
            y="ks",
            color="model_id",
            title="Validation KS Curve",
        )

        fig.update_xaxes(title="Cumulative population share")
        fig.update_yaxes(title="KS separation")
        fig.update_layout(height=620)

        st.plotly_chart(fig, use_container_width=True)

        ks_summary = (
            ks.groupby("model_id", as_index=False)["ks"]
            .max()
            .rename(columns={"ks": "max_validation_ks_from_curve"})
            .sort_values("max_validation_ks_from_curve", ascending=False)
        )

        st.markdown("### Maximum KS from Curve")
        build_table(safe_round_cols(ks_summary), height=240)



# Calibration Curve


elif section == "Calibration Curve":
    st.subheader("Calibration Curve")

    st.markdown(
        """
    Calibration checks whether predicted probabilities behave like real probabilities.

    How to read it:

    - The model produces a **predicted probability of default (PD)** for each borrower.
    - The validation data is grouped into probability bins (e.g., borrowers with similar predicted PDs).
    - The x-axis shows the **average predicted probability of default (PD)** in each bin.
    - The y-axis shows the **actual observed default rate** in that bin.
    - The dashed diagonal line represents perfect calibration (predicted PD = actual default rate).

    A well-calibrated model should sit close to the diagonal line. 

    - If the model is **below the line**, it is over-predicting risk (PD too high).
    - If the model is **above the line**, it is under-predicting risk (PD too low).

    This chart is critical because:

    - A model can rank well (high AUC/KS) but still produce poor probabilities.
    - Poor calibration leads to bad decisions in **pricing, limits, provisioning, and expected loss calculations**.
    """
    )

    cal = validation_only(calibration_df, comparison_model_ids)

    if cal.empty:
        st.info("No validation calibration data found for the selected models.")
    else:
        fig = px.line(
            cal.sort_values(["model_id", "predicted_prob"]),
            x="predicted_prob",
            y="actual_rate",
            color="model_id",
            markers=True,
            title="Validation Calibration Curve",
        )

        max_axis = float(
            max(
                cal["predicted_prob"].max(),
                cal["actual_rate"].max(),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, max_axis],
                y=[0, max_axis],
                mode="lines",
                name="Perfect calibration",
                line=dict(dash="dash"),
            )
        )

        fig.update_xaxes(title="Average predicted probability")
        fig.update_yaxes(title="Observed default rate")
        fig.update_layout(height=620)

        st.plotly_chart(fig, use_container_width=True)

        display_cols = [
            "model_id",
            "bin",
            "predicted_prob",
            "actual_rate",
        ]
        display_cols = [c for c in display_cols if c in cal.columns]

        build_table(
            safe_round_cols(cal[display_cols].sort_values(["model_id", "bin"])),
            height=330,
        )


# Calibration Error


elif section == "Calibration Error":
    st.subheader("Calibration Error")

    st.markdown(
        """
This section converts the calibration curve into summary metrics.

How it is calculated:

- For each calibration bin, we compare average predicted PD against actual observed default rate.
- **Mean absolute calibration error** averages the absolute difference across bins.
- **Max absolute calibration error** shows the worst bin-level calibration miss.
- **Mean squared calibration error** penalizes larger calibration misses more heavily.

Lower values are better.

This gives a compact way to compare calibration quality across models without relying only on the visual curve.
"""
    )

    cal_error = calibration_error_df[
        (calibration_error_df["model_id"].astype(str).isin(comparison_model_ids))
        & (calibration_error_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if cal_error.empty:
        st.info("No calibration error summary found. Run the performance exporter first.")
    else:
        build_table(safe_round_cols(cal_error), height=280)

        fig = px.bar(
            cal_error.sort_values("mean_abs_calibration_error"),
            x="model_id",
            y="mean_abs_calibration_error",
            title="Mean Absolute Calibration Error (lower is better)",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title="Mean absolute calibration error")
        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)


# Threshold Analysis


elif section == "Threshold Analysis":
    st.subheader("Threshold Analysis")

    st.markdown(
    """
    **PD (Probability of Default)** is the model’s estimate of how likely a borrower is to default.

    - A PD of **0.05 (5%)** means the model estimates a 5% chance of default.
    - A PD of **0.30 (30%)** means the borrower is considered much riskier.

    The PD threshold is used to make decisions:

    - Borrowers with **PD below the threshold** → approved  
    - Borrowers with **PD above the threshold** → rejected or flagged as high risk  

    All charts below show how changing this threshold impacts approvals, risk, and default capture.
    """
    )

    st.markdown(
        """
This section shows what happens if the model is used as a decision rule.

Assumption used here:

- Higher predicted PD means higher default risk.
- A borrower is treated as **approved** when predicted PD is below the selected threshold.
- A borrower is treated as **rejected / high-risk** when predicted PD is at or above the selected threshold.

How to read it:

- **Approval rate** shows the share of borrowers below the PD cutoff.
- **Approved bad rate** shows the observed default rate among approved borrowers.
- **Rejected bad rate** shows the observed default rate among rejected borrowers.
- **Captured bad share rejected** shows how many total defaults are captured by the rejected group.

This does not set the final business cutoff. It shows the tradeoff between growth and risk at different PD thresholds.
"""
    )

    threshold = threshold_df[
        (threshold_df["model_id"].astype(str).isin(comparison_model_ids))
        & (threshold_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if threshold.empty:
        st.info("No threshold analysis found. Run the performance exporter first.")
    else:
        selected_threshold = st.slider(
            "Select PD threshold for table view",
            min_value=0.01,
            max_value=0.99,
            value=0.20,
            step=0.01,
        )

        nearest = (
            threshold.assign(abs_diff=(threshold["threshold"] - selected_threshold).abs())
            .sort_values("abs_diff")
            .groupby("model_id", as_index=False)
            .first()
        )

        st.markdown("### Selected Threshold Summary")
        build_table(safe_round_cols(nearest.drop(columns=["abs_diff"])), height=300)

        fig = px.line(
            threshold.sort_values(["model_id", "threshold"]),
            x="threshold",
            y="approval_rate",
            color="model_id",
            title="Approval Rate by PD Threshold",
        )

        fig.update_xaxes(title="PD threshold")
        fig.update_yaxes(title="Approval rate")
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        """
        **How to interpret this chart:**  
        As the PD threshold increases, more borrowers fall below the cutoff, so the approval rate rises.  
        This chart shows the growth side of the tradeoff: a higher cutoff approves more applicants, but may also allow more risky borrowers through.
        """
        )

        fig = px.line(
            threshold.sort_values(["model_id", "threshold"]),
            x="threshold",
            y="approved_bad_rate",
            color="model_id",
            title="Approved Bad Rate by PD Threshold",
        )

        fig.update_xaxes(title="PD threshold")
        fig.update_yaxes(title="Observed bad rate among approved borrowers")
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        """
        **How to interpret this chart:**  
        This shows the observed default rate among borrowers who would be approved at each PD threshold.  
        A lower approved bad rate means the accepted population is cleaner. As the cutoff increases, the model approves more borrowers, and the approved bad rate usually rises.
        """
        )

        fig = px.line(
            threshold.sort_values(["model_id", "threshold"]),
            x="threshold",
            y="captured_bad_share_rejected",
            color="model_id",
            title="Default Capture in Rejected / High-Risk Group",
        )

        fig.update_xaxes(title="PD threshold")
        fig.update_yaxes(title="Share of total defaults captured by rejected group")
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        """
        **How to interpret this chart:**  
        This shows the share of all validation defaults captured by the rejected or high-risk group.  
        A higher value means the cutoff is successfully isolating more defaults, but it usually comes with a lower approval rate. This is the core risk-control tradeoff.
        """
        )



# Predicted vs Actual (Decile Analysis)


elif section == "Predicted vs Actual (Deciles)":
    st.subheader("Predicted vs Actual (Decile Analysis)")

    st.markdown(
        """
This section compares **model-predicted risk vs actual observed defaults**.

How it works:

- Each model assigns a **PD (Probability of Default)** to every borrower.
- Borrowers are sorted from **highest PD (riskiest)** to **lowest PD (safest)**.
- The population is split into **10 equal-sized groups (deciles)**.
- For each decile:
    - **Average predicted PD** is calculated
    - **Actual default rate** is calculated

How to read it:

- If the model is accurate:
    - Predicted PD ≈ Actual default rate
- If predicted PD is higher than actual:
    - Model is **overestimating risk**
- If predicted PD is lower than actual:
    - Model is **underestimating risk**

This is one of the most important checks because it directly compares **model outputs to real outcomes**.
"""
    )

    decile_df = pva_decile_df[
        (pva_decile_df["model_id"].astype(str).isin(comparison_model_ids))
        & (pva_decile_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if decile_df.empty:
        st.info("No predicted-vs-actual decile summary found. Run the app summary exporter first.")
    else:
        fig = px.line(
            decile_df.sort_values(["model_id", "decile"]),
            x="decile",
            y="predicted_pd",
            color="model_id",
            markers=True,
            title="Predicted PD vs Actual Default Rate by Decile",
        )

        for model_id in decile_df["model_id"].unique():
            sub = decile_df[decile_df["model_id"] == model_id]
            fig.add_trace(
                go.Scatter(
                    x=sub["decile"],
                    y=sub["actual_default_rate"],
                    mode="lines+markers",
                    name=f"{model_id} (Actual)",
                    line=dict(dash="dash"),
                )
            )

        fig.update_xaxes(title="Risk decile (0 = highest risk)")
        fig.update_yaxes(title="Rate")
        fig.update_layout(height=600)

        st.plotly_chart(fig, width="stretch")

        build_table(
            safe_round_cols(decile_df),
            height=350,
            key="pva_decile_table",
        )

        st.markdown(
        """
**How to interpret this chart:**

- Solid lines = model predicted PD  
- Dashed lines = actual observed defaults  

A strong model should:

- Show a **clear downward trend** (risk decreases across deciles)
- Have predicted and actual lines **closely aligned**

Large gaps indicate calibration or probability issues.
"""
        )

        build_table(safe_round_cols(decile_df), height=350)


# Error by PD band


elif section == "Error by PD Band":
    st.subheader("Error by PD Band")

    st.markdown(
        """
This section shows **where the model is making prediction errors** across risk levels.

How it works:

- Borrowers are grouped into **PD bands** based on predicted probability.
- For each band:
    - Average predicted PD is calculated
    - Actual default rate is calculated
    - Error = |Predicted PD − Actual Default Rate|

How to read it:

- Low error → model is accurate in that risk range
- High error → model is misestimating risk

This helps answer:

**Is the model wrong for high-risk borrowers, low-risk borrowers, or both?**
"""
    )

    error_df = pd_band_error_df[
        (pd_band_error_df["model_id"].astype(str).isin(comparison_model_ids))
        & (pd_band_error_df["split"].astype(str).str.lower() == "validation")
    ].copy()

    if error_df.empty:
        st.info("No PD-band error summary found. Run the app summary exporter first.")
    else:
        fig = px.bar(
            error_df,
            x="pd_band",
            y="abs_error",
            color="model_id",
            barmode="group",
            title="Absolute Prediction Error by PD Band",
        )

        fig.update_xaxes(title="PD band")
        fig.update_yaxes(title="Absolute error")
        fig.update_layout(height=500)

        st.plotly_chart(fig, width="stretch")

        build_table(
            safe_round_cols(error_df),
            height=350,
            key="pd_band_error_table",
        )

        st.markdown(
        """
**How to interpret this chart:**

- Taller bars = larger prediction errors
- Compare models across the same PD band

Typical patterns:

- Errors in **high PD bands** → model struggles with risky borrowers  
- Errors in **low PD bands** → model struggles with safe borrowers  
- Flat error profile → stable model across segments
"""
        )

        build_table(safe_round_cols(error_df), height=350)


# Ranking Consistency


elif section == "Ranking Consistency":
    st.subheader("Ranking Consistency")

    st.markdown(
        """
    Ranking consistency checks whether the same models remain strong across different validation metrics.

    How it is calculated:

    - Models are ranked separately on AUC, KS, Brier Score, and Log Loss.
    - AUC and KS are ranked with higher values better.
    - Brier Score and Log Loss are ranked with lower values better.
    - The final consistency score is a simple average of those metric ranks.
    - Lower average rank is better.

    This is not a weighted score and not the final champion decision. It is a review tool that shows whether a model is broadly strong or only strong on one metric.
    """
    )

    rank_df = comparison_df.copy()

    if "validation_auc" in rank_df.columns:
        rank_df["rank_auc"] = rank_df["validation_auc"].rank(ascending=False, method="min")

    if "validation_ks" in rank_df.columns:
        rank_df["rank_ks"] = rank_df["validation_ks"].rank(ascending=False, method="min")

    if "validation_brier_score" in rank_df.columns:
        rank_df["rank_brier"] = rank_df["validation_brier_score"].rank(ascending=True, method="min")

    if "validation_log_loss" in rank_df.columns:
        rank_df["rank_log_loss"] = rank_df["validation_log_loss"].rank(ascending=True, method="min")

    rank_cols = [
        "reference_role",
        "model_id",
        "model_family",
        "validation_auc",
        "rank_auc",
        "validation_ks",
        "rank_ks",
        "validation_brier_score",
        "rank_brier",
        "validation_log_loss",
        "rank_log_loss",
        "auc_gap",
    ]

    rank_cols = [c for c in rank_cols if c in rank_df.columns]

    rank_display = rank_df[rank_cols].copy()

    rank_number_cols = [c for c in rank_display.columns if c.startswith("rank_")]
    if rank_number_cols:
        rank_display["average_metric_rank"] = rank_display[rank_number_cols].mean(axis=1)
        rank_display = rank_display.sort_values("average_metric_rank")

    build_table(safe_round_cols(rank_display), height=380)

    if "average_metric_rank" in rank_display.columns:
        fig = px.bar(
            rank_display.sort_values("average_metric_rank", ascending=True),
            x="model_id",
            y="average_metric_rank",
            color="model_family",
            title="Average Validation Metric Rank (lower is better)",
        )

        fig.update_xaxes(title="Model")
        fig.update_yaxes(title="Average rank")
        fig.update_layout(height=420)

        st.plotly_chart(fig, use_container_width=True)