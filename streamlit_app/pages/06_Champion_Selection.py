from pathlib import Path

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(page_title="Champion Selection", layout="wide")

st.title("Champion Selection")
st.caption(
    "Final model selection view combining performance, interpretability, scoring behavior, and monitoring evidence."
)



# Paths


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

REGISTRY_PATH = OUTPUTS_DIR / "registry" / "model_summary.csv"
ENSEMBLE_REGISTRY_PATH = OUTPUTS_DIR / "ensemble_registry.xlsx"

SCORING_DIR = OUTPUTS_DIR / "scoring"
SCORE_CORR_PATH = SCORING_DIR / "model_score_correlations.csv"
RISK_BAND_SUMMARY_PATH = SCORING_DIR / "model_risk_band_summary.csv"
TOP_RISK_CAPTURE_APP_PATH = SCORING_DIR / "model_top_risk_capture_app.csv"
THRESHOLD_SUMMARY_APP_PATH = SCORING_DIR / "model_threshold_summary_app.csv"

MONITORING_DIR = OUTPUTS_DIR / "monitoring"
FEATURE_DRIFT_PATH = MONITORING_DIR / "feature_drift_summary.csv"

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


registry_df = load_registry()
score_corr_df = load_csv(SCORE_CORR_PATH)
risk_band_df = load_csv(RISK_BAND_SUMMARY_PATH)
top_risk_capture_app_df = load_csv(TOP_RISK_CAPTURE_APP_PATH)
threshold_summary_app_df = load_csv(THRESHOLD_SUMMARY_APP_PATH)
feature_drift_df = load_csv(FEATURE_DRIFT_PATH)

if registry_df.empty:
    st.error("Model registry not found. Run the model pipelines first.")
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
    if "decision_role" in data.columns:
        gb.configure_column("decision_role", pinned="left", width=220)
    if "selection_rationale" in data.columns:
        gb.configure_column("selection_rationale", width=650)
    if "pros" in data.columns:
        gb.configure_column("pros", width=560)
    if "cons" in data.columns:
        gb.configure_column("cons", width=560)

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


def get_model_row(model_id: str) -> pd.Series:
    rows = registry_df[registry_df["model_id"].astype(str) == str(model_id)]
    if rows.empty:
        return pd.Series(dtype="object")
    return rows.iloc[0]


def get_score_correlation(model_a: str, model_b: str) -> float | None:
    if score_corr_df.empty:
        return None

    corr = score_corr_df.copy()

    if "split" in corr.columns:
        corr = corr[corr["split"].astype(str).str.lower() == "validation"]

    rows = corr[
        (corr["score_col_1"].astype(str) == str(model_a))
        & (corr["score_col_2"].astype(str) == str(model_b))
    ]

    if rows.empty:
        rows = corr[
            (corr["score_col_1"].astype(str) == str(model_b))
            & (corr["score_col_2"].astype(str) == str(model_a))
        ]

    if rows.empty:
        return None

    return float(rows.iloc[0]["correlation"])


def top_risk_capture(model_id: str, top_pct: float = 0.10) -> dict:
    rows = top_risk_capture_app_df[
        (top_risk_capture_app_df["model_id"].astype(str) == str(model_id))
        & (top_risk_capture_app_df["split"].astype(str).str.lower() == "validation")
        & (top_risk_capture_app_df["top_population_share"].round(2) == round(top_pct, 2))
    ]

    if rows.empty:
        return {
            "top_default_capture": None,
            "top_default_rate": None,
            "lift_vs_overall": None,
        }

    row = rows.iloc[0]

    return {
        "top_default_capture": row.get("default_capture_share"),
        "top_default_rate": row.get("top_group_default_rate"),
        "lift_vs_overall": row.get("lift_vs_overall"),
    }


def threshold_behavior(model_id: str, threshold: float = 0.25) -> dict:
    rows = threshold_summary_app_df[
        (threshold_summary_app_df["model_id"].astype(str) == str(model_id))
        & (threshold_summary_app_df["split"].astype(str).str.lower() == "validation")
        & (threshold_summary_app_df["threshold"].round(2) == round(threshold, 2))
    ]

    if rows.empty:
        return {
            "approval_rate": None,
            "predicted_default_rate": None,
        }

    row = rows.iloc[0]

    return {
        "approval_rate": 1 - row.get("predicted_1_share"),
        "predicted_default_rate": row.get("predicted_1_share"),
    }


def get_family_complexity(model_id: str, family: str) -> str:
    model_id = str(model_id).upper()
    family = str(family).lower()

    if family == "logistic":
        return "Low"
    if family in {"tree", "boosting"}:
        return "Medium"
    if family == "ensemble" or "STACK" in model_id:
        return "High"
    if family == "neural_network":
        return "High"
    return "Medium"


def build_model_summary(model_id: str) -> dict:
    row = get_model_row(model_id)

    if row.empty:
        return {}

    family = row.get("model_family", "")
    desc = model_description(model_id, family, row.get("model_name", ""))

    capture = top_risk_capture(model_id)
    threshold = threshold_behavior(model_id)

    corr_cat = get_score_correlation(model_id, MODELERS_DEFAULT_MODEL_ID)
    corr_xgb = get_score_correlation(model_id, BEST_AUC_MODEL_ID)

    return {
        "model_id": model_id,
        "model_family": family,
        "model_description": desc,
        "validation_auc": row.get("validation_auc"),
        "validation_ks": row.get("validation_ks"),
        "validation_brier_score": row.get("validation_brier_score"),
        "validation_log_loss": row.get("validation_log_loss"),
        "auc_gap": row.get("auc_gap"),
        "feature_count": row.get("feature_count"),
        "complexity_level": get_family_complexity(model_id, family),
        "top_10_default_capture": capture["top_default_capture"],
        "top_10_default_rate": capture["top_default_rate"],
        "top_10_lift": capture["lift_vs_overall"],
        "approval_rate_at_25pct_pd": threshold["approval_rate"],
        "predicted_default_rate_at_25pct_pd": threshold["predicted_default_rate"],
        "score_corr_with_CAT002": corr_cat,
        "score_corr_with_XGBSTACK001": corr_xgb,
    }


def compare_to_benchmark(model_id: str, benchmark_id: str) -> dict:
    model = build_model_summary(model_id)
    benchmark = build_model_summary(benchmark_id)

    if not model or not benchmark:
        return {}

    def diff(metric):
        a = model.get(metric)
        b = benchmark.get(metric)
        if pd.isna(a) or pd.isna(b):
            return None
        return a - b

    return {
        "model_id": model_id,
        "benchmark_model": benchmark_id,
        "auc_diff": diff("validation_auc"),
        "ks_diff": diff("validation_ks"),
        "brier_diff": diff("validation_brier_score"),
        "log_loss_diff": diff("validation_log_loss"),
        "top_10_capture_diff": diff("top_10_default_capture"),
        "approval_rate_diff": diff("approval_rate_at_25pct_pd"),
    }


def generate_pros_cons(model_id: str) -> tuple[list[str], list[str], str]:
    summary = build_model_summary(model_id)

    if not summary:
        return [], [], "Model evidence unavailable."

    pros = []
    cons = []

    auc = summary.get("validation_auc")
    ks = summary.get("validation_ks")
    complexity = summary.get("complexity_level")
    family = str(summary.get("model_family", "")).lower()
    model_upper = str(model_id).upper()

    xgb_diff = compare_to_benchmark(model_id, BEST_AUC_MODEL_ID)

    if pd.notna(auc):
        if auc >= 0.86:
            pros.append("Strong validation AUC; competitive discrimination power.")
        elif auc >= 0.80:
            pros.append("Acceptable validation AUC for a credit-risk challenger model.")
        else:
            cons.append("Validation AUC is weaker than top-tier candidates.")

    if pd.notna(ks):
        if ks >= 0.55:
            pros.append("Strong KS statistic; separates defaults and non-defaults well.")
        else:
            cons.append("KS statistic is weaker than preferred champion candidates.")

    if xgb_diff and xgb_diff.get("auc_diff") is not None:
        if xgb_diff["auc_diff"] >= -0.005:
            pros.append("Performance is close to the best-AUC benchmark.")
        elif xgb_diff["auc_diff"] < -0.02:
            cons.append("Meaningfully trails the best-AUC benchmark on validation AUC.")

    top_capture = summary.get("top_10_default_capture")
    if pd.notna(top_capture):
        if top_capture >= 0.50:
            pros.append("Captures a strong share of defaults in the top-risk segment.")
        else:
            cons.append("Top-risk default capture is not clearly leading.")

    corr_cat = summary.get("score_corr_with_CAT002")
    if pd.notna(corr_cat) and model_id != MODELERS_DEFAULT_MODEL_ID:
        if corr_cat >= 0.95:
            pros.append("Scores applicants similarly to CAT002, suggesting operational consistency.")
        elif corr_cat < 0.80:
            cons.append("Scores applicants materially differently from CAT002; decision impact should be reviewed.")

    if complexity == "Low":
        pros.append("Lower model complexity; easier to explain and govern.")
    elif complexity == "Medium":
        pros.append("Moderate complexity; manageable with feature-importance and monitoring evidence.")
    elif complexity == "High":
        cons.append("Higher governance burden due to ensemble, stack, or neural-network complexity.")

    if family == "ensemble" or "STACK" in model_upper:
        pros.append("Useful benchmark for maximum predictive performance.")
        cons.append("Harder to explain and operationalize than a single-model champion.")

    if model_upper.startswith("CAT"):
        pros.append("Strong tabular modeling choice with practical performance and manageable governance burden.")

    if model_id == MODELERS_DEFAULT_MODEL_ID:
        conclusion = (
            "Best suited as the practical champion when the goal is balancing performance, stability, "
            "interpretability, and operational usability."
        )
    elif model_id == BEST_AUC_MODEL_ID:
        conclusion = (
            "Best suited as the statistical benchmark challenger when the goal is maximizing validation discrimination, "
            "but complexity and governance burden should be weighed carefully."
        )
    else:
        conclusion = (
            "Best treated as a challenger candidate. It should only replace the recommended champion if its incremental "
            "performance or business behavior is clearly better after reviewing tradeoffs."
        )

    return pros, cons, conclusion


def build_candidate_decision_row(model_id: str, decision_role: str) -> dict:
    summary = build_model_summary(model_id)
    pros, cons, conclusion = generate_pros_cons(model_id)

    return {
        "decision_role": decision_role,
        "model_id": model_id,
        "model_family": summary.get("model_family"),
        "model_description": summary.get("model_description"),
        "validation_auc": summary.get("validation_auc"),
        "validation_ks": summary.get("validation_ks"),
        "validation_brier_score": summary.get("validation_brier_score"),
        "top_10_default_capture": summary.get("top_10_default_capture"),
        "approval_rate_at_25pct_pd": summary.get("approval_rate_at_25pct_pd"),
        "complexity_level": summary.get("complexity_level"),
        "pros": " | ".join(pros),
        "cons": " | ".join(cons),
        "selection_rationale": conclusion,
    }


def render_candidate_expander(model_id: str, role: str, expanded: bool = False):
    pros, cons, conclusion = generate_pros_cons(model_id)

    with st.expander(f"{model_id} — {role}", expanded=expanded):
        st.markdown("#### Why this model may be chosen")
        if pros:
            for item in pros:
                st.markdown(f"- {item}")
        else:
            st.markdown("- No clear advantages were identified from available evidence.")

        st.markdown("#### Why this model may not be chosen")
        if cons:
            for item in cons:
                st.markdown(f"- {item}")
        else:
            st.markdown("- No major concerns were identified from available evidence.")

        st.markdown("#### Selection interpretation")
        st.markdown(conclusion)



# Sidebar: user-selected model


st.sidebar.header("Champion Review Controls")

families = sorted(registry_df["model_family"].dropna().unique().tolist())

selected_family = st.sidebar.selectbox(
    "Step 1: Select model family",
    families,
    index=0,
)

family_models = registry_df[
    registry_df["model_family"].astype(str) == str(selected_family)
].copy()

family_models = family_models.sort_values("validation_auc", ascending=False)
family_models["label"] = family_models.apply(model_label, axis=1)

selected_label = st.sidebar.selectbox(
    "Step 2: Select model",
    family_models["label"].tolist(),
)

USER_SELECTED_MODEL_ID = selected_label.split(" | ")[0]



# Navigation


section = st.radio(
    "Champion Selection Sections",
    [
        "Overview",
        "Candidate Comparison",
        "Final Recommendation",
    ],
    horizontal=True,
    label_visibility="collapsed",
)



# Overview


if section == "Overview":
    st.subheader("Champion Selection Overview")

    st.markdown(
        """
This page does not introduce new diagnostics.

It synthesizes evidence from the previous pages:

- **Model Leaderboard** → overall ranking and validation metrics
- **Model Interpretability Review** → explanation evidence and model complexity
- **Model Performance Deep Dive** → AUC, KS, calibration, lift, gains, and threshold behavior
- **Scoring & Risk Bands** → score distributions, risk bands, top-risk capture, and decision impact
- **Monitoring** → stability under synthetic stress and monitoring readiness

The purpose is to move from analysis to decision.

A good champion model is not simply the model with the highest AUC.  
It should also be explainable, stable, usable for decisioning, and defensible in model-risk review.
"""
    )

    overview_rows = [
        build_candidate_decision_row(BEST_AUC_MODEL_ID, "Best AUC Benchmark"),
        build_candidate_decision_row(MODELERS_DEFAULT_MODEL_ID, "Modeler Recommended Champion"),
        build_candidate_decision_row(USER_SELECTED_MODEL_ID, "User-Selected Candidate"),
    ]

    overview_df = pd.DataFrame(overview_rows).drop_duplicates(subset=["model_id"])

    build_table(safe_round_cols(overview_df), height=420)



# Candidate Comparison


elif section == "Candidate Comparison":
    st.subheader("Candidate Comparison")

    st.markdown(
        """
This section compares the three key decision candidates in one place:

- **Best AUC Benchmark**: the strongest statistical benchmark
- **Modeler Recommended Champion**: the practical production-style recommendation
- **User-Selected Candidate**: any model the reviewer wants to test against the same logic

This avoids reviewing one-row tables separately and keeps the tradeoff discussion in one place.
"""
    )

    candidate_rows = [
        build_candidate_decision_row(BEST_AUC_MODEL_ID, "Best AUC Benchmark"),
        build_candidate_decision_row(MODELERS_DEFAULT_MODEL_ID, "Modeler Recommended Champion"),
        build_candidate_decision_row(USER_SELECTED_MODEL_ID, "User-Selected Candidate"),
    ]

    candidate_df = pd.DataFrame(candidate_rows).drop_duplicates(
        subset=["decision_role", "model_id"]
    )

    build_table(safe_round_cols(candidate_df), height=430)

    st.markdown("### Difference vs Reference Models")

    comparison_rows = [
        compare_to_benchmark(USER_SELECTED_MODEL_ID, MODELERS_DEFAULT_MODEL_ID),
        compare_to_benchmark(USER_SELECTED_MODEL_ID, BEST_AUC_MODEL_ID),
    ]
    comparison_rows = [r for r in comparison_rows if r]

    if comparison_rows:
        build_table(safe_round_cols(pd.DataFrame(comparison_rows)), height=240)
    else:
        st.info("Reference comparison is unavailable for the selected model.")

    st.markdown("### Candidate Rationale")

    candidates = [
        (BEST_AUC_MODEL_ID, "Best AUC Benchmark"),
        (MODELERS_DEFAULT_MODEL_ID, "Modeler Recommended Champion"),
        (USER_SELECTED_MODEL_ID, "User-Selected Candidate"),
    ]

    seen = set()

    for model_id, role in candidates:
        key = (model_id, role)
        if key in seen:
            continue
        seen.add(key)

        render_candidate_expander(
            model_id=model_id,
            role=role,
            expanded=(model_id == MODELERS_DEFAULT_MODEL_ID),
        )



# Final Recommendation


elif section == "Final Recommendation":
    st.subheader("Final Recommendation")

    final_rows = [
        build_candidate_decision_row(MODELERS_DEFAULT_MODEL_ID, "Recommended Champion"),
        build_candidate_decision_row(BEST_AUC_MODEL_ID, "Benchmark Challenger"),
        build_candidate_decision_row(USER_SELECTED_MODEL_ID, "User-Selected Candidate"),
    ]

    final_df = pd.DataFrame(final_rows).drop_duplicates(
        subset=["decision_role", "model_id"]
    )

    build_table(safe_round_cols(final_df), height=430)

    st.markdown(
        f"""
### Decision guidance

**Recommended Champion:** `{MODELERS_DEFAULT_MODEL_ID}`

`CAT002` is recommended as the practical champion because it offers a strong balance of:

- validation performance
- model usability
- score behavior
- monitoring stability
- governance defensibility

**Benchmark Challenger:** `{BEST_AUC_MODEL_ID}`

`XGBSTACK001` should remain the benchmark challenger because it represents the highest validation-AUC model.  
It is valuable as a performance ceiling, but its stacked-ensemble complexity creates a higher documentation and governance burden.

**User-Selected Candidate:** `{USER_SELECTED_MODEL_ID}`

The selected model should be considered only if its incremental benefit is clear relative to both `CAT002` and `XGBSTACK001`.

Use this model when:

- it materially improves business-relevant metrics
- it produces stable scoring behavior
- its complexity is justified
- its decision impact is acceptable

### Final position

The strongest production-style decision is:

- Use **CAT002** as the recommended champion.
- Keep **XGBSTACK001** as the benchmark challenger.
- Use the selected model as a controlled challenger only if it beats the champion on a clearly defined business objective.
"""
    )