from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

PERF_DIR = OUTPUTS_DIR / "performance"
SCORING_DIR = OUTPUTS_DIR / "scoring"
MONITORING_DIR = OUTPUTS_DIR / "monitoring"

PREDICTIONS_PATH = PERF_DIR / "model_predictions.csv"
SCORES_WIDE_PATH = SCORING_DIR / "model_scores_wide.csv"

MODELERS_DEFAULT_MODEL_ID = "CAT002"
BEST_AUC_MODEL_ID = "XGBSTACK001"


def get_model_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"split", "row_number", "row_id", "y_true"}
    return [c for c in df.columns if c not in exclude]


def assign_risk_band(score_percentile: pd.Series) -> pd.Series:
    return pd.cut(
        score_percentile,
        bins=[0.00, 0.20, 0.50, 0.77, 0.95, 1.00],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Very High Risk",
            "Extreme Risk",
        ],
        include_lowest=True,
    ).astype(str)


def build_performance_summaries() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Missing: {PREDICTIONS_PATH}")
        return

    df = pd.read_csv(PREDICTIONS_PATH)

    # Predicted vs actual deciles
    decile_records = []

    for (split, model_id), sub in df.groupby(["split", "model_id"]):
        sub = sub.copy()

        sub["decile"] = pd.qcut(
            sub["y_pred"],
            10,
            labels=False,
            duplicates="drop",
        )
        sub["decile"] = 9 - sub["decile"]

        agg = (
            sub.groupby("decile", observed=True)
            .agg(
                predicted_pd=("y_pred", "mean"),
                actual_default_rate=("y_true", "mean"),
                count=("y_true", "size"),
            )
            .reset_index()
        )

        agg["split"] = split
        agg["model_id"] = model_id
        decile_records.append(agg)

    decile_df = pd.concat(decile_records, ignore_index=True)
    decile_df.to_csv(PERF_DIR / "predicted_actual_decile_summary.csv", index=False)

    # Error by PD band
    error_records = []

    for (split, model_id), sub in df.groupby(["split", "model_id"]):
        sub = sub.copy()

        sub["pd_band"] = pd.cut(
            sub["y_pred"],
            bins=[0, 0.05, 0.10, 0.20, 0.40, 1.00],
            include_lowest=True,
        ).astype(str)

        agg = (
            sub.groupby("pd_band", observed=True)
            .agg(
                predicted_pd=("y_pred", "mean"),
                actual_default_rate=("y_true", "mean"),
                count=("y_true", "size"),
            )
            .reset_index()
        )

        agg["abs_error"] = (
            agg["predicted_pd"] - agg["actual_default_rate"]
        ).abs()

        agg["split"] = split
        agg["model_id"] = model_id
        error_records.append(agg)

    error_df = pd.concat(error_records, ignore_index=True)
    error_df.to_csv(PERF_DIR / "pd_band_error_summary.csv", index=False)

    print("Created performance summary files.")


def build_scoring_summaries() -> None:
    if not SCORES_WIDE_PATH.exists():
        print(f"Missing: {SCORES_WIDE_PATH}")
        return

    wide = pd.read_csv(SCORES_WIDE_PATH)

    model_cols = get_model_cols(wide)
    validation = wide[wide["split"].astype(str).str.lower() == "validation"].copy()

    # Actual outcome summary
    actual_summary = pd.DataFrame(
        {
            "actual_class": ["Actual 0", "Actual 1"],
            "count": [
                int((validation["y_true"] == 0).sum()),
                int((validation["y_true"] == 1).sum()),
            ],
        }
    )
    actual_summary["share"] = actual_summary["count"] / actual_summary["count"].sum()
    actual_summary.to_csv(SCORING_DIR / "actual_outcome_summary.csv", index=False)

    # Score distribution sample for box plots
    sample_records = []
    sample_n = 5000

    for model_id in model_cols:
        sub = validation[["split", "row_number", "y_true", model_id]].copy()
        sub = sub.rename(columns={model_id: "predicted_pd"})
        sub["model_id"] = model_id

        if len(sub) > sample_n:
            sub = sub.sample(sample_n, random_state=42)

        sample_records.append(
            sub[["split", "row_number", "y_true", "model_id", "predicted_pd"]]
        )

    score_sample = pd.concat(sample_records, ignore_index=True)
    score_sample.to_csv(SCORING_DIR / "model_score_distribution_sample.csv", index=False)

    # Threshold summary
    threshold_records = []
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    for model_id in model_cols:
        scores = validation[model_id]

        for threshold in thresholds:
            pred_flag = (scores >= threshold).astype(int)

            threshold_records.append(
                {
                    "model_id": model_id,
                    "split": "validation",
                    "threshold": threshold,
                    "predicted_0_count": int((pred_flag == 0).sum()),
                    "predicted_1_count": int((pred_flag == 1).sum()),
                    "predicted_1_share": float(pred_flag.mean()),
                    "actual_1_count": int((validation["y_true"] == 1).sum()),
                    "true_positive_count": int(
                        ((pred_flag == 1) & (validation["y_true"] == 1)).sum()
                    ),
                    "false_positive_count": int(
                        ((pred_flag == 1) & (validation["y_true"] == 0)).sum()
                    ),
                    "false_negative_count": int(
                        ((pred_flag == 0) & (validation["y_true"] == 1)).sum()
                    ),
                    "true_negative_count": int(
                        ((pred_flag == 0) & (validation["y_true"] == 0)).sum()
                    ),
                }
            )

    threshold_df = pd.DataFrame(threshold_records)
    threshold_df.to_csv(SCORING_DIR / "model_threshold_summary_app.csv", index=False)

    # Top-risk capture summary
    top_capture_records = []
    top_pcts = np.round(np.arange(0.01, 0.51, 0.01), 2)

    total_defaults = validation["y_true"].sum()
    overall_default_rate = validation["y_true"].mean()

    for model_id in model_cols:
        for top_pct in top_pcts:
            n_top = max(1, int(len(validation) * top_pct))
            top_group = validation.nlargest(n_top, model_id)

            top_defaults = top_group["y_true"].sum()
            top_default_rate = top_group["y_true"].mean()

            top_capture_records.append(
                {
                    "model_id": model_id,
                    "split": "validation",
                    "top_population_share": top_pct,
                    "top_population_count": n_top,
                    "overall_default_rate": overall_default_rate,
                    "top_group_default_rate": top_default_rate,
                    "top_group_default_count": int(top_defaults),
                    "default_capture_share": (
                        top_defaults / total_defaults if total_defaults > 0 else 0
                    ),
                    "lift_vs_overall": (
                        top_default_rate / overall_default_rate
                        if overall_default_rate > 0
                        else 0
                    ),
                }
            )

    top_capture_df = pd.DataFrame(top_capture_records)
    top_capture_df.to_csv(SCORING_DIR / "model_top_risk_capture_app.csv", index=False)

    # Risk band assignments for transition summaries
    band_lookup = {}

    for model_id in model_cols:
        score_percentile = validation[model_id].rank(pct=True)
        band_lookup[model_id] = pd.DataFrame(
            {
                "row_number": validation["row_number"],
                "risk_band": assign_risk_band(score_percentile),
            }
        )

    # Pairwise risk-band transitions
    transition_records = []

    for reference_model in model_cols:
        ref = band_lookup[reference_model].rename(
            columns={"risk_band": "reference_risk_band"}
        )

        for comparison_model in model_cols:
            comp = band_lookup[comparison_model].rename(
                columns={"risk_band": "comparison_risk_band"}
            )

            merged = ref.merge(comp, on="row_number", how="inner")

            grouped = (
                merged.groupby(
                    ["reference_risk_band", "comparison_risk_band"],
                    observed=True,
                )
                .size()
                .reset_index(name="applicant_count")
            )

            grouped["reference_model"] = reference_model
            grouped["comparison_model"] = comparison_model
            grouped["split"] = "validation"
            grouped["applicant_pct"] = grouped["applicant_count"] / len(merged)

            transition_records.append(grouped)

    transition_df = pd.concat(transition_records, ignore_index=True)
    transition_df.to_csv(SCORING_DIR / "model_risk_band_transitions_app.csv", index=False)

    # Top-K overlap summaries
    overlap_records = []

    for top_pct in top_pcts:
        n_top = max(1, int(len(validation) * top_pct))

        top_sets = {
            model_id: set(validation.nlargest(n_top, model_id)["row_number"])
            for model_id in model_cols
        }

        for reference_model in model_cols:
            reference_top = top_sets[reference_model]

            for comparison_model in model_cols:
                comparison_top = top_sets[comparison_model]
                overlap_count = len(reference_top.intersection(comparison_top))

                overlap_records.append(
                    {
                        "reference_model": reference_model,
                        "comparison_model": comparison_model,
                        "split": "validation",
                        "top_population_share": top_pct,
                        "top_population_count": n_top,
                        "shared_high_risk_applicants": overlap_count,
                        "overlap_share": overlap_count / n_top,
                        "different_applicants": n_top - overlap_count,
                    }
                )

    overlap_df = pd.DataFrame(overlap_records)
    overlap_df.to_csv(SCORING_DIR / "model_topk_overlap_app.csv", index=False)

    print("Created scoring app summary files.")


def build_monitoring_stress_summary() -> None:
    if not SCORES_WIDE_PATH.exists():
        print(f"Missing: {SCORES_WIDE_PATH}")
        return

    wide = pd.read_csv(SCORES_WIDE_PATH)
    validation = wide[wide["split"].astype(str).str.lower() == "validation"].copy()

    model_cols = get_model_cols(validation)
    reference_model = (
        BEST_AUC_MODEL_ID
        if BEST_AUC_MODEL_ID in validation.columns
        else model_cols[0]
    )

    validation["reference_rank_pct"] = validation[reference_model].rank(pct=True)

    low_pool = validation[validation["reference_rank_pct"] <= 0.60]
    mid_pool = validation[
        (validation["reference_rank_pct"] > 0.60)
        & (validation["reference_rank_pct"] <= 0.85)
    ]
    high_pool = validation[validation["reference_rank_pct"] > 0.85]

    cohort_plan = [
        ("baseline_mix", 0.15),
        ("mild_risk_shift", 0.20),
        ("moderate_risk_shift", 0.25),
        ("high_risk_shift", 0.30),
        ("severe_risk_shift", 0.35),
    ]

    cohort_size = min(8000, len(validation))
    threshold = 0.25

    records = []
    cohort_records = []

    for i, (cohort_name, high_share) in enumerate(cohort_plan):
        high_n = int(cohort_size * high_share)
        mid_n = int(cohort_size * 0.30)
        low_n = cohort_size - high_n - mid_n

        cohort = pd.concat(
            [
                high_pool.sample(
                    n=min(high_n, len(high_pool)),
                    replace=True,
                    random_state=100 + i,
                ),
                mid_pool.sample(
                    n=min(mid_n, len(mid_pool)),
                    replace=True,
                    random_state=200 + i,
                ),
                low_pool.sample(
                    n=min(low_n, len(low_pool)),
                    replace=True,
                    random_state=300 + i,
                ),
            ],
            ignore_index=True,
        )

        cohort_records.append(
            {
                "cohort": cohort_name,
                "cohort_size": len(cohort),
                "high_risk_mix": high_share,
                "actual_default_rate": cohort["y_true"].mean(),
            }
        )

        for model_id in model_cols:
            pred_flag = (cohort[model_id] >= threshold).astype(int)

            try:
                auc = roc_auc_score(cohort["y_true"], cohort[model_id])
            except ValueError:
                auc = None

            records.append(
                {
                    "cohort": cohort_name,
                    "model_id": model_id,
                    "cohort_size": len(cohort),
                    "high_risk_mix": high_share,
                    "mean_pd": cohort[model_id].mean(),
                    "p50_pd": cohort[model_id].median(),
                    "p90_pd": cohort[model_id].quantile(0.90),
                    "approval_rate": 1 - pred_flag.mean(),
                    "predicted_default_rate": pred_flag.mean(),
                    "actual_default_rate": cohort["y_true"].mean(),
                    "auc": auc,
                }
            )

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        MONITORING_DIR / "synthetic_stress_summary_app.csv",
        index=False,
    )
    pd.DataFrame(cohort_records).to_csv(
        MONITORING_DIR / "synthetic_stress_cohort_design_app.csv",
        index=False,
    )

    print("Created monitoring stress summary files.")


if __name__ == "__main__":
    build_performance_summaries()
    build_scoring_summaries()
    build_monitoring_stress_summary()