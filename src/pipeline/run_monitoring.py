import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.monitoring.drift import monitor_features
from src.monitoring.monitoring_report import create_monitoring_recommendations
from src.monitoring.score_monitoring import monitor_model_scores
from src.monitoring.stability import (
    performance_stability_over_time,
    score_stability_over_time,
)


SCORING_DIR = OUTPUTS_SRC_DIR / "scoring"
MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"
MONITORING_DIR = OUTPUTS_SRC_DIR / "monitoring"


def main() -> None:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(SCORING_DIR / "scored_applicants.csv")
    all_model_scores = pd.read_csv(SCORING_DIR / "all_model_scores.csv")

    reference_tree = pd.read_csv(MODEL_READY_DIR / "df_tree_model_ready_src.csv")
    actual_tree = reference_tree.copy()

    target_col = "SeriousDlqin2yrs"

    reference_features = reference_tree.drop(columns=[target_col], errors="ignore")
    actual_features = actual_tree.drop(columns=[target_col], errors="ignore")

    numeric_features = reference_features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = reference_features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    score_cols = [
        col for col in all_model_scores.columns if col != "row_id"
    ]

    score_summary, score_buckets = monitor_model_scores(
        reference_scores=all_model_scores,
        actual_scores=all_model_scores,
        score_cols=score_cols,
    )

    feature_summary, feature_buckets = monitor_features(
        expected_df=reference_features,
        actual_df=actual_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    recommendations = create_monitoring_recommendations(
        feature_summary=feature_summary,
        score_summary=score_summary,
    )

    score_summary.to_csv(
        MONITORING_DIR / "score_drift_summary.csv",
        index=False,
    )

    score_buckets.to_csv(
        MONITORING_DIR / "score_drift_buckets.csv",
        index=False,
    )

    feature_summary.to_csv(
        MONITORING_DIR / "feature_drift_summary.csv",
        index=False,
    )

    feature_buckets.to_csv(
        MONITORING_DIR / "feature_drift_buckets.csv",
        index=False,
    )

    recommendations.to_csv(
        MONITORING_DIR / "monitoring_recommendations.csv",
        index=False,
    )

    scored_with_target = scored.copy()
    scored_with_target[target_col] = reference_tree[target_col].values

    scored_with_target["synthetic_cohort"] = pd.qcut(
        scored_with_target.index,
        q=10,
        labels=[f"cohort_{i}" for i in range(1, 11)],
    )

    score_stability = score_stability_over_time(
        df=scored_with_target,
        score_col="production_pd",
        time_col="synthetic_cohort",
    )

    performance_stability = performance_stability_over_time(
        df=scored_with_target,
        score_col="production_pd",
        target_col=target_col,
        time_col="synthetic_cohort",
    )

    score_stability.to_csv(
        MONITORING_DIR / "score_stability_synthetic_cohort.csv",
        index=False,
    )

    performance_stability.to_csv(
        MONITORING_DIR / "performance_stability_synthetic_cohort.csv",
        index=False,
    )

    print("Monitoring pipeline complete.")
    print(f"Outputs saved to: {MONITORING_DIR}")


if __name__ == "__main__":
    main()