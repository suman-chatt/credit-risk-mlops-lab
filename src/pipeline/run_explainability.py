import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import OUTPUTS_SRC_DIR
from src.explainability.fairness import run_fairness_analysis
from src.explainability.feature_importance import create_feature_importance_summary
from src.explainability.reason_codes import (
    create_applicant_reason_codes_from_scores,
    create_reason_code_table_from_feature_importance,
)
from src.explainability.shap_utils import (
    compute_tree_shap_values,
    create_applicant_shap_reasons,
    create_global_shap_importance,
    sample_for_explainability,
)
from src.scoring.score_models import load_model_registry, select_champion_models


REGISTRY_DIR = OUTPUTS_SRC_DIR / "registry"
DIAGNOSTICS_DIR = OUTPUTS_SRC_DIR / "diagnostics"
SCORING_DIR = OUTPUTS_SRC_DIR / "scoring"
MODEL_READY_DIR = OUTPUTS_SRC_DIR / "model_ready"
EXPLAINABILITY_DIR = OUTPUTS_SRC_DIR / "explainability"


def load_artifact_paths(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

    model_summary = load_model_registry(REGISTRY_DIR / "model_summary.parquet")
    artifact_paths = load_artifact_paths(REGISTRY_DIR / "artifact_paths.json")

    scored_df = pd.read_csv(SCORING_DIR / "scored_applicants.csv")
    scored_all_models = pd.read_csv(SCORING_DIR / "all_model_scores.csv")

    df_tree = pd.read_csv(MODEL_READY_DIR / "df_tree_model_ready_src.csv")
    df_binned = pd.read_csv(MODEL_READY_DIR / "df_logit_bins_ready_src.csv")

    champions = select_champion_models(model_summary)

    practical_model_id = champions["practical_champion"]
    production_model_id = champions.get("performance_champion", practical_model_id)

    print("Explainability models:")
    print(f"Production champion: {production_model_id}")
    print(f"Practical explainable champion: {practical_model_id}")

    # -----------------------------
    # Feature importance
    # -----------------------------

    feature_importance_path = DIAGNOSTICS_DIR / "feature_importance_tables.parquet"

    if feature_importance_path.exists():
        feature_importance_df = pd.read_parquet(feature_importance_path)

        fi_outputs = create_feature_importance_summary(
            feature_importance_df=feature_importance_df,
            champion_model_id=practical_model_id,
            top_n=20,
        )

        fi_outputs["champion_feature_importance"].to_csv(
            EXPLAINABILITY_DIR / "champion_feature_importance.csv",
            index=False,
        )

        fi_outputs["global_feature_importance"].to_csv(
            EXPLAINABILITY_DIR / "global_feature_importance.csv",
            index=False,
        )

        reason_summary = create_reason_code_table_from_feature_importance(
            fi_outputs["champion_feature_importance"],
            top_n=10,
        )

        reason_summary.to_csv(
            EXPLAINABILITY_DIR / "global_reason_code_summary.csv",
            index=False,
        )

    # -----------------------------
    # Applicant-level score proxy explanations
    # -----------------------------

    score_cols = [
        col
        for col in scored_df.columns
        if col not in ["row_id", "production_pd", "score_percentile", "risk_band"]
    ]

    applicant_reason_proxy = create_applicant_reason_codes_from_scores(
        scored_df=scored_df,
        score_cols=score_cols,
        production_score_col="production_pd",
    )

    applicant_reason_proxy.to_csv(
        EXPLAINABILITY_DIR / "applicant_reason_score_proxy.csv",
        index=False,
    )

    # -----------------------------
    # SHAP for practical champion
    # -----------------------------

    practical_model = joblib.load(artifact_paths[practical_model_id])

    practical_row = model_summary[
        model_summary["model_id"] == practical_model_id
    ].iloc[0]

    features = [
        x.strip()
        for x in practical_row["features"].split(",")
        if x.strip()
    ]

    X_tree = df_tree[features].copy()

    X_sample = sample_for_explainability(
        X_tree,
        sample_size=1000,
        random_state=42,
    )

    try:
        shap_df = compute_tree_shap_values(
            model=practical_model,
            X=X_sample,
        )

        shap_importance = create_global_shap_importance(shap_df)

        shap_reasons = create_applicant_shap_reasons(
            shap_df=shap_df,
            X=X_sample,
            top_n=5,
        )

        shap_importance.to_csv(
            EXPLAINABILITY_DIR / "shap_global_importance.csv",
            index=False,
        )

        shap_reasons.to_csv(
            EXPLAINABILITY_DIR / "applicant_shap_reasons.csv",
            index=False,
        )

    except Exception as exc:
        print(f"SHAP skipped: {exc}")

    # -----------------------------
    # Fairness / proxy diagnostics
    # -----------------------------

    fairness_df = scored_df[["row_id", "production_pd"]].copy()

    fairness_df["SeriousDlqin2yrs"] = df_tree["SeriousDlqin2yrs"].values

    fairness_df["age_group"] = df_binned["age_final_bin_v2"].values
    fairness_df["income_group"] = df_binned["MonthlyIncome_final_bin"].values
    fairness_df["dependents_group"] = df_binned[
        "NumberOfDependents_final_bin"
    ].values

    age_fairness = run_fairness_analysis(
        df=fairness_df,
        group_col="age_group",
        target_col="SeriousDlqin2yrs",
        score_col="production_pd",
        threshold=0.5,
    )

    income_fairness = run_fairness_analysis(
        df=fairness_df,
        group_col="income_group",
        target_col="SeriousDlqin2yrs",
        score_col="production_pd",
        threshold=0.5,
    )

    dependents_fairness = run_fairness_analysis(
        df=fairness_df,
        group_col="dependents_group",
        target_col="SeriousDlqin2yrs",
        score_col="production_pd",
        threshold=0.5,
    )

    age_fairness.to_csv(
        EXPLAINABILITY_DIR / "fairness_age_group.csv",
        index=False,
    )

    income_fairness.to_csv(
        EXPLAINABILITY_DIR / "fairness_income_group.csv",
        index=False,
    )

    dependents_fairness.to_csv(
        EXPLAINABILITY_DIR / "fairness_dependents_group.csv",
        index=False,
    )

    scored_all_models.to_csv(
        EXPLAINABILITY_DIR / "champion_score_comparison.csv",
        index=False,
    )

    print("Explainability pipeline complete.")
    print(f"Outputs saved to: {EXPLAINABILITY_DIR}")


if __name__ == "__main__":
    main()