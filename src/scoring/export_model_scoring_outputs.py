from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs_src"

PERFORMANCE_DIR = OUTPUTS_DIR / "performance"
SCORING_DIR = OUTPUTS_DIR / "scoring"

PREDICTIONS_PATH = PERFORMANCE_DIR / "model_predictions.csv"

SCORING_DIR.mkdir(parents=True, exist_ok=True)


RISK_BAND_LABELS = [
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk",
    "Extreme Risk",
]


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}. "
            "Run: python -m src.performance.export_model_performance_outputs"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {"model_id", "split", "row_number", "y_true", "y_pred"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")

    df = df.rename(columns={"y_pred": "predicted_pd"})

    df["predicted_pd"] = pd.to_numeric(df["predicted_pd"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

    df = df.dropna(subset=["predicted_pd", "y_true"]).copy()

    return df


def assign_score_percentile_and_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["score_percentile"] = (
        out.groupby(["model_id", "split"])["predicted_pd"]
        .rank(method="average", pct=True)
    )

    out["risk_band"] = pd.cut(
        out["score_percentile"],
        bins=[0.00, 0.20, 0.50, 0.80, 0.95, 1.00],
        labels=RISK_BAND_LABELS,
        include_lowest=True,
    )

    return out


def build_model_scores_wide(scored: pd.DataFrame) -> pd.DataFrame:
    wide = scored.pivot_table(
        index=["split", "row_number", "y_true"],
        columns="model_id",
        values="predicted_pd",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None

    return wide


def build_score_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby(["model_id", "split"], as_index=False)["predicted_pd"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            max="max",
        )
    )

    percentiles = (
        scored.groupby(["model_id", "split"])["predicted_pd"]
        .quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.01: "p01",
                0.05: "p05",
                0.25: "p25",
                0.50: "p50",
                0.75: "p75",
                0.95: "p95",
                0.99: "p99",
            }
        )
    )

    return summary.merge(percentiles, on=["model_id", "split"], how="left")


def build_risk_band_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby(["model_id", "split", "risk_band"], observed=True)
        .agg(
            applicant_count=("row_number", "count"),
            default_count=("y_true", "sum"),
            actual_default_rate=("y_true", "mean"),
            min_pd=("predicted_pd", "min"),
            mean_pd=("predicted_pd", "mean"),
            max_pd=("predicted_pd", "max"),
            min_percentile=("score_percentile", "min"),
            max_percentile=("score_percentile", "max"),
        )
        .reset_index()
    )

    totals = (
        scored.groupby(["model_id", "split"], as_index=False)
        .agg(total_applicants=("row_number", "count"))
    )

    summary = summary.merge(totals, on=["model_id", "split"], how="left")
    summary["applicant_pct"] = summary["applicant_count"] / summary["total_applicants"]

    return summary


def build_score_correlations(scored_wide: pd.DataFrame) -> pd.DataFrame:
    records = []

    model_cols = [
        c for c in scored_wide.columns
        if c not in {"split", "row_number", "y_true"}
    ]

    for split, sub in scored_wide.groupby("split"):
        corr = sub[model_cols].corr()

        for m1 in model_cols:
            for m2 in model_cols:
                records.append(
                    {
                        "split": split,
                        "score_col_1": m1,
                        "score_col_2": m2,
                        "correlation": corr.loc[m1, m2],
                    }
                )

    return pd.DataFrame(records)


def build_band_transition_vs_reference(
    scored: pd.DataFrame,
    reference_model_id: str = "XGBSTACK001",
) -> pd.DataFrame:
    if reference_model_id not in scored["model_id"].unique():
        return pd.DataFrame()

    ref = scored[
        scored["model_id"] == reference_model_id
    ][["split", "row_number", "risk_band"]].rename(
        columns={"risk_band": "reference_risk_band"}
    )

    other = scored[["model_id", "split", "row_number", "risk_band"]].copy()

    merged = other.merge(ref, on=["split", "row_number"], how="left")

    transition = (
        merged.groupby(
            ["split", "model_id", "reference_risk_band", "risk_band"],
            observed=True,
        )
        .size()
        .reset_index(name="applicant_count")
    )

    totals = (
        merged.groupby(["split", "model_id"], observed=True)
        .size()
        .reset_index(name="total_applicants")
    )

    transition = transition.merge(totals, on=["split", "model_id"], how="left")
    transition["applicant_pct"] = (
        transition["applicant_count"] / transition["total_applicants"]
    )

    return transition


def main() -> None:
    print("Loading model predictions...")
    predictions = load_predictions()

    print("Assigning score percentiles and risk bands...")
    scored = assign_score_percentile_and_band(predictions)

    print("Building wide score file...")
    scored_wide = build_model_scores_wide(scored)

    print("Building score summary...")
    score_summary = build_score_summary(scored)

    print("Building risk band summary...")
    risk_band_summary = build_risk_band_summary(scored)

    print("Building score correlations...")
    score_correlations = build_score_correlations(scored_wide)

    print("Building risk band transition table...")
    band_transition = build_band_transition_vs_reference(
        scored,
        reference_model_id="XGBSTACK001",
    )

    scored.to_csv(SCORING_DIR / "model_scores_long.csv", index=False)
    scored_wide.to_csv(SCORING_DIR / "model_scores_wide.csv", index=False)
    score_summary.to_csv(SCORING_DIR / "model_score_summary.csv", index=False)
    risk_band_summary.to_csv(SCORING_DIR / "model_risk_band_summary.csv", index=False)
    score_correlations.to_csv(SCORING_DIR / "model_score_correlations.csv", index=False)
    band_transition.to_csv(SCORING_DIR / "model_risk_band_transitions.csv", index=False)

    print("\nScoring export complete.")
    print(f"Output directory: {SCORING_DIR}")
    print(f"Rows scored: {len(scored):,}")
    print(f"Models scored: {scored['model_id'].nunique():,}")


if __name__ == "__main__":
    main()