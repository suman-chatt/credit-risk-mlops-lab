import pandas as pd


def create_monitoring_recommendations(
    feature_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    combined = pd.concat([score_summary, feature_summary], ignore_index=True)

    for _, row in combined.iterrows():
        flags = [row.get("psi_flag"), row.get("js_flag")]

        if "red" in flags:
            action = "Investigate immediately"
        elif "amber" in flags:
            action = "Review in next monitoring cycle"
        else:
            action = "No action required"

        rows.append(
            {
                "feature": row.get("feature"),
                "feature_type": row.get("feature_type"),
                "psi": row.get("psi"),
                "js_distance": row.get("js_distance"),
                "psi_flag": row.get("psi_flag"),
                "js_flag": row.get("js_flag"),
                "recommended_action": action,
            }
        )

    return pd.DataFrame(rows)