from pathlib import Path
import pandas as pd

SPLIT = "validation"
N_BUCKETS = 10

lift_path = Path("outputs_src/diagnostics/lift_tables.csv")
wide_path = Path("outputs_src/scoring/model_scores_wide.csv")
ensemble_registry_path = Path("outputs_src/ensemble_registry.xlsx")

lift_df = pd.read_csv(lift_path)
scores = pd.read_csv(wide_path)
ensemble_registry = pd.read_excel(ensemble_registry_path)

scores["split"] = scores["split"].astype(str).str.lower().str.strip()
scores = scores[scores["split"] == SPLIT].copy()

if "y_true" not in scores.columns:
    raise ValueError("y_true column not found in model_scores_wide.csv")

ensemble_model_ids = (
    ensemble_registry["model_id"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .unique()
    .tolist()
)

available_model_ids = set(scores.columns)

missing_models = [m for m in ensemble_model_ids if m not in available_model_ids]
usable_models = [m for m in ensemble_model_ids if m in available_model_ids]

if not usable_models:
    raise ValueError("No ensemble model columns found in model_scores_wide.csv")

all_new_rows = []

for model_id in usable_models:
    model_scores = scores[["split", "y_true", model_id]].dropna().copy()
    model_scores = model_scores.rename(columns={model_id: "predicted_pd"})
    model_scores = model_scores.sort_values("predicted_pd", ascending=False).reset_index(drop=True)

    model_scores["bucket_num"] = (
        pd.qcut(
            model_scores.index,
            q=N_BUCKETS,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    total_bad = model_scores["y_true"].sum()
    total_records = len(model_scores)
    overall_bad_rate = total_bad / total_records if total_records else 0

    rows = []

    for bucket_num, g in model_scores.groupby("bucket_num", sort=True):
        total = len(g)
        bad = int(g["y_true"].sum())
        good = total - bad

        rows.append(
            {
                "table_name": "lift_table",
                "model_id": model_id,
                "split": SPLIT,
                "bucket": bucket_num,
                "total": total,
                "bad": bad,
                "good": good,
                "bad_rate": bad / total if total else 0,
            }
        )

    new_df = pd.DataFrame(rows)

    new_df["cum_bad"] = new_df["bad"].cumsum()
    new_df["cum_total"] = new_df["total"].cumsum()
    new_df["cum_bad_pct"] = new_df["cum_bad"] / total_bad if total_bad else 0
    new_df["cum_total_pct"] = new_df["cum_total"] / total_records if total_records else 0
    new_df["lift"] = new_df["bad_rate"] / overall_bad_rate if overall_bad_rate else 0

    all_new_rows.append(new_df)

ensemble_lift_df = pd.concat(all_new_rows, ignore_index=True)

lift_df["model_id"] = lift_df["model_id"].astype(str).str.upper().str.strip()
lift_df = lift_df[~lift_df["model_id"].isin(usable_models)].copy()

final_df = pd.concat([lift_df, ensemble_lift_df], ignore_index=True)
final_df.to_csv(lift_path, index=False)

print(f"Added lift/gains rows for {len(usable_models)} ensemble models.")
print(f"Updated file: {lift_path}")

if missing_models:
    print("\nSkipped ensemble models not found in model_scores_wide.csv:")
    print(missing_models)

print("\nAdded models:")
print(usable_models)