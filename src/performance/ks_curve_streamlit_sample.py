import pandas as pd
from pathlib import Path

INPUT_PATH = Path("outputs_src/diagnostics/ks_curves.csv")
OUTPUT_PATH = Path("outputs_src/diagnostics/ks_curves_sample.csv")

df = pd.read_csv(INPUT_PATH)

# Keep only validation (you only use validation anyway)
df = df[df["split"].str.lower() == "validation"].copy()

# Downsample per model (keep ~50–100 points per model)
def sample_per_model(group, n=75):
    if len(group) <= n:
        return group
    return group.sort_values("population_pct").iloc[
        :: max(1, len(group) // n)
    ]

df_sample = (
    df.groupby("model_id", group_keys=False)
    .apply(sample_per_model)
    .reset_index(drop=True)
)

df_sample.to_csv(OUTPUT_PATH, index=False)

print("Sampled KS file saved:", OUTPUT_PATH)
print("Original size:", len(df))
print("Sampled size:", len(df_sample))