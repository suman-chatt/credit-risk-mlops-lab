import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


EPSILON = 1e-6


def assign_traffic_light(
    value: float,
    green_threshold: float,
    amber_threshold: float,
) -> str:
    """
    Assign traffic-light monitoring flag.
    """

    if value < green_threshold:
        return "green"

    if value < amber_threshold:
        return "amber"

    return "red"


def calculate_psi_from_distributions(
    expected_pct: pd.Series,
    actual_pct: pd.Series,
) -> float:
    """
    Calculate PSI from aligned expected/actual percentage distributions.
    """

    expected = expected_pct.replace(0, EPSILON).fillna(EPSILON)
    actual = actual_pct.replace(0, EPSILON).fillna(EPSILON)

    psi_values = (actual - expected) * np.log(actual / expected)

    return float(psi_values.sum())


def calculate_numeric_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """
    Calculate PSI for numeric variables using expected-data quantile bins.
    """

    expected_clean = pd.Series(expected).dropna()
    actual_clean = pd.Series(actual).dropna()

    _, bin_edges = pd.qcut(
        expected_clean,
        q=bins,
        retbins=True,
        duplicates="drop",
    )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_bins = pd.cut(expected_clean, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual_clean, bins=bin_edges, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True).sort_index()
    actual_dist = actual_bins.value_counts(normalize=True).sort_index()

    bucket_table = pd.concat(
        [expected_dist, actual_dist],
        axis=1,
        keys=["expected_pct", "actual_pct"],
    ).fillna(0)

    bucket_table["psi_component"] = (
        (bucket_table["actual_pct"].replace(0, EPSILON)
         - bucket_table["expected_pct"].replace(0, EPSILON))
        * np.log(
            bucket_table["actual_pct"].replace(0, EPSILON)
            / bucket_table["expected_pct"].replace(0, EPSILON)
        )
    )

    psi_value = float(bucket_table["psi_component"].sum())

    bucket_table = bucket_table.reset_index().rename(columns={"index": "bucket"})

    return psi_value, bucket_table


def calculate_categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """
    Calculate PSI / CSI-style drift for categorical variables.
    """

    expected_clean = pd.Series(expected).fillna("missing").astype(str)
    actual_clean = pd.Series(actual).fillna("missing").astype(str)

    expected_dist = expected_clean.value_counts(normalize=True)
    actual_dist = actual_clean.value_counts(normalize=True)

    all_categories = sorted(set(expected_dist.index) | set(actual_dist.index))

    expected_dist = expected_dist.reindex(all_categories, fill_value=0)
    actual_dist = actual_dist.reindex(all_categories, fill_value=0)

    bucket_table = pd.DataFrame(
        {
            "bucket": all_categories,
            "expected_pct": expected_dist.values,
            "actual_pct": actual_dist.values,
        }
    )

    bucket_table["psi_component"] = (
        (bucket_table["actual_pct"].replace(0, EPSILON)
         - bucket_table["expected_pct"].replace(0, EPSILON))
        * np.log(
            bucket_table["actual_pct"].replace(0, EPSILON)
            / bucket_table["expected_pct"].replace(0, EPSILON)
        )
    )

    psi_value = float(bucket_table["psi_component"].sum())

    return psi_value, bucket_table


def calculate_js_distance(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    is_numeric: bool = True,
) -> float:
    """
    Calculate Jensen-Shannon distance between two distributions.
    """

    if is_numeric:
        expected_clean = pd.Series(expected).dropna()
        actual_clean = pd.Series(actual).dropna()

        _, bin_edges = pd.qcut(
            expected_clean,
            q=bins,
            retbins=True,
            duplicates="drop",
        )

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        expected_bins = pd.cut(expected_clean, bins=bin_edges, include_lowest=True)
        actual_bins = pd.cut(actual_clean, bins=bin_edges, include_lowest=True)

        expected_dist = expected_bins.value_counts(normalize=True).sort_index()
        actual_dist = actual_bins.value_counts(normalize=True).sort_index()

    else:
        expected_clean = pd.Series(expected).fillna("missing").astype(str)
        actual_clean = pd.Series(actual).fillna("missing").astype(str)

        expected_dist = expected_clean.value_counts(normalize=True)
        actual_dist = actual_clean.value_counts(normalize=True)

        all_categories = sorted(set(expected_dist.index) | set(actual_dist.index))

        expected_dist = expected_dist.reindex(all_categories, fill_value=0)
        actual_dist = actual_dist.reindex(all_categories, fill_value=0)

    expected_array = expected_dist.replace(0, EPSILON).to_numpy()
    actual_array = actual_dist.replace(0, EPSILON).to_numpy()

    expected_array = expected_array / expected_array.sum()
    actual_array = actual_array / actual_array.sum()

    return float(jensenshannon(expected_array, actual_array))


def calculate_wasserstein(
    expected: pd.Series,
    actual: pd.Series,
) -> float:
    """
    Calculate Wasserstein distance for numeric distributions.
    """

    expected_clean = pd.Series(expected).dropna()
    actual_clean = pd.Series(actual).dropna()

    return float(wasserstein_distance(expected_clean, actual_clean))


def monitor_numeric_feature(
    expected: pd.Series,
    actual: pd.Series,
    feature_name: str,
    bins: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """
    Calculate numeric feature drift metrics.
    """

    psi_value, bucket_table = calculate_numeric_psi(
        expected=expected,
        actual=actual,
        bins=bins,
    )

    js_value = calculate_js_distance(
        expected=expected,
        actual=actual,
        bins=bins,
        is_numeric=True,
    )

    wasserstein_value = calculate_wasserstein(
        expected=expected,
        actual=actual,
    )

    summary = {
        "feature": feature_name,
        "feature_type": "numeric",
        "psi": psi_value,
        "js_distance": js_value,
        "wasserstein_distance": wasserstein_value,
        "psi_flag": assign_traffic_light(psi_value, 0.10, 0.25),
        "js_flag": assign_traffic_light(js_value, 0.05, 0.10),
        "wasserstein_flag": "review",
    }

    bucket_table.insert(0, "feature", feature_name)

    return summary, bucket_table


def monitor_categorical_feature(
    expected: pd.Series,
    actual: pd.Series,
    feature_name: str,
) -> tuple[dict, pd.DataFrame]:
    """
    Calculate categorical feature drift metrics.
    """

    psi_value, bucket_table = calculate_categorical_psi(
        expected=expected,
        actual=actual,
    )

    js_value = calculate_js_distance(
        expected=expected,
        actual=actual,
        is_numeric=False,
    )

    summary = {
        "feature": feature_name,
        "feature_type": "categorical",
        "psi": psi_value,
        "js_distance": js_value,
        "wasserstein_distance": None,
        "psi_flag": assign_traffic_light(psi_value, 0.10, 0.25),
        "js_flag": assign_traffic_light(js_value, 0.05, 0.10),
        "wasserstein_flag": "not_applicable",
    }

    bucket_table.insert(0, "feature", feature_name)

    return summary, bucket_table


def monitor_features(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monitor numeric and categorical feature drift.
    """

    summary_rows = []
    bucket_tables = []

    for feature in numeric_features:
        if feature not in expected_df.columns or feature not in actual_df.columns:
            continue

        summary, bucket_table = monitor_numeric_feature(
            expected=expected_df[feature],
            actual=actual_df[feature],
            feature_name=feature,
            bins=bins,
        )

        summary_rows.append(summary)
        bucket_tables.append(bucket_table)

    for feature in categorical_features:
        if feature not in expected_df.columns or feature not in actual_df.columns:
            continue

        summary, bucket_table = monitor_categorical_feature(
            expected=expected_df[feature],
            actual=actual_df[feature],
            feature_name=feature,
        )

        summary_rows.append(summary)
        bucket_tables.append(bucket_table)

    summary_df = pd.DataFrame(summary_rows)

    if bucket_tables:
        bucket_df = pd.concat(bucket_tables, ignore_index=True)
    else:
        bucket_df = pd.DataFrame()

    return summary_df, bucket_df


def monitor_score_drift(
    expected_scores: pd.Series,
    actual_scores: pd.Series,
    score_name: str = "production_pd",
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monitor score distribution drift.
    """

    summary, bucket_table = monitor_numeric_feature(
        expected=expected_scores,
        actual=actual_scores,
        feature_name=score_name,
        bins=bins,
    )

    summary_df = pd.DataFrame([summary])

    return summary_df, bucket_table