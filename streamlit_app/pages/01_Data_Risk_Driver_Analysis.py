from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(page_title="Risk Driver Analysis", layout="wide")

st.title("Risk Driver Analysis")
st.caption("Dynamic EDA for borrower risk drivers, default behavior, missingness, outliers, and transformations.")



# Paths / constants


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"

TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

VARIABLE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "Total balance on unsecured revolving credit lines divided by total credit limits.",
    "age": "Borrower age in years.",
    "NumberOfTime30-59DaysPastDueNotWorse": "Number of times borrower was 30–59 days past due, but not worse, in the past two years.",
    "DebtRatio": "Monthly debt payments, alimony, and living costs divided by monthly gross income.",
    "MonthlyIncome": "Borrower monthly income.",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit lines and loans.",
    "NumberOfTimes90DaysLate": "Number of times borrower was 90+ days past due.",
    "NumberRealEstateLoansOrLines": "Number of mortgage and real estate loans, including home equity lines.",
    "NumberOfTime60-89DaysPastDueNotWorse": "Number of times borrower was 60–89 days past due, but not worse, in the past two years.",
    "NumberOfDependents": "Number of dependents in the household, excluding the borrower.",
}



# Loaders


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    return df



# Helpers


def build_table(data: pd.DataFrame, height: int = 350):
    if data.empty:
        st.info("No data available.")
        return

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)

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


def get_numeric_variables(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.select_dtypes(include="number").columns
        if c != TARGET_COL
    ]


def variable_description(var: str) -> str:
    return VARIABLE_DESCRIPTIONS.get(var, "No variable description available.")


def create_business_bins(df: pd.DataFrame, var: str) -> pd.Series:
    s = df[var]

    delinquency_vars = {
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    }

    if var in delinquency_vars:
        def delinquency_bin(x):
            if pd.isna(x):
                return "Missing"
            if x >= 90:
                return "Special code"
            if x == 0:
                return "0"
            if x == 1:
                return "1"
            if x == 2:
                return "2"
            return "3+"

        return s.apply(delinquency_bin)

    if var == "RevolvingUtilizationOfUnsecuredLines":
        bins = [-np.inf, 0, 0.10, 0.30, 0.60, 0.90, 1.00, np.inf]
        labels = ["0", "0-10%", "10-30%", "30-60%", "60-90%", "90-100%", "100%+"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", "Missing")

    if var == "DebtRatio":
        cap = s.quantile(0.99)
        bins = [-np.inf, 0.10, 0.30, 0.50, 1.00, cap, np.inf]
        labels = ["0-10%", "10-30%", "30-50%", "50-100%", "100%-p99", "Above p99"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", "Missing")

    if var == "MonthlyIncome":
        bins = [-np.inf, 0, 2500, 5000, 7500, 10000, 15000, np.inf]
        labels = ["0", "0-2.5k", "2.5k-5k", "5k-7.5k", "7.5k-10k", "10k-15k", "15k+"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", "Missing")

    if var == "age":
        bins = [-np.inf, 25, 35, 45, 55, 65, 75, np.inf]
        labels = ["<25", "25-35", "35-45", "45-55", "55-65", "65-75", "75+"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str).replace("nan", "Missing")

    if var == "NumberOfDependents":
        def dependent_bin(x):
            if pd.isna(x):
                return "Missing"
            if x == 0:
                return "0"
            if x == 1:
                return "1"
            if x == 2:
                return "2"
            return "3+"

        return s.apply(dependent_bin)

    if var in ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]:
        def count_bin(x):
            if pd.isna(x):
                return "Missing"
            if x == 0:
                return "0"
            if x == 1:
                return "1"
            if x == 2:
                return "2"
            if x <= 5:
                return "3-5"
            if x <= 10:
                return "6-10"
            return "10+"

        return s.apply(count_bin)

    try:
        return pd.qcut(s, q=10, duplicates="drop").astype(str).replace("nan", "Missing")
    except ValueError:
        return s.astype(str).replace("nan", "Missing")


def build_volume_bad_rate_table(df: pd.DataFrame, var: str) -> pd.DataFrame:
    work = df[[TARGET_COL, var]].copy()
    work["bin"] = create_business_bins(work, var).astype(str)

    out = (
        work.groupby("bin", observed=True)
        .agg(
            observation_count=(TARGET_COL, "size"),
            bad_count=(TARGET_COL, "sum"),
            bad_rate=(TARGET_COL, "mean"),
            average_value=(var, "mean"),
        )
        .reset_index()
    )

    out["population_share"] = out["observation_count"] / out["observation_count"].sum()

    bin_order = get_bin_order(var)

    if bin_order:
        out["bin"] = pd.Categorical(out["bin"], categories=bin_order, ordered=True)
        out = out.sort_values("bin").reset_index(drop=True)
        out["bin"] = out["bin"].astype(str)
    else:
        out = out.sort_values("average_value").reset_index(drop=True)

    return out


def plot_volume_bad_rate(table: pd.DataFrame, var: str):
    table = table.copy()
    table["bin"] = table["bin"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=table["bin"],
            y=table["observation_count"],
            name="Observation Count",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=table["bin"],
            y=table["bad_rate"],
            name="Bad Rate",
            mode="lines+markers",
            connectgaps=True,
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"Volume and Bad Rate by {var}",
        xaxis=dict(
            title=f"{var} band",
            type="category",
            categoryorder="array",
            categoryarray=table["bin"].tolist(),
        ),
        yaxis=dict(title="Observation Count"),
        yaxis2=dict(
            title="Bad Rate",
            overlaying="y",
            side="right",
            tickformat=".0%",
        ),
        height=560,
        legend=dict(orientation="h"),
    )

    return fig


def cap_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)

def get_bin_order(var: str) -> list[str]:
    if var == "age":
        return ["<25", "25-35", "35-45", "45-55", "55-65", "65-75", "75+"]

    if var == "RevolvingUtilizationOfUnsecuredLines":
        return ["0", "0-10%", "10-30%", "30-60%", "60-90%", "90-100%", "100%+"]

    if var == "DebtRatio":
        return ["0-10%", "10-30%", "30-50%", "50-100%", "100%-p99", "Above p99"]

    if var == "MonthlyIncome":
        return ["0", "0-2.5k", "2.5k-5k", "5k-7.5k", "7.5k-10k", "10k-15k", "15k+", "Missing"]

    if var in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]:
        return ["0", "1", "2", "3+", "Special code", "Missing"]

    if var == "NumberOfDependents":
        return ["0", "1", "2", "3+", "Missing"]

    if var in ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]:
        return ["0", "1", "2", "3-5", "6-10", "10+", "Missing"]

    return []

# Load data


df = load_raw_data()

if df.empty:
    st.error(
        f"""
Raw training data not found.

Expected file:

`{DATA_PATH}`
"""
    )
    st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Target column `{TARGET_COL}` not found.")
    st.stop()

numeric_vars = get_numeric_variables(df)



# Navigation


section = st.radio(
    "Risk Driver Sections",
    [
        "Dataset Overview",
        "Target Distribution",
        "Variable Distribution",
        "Missingness vs Bad Rate",
        "Volume and Bad Rate by Band",
        "Outlier and Tail Review",
        "Transformation Review",
        "Preprocessing Summary",
    ],
    horizontal=True,
    label_visibility="collapsed",
)



# Dataset Overview


if section == "Dataset Overview":
    st.subheader("Dataset Overview")

    st.markdown(
        """
This section summarizes the raw training dataset before modeling.

It checks basic data quality, missingness, target rate, and variable coverage.  
This is the starting point for deciding which variables need imputation, binning, capping, or transformation.
"""
    )

    overview = pd.DataFrame(
        {
            "metric": [
                "Rows",
                "Columns",
                "Duplicate rows",
                "Target bad count",
                "Target good count",
                "Target bad rate",
            ],
            "value": [
                len(df),
                df.shape[1],
                df.duplicated().sum(),
                int(df[TARGET_COL].sum()),
                int((df[TARGET_COL] == 0).sum()),
                df[TARGET_COL].mean(),
            ],
        }
    )

    build_table(safe_round_cols(overview), height=220)

    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "variable", 0: "missing_count"})
    )
    missing["missing_pct"] = missing["missing_count"] / len(df)
    missing = missing.sort_values("missing_count", ascending=False)

    st.markdown("### Missing Value Summary")
    build_table(safe_round_cols(missing), height=320)



# Target Distribution


elif section == "Target Distribution":
    st.subheader("Target Distribution")

    st.markdown(
        """
This chart shows the target class balance.

- x-axis: target class  
- y-axis: borrower count  

`1` means the borrower experienced serious delinquency.  
A low event rate confirms this is an imbalanced credit-risk classification problem.
"""
    )

    target_summary = (
        df[TARGET_COL]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="count")
        .sort_values("target")
    )
    target_summary["share"] = target_summary["count"] / target_summary["count"].sum()

    fig = px.bar(
        target_summary,
        x="target",
        y="count",
        text="count",
        title="Target Class Distribution",
    )
    fig.update_xaxes(title="Serious delinquency flag")
    fig.update_yaxes(title="Borrower count")
    fig.update_layout(height=460)

    st.plotly_chart(fig, use_container_width=True)
    build_table(safe_round_cols(target_summary), height=180)



# Variable Distribution


elif section == "Variable Distribution":
    st.subheader("Variable Distribution")

    selected_var = st.selectbox("Select variable", numeric_vars)

    st.markdown(
        f"""
**Variable:** `{selected_var}`  

**Description:** {variable_description(selected_var)}

This section reviews the raw distribution of the selected variable.

- Histogram shows frequency across values  
- Box plot highlights spread and extreme values  
- Summary statistics show central tendency, tails, and missingness  

This helps identify skew, outliers, and variables that may need capping or transformation.
"""
    )

    s = df[selected_var]

    fig = px.histogram(
        df,
        x=selected_var,
        nbins=60,
        title=f"Distribution of {selected_var}",
    )
    fig.update_xaxes(title=selected_var)
    fig.update_yaxes(title="Count")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        x=selected_var,
        title=f"Box Plot of {selected_var}",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(
        {
            "metric": [
                "count",
                "missing_count",
                "missing_pct",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99",
                "max",
            ],
            "value": [
                s.count(),
                s.isna().sum(),
                s.isna().mean(),
                s.mean(),
                s.std(),
                s.min(),
                s.quantile(0.01),
                s.quantile(0.05),
                s.quantile(0.25),
                s.quantile(0.50),
                s.quantile(0.75),
                s.quantile(0.95),
                s.quantile(0.99),
                s.max(),
            ],
        }
    )

    build_table(safe_round_cols(summary), height=360)



# Missingness vs Bad Rate


elif section == "Missingness vs Bad Rate":
    st.subheader("Missingness vs Bad Rate")

    selected_var = st.selectbox("Select variable", numeric_vars)

    st.markdown(
        f"""
**Variable:** `{selected_var}`  

**Description:** {variable_description(selected_var)}

This section checks whether missingness itself carries risk information.

- x-axis: missing vs non-missing group  
- left y-axis: observation count  
- right y-axis: actual bad rate  

If the missing group has a materially different bad rate, the missing flag may be useful as a model feature.
"""
    )

    work = df[[TARGET_COL, selected_var]].copy()
    work["missing_group"] = np.where(work[selected_var].isna(), "Missing", "Not Missing")

    miss_table = (
        work.groupby("missing_group")
        .agg(
            observation_count=(TARGET_COL, "size"),
            bad_count=(TARGET_COL, "sum"),
            bad_rate=(TARGET_COL, "mean"),
        )
        .reset_index()
    )
    miss_table["population_share"] = miss_table["observation_count"] / len(work)

    if work[selected_var].isna().sum() == 0:
        st.info("This variable has no missing values in the raw training data.")
    else:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=miss_table["missing_group"],
                y=miss_table["observation_count"],
                name="Observation Count",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=miss_table["missing_group"],
                y=miss_table["bad_rate"],
                name="Bad Rate",
                mode="lines+markers",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title=f"Missingness and Bad Rate for {selected_var}",
            xaxis_title="Missingness group",
            yaxis=dict(title="Observation Count"),
            yaxis2=dict(title="Bad Rate", overlaying="y", side="right", tickformat=".0%"),
            height=480,
        )

        st.plotly_chart(fig, use_container_width=True)

    build_table(safe_round_cols(miss_table), height=220)



# Volume and Bad Rate by Band


elif section == "Volume and Bad Rate by Band":
    st.subheader("Volume and Bad Rate by Band")

    selected_var = st.selectbox("Select variable", numeric_vars)

    st.markdown(
        f"""
**Variable:** `{selected_var}`  

**Description:** {variable_description(selected_var)}

This is the main risk-driver chart.

- x-axis: business-aware bins for the selected variable  
- left y-axis: borrower count  
- right y-axis: observed bad rate  

A strong risk driver should show meaningful bad-rate separation across bands.  
For delinquency variables, risk should generally increase as past-due count increases.
"""
    )

    st.info(
    """
The selected variable is grouped into business-aware bins before charting.  
The bars show borrower volume in each bin. The line shows observed bad rate.
"""
)

    band_table = build_volume_bad_rate_table(df, selected_var)

    fig = plot_volume_bad_rate(band_table, selected_var)
    st.plotly_chart(fig, use_container_width=True)

    build_table(safe_round_cols(band_table), height=360)

    st.markdown(
        """
**Generic interpretation:**

- High-volume low-risk bands often represent the stable borrower base  
- Low-volume high-risk bands can still be very important for underwriting  
- Monotonic bad-rate patterns support binning and scorecard logic  
- Nonlinear patterns support tree-based models, transformations, or custom bins  
"""
    )



# Outlier and Tail Review


elif section == "Outlier and Tail Review":
    st.subheader("Outlier and Tail Review")

    selected_var = st.selectbox("Select variable", numeric_vars)

    st.markdown(
        f"""
**Variable:** `{selected_var}`  

**Description:** {variable_description(selected_var)}

This section reviews whether the selected variable has extreme tails.

- Raw distribution shows original values  
- Capped distribution clips values at the 1st and 99th percentile  
- The summary table shows how much the tail affects the variable  

This helps justify capping for variables such as utilization, debt ratio, and income.
"""
    )

    work = df[[selected_var]].copy()
    work[f"{selected_var}_capped"] = cap_series(work[selected_var])

    raw_cap = pd.DataFrame(
        {
            "metric": ["p01", "p50", "p99", "max"],
            "raw_value": [
                work[selected_var].quantile(0.01),
                work[selected_var].quantile(0.50),
                work[selected_var].quantile(0.99),
                work[selected_var].max(),
            ],
            "capped_value": [
                work[f"{selected_var}_capped"].quantile(0.01),
                work[f"{selected_var}_capped"].quantile(0.50),
                work[f"{selected_var}_capped"].quantile(0.99),
                work[f"{selected_var}_capped"].max(),
            ],
        }
    )

    fig = px.histogram(
        work,
        x=selected_var,
        nbins=80,
        title=f"Raw Distribution: {selected_var}",
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        work,
        x=f"{selected_var}_capped",
        nbins=80,
        title=f"Capped Distribution: {selected_var} clipped at p01/p99",
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    build_table(safe_round_cols(raw_cap), height=220)



# Transformation Review


elif section == "Transformation Review":
    st.subheader("Transformation Review")

    selected_var = st.selectbox("Select variable", numeric_vars)

    st.markdown(
        f"""
**Variable:** `{selected_var}`  

**Description:** {variable_description(selected_var)}

This section compares raw values to a log-transformed version.

- Raw histogram shows the original scale  
- Log histogram compresses large values and makes skew easier to model  

This is useful for heavily skewed variables such as income, debt ratio, and utilization.
"""
    )

    work = df[[selected_var]].copy()
    work[selected_var] = work[selected_var].fillna(work[selected_var].median())

    min_value = work[selected_var].min()
    if min_value < 0:
        work["log_transformed"] = np.log1p(work[selected_var] - min_value)
    else:
        work["log_transformed"] = np.log1p(work[selected_var])

    fig = px.histogram(
        work,
        x=selected_var,
        nbins=80,
        title=f"Raw Distribution: {selected_var}",
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        work,
        x="log_transformed",
        nbins=80,
        title=f"Log1p Transformed Distribution: {selected_var}",
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    transform_summary = pd.DataFrame(
        {
            "metric": ["raw_skew", "log_skew", "raw_p99", "log_p99"],
            "value": [
                work[selected_var].skew(),
                work["log_transformed"].skew(),
                work[selected_var].quantile(0.99),
                work["log_transformed"].quantile(0.99),
            ],
        }
    )

    build_table(safe_round_cols(transform_summary), height=220)



# Preprocessing Summary


elif section == "Preprocessing Summary":
    st.subheader("Preprocessing Summary")

    st.markdown(
        """
This section summarizes the preprocessing logic implied by the EDA.

The goal is not to change the raw data inside this page.  
The goal is to explain why the modeling pipeline later creates capped variables, missing flags, binned variables, and transformed variables.
"""
    )

    summary = pd.DataFrame(
        [
            {
                "area": "Missing values",
                "treatment": "Median imputation plus missingness flags for variables with meaningful missingness.",
                "business_reason": "Missingness may carry risk information and should not be silently discarded.",
            },
            {
                "area": "Outliers",
                "treatment": "Cap extreme values for heavily skewed variables.",
                "business_reason": "Extreme tails can dominate linear models and distort calibration.",
            },
            {
                "area": "Log transforms",
                "treatment": "Use log1p-style transforms for skewed continuous variables.",
                "business_reason": "Compresses extreme values while preserving ordering.",
            },
            {
                "area": "Business bins",
                "treatment": "Use custom bins for delinquency, utilization, debt ratio, income, age, and dependents.",
                "business_reason": "Supports WOE/logistic modeling, scorecards, and interpretable risk segmentation.",
            },
            {
                "area": "Target imbalance",
                "treatment": "Use ranking, calibration, lift, and threshold diagnostics rather than accuracy alone.",
                "business_reason": "Default events are rare, so accuracy can be misleading.",
            },
        ]
    )

    build_table(summary, height=320)

    st.markdown(
        """
**Bottom line:**  
The EDA supports the modeling design. Delinquency variables show strong risk separation, utilization and debt variables show nonlinear tail behavior, and missingness requires controlled treatment.
"""
    )