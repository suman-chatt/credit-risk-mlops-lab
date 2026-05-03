import json
import numpy as np
import pandas as pd

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.features.binning import FINAL_BIN_COLS
from src.models.diagnostics import (
    calculate_gains_lift,
    generate_calibration_table,
    generate_ks_curve_data,
    generate_roc_curve_data,
)
from src.models.evaluate import evaluate_binary_classifier
from src.models.model_registry import (
    add_coefficient_table,
    add_diagnostics,
    add_model_record as registry_add_model_record,
    add_vif_table,
    add_feature_importance_table,
)

from src.models.vif import calculate_vif, summarize_vif

from src.models.mlflow_utils import log_model_to_mlflow

def add_model_record(
    registry: dict,
    model_id: str,
    model_family: str,
    model_name: str,
    dataset_type: str,
    features: list,
    params: dict,
    train_metrics: dict,
    validation_metrics: dict,
    model_object,
    notes: str = "",
    selected_flag: bool = False,
) -> dict:
    """
    Add model to internal registry and log the same model to MLflow.

    This wrapper means all existing train.py functions automatically log
    to MLflow without editing every model training block.
    """

    registry = registry_add_model_record(
        registry=registry,
        model_id=model_id,
        model_family=model_family,
        model_name=model_name,
        dataset_type=dataset_type,
        features=features,
        params=params,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        model_object=model_object,
        notes=notes,
        selected_flag=selected_flag,
    )

    try:
        log_model_to_mlflow(
            model_id=model_id,
            model=model_object,
            params=params,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            model_family=model_family,
            model_name=model_name,
            dataset_type=dataset_type,
            experiment_name="credit-risk-base-models",
        )
    except Exception as exc:
        print(f"[MLflow warning] Failed to log {model_id}: {exc}")

    return registry

def get_iv_ordered_bin_columns(iv_path) -> list[str]:
    """
    Load IV values and return bin columns ordered from highest IV to lowest IV.
    """

    with open(iv_path, "r") as f:
        iv_values = json.load(f)

    ordered_cols = sorted(
        FINAL_BIN_COLS,
        key=lambda col: iv_values.get(col, 0),
        reverse=True,
    )

    return ordered_cols


def fit_statsmodels_logit(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit statsmodels logistic regression with intercept.
    """

    X_train_const = sm.add_constant(X_train, has_constant="add")

    model = sm.Logit(y_train, X_train_const)
    result = model.fit(disp=False)

    return result


def predict_statsmodels_logit(result, X: pd.DataFrame) -> pd.Series:
    """
    Generate predicted probabilities from statsmodels logistic regression.
    """

    X_const = sm.add_constant(X, has_constant="add")
    return result.predict(X_const)


def build_coefficient_table(result) -> pd.DataFrame:
    """
    Build coefficient diagnostics table for statsmodels logistic regression.
    """

    coef_table = pd.DataFrame(
        {
            "feature": result.params.index,
            "coefficient": result.params.values,
            "std_error": result.bse.values,
            "z_stat": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )

    coef_table["odds_ratio"] = np.exp(coef_table["coefficient"])

    return coef_table


def prepare_binned_design_matrix(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    selected_bin_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode selected binned categorical variables for binned logistic.

    Uses drop_first=True to avoid exact dummy-variable multicollinearity.
    """

    X_train_selected = X_train_raw[selected_bin_cols].astype(str)
    X_val_selected = X_val_raw[selected_bin_cols].astype(str)

    X_train_ohe = pd.get_dummies(
        X_train_selected,
        drop_first=True,
        dtype=int,
    )

    X_val_ohe = pd.get_dummies(
        X_val_selected,
        drop_first=True,
        dtype=int,
    )

    X_val_ohe = X_val_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

    return X_train_ohe, X_val_ohe


def add_standard_model_outputs(
    registry: dict,
    model_id: str,
    split_name: str,
    y_true,
    y_score,
) -> dict:
    """
    Add standard diagnostics tables for one model/split.
    """

    lift_table = calculate_gains_lift(y_true, y_score)
    roc_curve = generate_roc_curve_data(y_true, y_score)
    ks_curve = generate_ks_curve_data(y_true, y_score)
    calibration_table = generate_calibration_table(y_true, y_score)

    registry = add_diagnostics(
        registry=registry,
        model_id=model_id,
        split=split_name,
        lift_table=lift_table,
        roc_curve=roc_curve,
        ks_curve=ks_curve,
        calibration_table=calibration_table,
    )

    return registry

def get_permutation_ranked_features(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[str]:
    """
    Rank features using validation-set permutation importance.

    Used for pruned Naive Bayes experiments.
    """

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    )

    return importance_df["feature"].tolist()


def build_feature_importance_table(
    model,
    features: list[str],
) -> pd.DataFrame:
    """
    Build feature importance table when model exposes feature_importances_.
    """

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_staged_woe_logistic(
    registry: dict,
    datasets: dict,
    iv_path,
    auc_stop_threshold: float = 0.005,
    vif_threshold: float = 3.0,
    model_prefix: str = "WLOG",
) -> dict:
    """
    Train staged WOE logistic models.

    Logic:
    - order variables by IV descending
    - add one variable at a time
    - evaluate validation AUC increment
    - calculate VIF
    - keep every model in registry
    - stop once validation AUC improvement is below threshold
    """

    ordered_bin_cols = get_iv_ordered_bin_columns(iv_path)
    ordered_woe_features = [f"{col}_woe" for col in ordered_bin_cols]

    split = datasets["woe"]

    previous_val_auc = None
    selected_features = []

    for idx, feature in enumerate(ordered_woe_features, start=1):
        selected_features.append(feature)

        model_id = f"{model_prefix}{idx:03d}"

        X_train = split["X_train"][selected_features]
        X_val = split["X_val"][selected_features]

        y_train = split["y_train"]
        y_val = split["y_val"]

        result = fit_statsmodels_logit(X_train, y_train)

        train_score = predict_statsmodels_logit(result, X_train)
        val_score = predict_statsmodels_logit(result, X_val)

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        val_auc = val_metrics["auc"]

        if previous_val_auc is None:
            auc_increment = None
            stop_reason = ""
        else:
            auc_increment = val_auc - previous_val_auc
            stop_reason = (
                "validation_auc_increment_below_threshold"
                if auc_increment < auc_stop_threshold
                else ""
            )

        vif_table = calculate_vif(X_train, vif_threshold=vif_threshold)
        vif_summary = summarize_vif(vif_table, vif_threshold=vif_threshold)

        val_metrics["auc_increment"] = auc_increment
        val_metrics.update(vif_summary)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="logistic",
            model_name="WOE staged logistic",
            dataset_type="woe",
            features=selected_features.copy(),
            params={
                "auc_stop_threshold": auc_stop_threshold,
                "vif_threshold": vif_threshold,
            },
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=result,
            notes=stop_reason,
            selected_flag=False,
        )

        registry = add_vif_table(registry, model_id, vif_table)
        registry = add_coefficient_table(registry, model_id, build_coefficient_table(result))

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="train",
            y_true=y_train,
            y_score=train_score,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="validation",
            y_true=y_val,
            y_score=val_score,
        )

        if previous_val_auc is not None and auc_increment < auc_stop_threshold:
            break

        previous_val_auc = val_auc

    return registry


def train_staged_binned_logistic(
    registry: dict,
    datasets: dict,
    iv_path,
    auc_stop_threshold: float = 0.005,
    vif_threshold: float = 3.0,
    model_prefix: str = "BLOG",
) -> dict:
    """
    Train staged binned logistic models.

    Binned variables are added in IV order.
    Selected categorical bins are one-hot encoded with drop_first=True.
    """

    ordered_bin_cols = get_iv_ordered_bin_columns(iv_path)

    split = datasets["binned_logit"]

    previous_val_auc = None
    selected_bin_cols = []

    for idx, bin_col in enumerate(ordered_bin_cols, start=1):
        selected_bin_cols.append(bin_col)

        model_id = f"{model_prefix}{idx:03d}"

        X_train, X_val = prepare_binned_design_matrix(
            X_train_raw=split["X_train"],
            X_val_raw=split["X_val"],
            selected_bin_cols=selected_bin_cols,
        )

        y_train = split["y_train"]
        y_val = split["y_val"]

        result = fit_statsmodels_logit(X_train, y_train)

        train_score = predict_statsmodels_logit(result, X_train)
        val_score = predict_statsmodels_logit(result, X_val)

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        val_auc = val_metrics["auc"]

        if previous_val_auc is None:
            auc_increment = None
            stop_reason = ""
        else:
            auc_increment = val_auc - previous_val_auc
            stop_reason = (
                "validation_auc_increment_below_threshold"
                if auc_increment < auc_stop_threshold
                else ""
            )

        vif_table = calculate_vif(X_train, vif_threshold=vif_threshold)
        vif_summary = summarize_vif(vif_table, vif_threshold=vif_threshold)

        val_metrics["auc_increment"] = auc_increment
        val_metrics.update(vif_summary)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="logistic",
            model_name="Binned staged logistic",
            dataset_type="binned_logit",
            features=selected_bin_cols.copy(),
            params={
                "auc_stop_threshold": auc_stop_threshold,
                "vif_threshold": vif_threshold,
                "encoding": "one_hot_drop_first",
            },
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=result,
            notes=stop_reason,
            selected_flag=False,
        )

        registry = add_vif_table(registry, model_id, vif_table)
        registry = add_coefficient_table(registry, model_id, build_coefficient_table(result))

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="train",
            y_true=y_train,
            y_score=train_score,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="validation",
            y_true=y_val,
            y_score=val_score,
        )

        if previous_val_auc is not None and auc_increment < auc_stop_threshold:
            break

        previous_val_auc = val_auc

    return registry

def train_regularized_logistic(
    registry: dict,
    datasets: dict,
    model_prefix: str = "RLOG",
) -> dict:
    """
    Train regularized logistic regression models on scaled features.

    This is the ML-style logistic branch:
    - uses scaled numeric features
    - tests L1 and L2 regularization
    - keeps all candidates in the registry
    """

    split = datasets["scaled"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    features = split["features"]

    model_configs = [
        {"penalty": "l1", "C": 0.01, "solver": "liblinear"},
        {"penalty": "l1", "C": 0.1, "solver": "liblinear"},
        {"penalty": "l1", "C": 1.0, "solver": "liblinear"},
        {"penalty": "l1", "C": 10.0, "solver": "liblinear"},
        {"penalty": "l2", "C": 0.01, "solver": "liblinear"},
        {"penalty": "l2", "C": 0.1, "solver": "liblinear"},
        {"penalty": "l2", "C": 1.0, "solver": "liblinear"},
        {"penalty": "l2", "C": 10.0, "solver": "liblinear"},
    ]

    for idx, params in enumerate(model_configs, start=1):
        model_id = f"{model_prefix}{idx:03d}"

        model = LogisticRegression(
            penalty=params["penalty"],
            C=params["C"],
            solver=params["solver"],
            max_iter=1000,
            random_state=42,
        )

        model.fit(X_train, y_train)

        train_score = model.predict_proba(X_train)[:, 1]
        val_score = model.predict_proba(X_val)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        coef_table = pd.DataFrame(
            {
                "feature": features,
                "coefficient": model.coef_[0],
            }
        )

        coef_table["odds_ratio"] = np.exp(coef_table["coefficient"])
        coef_table["intercept"] = model.intercept_[0]
        coef_table["penalty"] = params["penalty"]
        coef_table["C"] = params["C"]
        coef_table["solver"] = params["solver"]

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="logistic",
            model_name="Regularized logistic",
            dataset_type="scaled",
            features=features,
            params=params,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="regularized_logistic_scaled_features",
            selected_flag=False,
        )

        registry = add_coefficient_table(
            registry=registry,
            model_id=model_id,
            coefficient_table=coef_table,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="train",
            y_true=y_train,
            y_score=train_score,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id=model_id,
            split_name="validation",
            y_true=y_val,
            y_score=val_score,
        )

    return registry

def train_gaussian_nb_models(
    registry: dict,
    datasets: dict,
    model_prefix: str = "GNB",
) -> dict:
    """
    Train GaussianNB models:
    - baseline using all scaled features
    - PCA variants using selected explained-variance thresholds
    - pruned feature variants using validation permutation-importance ranking
    """

    split = datasets["scaled"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    features = split["features"]

    model_counter = 1

    # 1. Baseline: all scaled features
    model_id = f"{model_prefix}{model_counter:03d}"

    baseline_model = GaussianNB()
    baseline_model.fit(X_train, y_train)

    train_score = baseline_model.predict_proba(X_train)[:, 1]
    val_score = baseline_model.predict_proba(X_val)[:, 1]

    train_metrics = evaluate_binary_classifier(y_train, train_score)
    val_metrics = evaluate_binary_classifier(y_val, val_score)

    registry = add_model_record(
        registry=registry,
        model_id=model_id,
        model_family="naive_bayes",
        model_name="GaussianNB baseline",
        dataset_type="scaled",
        features=features,
        params=baseline_model.get_params(),
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        model_object=baseline_model,
        notes="all_scaled_features",
    )

    registry = add_standard_model_outputs(
        registry, model_id, "train", y_train, train_score
    )
    registry = add_standard_model_outputs(
        registry, model_id, "validation", y_val, val_score
    )

    model_counter += 1

    # 2. PCA grid
    pca_variances = [0.70, 0.80, 0.90, 0.95, 0.99]

    for variance_threshold in pca_variances:
        model_id = f"{model_prefix}{model_counter:03d}"

        pca = PCA(
            n_components=variance_threshold,
            random_state=42,
        )

        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        model = GaussianNB()
        model.fit(X_train_pca, y_train)

        train_score = model.predict_proba(X_train_pca)[:, 1]
        val_score = model.predict_proba(X_val_pca)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        pca_features = [
            f"PC{i + 1}" for i in range(X_train_pca.shape[1])
        ]

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="naive_bayes",
            model_name=f"GaussianNB PCA {variance_threshold}",
            dataset_type="scaled_pca",
            features=pca_features,
            params={
                "pca_variance_threshold": variance_threshold,
                "n_components": X_train_pca.shape[1],
            },
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="pca_variant",
        )

        registry = add_standard_model_outputs(
            registry, model_id, "train", y_train, train_score
        )
        registry = add_standard_model_outputs(
            registry, model_id, "validation", y_val, val_score
        )

        model_counter += 1

    # 3. Pruned feature grid using permutation importance
    ranked_features = get_permutation_ranked_features(
        model=baseline_model,
        X_val=X_val,
        y_val=y_val,
    )

    prune_sizes = [5, 8, 10, 12]

    for k in prune_sizes:
        selected_features = ranked_features[:k]

        model_id = f"{model_prefix}{model_counter:03d}"

        X_train_sub = X_train[selected_features]
        X_val_sub = X_val[selected_features]

        model = GaussianNB()
        model.fit(X_train_sub, y_train)

        train_score = model.predict_proba(X_train_sub)[:, 1]
        val_score = model.predict_proba(X_val_sub)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="naive_bayes",
            model_name=f"GaussianNB top_{k}_permutation_features",
            dataset_type="scaled_pruned",
            features=selected_features,
            params={
                "top_k": k,
                "ranking_method": "validation_permutation_importance",
            },
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="feature_pruning_from_permutation_importance",
        )

        registry = add_standard_model_outputs(
            registry, model_id, "train", y_train, train_score
        )
        registry = add_standard_model_outputs(
            registry, model_id, "validation", y_val, val_score
        )

        model_counter += 1

    return registry


def train_bernoulli_nb_models(
    registry: dict,
    datasets: dict,
    model_prefix: str = "BNB",
) -> dict:
    """
    Train BernoulliNB models on one-hot encoded binned features.

    Grid follows the notebook structure:
    - alpha grid
    - fit_prior True/False
    """

    split = datasets["binned_ohe"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    features = split["features"]

    alpha_grid = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0]
    fit_prior_grid = [True, False]

    model_counter = 1

    for alpha in alpha_grid:
        for fit_prior in fit_prior_grid:
            model_id = f"{model_prefix}{model_counter:03d}"

            model = BernoulliNB(
                alpha=alpha,
                fit_prior=fit_prior,
            )

            model.fit(X_train, y_train)

            train_score = model.predict_proba(X_train)[:, 1]
            val_score = model.predict_proba(X_val)[:, 1]

            train_metrics = evaluate_binary_classifier(y_train, train_score)
            val_metrics = evaluate_binary_classifier(y_val, val_score)

            registry = add_model_record(
                registry=registry,
                model_id=model_id,
                model_family="naive_bayes",
                model_name="BernoulliNB binned one-hot",
                dataset_type="binned_ohe",
                features=features,
                params={
                    "alpha": alpha,
                    "fit_prior": fit_prior,
                },
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                model_object=model,
                notes="bernoulli_nb_alpha_fit_prior_grid",
                selected_flag=False,
            )

            registry = add_standard_model_outputs(
                registry,
                model_id,
                "train",
                y_train,
                train_score,
            )

            registry = add_standard_model_outputs(
                registry,
                model_id,
                "validation",
                y_val,
                val_score,
            )

            model_counter += 1

    return registry

def train_woe_gaussian_nb_models(
    registry: dict,
    datasets: dict,
    model_prefix: str = "WGNB",
) -> dict:
    """
    Train GaussianNB variants on WOE features.

    Grid follows the notebook structure:
    - baseline all WOE features
    - var_smoothing grid
    - pruned WOE feature grid using permutation importance
    """

    split = datasets["woe"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    features = split["features"]

    model_counter = 1

    # Baseline
    baseline_model = GaussianNB()

    model_id = f"{model_prefix}{model_counter:03d}"
    baseline_model.fit(X_train, y_train)

    train_score = baseline_model.predict_proba(X_train)[:, 1]
    val_score = baseline_model.predict_proba(X_val)[:, 1]

    train_metrics = evaluate_binary_classifier(y_train, train_score)
    val_metrics = evaluate_binary_classifier(y_val, val_score)

    registry = add_model_record(
        registry=registry,
        model_id=model_id,
        model_family="naive_bayes",
        model_name="WOE GaussianNB baseline",
        dataset_type="woe",
        features=features,
        params=baseline_model.get_params(),
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        model_object=baseline_model,
        notes="all_woe_features",
        selected_flag=False,
    )

    registry = add_standard_model_outputs(
        registry, model_id, "train", y_train, train_score
    )
    registry = add_standard_model_outputs(
        registry, model_id, "validation", y_val, val_score
    )

    model_counter += 1

    # var_smoothing grid
    var_smoothing_grid = [
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
    ]

    for var_smoothing in var_smoothing_grid:
        model_id = f"{model_prefix}{model_counter:03d}"

        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)

        train_score = model.predict_proba(X_train)[:, 1]
        val_score = model.predict_proba(X_val)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="naive_bayes",
            model_name="WOE GaussianNB var_smoothing",
            dataset_type="woe",
            features=features,
            params={"var_smoothing": var_smoothing},
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="woe_gaussian_nb_var_smoothing_grid",
            selected_flag=False,
        )

        registry = add_standard_model_outputs(
            registry, model_id, "train", y_train, train_score
        )
        registry = add_standard_model_outputs(
            registry, model_id, "validation", y_val, val_score
        )

        model_counter += 1

    # Pruned WOE feature grid using permutation importance
    ranked_features = get_permutation_ranked_features(
        model=baseline_model,
        X_val=X_val,
        y_val=y_val,
    )

    prune_sizes = [3, 5, 8, 10]

    for k in prune_sizes:
        selected_features = ranked_features[:k]

        model_id = f"{model_prefix}{model_counter:03d}"

        X_train_sub = X_train[selected_features]
        X_val_sub = X_val[selected_features]

        model = GaussianNB()
        model.fit(X_train_sub, y_train)

        train_score = model.predict_proba(X_train_sub)[:, 1]
        val_score = model.predict_proba(X_val_sub)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="naive_bayes",
            model_name=f"WOE GaussianNB top_{k}_permutation_features",
            dataset_type="woe_pruned",
            features=selected_features,
            params={
                "top_k": k,
                "ranking_method": "validation_permutation_importance",
            },
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="woe_feature_pruning_from_permutation_importance",
            selected_flag=False,
        )

        registry = add_standard_model_outputs(
            registry, model_id, "train", y_train, train_score
        )
        registry = add_standard_model_outputs(
            registry, model_id, "validation", y_val, val_score
        )

        model_counter += 1

    return registry


def train_neural_network_models(
    registry: dict,
    datasets: dict,
    model_prefix: str = "NN",
) -> dict:
    """
    Train neural network models using sklearn MLPClassifier.

    Design:
    - uses scaled features
    - keeps architecture grid controlled but broad enough
    - includes all-feature models
    - includes permutation-ranked top feature subsets
    - uses early stopping consistently
    """

    split = datasets["scaled"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    all_features = split["features"]

    # Baseline model used only to create permutation ranking
    ranking_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=256,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
    )

    ranking_model.fit(X_train, y_train)

    ranked_features = get_permutation_ranked_features(
        model=ranking_model,
        X_val=X_val,
        y_val=y_val,
    )

    feature_sets = {
        "all_features": all_features,
        "top_5_permutation_features": ranked_features[:5],
        "top_10_permutation_features": ranked_features[:10],
        "top_12_permutation_features": ranked_features[:12],
    }

    model_configs = [
        # Notebook-aligned models
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 100,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (32, 16),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_10_permutation_features",
        },
        {
            "hidden_layer_sizes": (16,),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.0001,
            "learning_rate_init": 0.0005,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        # Expanded controlled grid
        {
            "hidden_layer_sizes": (64,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (128,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (128, 64),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (64, 32, 16),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.01,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "learning_rate_init": 0.0005,
            "max_iter": 200,
            "feature_set": "all_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "learning_rate_init": 0.0001,
            "max_iter": 200,
            "feature_set": "all_features",
        },
        # Feature subset tests
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_5_permutation_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_5_permutation_features",
        },
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_10_permutation_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_10_permutation_features",
        },
        {
            "hidden_layer_sizes": (32,),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_12_permutation_features",
        },
        {
            "hidden_layer_sizes": (64, 32),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "feature_set": "top_12_permutation_features",
        },
    ]

    for idx, config in enumerate(model_configs, start=1):
        model_id = f"{model_prefix}{idx:03d}"

        selected_features = feature_sets[config["feature_set"]]

        X_train_model = X_train[selected_features]
        X_val_model = X_val[selected_features]

        model = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=config["alpha"],
            learning_rate_init=config["learning_rate_init"],
            batch_size=256,
            max_iter=config["max_iter"],
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
        )

        model.fit(X_train_model, y_train)

        train_score = model.predict_proba(X_train_model)[:, 1]
        val_score = model.predict_proba(X_val_model)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        params = {
            "hidden_layer_sizes": config["hidden_layer_sizes"],
            "activation": "relu",
            "solver": "adam",
            "alpha": config["alpha"],
            "learning_rate_init": config["learning_rate_init"],
            "batch_size": 256,
            "max_iter": config["max_iter"],
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 10,
            "feature_set": config["feature_set"],
            "n_layers": len(config["hidden_layer_sizes"]),
            "n_iter_actual": model.n_iter_,
            "loss": model.loss_,
        }

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family="neural_network",
            model_name="MLPClassifier",
            dataset_type="scaled",
            features=selected_features,
            params=params,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes="mlp_classifier_scaled_features",
            selected_flag=False,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id,
            "train",
            y_train,
            train_score,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id,
            "validation",
            y_val,
            val_score,
        )

    return registry


def train_tree_models(
    registry: dict,
    datasets: dict,
) -> dict:
    """
    Train tree and boosting model families.

    Includes:
    - Decision Tree
    - Random Forest
    - Extra Trees
    - HistGradientBoosting
    - XGBoost, if installed
    - LightGBM, if installed
    - CatBoost, if installed

    The grid includes notebook-style candidates plus conservative
    anti-overfit challengers.
    """

    split = datasets["tree"]

    X_train = split["X_train"]
    X_val = split["X_val"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    features = split["features"]

    model_specs = []

    # Decision Tree
    decision_tree_configs = [
        {"max_depth": 3, "min_samples_leaf": 100, "class_weight": None},
        {"max_depth": 5, "min_samples_leaf": 100, "class_weight": None},
        {"max_depth": 7, "min_samples_leaf": 100, "class_weight": None},
        {"max_depth": 5, "min_samples_leaf": 250, "class_weight": None},
        {"max_depth": 7, "min_samples_leaf": 250, "class_weight": None},
        {"max_depth": 5, "min_samples_leaf": 100, "class_weight": "balanced"},
        {"max_depth": 7, "min_samples_leaf": 100, "class_weight": "balanced"},
    ]

    for idx, params in enumerate(decision_tree_configs, start=1):
        model_specs.append(
            {
                "model_id": f"T{idx:03d}",
                "model_family": "tree",
                "model_name": "DecisionTreeClassifier",
                "dataset_type": "tree",
                "model": DecisionTreeClassifier(
                    **params,
                    random_state=42,
                ),
                "params": params,
                "notes": "decision_tree_grid",
            }
        )

    # Random Forest
    random_forest_configs = [
        {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 250,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 100,
            "max_features": 0.5,
            "class_weight": "balanced_subsample",
        },
    ]

    for idx, params in enumerate(random_forest_configs, start=1):
        model_specs.append(
            {
                "model_id": f"RF{idx:03d}",
                "model_family": "tree",
                "model_name": "RandomForestClassifier",
                "dataset_type": "tree",
                "model": RandomForestClassifier(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                ),
                "params": params,
                "notes": "random_forest_grid_plus_anti_overfit",
            }
        )

    # Extra Trees
    extra_trees_configs = [
        {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 100,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 250,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 100,
            "max_features": 0.5,
            "class_weight": "balanced",
        },
    ]

    for idx, params in enumerate(extra_trees_configs, start=1):
        model_specs.append(
            {
                "model_id": f"ET{idx:03d}",
                "model_family": "tree",
                "model_name": "ExtraTreesClassifier",
                "dataset_type": "tree",
                "model": ExtraTreesClassifier(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                ),
                "params": params,
                "notes": "extra_trees_grid_plus_anti_overfit",
            }
        )

    # HistGradientBoosting
    hgb_configs = [
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 400,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 250,
            "l2_regularization": 0.1,
        },
        {
            "learning_rate": 0.02,
            "max_iter": 500,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 250,
            "l2_regularization": 1.0,
        },
    ]

    for idx, params in enumerate(hgb_configs, start=1):
        model_specs.append(
            {
                "model_id": f"GB{idx:03d}",
                "model_family": "boosting",
                "model_name": "HistGradientBoostingClassifier",
                "dataset_type": "tree",
                "model": HistGradientBoostingClassifier(
                    **params,
                    random_state=42,
                ),
                "params": params,
                "notes": "hist_gradient_boosting_grid_plus_anti_overfit",
            }
        )

    # XGBoost
    if XGBClassifier is not None:
        xgb_configs = [
            {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 2,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "reg_alpha": 0.1,
            },
            {
                "n_estimators": 700,
                "max_depth": 2,
                "learning_rate": 0.02,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10.0,
                "reg_alpha": 0.5,
            },
        ]

        for idx, params in enumerate(xgb_configs, start=1):
            model_specs.append(
                {
                    "model_id": f"XGB{idx:03d}",
                    "model_family": "boosting",
                    "model_name": "XGBClassifier",
                    "dataset_type": "tree",
                    "model": XGBClassifier(
                        **params,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                    "params": params,
                    "notes": "xgboost_grid_plus_anti_overfit",
                }
            )

    # LightGBM
    if LGBMClassifier is not None:
        lgb_configs = [
            {
                "n_estimators": 300,
                "num_leaves": 15,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "min_child_samples": 100,
            },
            {
                "n_estimators": 500,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "min_child_samples": 100,
            },
            {
                "n_estimators": 500,
                "num_leaves": 15,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "reg_alpha": 0.1,
                "min_child_samples": 250,
            },
            {
                "n_estimators": 700,
                "num_leaves": 15,
                "learning_rate": 0.02,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10.0,
                "reg_alpha": 0.5,
                "min_child_samples": 300,
            },
        ]

        for idx, params in enumerate(lgb_configs, start=1):
            model_specs.append(
                {
                    "model_id": f"LGB{idx:03d}",
                    "model_family": "boosting",
                    "model_name": "LGBMClassifier",
                    "dataset_type": "tree",
                    "model": LGBMClassifier(
                        **params,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    "params": params,
                    "notes": "lightgbm_grid_plus_anti_overfit",
                }
            )

    # CatBoost
    if CatBoostClassifier is not None:
        cat_configs = [
            {
                "iterations": 300,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
            },
            {
                "iterations": 500,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
            },
            {
                "iterations": 500,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 10.0,
            },
            {
                "iterations": 700,
                "depth": 4,
                "learning_rate": 0.02,
                "l2_leaf_reg": 15.0,
            },
        ]

        for idx, params in enumerate(cat_configs, start=1):
            model_specs.append(
                {
                    "model_id": f"CAT{idx:03d}",
                    "model_family": "boosting",
                    "model_name": "CatBoostClassifier",
                    "dataset_type": "tree",
                    "model": CatBoostClassifier(
                        **params,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=42,
                        verbose=False,
                    ),
                    "params": params,
                    "notes": "catboost_grid_plus_anti_overfit",
                }
            )

    for spec in model_specs:
        model_id = spec["model_id"]
        model = spec["model"]

        model.fit(X_train, y_train)

        train_score = model.predict_proba(X_train)[:, 1]
        val_score = model.predict_proba(X_val)[:, 1]

        train_metrics = evaluate_binary_classifier(y_train, train_score)
        val_metrics = evaluate_binary_classifier(y_val, val_score)

        validation_gap = train_metrics["auc"] - val_metrics["auc"]
        val_metrics["auc_train_validation_gap"] = float(validation_gap)

        registry = add_model_record(
            registry=registry,
            model_id=model_id,
            model_family=spec["model_family"],
            model_name=spec["model_name"],
            dataset_type=spec["dataset_type"],
            features=features,
            params=spec["params"],
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_object=model,
            notes=spec["notes"],
            selected_flag=False,
        )

        importance_table = build_feature_importance_table(model, features)

        if not importance_table.empty:
            registry = add_feature_importance_table(
                registry=registry,
                model_id=model_id,
                feature_importance_table=importance_table,
            )

        registry = add_standard_model_outputs(
            registry,
            model_id,
            "train",
            y_train,
            train_score,
        )

        registry = add_standard_model_outputs(
            registry,
            model_id,
            "validation",
            y_val,
            val_score,
        )

    return registry

