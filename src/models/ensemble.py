import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.models.diagnostics import (
    calculate_gains_lift,
    generate_calibration_table,
    generate_ks_curve_data,
    generate_roc_curve_data,
)
from src.models.evaluate import evaluate_binary_classifier
from src.models.model_registry import add_diagnostics, add_model_record
from src.models.mlflow_utils import log_model_to_mlflow


# --------------------------------------------------
# Serializable ensemble model classes
# --------------------------------------------------

class EqualAverageEnsemble:
    """
    Serializable equal-weight ensemble over base model prediction columns.

    Expected input X:
        DataFrame where columns are base model IDs and values are predicted probabilities.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names_in_ = list(feature_names)
        self.weights_ = np.repeat(1 / len(self.feature_names_in_), len(self.feature_names_in_))

    def predict_proba(self, X):
        X_aligned = X[self.feature_names_in_]
        score = X_aligned.mean(axis=1).to_numpy()
        score = np.clip(score, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - score, score])


class WeightedAverageEnsemble:
    """
    Serializable weighted-average ensemble over base model prediction columns.

    Expected input X:
        DataFrame where columns are base model IDs and values are predicted probabilities.
    """

    def __init__(self, weights: dict[str, float]):
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.feature_names_in_ = list(self.weights.keys())
        self.weights_ = np.array([self.weights[col] for col in self.feature_names_in_], dtype=float)

    def predict_proba(self, X):
        X_aligned = X[self.feature_names_in_]
        score = np.dot(X_aligned.to_numpy(), self.weights_)
        score = np.clip(score, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - score, score])


class RankAverageEnsemble:
    """
    Serializable rank-average ensemble over base model prediction columns.

    Expected input X:
        DataFrame where columns are base model IDs and values are predicted probabilities.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names_in_ = list(feature_names)

    def predict_proba(self, X):
        X_aligned = X[self.feature_names_in_]
        score = X_aligned.rank(pct=True).mean(axis=1).to_numpy()
        score = np.clip(score, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - score, score])


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def parse_features(feature_string: str) -> list[str]:
    return [x.strip() for x in feature_string.split(",") if x.strip()]


def select_ensemble_candidates(
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one strong candidate per major model family/type.
    """

    candidates = []

    selection_rules = [
        ("best_boosting", model_summary["model_family"] == "boosting"),
        ("best_nn", model_summary["model_family"] == "neural_network"),
        ("best_bnb", model_summary["model_id"].str.startswith("BNB")),
        ("best_wlog", model_summary["model_id"].str.startswith("WLOG")),
        ("best_blog", model_summary["model_id"].str.startswith("BLOG")),
        ("best_rlog", model_summary["model_id"].str.startswith("RLOG")),
        ("best_wgnb", model_summary["model_id"].str.startswith("WGNB")),
        ("best_gnb", model_summary["model_id"].str.startswith("GNB")),
    ]

    for candidate_group, mask in selection_rules:
        subset = model_summary[mask].copy()

        # PCA GNB models were trained without saving the PCA transformer,
        # so exclude them from ensemble prediction.
        subset = subset[subset["dataset_type"] != "scaled_pca"]

        if subset.empty:
            continue

        best_row = subset.sort_values(
            "validation_auc",
            ascending=False,
        ).iloc[0].copy()

        best_row["candidate_group"] = candidate_group
        candidates.append(best_row)

    return pd.DataFrame(candidates).reset_index(drop=True)


def predict_saved_model(
    model,
    model_row: pd.Series,
    datasets: dict,
    split_name: str,
) -> np.ndarray:
    """
    Generate predictions from a saved base model using the correct dataset.
    """

    model_id = model_row["model_id"]
    dataset_type = model_row["dataset_type"]
    model_name = model_row["model_name"]
    features = parse_features(model_row["features"])

    if dataset_type == "tree":
        split = datasets["tree"]
        X = split[f"X_{split_name}"][features]

    elif dataset_type in ["scaled", "scaled_pruned"]:
        split = datasets["scaled"]
        X = split[f"X_{split_name}"][features]

    elif dataset_type in ["woe", "woe_pruned"]:
        split = datasets["woe"]
        X = split[f"X_{split_name}"][features]

    elif dataset_type == "binned_ohe":
        split = datasets["binned_ohe"]
        X = split[f"X_{split_name}"][features]

    elif dataset_type == "binned_logit":
        split = datasets["binned_logit"]
        X_raw = split[f"X_{split_name}"][features].astype(str)
        X = pd.get_dummies(X_raw, drop_first=True, dtype=int)

        expected_cols = [x for x in model.params.index if x != "const"]
        X = X.reindex(columns=expected_cols, fill_value=0)

    else:
        raise ValueError(f"Unsupported dataset_type for ensemble: {dataset_type}")

    if model_id.startswith(("WLOG", "BLOG")) or "staged logistic" in model_name:
        X_const = sm.add_constant(X, has_constant="add")
        return np.asarray(model.predict(X_const))

    return model.predict_proba(X)[:, 1]


def build_base_prediction_matrices(
    candidates: pd.DataFrame,
    artifact_paths: dict,
    datasets: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train and validation prediction matrices from selected base models.
    """

    train_preds = {}
    val_preds = {}

    for _, row in candidates.iterrows():
        model_id = row["model_id"]

        model_path = artifact_paths.get(model_id)
        if model_path is None:
            raise ValueError(f"Missing artifact path for model_id: {model_id}")

        model = joblib.load(model_path)

        train_preds[model_id] = predict_saved_model(
            model=model,
            model_row=row,
            datasets=datasets,
            split_name="train",
        )

        val_preds[model_id] = predict_saved_model(
            model=model,
            model_row=row,
            datasets=datasets,
            split_name="val",
        )

    return pd.DataFrame(train_preds), pd.DataFrame(val_preds)


def add_ensemble_diagnostics(
    registry: dict,
    model_id: str,
    split: str,
    y_true,
    y_score,
) -> dict:
    registry = add_diagnostics(
        registry=registry,
        model_id=model_id,
        split=split,
        lift_table=calculate_gains_lift(y_true, y_score),
        roc_curve=generate_roc_curve_data(y_true, y_score),
        ks_curve=generate_ks_curve_data(y_true, y_score),
        calibration_table=generate_calibration_table(y_true, y_score),
    )

    return registry


def register_ensemble_model(
    registry: dict,
    model_id: str,
    model_name: str,
    features: list[str],
    params: dict,
    y_train,
    train_score,
    y_val,
    val_score,
    model_object=None,
    notes: str = "",
) -> dict:
    train_metrics = evaluate_binary_classifier(y_train, train_score)
    val_metrics = evaluate_binary_classifier(y_val, val_score)

    registry = add_model_record(
        registry=registry,
        model_id=model_id,
        model_family="ensemble",
        model_name=model_name,
        dataset_type="base_model_predictions",
        features=features,
        params=params,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        model_object=model_object,
        notes=notes,
        selected_flag=False,
    )

    try:
        log_model_to_mlflow(
            model_id=model_id,
            model=model_object,
            params=params,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            model_family="ensemble",
            model_name=model_name,
            dataset_type="base_model_predictions",
            experiment_name="credit-risk-ensembles",
        )
    except Exception as exc:
        print(f"[MLflow warning] Failed to log ensemble {model_id}: {exc}")

    registry = add_ensemble_diagnostics(
        registry,
        model_id,
        "train",
        y_train,
        train_score,
    )

    registry = add_ensemble_diagnostics(
        registry,
        model_id,
        "validation",
        y_val,
        val_score,
    )

    return registry


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 0)
    if weights.sum() == 0:
        return np.repeat(1 / len(weights), len(weights))
    return weights / weights.sum()


def optimize_blend_weights(
    train_matrix: pd.DataFrame,
    y_train,
) -> np.ndarray:
    """
    Optimize ensemble weights using train log loss.
    """

    n_models = train_matrix.shape[1]

    if minimize is None:
        return np.repeat(1 / n_models, n_models)

    initial_weights = np.repeat(1 / n_models, n_models)

    bounds = [(0, 1) for _ in range(n_models)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def objective(weights):
        weights = normalize_weights(weights)
        score = np.dot(train_matrix.values, weights)
        score = np.clip(score, 1e-6, 1 - 1e-6)
        return log_loss(y_train, score)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return initial_weights

    return normalize_weights(result.x)


# --------------------------------------------------
# Ensemble tournament
# --------------------------------------------------

def run_ensemble_tournament(
    registry: dict,
    model_summary: pd.DataFrame,
    artifact_paths: dict,
    datasets: dict,
) -> dict:
    """
    Build ensemble models from saved base model predictions.

    All ensembles are registered as first-class, serializable model objects:
    - EqualAverageEnsemble
    - WeightedAverageEnsemble
    - RankAverageEnsemble
    - sklearn / xgboost stacking meta-learners
    """

    candidates = select_ensemble_candidates(model_summary)

    train_matrix, val_matrix = build_base_prediction_matrices(
        candidates=candidates,
        artifact_paths=artifact_paths,
        datasets=datasets,
    )

    y_train = datasets["tree"]["y_train"]
    y_val = datasets["tree"]["y_val"]

    features = train_matrix.columns.tolist()

    candidate_auc = dict(zip(candidates["model_id"], candidates["validation_auc"]))
    candidate_ks = dict(zip(candidates["model_id"], candidates["validation_ks"]))
    candidate_log_loss = dict(zip(candidates["model_id"], candidates["validation_log_loss"]))

    # 1. Equal average
    equal_model = EqualAverageEnsemble(features)

    train_score = equal_model.predict_proba(train_matrix)[:, 1]
    val_score = equal_model.predict_proba(val_matrix)[:, 1]

    registry = register_ensemble_model(
        registry,
        "EASTACK001",
        "Equal average ensemble",
        features,
        {"weights": dict(zip(features, equal_model.weights_))},
        y_train,
        train_score,
        y_val,
        val_score,
        model_object=equal_model,
        notes="equal_average_of_selected_family_champions",
    )

    # 2. AUC-weighted
    auc_weights = np.array([max(candidate_auc[m] - 0.5, 1e-6) for m in features])
    auc_weights = normalize_weights(auc_weights)
    auc_weight_dict = dict(zip(features, auc_weights))

    auc_model = WeightedAverageEnsemble(auc_weight_dict)

    registry = register_ensemble_model(
        registry,
        "WASTACK001",
        "AUC-weighted ensemble",
        features,
        {"weights": auc_weight_dict},
        y_train,
        auc_model.predict_proba(train_matrix)[:, 1],
        y_val,
        auc_model.predict_proba(val_matrix)[:, 1],
        model_object=auc_model,
        notes="validation_auc_weighted_average",
    )

    # 3. KS-weighted
    ks_weights = np.array([max(candidate_ks[m], 1e-6) for m in features])
    ks_weights = normalize_weights(ks_weights)
    ks_weight_dict = dict(zip(features, ks_weights))

    ks_model = WeightedAverageEnsemble(ks_weight_dict)

    registry = register_ensemble_model(
        registry,
        "WASTACK002",
        "KS-weighted ensemble",
        features,
        {"weights": ks_weight_dict},
        y_train,
        ks_model.predict_proba(train_matrix)[:, 1],
        y_val,
        ks_model.predict_proba(val_matrix)[:, 1],
        model_object=ks_model,
        notes="validation_ks_weighted_average",
    )

    # 4. Inverse log-loss weighted
    inv_logloss_weights = np.array([1 / max(candidate_log_loss[m], 1e-6) for m in features])
    inv_logloss_weights = normalize_weights(inv_logloss_weights)
    inv_logloss_weight_dict = dict(zip(features, inv_logloss_weights))

    inv_logloss_model = WeightedAverageEnsemble(inv_logloss_weight_dict)

    registry = register_ensemble_model(
        registry,
        "WASTACK003",
        "Inverse log-loss weighted ensemble",
        features,
        {"weights": inv_logloss_weight_dict},
        y_train,
        inv_logloss_model.predict_proba(train_matrix)[:, 1],
        y_val,
        inv_logloss_model.predict_proba(val_matrix)[:, 1],
        model_object=inv_logloss_model,
        notes="inverse_validation_logloss_weighted_average",
    )

    # 5. Optimized log-loss weights
    opt_weights = optimize_blend_weights(train_matrix, y_train)
    opt_weight_dict = dict(zip(features, opt_weights))

    opt_model = WeightedAverageEnsemble(opt_weight_dict)

    registry = register_ensemble_model(
        registry,
        "WASTACK004",
        "Optimized weighted ensemble",
        features,
        {"weights": opt_weight_dict},
        y_train,
        opt_model.predict_proba(train_matrix)[:, 1],
        y_val,
        opt_model.predict_proba(val_matrix)[:, 1],
        model_object=opt_model,
        notes="train_logloss_optimized_weights",
    )

    # 6. Rank average
    rank_model = RankAverageEnsemble(features)

    registry = register_ensemble_model(
        registry,
        "RASTACK001",
        "Rank average ensemble",
        features,
        {"rank_method": "percentile_average"},
        y_train,
        rank_model.predict_proba(train_matrix)[:, 1],
        y_val,
        rank_model.predict_proba(val_matrix)[:, 1],
        model_object=rank_model,
        notes="rank_average_of_base_predictions",
    )

    # 7. Stacking meta-learners
    stack_configs = [
        (
            "LRSTACK001",
            "Logistic stacking L2",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
        ),
        (
            "RIDGESTACK001",
            "Ridge logistic stacking",
            LogisticRegression(
                penalty="l2",
                C=0.1,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
        ),
        (
            "LASSOSTACK001",
            "Lasso logistic stacking",
            LogisticRegression(
                penalty="l1",
                C=0.1,
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
        ),
        (
            "RFSTACK001",
            "Random forest stacking",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "NNSTACK001",
            "Neural network stacking",
            MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
            ),
        ),
    ]

    if XGBClassifier is not None:
        stack_configs.append(
            (
                "XGBSTACK001",
                "XGBoost stacking",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

    for model_id, model_name, model in stack_configs:
        model.fit(train_matrix, y_train)

        # Store input names explicitly for consistent interpretability/scoring later.
        if not hasattr(model, "feature_names_in_"):
            model.feature_names_in_ = np.array(features)

        train_score = model.predict_proba(train_matrix)[:, 1]
        val_score = model.predict_proba(val_matrix)[:, 1]

        registry = register_ensemble_model(
            registry,
            model_id,
            model_name,
            features,
            model.get_params(),
            y_train,
            train_score,
            y_val,
            val_score,
            model_object=model,
            notes="meta_learner_on_base_model_predictions",
        )

    return registry