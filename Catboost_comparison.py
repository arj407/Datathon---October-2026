from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_metrics(name, actual, predicted):
    return {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }


def prepare_catboost_features(data, categorical_features):
    """Keep numeric NaNs; represent categorical values as strings."""
    result = data.copy()

    for column in categorical_features:
        result[column] = (
            result[column]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    return result


def main():
    # ==========================================================
    # 1. LOAD DATA
    # ==========================================================

    csv_path = Path(
        "/Users/ArjunMittal/Downloads/Track 1 Dataset/"
        "Track_1_Training_Cleaned.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded file: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isna().sum())

    assert df["id"].is_unique, "Duplicate row IDs found."

    assert df[
        [
            "temperature",
            "humidity",
            "occupancy",
            "previous_usage",
            "energy_usage"
        ]
    ].notna().all().all(), "Some required numeric values are still missing."

    numeric_features = [
        "temperature",
        "humidity",
        "occupancy",
        "previous_usage",
    ]

    categorical_features = [
        "building_id",
        "hour",
        "day_of_week",
        "month",
    ]

    # Use the same input columns as the linear baseline.
    features = numeric_features + categorical_features

    X = df[features].copy()
    y = df["energy_usage"].copy()

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Target contains missing or infinite values.")

    # ==========================================================
    # 2. REPRODUCE THE ORIGINAL 80/20 SPLIT
    # ==========================================================

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=2000,
        random_state=72,
        stratify=X["building_id"],
    )

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_valid)}")

    # ==========================================================
    # 3. TRAIN THE ORIGINAL LINEAR REGRESSION BASELINE
    # ==========================================================

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    linear_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    print("\nTraining linear regression...")
    linear_model.fit(X_train, y_train)

    # ==========================================================
    # 4. PREPARE CATBOOST INPUTS
    # ==========================================================

    # CatBoost receives the original columns rather than the
    # scaled / one-hot-encoded linear regression inputs.
    # Numeric missing values remain NaN.
    X_train_cb = prepare_catboost_features(
        X_train, categorical_features
    )
    X_valid_cb = prepare_catboost_features(
        X_valid, categorical_features
    )

    # Split ONLY the training portion for early stopping.
    # The final 1,600 validation rows are not used here.
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train_cb,
        y_train,
        test_size=0.2,
        random_state=24,
        stratify=X_train_cb["building_id"],
    )

    print(f"\nCatBoost initial fitting rows: {len(X_fit)}")
    print(f"CatBoost early-stopping rows: {len(X_stop)}")

    parameters = {
        "depth": 4,
        "learning_rate": 0.07,
        "l2_leaf_reg": 10,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "task_type": "CPU",
        "allow_writing_files": False,
    }

    # ==========================================================
    # 5. SELECT CATBOOST'S TREE COUNT
    # ==========================================================

    print("\nSelecting CatBoost tree count...")

    selection_model = CatBoostRegressor(
        iterations=3000,
        **parameters,
    )

    selection_model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_features,
        eval_set=(X_stop, y_stop),
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=250,
    )

    # CatBoost returns a zero-based iteration index.
    best_tree_count = selection_model.get_best_iteration() + 1

    print(f"\nSelected number of trees: {best_tree_count}")

    # ==========================================================
    # 6. RETRAIN CATBOOST ON ALL 6,400 TRAINING ROWS
    # ==========================================================

    # Fix the tree count chosen above.
    # No final validation data is supplied during this fit.
    catboost_model = CatBoostRegressor(
        iterations=best_tree_count,
        **parameters,
    )

    print("\nRetraining CatBoost on all training rows...")

    catboost_model.fit(
        X_train_cb,
        y_train,
        cat_features=categorical_features,
        verbose=250,
    )

    # ==========================================================
    # 7. COMPARE ON THE SAME VALIDATION ROWS
    # ==========================================================

    linear_predictions = linear_model.predict(X_valid)
    catboost_predictions = catboost_model.predict(X_valid_cb)

    results = pd.DataFrame([
        calculate_metrics(
            "Linear regression",
            y_valid,
            linear_predictions,
        ),
        calculate_metrics(
            "CatBoost",
            y_valid,
            catboost_predictions,
        ),
    ])

    print("\nVALIDATION RESULTS")
    print(
        results.sort_values("RMSE").to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    linear_rmse = np.sqrt(
        mean_squared_error(y_valid, linear_predictions)
    )
    catboost_rmse = np.sqrt(
        mean_squared_error(y_valid, catboost_predictions)
    )

    improvement = (
        100 * (linear_rmse - catboost_rmse) / linear_rmse
    )

    print(
        f"\nCatBoost RMSE improvement over linear regression: "
        f"{improvement:.2f}%"
    )
    print("Positive = CatBoost is better; negative = worse.")

    print("\nTRAINING VS VALIDATION RMSE")

    for name, model, train_inputs, valid_rmse in [
        (
            "Linear regression",
            linear_model,
            X_train,
            linear_rmse,
        ),
        (
            "CatBoost",
            catboost_model,
            X_train_cb,
            catboost_rmse,
        ),
    ]:
        train_rmse = np.sqrt(
            mean_squared_error(
                y_train,
                model.predict(train_inputs),
            )
        )
        print(
            f"{name}: training={train_rmse:.4f}, "
            f"validation={valid_rmse:.4f}"
        )

    # ==========================================================
    # 8. CHECK MISSING-PREVIOUS-USAGE PERFORMANCE
    # ==========================================================

    prediction_table = pd.DataFrame({
        "id": df.loc[X_valid.index, "id"].to_numpy(),
        "building_id": X_valid["building_id"].to_numpy(),
        "actual": y_valid.to_numpy(),
        "linear_prediction": linear_predictions,
        "catboost_prediction": catboost_predictions,
        "previous_usage_missing": (
            X_valid["previous_usage"].isna().to_numpy()
        ),
    })

    for name in ["linear", "catboost"]:
        prediction_table[f"{name}_absolute_error"] = np.abs(
            prediction_table["actual"]
            - prediction_table[f"{name}_prediction"]
        )

    missingness_comparison = (
        prediction_table.groupby("previous_usage_missing")
        .agg(
            rows=("actual", "size"),
            linear_MAE=("linear_absolute_error", "mean"),
            catboost_MAE=("catboost_absolute_error", "mean"),
        )
    )

    print("\nERROR BY AVAILABILITY OF PREVIOUS USAGE")
    print(missingness_comparison.round(4).to_string())

    # ==========================================================
    # 9. SAVE RESULTS
    # ==========================================================

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "catboost_comparison_metrics.csv",
        index=False,
    )
    prediction_table.to_csv(
        output_dir / "catboost_validation_predictions.csv",
        index=False,
    )
    missingness_comparison.to_csv(
        output_dir / "catboost_missingness_comparison.csv",
    )

    # ==========================================================
    # 10. PLOT VALIDATION PREDICTIONS AND RESIDUALS
    # ==========================================================

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    low = min(
        y_valid.min(),
        linear_predictions.min(),
        catboost_predictions.min(),
    )
    high = max(
        y_valid.max(),
        linear_predictions.max(),
        catboost_predictions.max(),
    )

    residual_limit = max(
        np.abs(y_valid.to_numpy() - linear_predictions).max(),
        np.abs(y_valid.to_numpy() - catboost_predictions).max(),
        1.0,
    ) * 1.05

    for column, (name, predictions) in enumerate([
        ("Linear regression", linear_predictions),
        ("CatBoost", catboost_predictions),
    ]):
        ax = axes[0, column]
        ax.scatter(y_valid, predictions, alpha=0.25, s=15)
        ax.plot([low, high], [low, high], "r--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Actual energy usage")
        ax.set_ylabel("Predicted energy usage")
        ax.set_title(f"{name}: validation predictions")

        ax = axes[1, column]
        residuals = y_valid.to_numpy() - predictions
        ax.scatter(predictions, residuals, alpha=0.25, s=15)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlim(low, high)
        ax.set_ylim(-residual_limit, residual_limit)
        ax.set_xlabel("Predicted energy usage")
        ax.set_ylabel("Actual minus predicted")
        ax.set_title(f"{name}: validation residuals")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        output_dir / "catboost_comparison_plots.png",
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nResults saved to: {output_dir}")
    plt.show()


if __name__ == "__main__":
    main()