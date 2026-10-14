from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_metrics(name, actual, predicted):
    return {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }


def main():
    # ==========================================================
    # 1. LOAD THE TRAINING DATA
    # ==========================================================

    csv_path = Path(
        "/Users/ArjunMittal/Downloads/Track 1 Dataset/"
        "Track 1 Training Dataset.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find: {csv_path}")

    df = pd.read_csv(csv_path)

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

    features = numeric_features + categorical_features

    X = df[features].copy()
    y = df["energy_usage"].copy()

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Target contains missing or infinite values.")

    # ==========================================================
    # 2. USE EXACTLY THE SAME SPLIT AS THE LINEAR BASELINE
    # ==========================================================

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=X["building_id"],
    )

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_valid)}")

    # ==========================================================
    # 3. BUILD THE SAME PREPROCESSING PIPELINE
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

    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge()),
    ])

    # ==========================================================
    # 4. SELECT ALPHA USING TRAINING DATA ONLY
    # ==========================================================

    # Smaller alpha: closer to ordinary linear regression.
    # Larger alpha: stronger shrinkage of coefficients.
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Inner folds preserve the representation of each building.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    inner_folds = list(
        splitter.split(X_train, X_train["building_id"])
    )

    # The ENTIRE pipeline is cross-validated:
    # imputation, scaling, and encoding are fitted separately
    # inside each training fold.
    search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid={"regressor__alpha": alpha_values},
        scoring="neg_root_mean_squared_error",
        cv=inner_folds,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )

    print("\nSelecting alpha using five-fold training cross-validation...")
    search.fit(X_train, y_train)

    # refit=True retrains the winning pipeline on all 6,400
    # training rows, without using the 1,600 validation targets.
    ridge_model = search.best_estimator_

    cv_results = pd.DataFrame({
        "alpha": [
            float(value)
            for value in search.cv_results_["param_regressor__alpha"]
        ],
        "mean_CV_RMSE": -search.cv_results_["mean_test_score"],
        "std_CV_RMSE": search.cv_results_["std_test_score"],
    }).sort_values("mean_CV_RMSE")

    print("\nALPHA COMPARISON — TRAINING CROSS-VALIDATION")
    print(cv_results.round(4).to_string(index=False))

    print(f"\nSelected alpha: {search.best_params_['regressor__alpha']}")

    # ==========================================================
    # 5. FIT ORDINARY LINEAR REGRESSION FOR DIRECT COMPARISON
    # ==========================================================

    linear_model = clone(ridge_pipeline).set_params(
        regressor=LinearRegression()
    )
    linear_model.fit(X_train, y_train)

    # ==========================================================
    # 6. EVALUATE BOTH ON THE SAME VALIDATION ROWS
    # ==========================================================

    ridge_predictions = ridge_model.predict(X_valid)
    linear_predictions = linear_model.predict(X_valid)

    previous_predictions = (
        X_valid["previous_usage"]
        .fillna(X_train["previous_usage"].median())
        .to_numpy()
    )

    results = pd.DataFrame([
        calculate_metrics(
            "Previous usage",
            y_valid,
            previous_predictions,
        ),
        calculate_metrics(
            "Linear regression",
            y_valid,
            linear_predictions,
        ),
        calculate_metrics(
            "Ridge regression",
            y_valid,
            ridge_predictions,
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
    ridge_rmse = np.sqrt(
        mean_squared_error(y_valid, ridge_predictions)
    )

    improvement = 100 * (linear_rmse - ridge_rmse) / linear_rmse

    print(
        f"\nRidge RMSE improvement over linear regression: "
        f"{improvement:.2f}%"
    )
    print("Positive = Ridge is better; negative = Ridge is worse.")

    ridge_train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            ridge_model.predict(X_train),
        )
    )

    print(f"\nRidge training RMSE:   {ridge_train_rmse:.4f}")
    print(f"Ridge validation RMSE: {ridge_rmse:.4f}")

    # ==========================================================
    # 7. COMPARE ERRORS WHEN PREVIOUS USAGE IS MISSING
    # ==========================================================

    predictions_table = pd.DataFrame({
        "id": df.loc[X_valid.index, "id"].to_numpy(),
        "actual_energy_usage": y_valid.to_numpy(),
        "linear_prediction": linear_predictions,
        "ridge_prediction": ridge_predictions,
        "previous_usage_missing": (
            X_valid["previous_usage"].isna().to_numpy()
        ),
    })

    predictions_table["linear_absolute_error"] = np.abs(
        predictions_table["actual_energy_usage"]
        - predictions_table["linear_prediction"]
    )

    predictions_table["ridge_absolute_error"] = np.abs(
        predictions_table["actual_energy_usage"]
        - predictions_table["ridge_prediction"]
    )

    missingness_comparison = (
        predictions_table.groupby("previous_usage_missing")
        .agg(
            rows=("actual_energy_usage", "size"),
            linear_MAE=("linear_absolute_error", "mean"),
            ridge_MAE=("ridge_absolute_error", "mean"),
        )
    )

    print("\nERROR BY AVAILABILITY OF PREVIOUS USAGE")
    print(missingness_comparison.round(4).to_string())

    # ==========================================================
    # 8. SAVE THE RESULTS
    # ==========================================================

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "ridge_comparison_metrics.csv",
        index=False,
    )

    cv_results.to_csv(
        output_dir / "ridge_alpha_search.csv",
        index=False,
    )

    predictions_table.to_csv(
        output_dir / "ridge_validation_predictions.csv",
        index=False,
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()