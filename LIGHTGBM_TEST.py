from pathlib import Path
import inspect

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# ==============================================================
# 1. FILE PATHS — CHANGE THESE IF NEEDED
# ==============================================================

DATA_DIR = Path("/Users/ArjunMittal/Downloads/Track 1 Dataset")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "lightgbm_submission"
)

TARGET = "energy_usage"

# Same input choices as the earlier LightGBM comparison.
NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "occupancy",
    "previous_usage",
]

CATEGORICAL_FEATURES = [
    "building_id",
    "hour",
    "day_of_week",
    "month",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_PARAMETERS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

MODEL_PARAMETERS.update({
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 50,
    "reg_lambda": 10.0,
})


def normalize_features(data):
    """
    Validate inputs and standardize their types.

    Numeric missing values remain NaN.
    Categorical values become strings, with missing values preserved.
    """
    missing_columns = [
        column for column in FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required input columns are missing: {missing_columns}"
        )

    X = data[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        ).astype(float)

        if np.isinf(X[column].to_numpy()).any():
            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

    # Normalize integer-valued time categories so that, for example,
    # an hour read as 5.0 matches an hour read as 5.
    for column in ["hour", "month"]:
        values = pd.to_numeric(X[column], errors="raise")

        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Column '{column}' contains infinity.")

        observed = values.dropna()

        if not np.equal(observed, np.floor(observed)).all():
            raise ValueError(f"Column '{column}' must contain integers.")

        X[column] = values.astype("Int64").astype("string")

    for column in ["building_id", "day_of_week"]:
        X[column] = X[column].astype("string")

    return X


def learn_categories(training_features):
    """Learn category lists from training inputs only."""
    return {
        column: training_features[column].dropna().unique().tolist()
        for column in CATEGORICAL_FEATURES
    }


def apply_categories(features, category_levels):
    """
    Apply fixed category lists.

    Missing or previously unseen categories become NaN,
    which LightGBM can handle.
    """
    X = features.copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = pd.Categorical(
            X[column],
            categories=category_levels[column],
        )

    return X


def main():
    # ==========================================================
    # 2. LOAD AND CHECK THE FILES
    # ==========================================================

    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find file: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if len(train_df) != 8000:
        raise ValueError(
            f"Expected 8,000 training rows, found {len(train_df):,}."
        )

    if len(test_df) != 3000:
        raise ValueError(
            f"Expected 3,000 testing rows, found {len(test_df):,}."
        )

    if TARGET not in train_df.columns:
        raise ValueError(
            f"Training CSV must contain '{TARGET}'."
        )

    if TARGET in test_df.columns:
        print(
            f"Note: '{TARGET}' exists in the testing CSV "
            "but will not be used as an input."
        )

    for name, data in [("Training", train_df), ("Testing", test_df)]:
        if "id" not in data.columns:
            raise ValueError(f"{name} CSV must contain an 'id' column.")

        if data["id"].isna().any() or not data["id"].is_unique:
            raise ValueError(
                f"{name} CSV has missing or duplicate IDs."
            )

    overlap = set(train_df["id"]) & set(test_df["id"])

    if overlap:
        raise ValueError(
            f"Training and testing CSVs share {len(overlap)} IDs. "
            "Check that you selected the correct files."
        )

    X = normalize_features(train_df)
    X_test = normalize_features(test_df)

    y = pd.to_numeric(
        train_df[TARGET],
        errors="raise",
    ).astype(float)

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError(
            "Training energy_usage contains missing or infinite values."
        )

    print(f"\nTraining file: {TRAIN_PATH}")
    print(f"Testing file:  {TEST_PATH}")
    print(f"Training rows: {len(X):,}")
    print(f"Testing rows:  {len(X_test):,}")

    print("\nMissing numeric inputs in the testing CSV:")
    print(X_test[NUMERIC_FEATURES].isna().sum().to_string())

    # ==========================================================
    # 3. SELECT TREE COUNT USING AN INTERNAL TRAINING SPLIT
    # ==========================================================

    # The independent testing CSV is never used for early stopping.
    X_fit_raw, X_stop_raw, y_fit, y_stop = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=123,
        stratify=X["building_id"].fillna("Unknown"),
    )

    internal_categories = learn_categories(X_fit_raw)

    X_fit = apply_categories(X_fit_raw, internal_categories)
    X_stop = apply_categories(X_stop_raw, internal_categories)

    print(f"\nInitial fitting rows: {len(X_fit):,}")
    print(f"Early-stopping rows:  {len(X_stop):,}")
    print("\nSelecting the number of trees...")

    selection_model = lgb.LGBMRegressor(
        n_estimators=5000,
        **MODEL_PARAMETERS,
    )

    # Support both newer and older LightGBM evaluation interfaces.
    if "eval_X" in inspect.signature(selection_model.fit).parameters:
        evaluation_arguments = {
            "eval_X": X_stop,
            "eval_y": y_stop,
        }
    else:
        evaluation_arguments = {
            "eval_set": [(X_stop, y_stop)],
        }

    selection_model.fit(
        X_fit,
        y_fit,
        categorical_feature=CATEGORICAL_FEATURES,
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(period=500),
        ],
        **evaluation_arguments,
    )

    # LightGBM's best_iteration_ is already a tree count.
    best_tree_count = int(selection_model.best_iteration_)

    if best_tree_count < 1:
        raise RuntimeError("Could not determine a valid tree count.")

    stop_predictions = selection_model.predict(
        X_stop,
        num_iteration=best_tree_count,
    )

    stop_rmse = np.sqrt(
        mean_squared_error(y_stop, stop_predictions)
    )
    stop_mae = mean_absolute_error(y_stop, stop_predictions)

    print(f"\nSelected trees: {best_tree_count}")
    print(f"Internal early-stopping RMSE: {stop_rmse:.4f}")
    print(f"Internal early-stopping MAE:  {stop_mae:.4f}")
    print(
        "These are internal selection scores, "
        "not scores on the independent testing CSV."
    )

    # ==========================================================
    # 4. TRAIN A FRESH MODEL ON ALL 8,000 LABELLED ROWS
    # ==========================================================

    final_categories = learn_categories(X)

    X_full = apply_categories(X, final_categories)
    X_test_final = apply_categories(X_test, final_categories)

    final_model = lgb.LGBMRegressor(
        n_estimators=best_tree_count,
        **MODEL_PARAMETERS,
    )

    print(f"\nTraining the final model on all {len(X_full):,} rows...")

    # No validation rows are withheld from this final fit.
    final_model.fit(
        X_full,
        y,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    # ==========================================================
    # 5. PREDICT THE INDEPENDENT TESTING CSV
    # ==========================================================

    # No sorting: predictions retain the original testing row order.
    predictions = np.asarray(
        final_model.predict(X_test_final),
        dtype=float,
    )

    if predictions.shape != (len(test_df),):
        raise RuntimeError("Unexpected prediction dimensions.")

    if not np.isfinite(predictions).all():
        raise RuntimeError("Predictions contain NaN or infinity.")

    # ==========================================================
    # 6. SAVE THE MODEL AND ITS CATEGORY MAPPINGS
    # ==========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "lightgbm_model.pkl"

    # Save the category mappings needed to reproduce preprocessing.
    bundle = {
        "model": final_model,
        "feature_names": FEATURES,
        "category_levels": final_categories,
    }

    joblib.dump(bundle, model_path)

    # Verify predictions after loading the saved model.
    loaded = joblib.load(model_path)

    reloaded_inputs = apply_categories(
        normalize_features(test_df),
        loaded["category_levels"],
    )[loaded["feature_names"]]

    reloaded_predictions = loaded["model"].predict(reloaded_inputs)

    np.testing.assert_allclose(
        predictions,
        reloaded_predictions,
        rtol=1e-10,
        atol=1e-10,
    )

    # ==========================================================
    # 7. SAVE THE PREDICTIONS
    # ==========================================================

    energy_predictions = pd.DataFrame({
        "id": test_df["id"].to_numpy(),
        "energy_usage": predictions,
    })

    energy_path = OUTPUT_DIR / "predicted_energy_usage.csv"
    energy_predictions.to_csv(energy_path, index=False)

    # Single-column format shown in the competition booklet.
    submission = pd.DataFrame({
        "prediction": predictions,
    })

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    # Verify the saved submission.
    saved_submission = pd.read_csv(submission_path)

    if saved_submission.shape != (3000, 1):
        raise RuntimeError("Saved submission has incorrect dimensions.")

    if saved_submission.columns.tolist() != ["prediction"]:
        raise RuntimeError("Saved submission has an incorrect header.")

    np.testing.assert_allclose(
        saved_submission["prediction"].to_numpy(),
        predictions,
    )

    print("\nFIRST 10 TEST PREDICTIONS")
    print(energy_predictions.head(10).round(4).to_string(index=False))

    print("\nFiles saved successfully:")
    print(f"ID + energy usage: {energy_path}")
    print(f"Submission CSV:    {submission_path}")
    print(f"Trained model:     {model_path}")

    print(
        "\nTest RMSE and MAE cannot be calculated without "
        "the testing dataset's actual energy_usage values."
    )


if __name__ == "__main__":
    main()