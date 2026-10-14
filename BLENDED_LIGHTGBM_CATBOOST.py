from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================
# 1. LOCATE THE TWO SAVED PREDICTION FILES
# ==============================================================

BASE_DIR = Path(__file__).resolve().parent

CATBOOST_PATH = (
    BASE_DIR
    / "outputs"
    / "catboost_submission"
    / "predicted_energy_usage.csv"
)

LIGHTGBM_PATH = (
    BASE_DIR
    / "outputs"
    / "lightgbm_submission"
    / "predicted_energy_usage.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "blended_submission"

# Start with an equal-weight average.
CATBOOST_WEIGHT = 0.25
LIGHTGBM_WEIGHT = 0.75


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find: {path}\n"
            "Run the corresponding model script first, "
            "or update this file path."
        )

    df = pd.read_csv(path, dtype={"id": str})

    required = {"id", "energy_usage"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    if len(df) != 3000:
        raise ValueError(
            f"Expected 3,000 predictions in {path}, "
            f"found {len(df):,}."
        )

    if df["id"].isna().any() or not df["id"].is_unique:
        raise ValueError(
            f"Missing or duplicate IDs found in {path}."
        )

    df["energy_usage"] = pd.to_numeric(
        df["energy_usage"],
        errors="raise",
    )

    if not np.isfinite(df["energy_usage"].to_numpy()).all():
        raise ValueError(
            f"Missing or infinite predictions found in {path}."
        )

    return df[["id", "energy_usage"]].copy()


def main():
    # ==========================================================
    # 2. VALIDATE THE WEIGHTS
    # ==========================================================

    weights = np.array(
        [CATBOOST_WEIGHT, LIGHTGBM_WEIGHT],
        dtype=float,
    )

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "Weights must be non-negative, finite, and sum to 1."
        )

    # ==========================================================
    # 3. LOAD AND MATCH PREDICTIONS BY ID
    # ==========================================================

    catboost_df = load_predictions(CATBOOST_PATH)
    lightgbm_df = load_predictions(LIGHTGBM_PATH)

    if set(catboost_df["id"]) != set(lightgbm_df["id"]):
        raise ValueError(
            "The prediction files contain different IDs. "
            "Both models must predict the same testing dataset."
        )

    # Preserve CatBoost's saved row order, which should be
    # the original testing CSV order.
    test_ids = catboost_df["id"].to_numpy()

    catboost_predictions = (
        catboost_df["energy_usage"].to_numpy()
    )

    # Reorder LightGBM predictions to match those IDs.
    lightgbm_predictions = (
        lightgbm_df
        .set_index("id")
        .loc[test_ids, "energy_usage"]
        .to_numpy()
    )

    # ==========================================================
    # 4. BLEND THE PREDICTIONS
    # ==========================================================

    blended_predictions = (
        CATBOOST_WEIGHT * catboost_predictions
        + LIGHTGBM_WEIGHT * lightgbm_predictions
    )

    if not np.isfinite(blended_predictions).all():
        raise RuntimeError("Blended predictions are not all finite.")

    # ==========================================================
    # 5. SAVE OUTPUT FILES
    # ==========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed file for checking each model's contribution.
    comparison = pd.DataFrame({
        "id": test_ids,
        "catboost_prediction": catboost_predictions,
        "lightgbm_prediction": lightgbm_predictions,
        "blended_prediction": blended_predictions,
    })

    comparison.to_csv(
        OUTPUT_DIR / "prediction_comparison.csv",
        index=False,
    )

    # Human-readable predictions mapped to IDs.
    energy_predictions = pd.DataFrame({
        "id": test_ids,
        "energy_usage": blended_predictions,
    })

    energy_predictions.to_csv(
        OUTPUT_DIR / "predicted_energy_usage.csv",
        index=False,
    )

    # Submission format matching the booklet's example.
    submission = pd.DataFrame({
        "prediction": blended_predictions,
    })

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    # Verify the saved file.
    saved = pd.read_csv(submission_path)

    if saved.shape != (3000, 1):
        raise RuntimeError("Saved submission has incorrect dimensions.")

    np.testing.assert_allclose(
        saved["prediction"].to_numpy(),
        blended_predictions,
    )

    print(f"\nCatBoost weight: {CATBOOST_WEIGHT:.0%}")
    print(f"LightGBM weight: {LIGHTGBM_WEIGHT:.0%}")

    print("\nFIRST 10 BLENDED PREDICTIONS")
    print(comparison.head(10).round(4).to_string(index=False))

    print(f"\nSaved blended submission: {submission_path}")
    print(f"All output files: {OUTPUT_DIR}")

    print(
        "\nRMSE cannot be calculated without actual test energy usage. "
        "Blending is not guaranteed to improve either model."
    )


if __name__ == "__main__":
    main()