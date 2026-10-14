"""CatBoost + engineered features + Optuna + five-fold validation.

Install: python3 -m pip install catboost optuna pandas numpy scikit-learn joblib
Run:     python3 catboost_optuna_energy.py --trials 50

The script reserves 20% of labelled rows for a development holdout.
Optuna uses five folds on the remaining 80%; early stopping uses a further
split INSIDE each fold's training rows. The scoring fold is never used
for early stopping. Finally the chosen model is fitted on all 8000 labels.
Optuna's winning CV score is a selection score, not an unbiased final test.
"""

from pathlib import Path
import argparse
import hashlib
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')
TARGET = 'energy_usage'
BASE_NUMERIC = ['temperature', 'humidity', 'occupancy', 'previous_usage', 'hour', 'month']
BASE_CATEGORICAL = ['building_id', 'building_type', 'day_of_week']
INPUT_COLUMNS = BASE_NUMERIC + BASE_CATEGORICAL
SEED = 42
FEATURE_VERSION = 'cyclic_interactions_v1'


def load_data(path, labelled=False):
    data = pd.read_csv(path, dtype={'id': str})
    required = ['id'] + INPUT_COLUMNS + ([TARGET] if labelled else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    if data.id.isna().any() or not data.id.is_unique:
        raise ValueError(f'{path}: IDs must be present and unique.')
    for c in BASE_NUMERIC + ([TARGET] if labelled else []):
        data[c] = pd.to_numeric(data[c], errors='raise')
        if np.isinf(data[c].to_numpy(dtype=float)).any():
            raise ValueError(f'{path}: infinite values in {c}.')
    if labelled and data[TARGET].isna().any():
        raise ValueError('Training targets cannot be missing.')
    for c, lower, upper in [('hour', 0, 23), ('month', 1, 12)]:
        valid = data[c].dropna()
        if not valid.between(lower, upper).all() or not (valid % 1 == 0).all():
            raise ValueError(f'Invalid {c} values.')
    return data


def make_features(data, use_cyclic=True, use_interactions=True):
    """Row-local transformations only: no target or cross-row statistics."""
    x = data[INPUT_COLUMNS].copy()
    categorical = list(BASE_CATEGORICAL)
    for c in categorical:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)

    if use_cyclic:
        x['hour_sin'] = np.sin(2 * np.pi * x['hour'] / 24)
        x['hour_cos'] = np.cos(2 * np.pi * x['hour'] / 24)
        x['month_sin'] = np.sin(2 * np.pi * (x['month'] - 1) / 12)
        x['month_cos'] = np.cos(2 * np.pi * (x['month'] - 1) / 12)

    if use_interactions:
        hour_token = x['hour'].astype('Int64').astype('string').fillna('Unknown')
        x['building_type_hour'] = x['building_type'] + '__' + hour_token
        x['building_type_weekday'] = x['building_type'] + '__' + x['day_of_week']
        categorical += ['building_type_hour', 'building_type_weekday']
        x['is_weekend'] = x['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        x['temperature_x_occupancy'] = x['temperature'] * x['occupancy']
        x['temperature_x_humidity'] = x['temperature'] * x['humidity']
    # Numeric NaNs deliberately remain NaN, including derived interactions.
    return x, categorical


def split_parameters(parameters):
    p = dict(parameters)
    flags = {
        'use_cyclic': bool(p.pop('use_cyclic', True)),
        'use_interactions': bool(p.pop('use_interactions', True)),
    }
    return p, flags


def new_model(parameters, iterations, threads, seed=SEED):
    return CatBoostRegressor(
        iterations=int(iterations), loss_function='RMSE', eval_metric='RMSE',
        nan_mode='Min', random_seed=seed, task_type='CPU',
        allow_writing_files=False, thread_count=threads,
        **parameters,
    )


def metrics(y, prediction):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
        'MAE': float(mean_absolute_error(y, prediction)),
        'R2': float(r2_score(y, prediction)),
    }


def choose_tree_count(data, parameters, args, seed=123):
    model_parameters, feature_flags = split_parameters(parameters)
    x, cats = make_features(data, **feature_flags)
    fit_idx, stop_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=seed,
        stratify=data['building_id'].fillna('Unknown'),
    )
    model = new_model(model_parameters, args.max_iterations, args.threads)
    model.fit(
        x.iloc[fit_idx], data[TARGET].iloc[fit_idx], cat_features=cats,
        eval_set=(x.iloc[stop_idx], data[TARGET].iloc[stop_idx]),
        early_stopping_rounds=args.patience, use_best_model=True, verbose=False,
    )
    return int(model.get_best_iteration() + 1)


def fit_fixed(data, parameters, trees, args):
    model_parameters, feature_flags = split_parameters(parameters)
    x, cats = make_features(data, **feature_flags)
    model = new_model(model_parameters, trees, args.threads)
    model.fit(x, data[TARGET], cat_features=cats, verbose=False)
    return {
        'model': model,
        'feature_names': list(x.columns),
        'categorical_features': cats,
        'feature_flags': feature_flags,
        'parameters': parameters,
        'trees': int(trees),
        'feature_version': FEATURE_VERSION,
    }


def predict_bundle(bundle, data):
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('Model and feature-preparation versions do not match.')
    x, _ = make_features(data, **bundle['feature_flags'])
    return np.asarray(bundle['model'].predict(x[bundle['feature_names']]))


def cross_validate(parameters, development, folds, args, trial=None):
    model_parameters, feature_flags = split_parameters(parameters)
    x, cats = make_features(development, **feature_flags)
    y = development[TARGET]
    predictions = np.full(len(development), np.nan)
    rows = []
    for fold_number, (training_idx, scoring_idx) in enumerate(folds, 1):
        # Early stopping is INSIDE this fold's training data.
        fit_idx, stopping_idx = train_test_split(
            training_idx, test_size=0.2, random_state=123,
            stratify=development['building_id'].iloc[training_idx].fillna('Unknown'),
        )
        selector = new_model(model_parameters, args.max_iterations, args.threads)
        selector.fit(
            x.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats,
            eval_set=(x.iloc[stopping_idx], y.iloc[stopping_idx]),
            early_stopping_rounds=args.patience, use_best_model=True, verbose=False,
        )
        trees = int(selector.get_best_iteration() + 1)
        # Refit using the full fold-training portion before scoring.
        model = new_model(model_parameters, trees, args.threads)
        model.fit(x.iloc[training_idx], y.iloc[training_idx], cat_features=cats,
                  verbose=False)
        pred = model.predict(x.iloc[scoring_idx])
        predictions[scoring_idx] = pred
        row = dict(fold=fold_number, trees=trees, **metrics(y.iloc[scoring_idx], pred))
        rows.append(row)
        label = 'baseline' if trial is None else f'trial {trial.number}'
        print(f"  {label}, fold {fold_number}/{len(folds)}: "
              f"RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, trees={trees}", flush=True)
        if trial is not None:
            partial_score = float(np.mean([r['RMSE'] for r in rows]))
            trial.report(partial_score, step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not np.isfinite(predictions).all():
        raise RuntimeError('Cross-validation failed to predict every scoring row.')
    return rows, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DATA_DIR / 'train.csv')
    parser.add_argument('--test', type=Path, default=DATA_DIR / 'test.csv')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'catboost_optuna_results')
    parser.add_argument('--trials', type=int, default=50,
                        help='Total attempted trials in the resumable study, including pruned ones.')
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--predict-only', type=Path, default=None)
    args = parser.parse_args()
    if args.trials < 2 or args.max_iterations < 1 or args.patience < 1 or args.threads < 1:
        parser.error('Use at least 2 trials and positive iteration/patience/thread counts.')
    args.output.mkdir(parents=True, exist_ok=True)
    test = load_data(args.test)
    if len(test) != 3000:
        raise ValueError(f'Expected 3000 testing rows, found {len(test)}.')

    if args.predict_only:
        final = joblib.load(args.predict_only)
    else:
        train = load_data(args.train, labelled=True)
        if len(train) != 8000:
            raise ValueError(f'Expected 8000 training rows, found {len(train)}.')
        if set(train.id) & set(test.id):
            raise ValueError('Training and testing IDs overlap.')
        development, holdout = train_test_split(
            train, test_size=0.2, random_state=SEED,
            stratify=train.building_id.fillna('Unknown'),
        )
        development = development.reset_index(drop=True)
        holdout = holdout.reset_index(drop=True)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            development, development.building_id.fillna('Unknown'),
        ))
        print(f'Development rows: {len(development)} | Holdout rows: {len(holdout)}',
              flush=True)
        print('Five-fold tuning; the holdout is excluded from the search.', flush=True)

        # This matches the previously tested shallow CatBoost configuration.
        baseline_parameters = dict(
            learning_rate=0.03, depth=4, l2_leaf_reg=10,
            random_strength=1.0, use_cyclic=False, use_interactions=False,
        )
        baseline_rows, baseline_oof = cross_validate(
            baseline_parameters, development, folds, args,
        )
        baseline_cv = float(np.mean([r['RMSE'] for r in baseline_rows]))
        pd.DataFrame(baseline_rows).to_csv(args.output / 'baseline_folds.csv', index=False)

        def objective(trial):
            parameters = dict(
                learning_rate=trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
                depth=trial.suggest_int('depth', 3, 8),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
                random_strength=trial.suggest_float('random_strength', 0.0, 3.0),
                # Bernoulli supports subsample. Do not combine subsample
                # with CatBoost's Bayesian-bootstrap bagging_temperature.
                bootstrap_type='Bernoulli',
                subsample=trial.suggest_float('subsample', 0.65, 1.0),
                # Test whether the extra features actually improve CV.
                use_cyclic=trial.suggest_categorical('use_cyclic', [False, True]),
                use_interactions=trial.suggest_categorical('use_interactions', [False, True]),
            )
            rows, oof = cross_validate(parameters, development, folds, args, trial)
            trial.set_user_attr('fold_results', rows)
            trial.set_user_attr('parameters', parameters)
            trial.set_user_attr('oof_metrics', metrics(development[TARGET], oof))
            np.save(args.output / f'trial_{trial.number}_oof.npy', oof)
            return float(np.mean([r['RMSE'] for r in rows]))

        # Fingerprint prevents accidental reuse of a study from different data
        # or a changed early-stopping budget. Keep the SQLite file to resume.
        signature = hashlib.sha256(
            args.train.read_bytes()
            + json.dumps([FEATURE_VERSION, args.max_iterations, args.patience, SEED]).encode()
        ).hexdigest()[:16]
        study = optuna.create_study(
            study_name=f'campus_energy_{signature}', direction='minimize',
            storage='sqlite:///' + (args.output / 'optuna_study.db').resolve().as_posix(),
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        if len(study.trials) == 0:
            initial = dict(learning_rate=0.03, depth=4, l2_leaf_reg=10.0,
                           random_strength=1.0, subsample=1.0)
            study.enqueue_trial(dict(initial, use_cyclic=False, use_interactions=False))
            study.enqueue_trial(dict(initial, use_cyclic=True, use_interactions=True))
        finished = sum(t.state.is_finished() for t in study.trials)
        remaining = max(0, args.trials - finished)
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
        best = study.best_trial
        tuned_parameters = best.user_attrs['parameters']
        study.trials_dataframe().to_csv(args.output / 'optuna_trials.csv', index=False)
        pd.DataFrame(best.user_attrs['fold_results']).to_csv(
            args.output / 'best_trial_folds.csv', index=False,
        )
        # Do not force a more complicated model when CV prefers the baseline.
        use_tuned = best.value < baseline_cv
        chosen_parameters = tuned_parameters if use_tuned else baseline_parameters
        chosen_name = 'Optuna-selected CatBoost' if use_tuned else 'Baseline retained'
        print(f'\nBaseline mean CV RMSE: {baseline_cv:.4f}', flush=True)
        print(f'Best Optuna mean CV RMSE: {best.value:.4f}', flush=True)
        print(f'Chosen by CV: {chosen_name}', flush=True)
        print(json.dumps(chosen_parameters, indent=2), flush=True)

        # Holdout comparison does NOT select the candidate or its parameters.
        holdout_rows = []
        holdout_predictions = pd.DataFrame({'id': holdout.id, 'actual': holdout[TARGET]})
        for name, parameters in [('Baseline', baseline_parameters),
                                 ('Best Optuna candidate', tuned_parameters)]:
            trees = choose_tree_count(development, parameters, args)
            model = fit_fixed(development, parameters, trees, args)
            pred = predict_bundle(model, holdout)
            holdout_rows.append(dict(model=name, trees=trees, **metrics(holdout[TARGET], pred)))
            holdout_predictions[name] = pred
        holdout_results = pd.DataFrame(holdout_rows)
        print('\nHOLDOUT RESULTS — NOT PORTAL SCORES', flush=True)
        print(holdout_results.round(4).to_string(index=False), flush=True)
        holdout_results.to_csv(args.output / 'holdout_metrics.csv', index=False)
        holdout_predictions.to_csv(args.output / 'holdout_predictions.csv', index=False)

        # All 8000 labelled rows can now contribute to the final model.
        final_trees = choose_tree_count(train, chosen_parameters, args)
        print(f'\nFinal tree count: {final_trees}. Training on all 8000 rows...', flush=True)
        final = fit_fixed(train, chosen_parameters, final_trees, args)
        final['package_versions'] = {
            name: importlib.metadata.version(name)
            for name in ['catboost', 'optuna', 'pandas', 'numpy', 'scikit-learn', 'joblib']
        }
        final['training_data_sha256'] = hashlib.sha256(args.train.read_bytes()).hexdigest()
        joblib.dump(final, args.output / 'model.pkl')
        final['model'].save_model(str(args.output / 'model.cbm'))
        (args.output / 'best_parameters.json').write_text(json.dumps({
            'chosen_by_cv': chosen_name, 'parameters': chosen_parameters,
            'final_trees': final_trees, 'baseline_mean_cv_rmse': baseline_cv,
            'best_optuna_mean_cv_rmse': best.value,
            'note': 'Winning CV is a tuning score. Holdout is excluded from this search; '
                    'prior manual experiments may already have examined these rows. '
                    'Neither result establishes competition test RMSE.',
        }, indent=2))
        (args.output / 'requirements.txt').write_text('\n'.join(
            f'{k}=={v}' for k, v in final['package_versions'].items()
        ) + '\n')
        reloaded = joblib.load(args.output / 'model.pkl')
        np.testing.assert_allclose(predict_bundle(final, test), predict_bundle(reloaded, test),
                                   rtol=1e-10, atol=1e-10)

    prediction = predict_bundle(final, test)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise RuntimeError('Invalid predictions.')
    pd.DataFrame({'id': test.id.to_numpy(), TARGET: prediction}).to_csv(
        args.output / 'predicted_energy_usage.csv', index=False,
    )
    pd.DataFrame({'prediction': prediction}).to_csv(args.output / 'submission.csv', index=False)
    saved = pd.read_csv(args.output / 'submission.csv')
    assert saved.shape == (3000, 1) and saved.columns.tolist() == ['prediction']
    np.testing.assert_allclose(saved.prediction, prediction)
    print('\nSaved:', args.output / 'submission.csv', flush=True)
    print('Actual test RMSE requires test labels or a competition portal score.', flush=True)


if __name__ == '__main__':
    main()
