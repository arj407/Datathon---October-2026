"""CatBoost + LightGBM + engineered features + Optuna + five-fold CV + stacking.

Install: python3 -m pip install catboost lightgbm optuna pandas numpy scikit-learn joblib
Run:     python3 catboost_optuna_energy_v2.py --trials 60 --lgb-trials 40

Pipeline
  1. Reserve 20% of labelled rows as a development holdout (never used to tune).
  2. Optuna + five-fold CV tunes CatBoost, then tunes LightGBM, on the other 80%.
     Early stopping always uses a split INSIDE each fold's training rows.
  3. Each study also chooses whether to use cyclic, interaction and lag/rolling
     features, so cross-validation (not hope) decides what helps.
  4. A non-negative linear meta-model (stacking) is fitted on the two models'
     out-of-fold predictions, then compared with each model on the holdout.
  5. Final members are refitted on all 8000 labelled rows.

Lag / rolling features
  * Built ONLY from input columns (temperature, occupancy, previous_usage),
    using PAST rows of the same building. They never touch energy_usage, so
    they cannot leak labels. Lag-1 of energy_usage already exists as
    `previous_usage`.
  * "Past" means previous rows in id order within a building. They are only
    useful if rows are genuinely sequential; the script prints a diagnostic and
    Optuna switches them off if they do not improve CV.
Optuna's winning CV score is a selection score, not an unbiased final test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.metadata
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore', message='.*eval_set.*')
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')
TARGET = 'energy_usage'
BASE_NUMERIC = ['temperature', 'humidity', 'occupancy', 'previous_usage', 'hour', 'month']
BASE_CATEGORICAL = ['building_id', 'building_type', 'day_of_week']
INPUT_COLUMNS = BASE_NUMERIC + BASE_CATEGORICAL
SEED = 42
FEATURE_VERSION = 'cyclic_interactions_lags_v2'

# Lag / rolling configuration (steps are PREVIOUS ROWS of the same building).
SEQUENCE_SOURCES = ['temperature', 'occupancy', 'previous_usage']
LAGS = [1, 24, 168]
WINDOWS = [3, 6, 24]
ROLL_STATS = ['mean', 'std', 'min', 'max']
SEQ_COLUMNS = (
    [f'{c}_lag{lag}' for c in SEQUENCE_SOURCES for lag in LAGS]
    + [f'{c}_roll{w}_{s}' for c in SEQUENCE_SOURCES for w in WINDOWS for s in ROLL_STATS]
)


# --------------------------------------------------------------------------- data
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


def add_sequence_features(train, test, verbose=True):
    """Lag and rolling-window features from INPUT columns only (past rows of the
    same building, ordered by id). Train and test rows are ordered together so a
    row's neighbours can be either; no target value is ever used."""
    full = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0)], ignore_index=True)
    digits = full['id'].str.extract(r'(\d+)\s*$')[0]
    if digits.notna().all():
        full['_order'] = digits.astype(float)
        how = 'numeric part of id'
    else:
        full['_order'] = np.arange(len(full), dtype=float)
        how = 'file order (train rows, then test rows)'
    full = full.sort_values(['building_id', '_order'], kind='mergesort')
    key = full['building_id'].astype('string').fillna('Unknown')
    g = full.groupby(key, sort=False)

    if verbose and TARGET in full:       # is the data actually sequential?
        previous_target = g[TARGET].shift(1)
        ok = previous_target.notna() & full[TARGET].notna() & full['previous_usage'].notna()
        if ok.sum() > 50:
            exact = float((full.loc[ok, 'previous_usage'] - previous_target[ok]).abs().lt(1e-6).mean())
            corr_prev = float(full.loc[ok, 'previous_usage'].corr(previous_target[ok]))
            same = full[TARGET].notna() & full['previous_usage'].notna()
            corr_same = float(full.loc[same, 'previous_usage'].corr(full.loc[same, TARGET]))
            print(f'Row order used: {how}.')
            print('Sequence check: previous_usage == previous row\'s energy_usage in '
                  f'{exact:.1%} of pairs (corr with prior row {corr_prev:.3f} vs '
                  f'same row {corr_same:.3f}).')
            print('  Near 0% exact matches means rows are NOT consecutive, so lag/rolling '
                  'features are probably noise and CV will switch them off.\n', flush=True)

    new = {}
    for c in SEQUENCE_SOURCES:
        for lag in LAGS:
            new[f'{c}_lag{lag}'] = g[c].shift(lag)
        past = g[c].shift(1)             # excludes the current row
        for w in WINDOWS:
            rolled = past.groupby(key, sort=False).rolling(w, min_periods=1)
            for s in ROLL_STATS:
                new[f'{c}_roll{w}_{s}'] = getattr(rolled, s)().reset_index(level=0, drop=True)
    full = full.assign(**new)

    n = len(train)
    keep = list(train.columns) + SEQ_COLUMNS
    train_out = full.loc[range(n), keep].reset_index(drop=True)
    test_out = full.loc[range(n, len(full)), [c for c in keep if c != TARGET]].reset_index(drop=True)
    return train_out, test_out


def make_features(data, use_cyclic=True, use_interactions=True, use_lags=False):
    """Row-local transformations plus pre-computed lag/rolling columns."""
    x = data[INPUT_COLUMNS + (SEQ_COLUMNS if use_lags else [])].copy()
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
    # Numeric NaNs deliberately remain NaN, including derived features.
    return x, categorical


def split_parameters(parameters):
    p = dict(parameters)
    flags = {
        'use_cyclic': bool(p.pop('use_cyclic', True)),
        'use_interactions': bool(p.pop('use_interactions', True)),
        'use_lags': bool(p.pop('use_lags', False)),
    }
    return p, flags


def lgb_encode(x, cats, categories=None):
    """LightGBM: base categoricals become pandas 'category'; high-cardinality
    interaction strings are dropped (they overfit LightGBM's categorical splits)."""
    x = x.copy()
    use = [c for c in cats if c in BASE_CATEGORICAL]
    x = x.drop(columns=[c for c in cats if c not in BASE_CATEGORICAL])
    if categories is None:
        categories = {c: sorted(x[c].unique()) for c in use}
    for c in use:
        x[c] = pd.Categorical(x[c], categories=categories[c])
    return x, categories


# --------------------------------------------------------------------------- models
def metrics(y, prediction):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
        'MAE': float(mean_absolute_error(y, prediction)),
        'R2': float(r2_score(y, prediction)),
    }


def fit_model(family, model_parameters, x, y, cats, iterations, args,
              seed=SEED, eval_set=None):
    """Fits one model. With eval_set, early-stops and returns the best tree count."""
    if family == 'catboost':
        model = CatBoostRegressor(
            iterations=int(iterations), loss_function='RMSE', eval_metric='RMSE',
            nan_mode='Min', random_seed=seed, task_type='CPU',
            allow_writing_files=False, thread_count=args.threads, **model_parameters,
        )
        if eval_set is None:
            model.fit(x, y, cat_features=cats, verbose=False)
            return model, int(iterations)
        model.fit(x, y, cat_features=cats, eval_set=eval_set,
                  early_stopping_rounds=args.patience, use_best_model=True, verbose=False)
        return model, int(model.get_best_iteration() + 1)

    model = lgb.LGBMRegressor(
        n_estimators=int(iterations), random_state=seed, n_jobs=args.threads,
        verbose=-1, subsample_freq=1, **model_parameters,
    )
    if eval_set is None:
        model.fit(x, y)
        return model, int(iterations)
    model.fit(x, y, eval_set=[eval_set],
              callbacks=[lgb.early_stopping(args.patience, verbose=False)])
    return model, int(model.best_iteration_ or iterations)


def choose_tree_count(family, data, parameters, args, seed=123):
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(data, **flags)
    if family == 'lightgbm':
        x, _ = lgb_encode(x, cats)
    fit_idx, stop_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=seed,
        stratify=data['building_id'].fillna('Unknown'),
    )
    _, trees = fit_model(
        family, model_parameters, x.iloc[fit_idx], data[TARGET].iloc[fit_idx], cats,
        args.max_iterations, args,
        eval_set=(x.iloc[stop_idx], data[TARGET].iloc[stop_idx]),
    )
    return trees


def fit_fixed(family, data, parameters, trees, args):
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(data, **flags)
    categories = None
    if family == 'lightgbm':
        x, categories = lgb_encode(x, cats)
    model, _ = fit_model(family, model_parameters, x, data[TARGET], cats, trees, args)
    return {
        'family': family, 'model': model, 'feature_names': list(x.columns),
        'categorical_features': cats, 'categories': categories,
        'feature_flags': flags, 'parameters': parameters, 'trees': int(trees),
        'feature_version': FEATURE_VERSION,
    }


def predict_bundle(bundle, data):
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('Model and feature-preparation versions do not match.')
    x, cats = make_features(data, **bundle['feature_flags'])
    if bundle['family'] == 'lightgbm':
        x, _ = lgb_encode(x, cats, bundle['categories'])
    return np.asarray(bundle['model'].predict(x[bundle['feature_names']]))


def predict_final(final, data):
    columns = [predict_bundle(final['members'][name], data) for name in final['member_names']]
    return final['intercept'] + np.column_stack(columns) @ final['weights']


def cross_validate(family, parameters, development, folds, args, trial=None, label=None):
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(development, **flags)
    if family == 'lightgbm':
        x, _ = lgb_encode(x, cats)       # categories come from inputs only, no labels
    y = development[TARGET]
    predictions = np.full(len(development), np.nan)
    rows = []
    for fold_number, (training_idx, scoring_idx) in enumerate(folds, 1):
        # Early stopping is INSIDE this fold's training data.
        fit_idx, stopping_idx = train_test_split(
            training_idx, test_size=0.2, random_state=123,
            stratify=development['building_id'].iloc[training_idx].fillna('Unknown'),
        )
        _, trees = fit_model(
            family, model_parameters, x.iloc[fit_idx], y.iloc[fit_idx], cats,
            args.max_iterations, args,
            eval_set=(x.iloc[stopping_idx], y.iloc[stopping_idx]),
        )
        # Refit using the full fold-training portion before scoring.
        model, _ = fit_model(family, model_parameters, x.iloc[training_idx],
                             y.iloc[training_idx], cats, trees, args)
        pred = model.predict(x.iloc[scoring_idx])
        predictions[scoring_idx] = pred
        row = dict(fold=fold_number, trees=trees, **metrics(y.iloc[scoring_idx], pred))
        rows.append(row)
        name = label or (f'{family} trial {trial.number}' if trial is not None else family)
        print(f"  {name}, fold {fold_number}/{len(folds)}: "
              f"RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, trees={trees}", flush=True)
        if trial is not None:
            trial.report(float(np.mean([r['RMSE'] for r in rows])), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not np.isfinite(predictions).all():
        raise RuntimeError('Cross-validation failed to predict every scoring row.')
    return rows, predictions


def run_study(family, prefix, suggest, initial_trials, signature, development, folds, args, n_trials):
    """Resumable Optuna study with five-fold CV inside every trial."""
    def objective(trial):
        parameters = suggest(trial)
        rows, oof = cross_validate(family, parameters, development, folds, args, trial)
        trial.set_user_attr('fold_results', rows)
        trial.set_user_attr('parameters', parameters)
        trial.set_user_attr('oof_metrics', metrics(development[TARGET], oof))
        np.save(args.output / f'{prefix}_trial_{trial.number}_oof.npy', oof)
        return float(np.mean([r['RMSE'] for r in rows]))

    study = optuna.create_study(
        study_name=f'campus_energy_{family}_{signature}', direction='minimize',
        storage='sqlite:///' + (args.output / 'optuna_study.db').resolve().as_posix(),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    if len(study.trials) == 0:
        for trial_params in initial_trials:
            study.enqueue_trial(trial_params)
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, n_trials - finished)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    study.trials_dataframe().to_csv(args.output / f'optuna_trials_{prefix}.csv', index=False)
    return study


def suggest_catboost(trial):
    return dict(
        learning_rate=trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        depth=trial.suggest_int('depth', 3, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
        random_strength=trial.suggest_float('random_strength', 0.0, 3.0),
        rsm=trial.suggest_float('rsm', 0.5, 1.0),     # feature subsampling per split
        # Bernoulli supports subsample. Do not combine subsample
        # with CatBoost's Bayesian-bootstrap bagging_temperature.
        bootstrap_type='Bernoulli',
        subsample=trial.suggest_float('subsample', 0.65, 1.0),
        use_cyclic=trial.suggest_categorical('use_cyclic', [False, True]),
        use_interactions=trial.suggest_categorical('use_interactions', [False, True]),
        use_lags=trial.suggest_categorical('use_lags', [False, True]),
    )


def suggest_lightgbm(trial):
    parameters = dict(
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int('num_leaves', 4, 48, log=True),
        min_child_samples=trial.suggest_int('min_child_samples', 10, 120, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 100.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        cat_smooth=trial.suggest_float('cat_smooth', 5.0, 50.0),
        linear_tree=trial.suggest_categorical('linear_tree', [False, True]),
        use_cyclic=trial.suggest_categorical('use_cyclic', [False, True]),
        use_interactions=trial.suggest_categorical('use_interactions', [False, True]),
        use_lags=trial.suggest_categorical('use_lags', [False, True]),
    )
    if parameters['linear_tree']:
        parameters['linear_lambda'] = trial.suggest_float('linear_lambda', 0.01, 10.0, log=True)
    return parameters


def oof_for_trial(family, prefix, trial, development, folds, args):
    path = args.output / f'{prefix}_trial_{trial.number}_oof.npy'
    if path.exists():
        return np.load(path)
    _, oof = cross_validate(family, trial.user_attrs['parameters'], development, folds, args)
    return oof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DATA_DIR / 'train.csv')
    parser.add_argument('--test', type=Path, default=DATA_DIR / 'test.csv')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'catboost_optuna_results_v2')
    parser.add_argument('--trials', type=int, default=60,
                        help='Total attempted CatBoost trials (resumable), including pruned ones.')
    parser.add_argument('--lgb-trials', type=int, default=40,
                        help='Total attempted LightGBM trials (resumable).')
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--predict-only', type=Path, default=None)
    args = parser.parse_args()
    if min(args.trials, args.lgb_trials) < 3 or min(args.max_iterations, args.patience, args.threads) < 1:
        parser.error('Use at least 3 trials per model and positive iteration/patience/thread counts.')
    args.output.mkdir(parents=True, exist_ok=True)

    train = load_data(args.train, labelled=True)
    test = load_data(args.test)
    if len(test) != 3000:
        raise ValueError(f'Expected 3000 testing rows, found {len(test)}.')
    train, test = add_sequence_features(train, test, verbose=not args.predict_only)

    if args.predict_only:
        final = joblib.load(args.predict_only)
    else:
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
        print(f'Development rows: {len(development)} | Holdout rows: {len(holdout)}', flush=True)
        print('Five-fold tuning; the holdout is excluded from every search.', flush=True)

        signature = hashlib.sha256(
            args.train.read_bytes()
            + json.dumps([FEATURE_VERSION, args.max_iterations, args.patience, SEED]).encode()
        ).hexdigest()[:16]

        # ---------------- CatBoost: baseline, then Optuna ----------------
        baseline_parameters = dict(
            learning_rate=0.03, depth=4, l2_leaf_reg=10,
            random_strength=1.0, use_cyclic=False, use_interactions=False, use_lags=False,
        )
        baseline_rows, baseline_oof = cross_validate(
            'catboost', baseline_parameters, development, folds, args, label='cb baseline',
        )
        baseline_cv = float(np.mean([r['RMSE'] for r in baseline_rows]))
        pd.DataFrame(baseline_rows).to_csv(args.output / 'baseline_folds.csv', index=False)

        cb_start = dict(learning_rate=0.03, depth=4, l2_leaf_reg=10.0,
                        random_strength=1.0, rsm=1.0, subsample=1.0)
        cb_study = run_study(
            'catboost', 'cb', suggest_catboost,
            [dict(cb_start, use_cyclic=False, use_interactions=False, use_lags=False),
             dict(cb_start, use_cyclic=True, use_interactions=True, use_lags=False),
             dict(cb_start, use_cyclic=True, use_interactions=True, use_lags=True)],
            signature, development, folds, args, args.trials,
        )
        cb_best = cb_study.best_trial
        cb_tuned = cb_best.user_attrs['parameters']

        # ---------------- LightGBM: Optuna ----------------
        lgb_start = dict(learning_rate=0.03, num_leaves=15, min_child_samples=30,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
                         reg_alpha=0.1, cat_smooth=20.0, linear_tree=False)
        lgb_study = run_study(
            'lightgbm', 'lgb', suggest_lightgbm,
            [dict(lgb_start, use_cyclic=False, use_interactions=False, use_lags=False),
             dict(lgb_start, use_cyclic=True, use_interactions=True, use_lags=True)],
            signature, development, folds, args, args.lgb_trials,
        )
        lgb_best = lgb_study.best_trial
        lgb_tuned = lgb_best.user_attrs['parameters']

        # Do not force a more complicated CatBoost when CV prefers the baseline.
        use_tuned = cb_best.value < baseline_cv
        cb_chosen = cb_tuned if use_tuned else baseline_parameters
        cb_name = 'CatBoost Optuna' if use_tuned else 'CatBoost baseline'
        print(f'\nCatBoost baseline mean CV RMSE: {baseline_cv:.4f}', flush=True)
        print(f'CatBoost best Optuna mean CV RMSE: {cb_best.value:.4f}', flush=True)
        print(f'LightGBM best Optuna mean CV RMSE: {lgb_best.value:.4f}', flush=True)
        print(f'CatBoost chosen by CV: {cb_name}', flush=True)
        print('CatBoost parameters:', json.dumps(cb_chosen, indent=2), flush=True)
        print('LightGBM parameters:', json.dumps(lgb_tuned, indent=2), flush=True)
        pd.DataFrame(cb_best.user_attrs['fold_results']).to_csv(
            args.output / 'best_catboost_trial_folds.csv', index=False)
        pd.DataFrame(lgb_best.user_attrs['fold_results']).to_csv(
            args.output / 'best_lightgbm_trial_folds.csv', index=False)

        # ---------------- Stacking meta-model on out-of-fold predictions ----------------
        cb_oof = oof_for_trial('catboost', 'cb', cb_best, development, folds, args) if use_tuned else baseline_oof
        lgb_oof = oof_for_trial('lightgbm', 'lgb', lgb_best, development, folds, args)
        meta = LinearRegression(positive=True)       # non-negative weights + intercept
        meta.fit(np.column_stack([cb_oof, lgb_oof]), development[TARGET])
        stack_oof = meta.predict(np.column_stack([cb_oof, lgb_oof]))
        print('\nOut-of-fold RMSE on development rows:', flush=True)
        print(f"  {cb_name}: {metrics(development[TARGET], cb_oof)['RMSE']:.4f}")
        print(f"  LightGBM Optuna: {metrics(development[TARGET], lgb_oof)['RMSE']:.4f}")
        print(f"  Stack: {metrics(development[TARGET], stack_oof)['RMSE']:.4f} "
              '(weights fitted on these rows, so slightly optimistic)')
        print(f'  Meta weights [CatBoost, LightGBM] = {np.round(meta.coef_, 3).tolist()}, '
              f'intercept = {meta.intercept_:.3f}', flush=True)

        # ---------------- Holdout comparison (selects nothing) ----------------
        holdout_rows = []
        holdout_predictions = pd.DataFrame({'id': holdout.id, 'actual': holdout[TARGET]})
        candidates = [('CatBoost baseline', 'catboost', baseline_parameters),
                      ('CatBoost Optuna', 'catboost', cb_tuned),
                      ('LightGBM Optuna', 'lightgbm', lgb_tuned)]
        for name, family, parameters in candidates:
            trees = choose_tree_count(family, development, parameters, args)
            model = fit_fixed(family, development, parameters, trees, args)
            pred = predict_bundle(model, holdout)
            holdout_rows.append(dict(model=name, trees=trees, **metrics(holdout[TARGET], pred)))
            holdout_predictions[name] = pred
        stack_pred = meta.predict(np.column_stack(
            [holdout_predictions[cb_name], holdout_predictions['LightGBM Optuna']]))
        holdout_rows.append(dict(model='Stack (weights from dev OOF)', trees=np.nan,
                                 **metrics(holdout[TARGET], stack_pred)))
        holdout_predictions['Stack'] = stack_pred
        holdout_results = pd.DataFrame(holdout_rows)
        print('\nHOLDOUT RESULTS — NOT PORTAL SCORES', flush=True)
        print(holdout_results.round(4).to_string(index=False), flush=True)
        holdout_results.to_csv(args.output / 'holdout_metrics.csv', index=False)
        holdout_predictions.to_csv(args.output / 'holdout_predictions.csv', index=False)

        # ---------------- Final members on all 8000 labelled rows ----------------
        members = {}
        for name, family, parameters in [('catboost', 'catboost', cb_chosen),
                                         ('lightgbm', 'lightgbm', lgb_tuned)]:
            trees = choose_tree_count(family, train, parameters, args)
            print(f'Final {name} tree count: {trees}. Training on all 8000 rows...', flush=True)
            members[name] = fit_fixed(family, train, parameters, trees, args)
        final = {
            'members': members, 'member_names': ['catboost', 'lightgbm'],
            'weights': np.asarray(meta.coef_, dtype=float), 'intercept': float(meta.intercept_),
            'feature_version': FEATURE_VERSION,
            'package_versions': {
                name: importlib.metadata.version(name)
                for name in ['catboost', 'lightgbm', 'optuna', 'pandas', 'numpy',
                             'scikit-learn', 'joblib']
            },
            'training_data_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        }
        joblib.dump(final, args.output / 'model.pkl')
        members['catboost']['model'].save_model(str(args.output / 'model_catboost.cbm'))
        members['lightgbm']['model'].booster_.save_model(str(args.output / 'model_lightgbm.txt'))
        (args.output / 'best_parameters.json').write_text(json.dumps({
            'catboost_chosen_by_cv': cb_name, 'catboost_parameters': cb_chosen,
            'lightgbm_parameters': lgb_tuned,
            'final_trees': {k: v['trees'] for k, v in members.items()},
            'meta_weights': {'catboost': float(meta.coef_[0]), 'lightgbm': float(meta.coef_[1]),
                             'intercept': float(meta.intercept_)},
            'catboost_baseline_mean_cv_rmse': baseline_cv,
            'catboost_best_optuna_mean_cv_rmse': cb_best.value,
            'lightgbm_best_optuna_mean_cv_rmse': lgb_best.value,
            'note': 'Winning CV is a tuning score. Holdout is excluded from the search; '
                    'prior manual experiments may already have examined these rows. '
                    'Neither result establishes competition test RMSE.',
        }, indent=2))
        (args.output / 'requirements.txt').write_text('\n'.join(
            f'{k}=={v}' for k, v in final['package_versions'].items()
        ) + '\n')
        reloaded = joblib.load(args.output / 'model.pkl')
        np.testing.assert_allclose(predict_final(final, test), predict_final(reloaded, test),
                                   rtol=1e-10, atol=1e-10)

    prediction = predict_final(final, test)
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
