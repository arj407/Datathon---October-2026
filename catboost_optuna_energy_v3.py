"""CatBoost + LightGBM + spline-Ridge, Optuna-tuned, stacked. Version 3.

Install: python3 -m pip install catboost lightgbm optuna pandas numpy scikit-learn joblib
Run:     python3 catboost_optuna_energy_v3.py --trials 60 --lgb-trials 40

What is new compared with v2
  * Feature groups that cross-validation switches on or off:
      time_encoding  none | cyclic | fourier (3 harmonics of hour, weekday, month)
      use_interactions  building_type x hour / weekday, weekend flag, products
      use_context    per-building / per-building-hour statistics of the INPUT
                     columns and each row's deviation from them (no labels)
      use_lags       lag, rolling and expanding statistics over earlier rows of
                     the same building (INPUT columns only; no labels)
  * log_target: train on log1p(energy_usage), predict expm1; scored on the raw
    scale, so it is kept only if it really lowers RMSE.
  * huber_mult: optional robust (Huber) training loss, for noisy labels.
  * Validation scheme (--cv auto|kfold|time). `auto` uses time-ordered
    TimeSeriesSplit only when test ids all come AFTER the training ids;
    otherwise stratified K-fold, which mimics a randomly split test set.
  * Three diverse members: tuned CatBoost, tuned LightGBM, spline-Ridge.
    Tree members are bagged over several seeds in their final fits.
  * Stacking: CV on the out-of-fold predictions picks between a non-negative
    linear blend, a plain average and a small non-linear CatBoost meta-model.

Optuna's winning CV score is a selection score, not an unbiased final test.
"""
from pathlib import Path
from functools import partial
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings('ignore', message='.*eval_set.*')
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')
TARGET = 'energy_usage'
BASE_NUMERIC = ['temperature', 'humidity', 'occupancy', 'previous_usage', 'hour', 'month']
BASE_CATEGORICAL = ['building_id', 'building_type', 'day_of_week']
INPUT_COLUMNS = BASE_NUMERIC + BASE_CATEGORICAL
SEED = 42
FEATURE_VERSION = 'v3_fourier_context_lags_logtarget'
FEATURE_FLAGS = ['time_encoding', 'use_interactions', 'use_context', 'use_lags']
DAY_INDEX = {d: i for i, d in enumerate(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}

# Lag / rolling / expanding configuration (steps are PREVIOUS ROWS of the same building).
SEQUENCE_SOURCES = ['temperature', 'occupancy', 'previous_usage']
LAGS = [1, 24, 168]
WINDOWS = [3, 6, 24]
ROLL_STATS = ['mean', 'std', 'min', 'max']
SEQ_COLUMNS = (
    [f'{c}_lag{lag}' for c in SEQUENCE_SOURCES for lag in LAGS]
    + [f'{c}_roll{w}_{s}' for c in SEQUENCE_SOURCES for w in WINDOWS for s in ROLL_STATS]
    + [f'{c}_exp_{s}' for c in SEQUENCE_SOURCES for s in ['mean', 'std']]
)
CONTEXT_COLUMNS = [
    'bld_prev_mean', 'bld_prev_std', 'bld_occ_mean', 'bld_temp_mean', 'bld_hum_mean',
    'bld_hour_prev_mean', 'bld_dow_prev_mean', 'bt_hour_prev_mean',
    'prev_dev', 'prev_z', 'prev_rel', 'prev_dev_bld_hour', 'prev_dev_bt_hour',
    'occ_dev', 'temp_dev',
]
RIDGE_SPLINE = ['temperature', 'humidity', 'occupancy', 'previous_usage']


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


def _group_mean(values, keys, min_count=1):
    grouped = values.groupby(keys)
    mean = grouped.transform('mean')
    return mean if min_count <= 1 else mean.where(grouped.transform('count') >= min_count)


def add_derived_columns(train, test, verbose=True):
    """Context, lag, rolling and expanding features computed from INPUT columns of
    train and test together. No target value is ever used. Also returns a
    `row_order` column (numeric part of id, or file order) used by time-aware CV."""
    full = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0)], ignore_index=True)
    digits = full['id'].str.extract(r'(\d+)\s*$')[0]
    if digits.notna().all():
        full['row_order'] = digits.astype(float)
        order_source, how = 'id', 'numeric part of id'
    else:
        full['row_order'] = np.arange(len(full), dtype=float)
        order_source, how = 'file', 'file order (train rows, then test rows)'
    full = full.sort_values(['building_id', 'row_order'], kind='mergesort')
    bk = full['building_id'].astype('string').fillna('Unknown')
    hk = full['hour'].astype('Int64').astype('string').fillna('Unknown')
    dk = full['day_of_week'].astype('string').fillna('Unknown')
    tk = full['building_type'].astype('string').fillna('Unknown')
    g = full.groupby(bk, sort=False)

    if verbose and TARGET in full:       # is the data actually sequential?
        previous_target = g[TARGET].shift(1)
        ok = previous_target.notna() & full[TARGET].notna() & full['previous_usage'].notna()
        if ok.sum() > 50:
            exact = float((full.loc[ok, 'previous_usage'] - previous_target[ok]).abs().lt(1e-6).mean())
            corr_prev = float(full.loc[ok, 'previous_usage'].corr(previous_target[ok]))
            same = full[TARGET].notna() & full['previous_usage'].notna()
            corr_same = float(full.loc[same, 'previous_usage'].corr(full.loc[same, TARGET]))
            print(f'Row order used: {how}.')
            print("Sequence check: previous_usage == previous row's energy_usage in "
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
            rolled = past.groupby(bk, sort=False).rolling(w, min_periods=1)
            for s in ROLL_STATS:
                new[f'{c}_roll{w}_{s}'] = getattr(rolled, s)().reset_index(level=0, drop=True)
        expanding = past.groupby(bk, sort=False).expanding(min_periods=1)
        for s in ['mean', 'std']:
            new[f'{c}_exp_{s}'] = getattr(expanding, s)().reset_index(level=0, drop=True)

    prev, occ, temp = full['previous_usage'], full['occupancy'], full['temperature']
    ctx = {}
    ctx['bld_prev_mean'] = _group_mean(prev, bk)
    ctx['bld_prev_std'] = prev.groupby(bk).transform('std')
    ctx['bld_occ_mean'] = _group_mean(occ, bk)
    ctx['bld_temp_mean'] = _group_mean(temp, bk)
    ctx['bld_hum_mean'] = _group_mean(full['humidity'], bk)
    ctx['bld_hour_prev_mean'] = _group_mean(prev, [bk, hk], min_count=3)
    ctx['bld_dow_prev_mean'] = _group_mean(prev, [bk, dk], min_count=3)
    ctx['bt_hour_prev_mean'] = _group_mean(prev, [tk, hk])
    ctx['prev_dev'] = prev - ctx['bld_prev_mean']
    ctx['prev_z'] = ctx['prev_dev'] / (ctx['bld_prev_std'] + 1e-6)
    ctx['prev_rel'] = prev / (ctx['bld_prev_mean'].abs() + 1e-6)
    ctx['prev_dev_bld_hour'] = prev - ctx['bld_hour_prev_mean']
    ctx['prev_dev_bt_hour'] = prev - ctx['bt_hour_prev_mean']
    ctx['occ_dev'] = occ - ctx['bld_occ_mean']
    ctx['temp_dev'] = temp - ctx['bld_temp_mean']
    new.update(ctx)
    full = full.assign(**new)

    n = len(train)
    keep = list(train.columns) + SEQ_COLUMNS + CONTEXT_COLUMNS + ['row_order']
    train_out = full.loc[range(n), keep].reset_index(drop=True)
    test_out = full.loc[range(n, len(full)), [c for c in keep if c != TARGET]].reset_index(drop=True)
    return train_out, test_out, order_source


def _add_harmonics(x, name, values, period, order):
    for k in range(1, order + 1):
        x[f'{name}_sin{k}'] = np.sin(2 * np.pi * k * values / period)
        x[f'{name}_cos{k}'] = np.cos(2 * np.pi * k * values / period)


def make_features(data, time_encoding='cyclic', use_interactions=True,
                  use_context=False, use_lags=False):
    """Row-local transformations plus pre-computed context / lag columns."""
    cols = (INPUT_COLUMNS + (CONTEXT_COLUMNS if use_context else [])
            + (SEQ_COLUMNS if use_lags else []))
    x = data[cols].copy()
    categorical = list(BASE_CATEGORICAL)
    for c in categorical:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)

    if time_encoding == 'cyclic':
        _add_harmonics(x, 'hour', x['hour'], 24, 1)
        _add_harmonics(x, 'month', x['month'] - 1, 12, 1)
    elif time_encoding == 'fourier':
        weekday = x['day_of_week'].str.strip().str.title().map(DAY_INDEX).astype(float)
        _add_harmonics(x, 'hour', x['hour'], 24, 3)
        _add_harmonics(x, 'weekday', weekday, 7, 3)
        _add_harmonics(x, 'month', x['month'] - 1, 12, 3)

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
        'time_encoding': str(p.pop('time_encoding', 'cyclic')),
        'use_interactions': bool(p.pop('use_interactions', True)),
        'use_context': bool(p.pop('use_context', False)),
        'use_lags': bool(p.pop('use_lags', False)),
        'log_target': bool(p.pop('log_target', False)),
    }
    return p, flags


def feature_kwargs(flags):
    return {k: flags[k] for k in FEATURE_FLAGS}


def transform_y(y, log_target):
    return np.log1p(y) if log_target else y


def inverse_y(pred, log_target):
    return np.expm1(pred) if log_target else pred


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


def add_calendar_categories(x):
    x = x.copy()
    x['hour_cat'] = x['hour'].astype('Int64').astype('string').fillna('Unknown').astype(str)
    x['month_cat'] = x['month'].astype('Int64').astype('string').fillna('Unknown').astype(str)
    return x


def build_ridge(columns, cats):
    spline_cols = [c for c in RIDGE_SPLINE if c in columns]
    other = [c for c in columns if c not in spline_cols and c not in cats]
    pre = ColumnTransformer([
        ('spline', make_pipeline(SimpleImputer(strategy='median'),
                                 SplineTransformer(n_knots=8, degree=3,
                                                   extrapolation='constant')), spline_cols),
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), other),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5),
         list(cats) + ['hour_cat', 'month_cat']),
    ])
    return make_pipeline(FunctionTransformer(add_calendar_categories, validate=False), pre,
                         RidgeCV(alphas=np.logspace(-2, 3, 25)))


class MeanBlend:
    """Plain average of the member predictions (no fitting, cannot overfit)."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X).mean(axis=1)


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
    mp = dict(model_parameters)
    huber = mp.pop('huber_mult', None)
    if family == 'ridge':
        model = build_ridge(list(x.columns), cats)
        model.fit(x, y)
        return model, 0
    delta = None if not huber else float(huber) * float(np.std(y))

    if family == 'catboost':
        model = CatBoostRegressor(
            iterations=int(iterations),
            loss_function='RMSE' if delta is None else f'Huber:delta={delta:.6f}',
            eval_metric='RMSE', nan_mode='Min', random_seed=seed, task_type='CPU',
            allow_writing_files=False, thread_count=args.threads, **mp,
        )
        if eval_set is None:
            model.fit(x, y, cat_features=cats, verbose=False)
            return model, int(iterations)
        model.fit(x, y, cat_features=cats, eval_set=eval_set,
                  early_stopping_rounds=args.patience, use_best_model=True, verbose=False)
        return model, int(model.get_best_iteration() + 1)

    extra = {} if delta is None else dict(objective='huber', alpha=delta)
    model = lgb.LGBMRegressor(
        n_estimators=int(iterations), random_state=seed, n_jobs=args.threads,
        verbose=-1, subsample_freq=1, metric='rmse', **mp, **extra,
    )
    if eval_set is None:
        model.fit(x, y)
        return model, int(iterations)
    model.fit(x, y, eval_set=[eval_set],
              callbacks=[lgb.early_stopping(args.patience, verbose=False)])
    return model, int(model.best_iteration_ or iterations)


def inner_split(indices, data, cv_mode, seed=123):
    """Early-stopping split INSIDE the given training rows."""
    indices = np.asarray(indices)
    if cv_mode == 'time':                # keep chronology: stop on the latest 20%
        k = max(1, int(round(len(indices) * 0.2)))
        return indices[:-k], indices[-k:]
    return train_test_split(
        indices, test_size=0.2, random_state=seed,
        stratify=data['building_id'].iloc[indices].fillna('Unknown'),
    )


def choose_tree_count(family, data, parameters, args):
    if family == 'ridge':
        return 0
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(data, **feature_kwargs(flags))
    if family == 'lightgbm':
        x, _ = lgb_encode(x, cats)
    y = transform_y(data[TARGET], flags['log_target'])
    fit_idx, stop_idx = inner_split(np.arange(len(data)), data, args.cv_mode)
    _, trees = fit_model(
        family, model_parameters, x.iloc[fit_idx], y.iloc[fit_idx], cats,
        args.max_iterations, args, eval_set=(x.iloc[stop_idx], y.iloc[stop_idx]),
    )
    return trees


def fit_fixed(family, data, parameters, trees, args):
    """Final fit. Tree models are bagged over --final-seeds seeds."""
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(data, **feature_kwargs(flags))
    categories = None
    if family == 'lightgbm':
        x, categories = lgb_encode(x, cats)
    y = transform_y(data[TARGET], flags['log_target'])
    seeds = [SEED] if family == 'ridge' else [SEED + i for i in range(args.final_seeds)]
    models = [fit_model(family, model_parameters, x, y, cats, trees, args, seed=s)[0]
              for s in seeds]
    return {
        'family': family, 'models': models, 'feature_names': list(x.columns),
        'categorical_features': cats, 'categories': categories,
        'feature_flags': flags, 'parameters': parameters, 'trees': int(trees),
        'feature_version': FEATURE_VERSION,
    }


def predict_bundle(bundle, data):
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('Model and feature-preparation versions do not match.')
    flags = bundle['feature_flags']
    x, cats = make_features(data, **feature_kwargs(flags))
    if bundle['family'] == 'lightgbm':
        x, _ = lgb_encode(x, cats, bundle['categories'])
    x = x[bundle['feature_names']]
    pred = np.mean([np.asarray(m.predict(x)) for m in bundle['models']], axis=0)
    return inverse_y(pred, flags['log_target'])


def predict_final(final, data):
    columns = [predict_bundle(final['members'][name], data) for name in final['member_names']]
    return np.asarray(final['meta'].predict(np.column_stack(columns)))


def cross_validate(family, parameters, development, folds, args, trial=None,
                   label=None, verbose=True):
    model_parameters, flags = split_parameters(parameters)
    x, cats = make_features(development, **feature_kwargs(flags))
    if family == 'lightgbm':
        x, _ = lgb_encode(x, cats)       # categories come from inputs only, no labels
    y_raw = development[TARGET]
    y = transform_y(y_raw, flags['log_target'])
    predictions = np.full(len(development), np.nan)
    rows = []
    for fold_number, (training_idx, scoring_idx) in enumerate(folds, 1):
        trees = 0
        if family == 'ridge':
            model, _ = fit_model(family, model_parameters, x.iloc[training_idx],
                                 y.iloc[training_idx], cats, 0, args)
        else:
            fit_idx, stopping_idx = inner_split(training_idx, development, args.cv_mode)
            _, trees = fit_model(
                family, model_parameters, x.iloc[fit_idx], y.iloc[fit_idx], cats,
                args.max_iterations, args,
                eval_set=(x.iloc[stopping_idx], y.iloc[stopping_idx]),
            )
            # Refit using the full fold-training portion before scoring.
            model, _ = fit_model(family, model_parameters, x.iloc[training_idx],
                                 y.iloc[training_idx], cats, trees, args)
        pred = inverse_y(np.asarray(model.predict(x.iloc[scoring_idx])), flags['log_target'])
        predictions[scoring_idx] = pred
        row = dict(fold=fold_number, trees=trees, **metrics(y_raw.iloc[scoring_idx], pred))
        rows.append(row)
        if verbose:
            name = label or (f'{family} trial {trial.number}' if trial is not None else family)
            print(f"  {name}, fold {fold_number}/{len(folds)}: "
                  f"RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, trees={trees}", flush=True)
        if trial is not None:
            trial.report(float(np.mean([r['RMSE'] for r in rows])), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if args.cv_mode == 'kfold' and not np.isfinite(predictions).all():
        raise RuntimeError('Cross-validation failed to predict every scoring row.')
    return rows, predictions


def make_splits(train, cv_mode):
    """Development / holdout split plus CV folds for the chosen validation scheme."""
    if cv_mode == 'time':
        train = train.sort_values('row_order', kind='mergesort').reset_index(drop=True)
        cut = int(len(train) * 0.8)
        development = train.iloc[:cut].reset_index(drop=True)
        holdout = train.iloc[cut:].reset_index(drop=True)
        folds = list(TimeSeriesSplit(n_splits=5).split(development))
    else:
        development, holdout = train_test_split(
            train, test_size=0.2, random_state=SEED,
            stratify=train.building_id.fillna('Unknown'),
        )
        development = development.reset_index(drop=True)
        holdout = holdout.reset_index(drop=True)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            development, development.building_id.fillna('Unknown'),
        ))
    return train, development, holdout, folds


def resolve_cv_mode(requested, train, test, order_source):
    if requested != 'auto':
        return requested, f'forced by --cv {requested}'
    if order_source != 'id':
        return 'kfold', 'ids are not numeric, so chronology is unknown'
    if test.row_order.min() >= train.row_order.quantile(0.9):
        return 'time', 'test ids all come after (nearly) all training ids'
    return 'kfold', 'test ids are interleaved with training ids (random split)'


# --------------------------------------------------------------------------- Optuna
def run_study(family, prefix, suggest, initial_trials, signature, development, folds, args, n_trials):
    """Resumable Optuna study with cross-validation inside every trial."""
    def objective(trial):
        parameters = suggest(trial)
        rows, oof = cross_validate(family, parameters, development, folds, args, trial)
        covered = np.isfinite(oof)
        trial.set_user_attr('fold_results', rows)
        trial.set_user_attr('parameters', parameters)
        trial.set_user_attr('oof_metrics', metrics(development[TARGET][covered], oof[covered]))
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


def suggest_shared(trial, allow_log):
    return dict(
        time_encoding=trial.suggest_categorical('time_encoding', ['none', 'cyclic', 'fourier']),
        use_interactions=trial.suggest_categorical('use_interactions', [False, True]),
        use_context=trial.suggest_categorical('use_context', [False, True]),
        use_lags=trial.suggest_categorical('use_lags', [False, True]),
        log_target=(trial.suggest_categorical('log_target', [False, True]) if allow_log else False),
        huber_mult=trial.suggest_categorical('huber_mult', [None, 0.7, 1.0, 1.5, 2.5]),
    )


def suggest_catboost(trial, allow_log=False):
    parameters = dict(
        learning_rate=trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        depth=trial.suggest_int('depth', 3, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
        random_strength=trial.suggest_float('random_strength', 0.0, 3.0),
        rsm=trial.suggest_float('rsm', 0.5, 1.0),     # feature subsampling per split
        max_ctr_complexity=trial.suggest_int('max_ctr_complexity', 1, 3),
        # Bernoulli supports subsample. Do not combine subsample
        # with CatBoost's Bayesian-bootstrap bagging_temperature.
        bootstrap_type='Bernoulli',
        subsample=trial.suggest_float('subsample', 0.65, 1.0),
        grow_policy=trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
    )
    if parameters['grow_policy'] == 'Depthwise':
        parameters['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 5, 60, log=True)
    parameters.update(suggest_shared(trial, allow_log))
    return parameters


def suggest_lightgbm(trial, allow_log=False):
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
    )
    if parameters['linear_tree']:
        parameters['linear_lambda'] = trial.suggest_float('linear_lambda', 0.01, 10.0, log=True)
    parameters.update(suggest_shared(trial, allow_log))
    return parameters


def oof_for_trial(family, prefix, trial, development, folds, args):
    path = args.output / f'{prefix}_trial_{trial.number}_oof.npy'
    if path.exists():
        return np.load(path)
    _, oof = cross_validate(family, trial.user_attrs['parameters'], development, folds, args)
    return oof


def search_ridge(development, folds, args):
    """Tiny exhaustive search for the Ridge member (it is fast)."""
    best = None
    print('\nSpline-Ridge search (mean CV RMSE):', flush=True)
    for time_encoding in ['cyclic', 'fourier']:
        for use_context in [False, True]:
            for use_lags in [False, True]:
                parameters = dict(time_encoding=time_encoding, use_interactions=False,
                                  use_context=use_context, use_lags=use_lags, log_target=False)
                rows, oof = cross_validate('ridge', parameters, development, folds, args,
                                           verbose=False)
                score = float(np.mean([r['RMSE'] for r in rows]))
                print(f'  {time_encoding:8s} context={use_context!s:5s} lags={use_lags!s:5s} '
                      f'-> {score:.4f}', flush=True)
                if best is None or score < best[0]:
                    best = (score, parameters, oof)
    print(f'  best Ridge CV RMSE: {best[0]:.4f}', flush=True)
    return best


def fit_meta(oof_matrix, y, args):
    """Chooses the stacking meta-model by cross-validation ON the out-of-fold
    predictions, so a flexible meta-model only wins if it genuinely helps."""
    candidates = {
        'mean_blend': MeanBlend,
        'nonnegative_linear': lambda: LinearRegression(positive=True),
        'catboost_meta': lambda: CatBoostRegressor(
            iterations=300, depth=3, learning_rate=0.03, l2_leaf_reg=10,
            loss_function='RMSE', random_seed=SEED, allow_writing_files=False,
            thread_count=args.threads, verbose=False),
    }
    scores = {}
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, make in candidates.items():
        pred = np.zeros(len(y))
        for tr, va in inner.split(oof_matrix):
            pred[va] = make().fit(oof_matrix[tr], y[tr]).predict(oof_matrix[va])
        scores[name] = float(np.sqrt(mean_squared_error(y, pred)))
    best = min(scores, key=scores.get)
    return best, candidates[best]().fit(oof_matrix, y), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=DATA_DIR / 'train.csv')
    parser.add_argument('--test', type=Path, default=DATA_DIR / 'test.csv')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'catboost_optuna_results_v3')
    parser.add_argument('--trials', type=int, default=60,
                        help='Total attempted CatBoost trials (resumable), including pruned ones.')
    parser.add_argument('--lgb-trials', type=int, default=40,
                        help='Total attempted LightGBM trials (resumable).')
    parser.add_argument('--cv', choices=['auto', 'kfold', 'time'], default='auto',
                        help='Validation scheme. auto = time-ordered only if test ids follow train ids.')
    parser.add_argument('--final-seeds', type=int, default=3,
                        help='Seeds averaged in the final tree models.')
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--predict-only', type=Path, default=None)
    args = parser.parse_args()
    if (args.trials < 4 or args.lgb_trials < 3 or args.final_seeds < 1
            or min(args.max_iterations, args.patience, args.threads) < 1):
        parser.error('Use >=4 CatBoost trials, >=3 LightGBM trials and positive counts.')
    args.output.mkdir(parents=True, exist_ok=True)

    train = load_data(args.train, labelled=True)
    test = load_data(args.test)
    if len(test) != 3000:
        raise ValueError(f'Expected 3000 testing rows, found {len(test)}.')
    train, test, order_source = add_derived_columns(train, test, verbose=not args.predict_only)

    if args.predict_only:
        final = joblib.load(args.predict_only)
    else:
        if len(train) != 8000:
            raise ValueError(f'Expected 8000 training rows, found {len(train)}.')
        if set(train.id) & set(test.id):
            raise ValueError('Training and testing IDs overlap.')
        args.cv_mode, reason = resolve_cv_mode(args.cv, train, test, order_source)
        print(f'Validation scheme: {args.cv_mode} ({reason}).', flush=True)
        train, development, holdout, folds = make_splits(train, args.cv_mode)
        allow_log = bool(train[TARGET].min() > -1)
        print(f'Target skew = {train[TARGET].skew():.2f}; log1p target '
              f'{"allowed" if allow_log else "NOT allowed (values <= -1)"}.', flush=True)
        print(f'Development rows: {len(development)} | Holdout rows: {len(holdout)}', flush=True)
        print('Cross-validated tuning; the holdout is excluded from every search.', flush=True)

        signature = hashlib.sha256(
            args.train.read_bytes()
            + json.dumps([FEATURE_VERSION, args.max_iterations, args.patience, SEED,
                          args.cv_mode]).encode()
        ).hexdigest()[:16]

        def with_log(d):
            return dict(d, log_target=False) if allow_log else d

        # ---------------- CatBoost: baseline, then Optuna ----------------
        baseline_parameters = dict(
            learning_rate=0.03, depth=4, l2_leaf_reg=10, random_strength=1.0,
            time_encoding='none', use_interactions=False, use_context=False,
            use_lags=False, log_target=False,
        )
        baseline_rows, baseline_oof = cross_validate(
            'catboost', baseline_parameters, development, folds, args, label='cb baseline',
        )
        baseline_cv = float(np.mean([r['RMSE'] for r in baseline_rows]))
        pd.DataFrame(baseline_rows).to_csv(args.output / 'baseline_folds.csv', index=False)

        cb_start = with_log(dict(
            learning_rate=0.03, depth=4, l2_leaf_reg=10.0, random_strength=1.0, rsm=1.0,
            max_ctr_complexity=2, subsample=1.0, grow_policy='SymmetricTree', huber_mult=None))
        flags_off = dict(time_encoding='none', use_interactions=False, use_context=False, use_lags=False)
        flags_basic = dict(time_encoding='cyclic', use_interactions=True, use_context=False, use_lags=False)
        flags_ctx = dict(time_encoding='fourier', use_interactions=True, use_context=True, use_lags=False)
        flags_all = dict(time_encoding='fourier', use_interactions=True, use_context=True, use_lags=True)
        cb_study = run_study(
            'catboost', 'cb', partial(suggest_catboost, allow_log=allow_log),
            [dict(cb_start, **f) for f in (flags_off, flags_basic, flags_ctx, flags_all)],
            signature, development, folds, args, args.trials,
        )
        cb_best = cb_study.best_trial
        cb_tuned = cb_best.user_attrs['parameters']

        # ---------------- LightGBM: Optuna ----------------
        lgb_start = with_log(dict(
            learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=0.1, cat_smooth=20.0,
            linear_tree=False, huber_mult=None))
        lgb_study = run_study(
            'lightgbm', 'lgb', partial(suggest_lightgbm, allow_log=allow_log),
            [dict(lgb_start, **f) for f in (flags_off, flags_ctx, flags_all)],
            signature, development, folds, args, args.lgb_trials,
        )
        lgb_best = lgb_study.best_trial
        lgb_tuned = lgb_best.user_attrs['parameters']

        # ---------------- Ridge member ----------------
        ridge_cv, ridge_parameters, ridge_oof = search_ridge(development, folds, args)

        # Do not force a more complicated CatBoost when CV prefers the baseline.
        use_tuned = cb_best.value < baseline_cv
        cb_chosen = cb_tuned if use_tuned else baseline_parameters
        cb_name = 'CatBoost Optuna' if use_tuned else 'CatBoost baseline'
        print(f'\nCatBoost baseline mean CV RMSE: {baseline_cv:.4f}', flush=True)
        print(f'CatBoost best Optuna mean CV RMSE: {cb_best.value:.4f}', flush=True)
        print(f'LightGBM best Optuna mean CV RMSE: {lgb_best.value:.4f}', flush=True)
        print(f'Spline-Ridge mean CV RMSE: {ridge_cv:.4f}', flush=True)
        print(f'CatBoost chosen by CV: {cb_name}', flush=True)
        print('CatBoost parameters:', json.dumps(cb_chosen, indent=2), flush=True)
        print('LightGBM parameters:', json.dumps(lgb_tuned, indent=2), flush=True)
        pd.DataFrame(cb_best.user_attrs['fold_results']).to_csv(
            args.output / 'best_catboost_trial_folds.csv', index=False)
        pd.DataFrame(lgb_best.user_attrs['fold_results']).to_csv(
            args.output / 'best_lightgbm_trial_folds.csv', index=False)

        # ---------------- Stacking on out-of-fold predictions ----------------
        cb_oof = (oof_for_trial('catboost', 'cb', cb_best, development, folds, args)
                  if use_tuned else baseline_oof)
        lgb_oof = oof_for_trial('lightgbm', 'lgb', lgb_best, development, folds, args)
        member_names = ['catboost', 'lightgbm', 'ridge']
        labels = [cb_name, 'LightGBM Optuna', 'Ridge splines']
        oof_matrix = np.column_stack([cb_oof, lgb_oof, ridge_oof])
        covered = np.isfinite(oof_matrix).all(axis=1)
        y_dev = development[TARGET].to_numpy()
        print(f'\nOut-of-fold RMSE ({int(covered.sum())} scored development rows):', flush=True)
        for label, column in zip(labels, oof_matrix.T):
            print(f'  {label}: {metrics(y_dev[covered], column[covered])["RMSE"]:.4f}')
        meta_name, meta, meta_scores = fit_meta(oof_matrix[covered], y_dev[covered], args)
        print('  Meta-model CV RMSE on OOF predictions:',
              {k: round(v, 4) for k, v in meta_scores.items()})
        print(f'  Meta-model chosen: {meta_name}', flush=True)
        if hasattr(meta, 'coef_'):
            print(f'  Weights {dict(zip(member_names, np.round(meta.coef_, 3).tolist()))}, '
                  f'intercept = {meta.intercept_:.3f}', flush=True)

        # ---------------- Holdout comparison (selects nothing) ----------------
        holdout_rows = []
        holdout_predictions = pd.DataFrame({'id': holdout.id, 'actual': holdout[TARGET]})
        candidates = [('CatBoost baseline', 'catboost', baseline_parameters),
                      ('CatBoost Optuna', 'catboost', cb_tuned),
                      ('LightGBM Optuna', 'lightgbm', lgb_tuned),
                      ('Ridge splines', 'ridge', ridge_parameters)]
        for name, family, parameters in candidates:
            trees = choose_tree_count(family, development, parameters, args)
            model = fit_fixed(family, development, parameters, trees, args)
            pred = predict_bundle(model, holdout)
            holdout_rows.append(dict(model=name, trees=trees, **metrics(holdout[TARGET], pred)))
            holdout_predictions[name] = pred
        stack_pred = meta.predict(np.column_stack([holdout_predictions[l] for l in labels]))
        holdout_rows.append(dict(model=f'Stack ({meta_name})', trees=np.nan,
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
                                         ('lightgbm', 'lightgbm', lgb_tuned),
                                         ('ridge', 'ridge', ridge_parameters)]:
            trees = choose_tree_count(family, train, parameters, args)
            print(f'Final {name}: trees={trees}. Training on all 8000 rows...', flush=True)
            members[name] = fit_fixed(family, train, parameters, trees, args)
        final = {
            'members': members, 'member_names': member_names,
            'meta': meta, 'meta_name': meta_name, 'cv_mode': args.cv_mode,
            'feature_version': FEATURE_VERSION,
            'package_versions': {
                name: importlib.metadata.version(name)
                for name in ['catboost', 'lightgbm', 'optuna', 'pandas', 'numpy',
                             'scikit-learn', 'joblib']
            },
            'training_data_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        }
        joblib.dump(final, args.output / 'model.pkl')
        members['catboost']['models'][0].save_model(str(args.output / 'model_catboost.cbm'))
        members['lightgbm']['models'][0].booster_.save_model(str(args.output / 'model_lightgbm.txt'))
        (args.output / 'best_parameters.json').write_text(json.dumps({
            'cv_mode': args.cv_mode, 'catboost_chosen_by_cv': cb_name,
            'catboost_parameters': cb_chosen, 'lightgbm_parameters': lgb_tuned,
            'ridge_parameters': ridge_parameters,
            'final_trees': {k: v['trees'] for k, v in members.items()},
            'meta_model': meta_name, 'meta_cv_rmse': meta_scores,
            'catboost_baseline_mean_cv_rmse': baseline_cv,
            'catboost_best_optuna_mean_cv_rmse': cb_best.value,
            'lightgbm_best_optuna_mean_cv_rmse': lgb_best.value,
            'ridge_mean_cv_rmse': ridge_cv,
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