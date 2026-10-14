"""Stacked ensemble for the campus-energy task.

Sits next to catboost_optuna_energy.py (imports its constants and loader).

Install: python3 -m pip install lightgbm catboost scikit-learn scipy pandas numpy
Run:     python3 stack_energy.py                # CatBoost + LightGBM + Ridge
         python3 stack_energy.py --tabpfn       # adds TabPFN (best on a GPU)

What it does
  1. Adds unlabelled, row-local or feature-only group features (no target used).
  2. Produces out-of-fold (OOF) predictions for several different model families
     on the SAME 5 folds over all 8000 labelled rows.
  3. Learns non-negative blend weights on the OOF predictions (NNLS).
  4. Averages each model's fold-predictions for the test set (no refit needed).
Early stopping always uses a split INSIDE each fold's training rows.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import catboost_optuna_energy as base

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*eval_set.*')

TARGET, SEED = base.TARGET, base.SEED
CATS = ['building_id', 'building_type', 'day_of_week', 'bt_hour', 'bt_weekday', 'bt_month']
LINEAR_ONLY = ['hour_cat', 'month_cat']          # one-hot only for the linear model
SPLINE_COLS = ['temperature', 'humidity', 'occupancy', 'previous_usage']


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --------------------------------------------------------------------------- features
def build_features(train, test):
    """Features for train+test together. Group statistics use ONLY input columns
    (never the target), so nothing leaks from labels."""
    full = pd.concat([train[base.INPUT_COLUMNS], test[base.INPUT_COLUMNS]], ignore_index=True)
    x = full.copy()
    for c in base.BASE_CATEGORICAL:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)

    for c in SPLINE_COLS:                         # missingness can itself be informative
        if x[c].isna().any():
            x[c + '_missing'] = x[c].isna().astype(int)

    x['hour_sin'] = np.sin(2 * np.pi * x.hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * x.hour / 24)
    x['month_sin'] = np.sin(2 * np.pi * (x.month - 1) / 12)
    x['month_cos'] = np.cos(2 * np.pi * (x.month - 1) / 12)
    x['is_weekend'] = x.day_of_week.isin(['Saturday', 'Sunday']).astype(int)

    hour_tok = x.hour.astype('Int64').astype('string').fillna('Unknown')
    month_tok = x.month.astype('Int64').astype('string').fillna('Unknown')
    x['bt_hour'] = x.building_type + '__' + hour_tok
    x['bt_weekday'] = x.building_type + '__' + x.day_of_week
    x['bt_month'] = x.building_type + '__' + month_tok
    x['hour_cat'], x['month_cat'] = hour_tok.astype(str), month_tok.astype(str)

    # Building "personality": how this building normally behaves, from inputs only.
    g = x.groupby('building_id')
    x['bld_prev_mean'] = g.previous_usage.transform('mean')
    x['bld_prev_std'] = g.previous_usage.transform('std')
    x['bld_occ_mean'] = g.occupancy.transform('mean')
    x['bld_temp_mean'] = g.temperature.transform('mean')
    x['prev_dev'] = x.previous_usage - x.bld_prev_mean
    x['prev_z'] = x.prev_dev / (x.bld_prev_std + 1e-6)
    x['occ_dev'] = x.occupancy - x.bld_occ_mean
    x['bt_hour_prev_mean'] = x.groupby('bt_hour').previous_usage.transform('mean')
    x['prev_dev_bt_hour'] = x.previous_usage - x.bt_hour_prev_mean

    # Heating / cooling degrees: energy is usually V- or hockey-stick-shaped in temperature.
    x['cdd'] = (x.temperature - 22).clip(lower=0)
    x['hdd'] = (18 - x.temperature).clip(lower=0)
    x['temp_x_occ'] = x.temperature * x.occupancy
    x['temp_x_hum'] = x.temperature * x.humidity
    x['prev_x_occ'] = x.previous_usage * x.occupancy
    x['occ_x_cdd'] = x.occupancy * x.cdd

    n = len(train)
    return x.iloc[:n].reset_index(drop=True), x.iloc[n:].reset_index(drop=True)


def tree_columns(x):
    return [c for c in x.columns if c not in LINEAR_ONLY]


# --------------------------------------------------------------------------- models
def early_stop_split(n, seed):
    return train_test_split(np.arange(n), test_size=0.15, random_state=seed)


def catboost_runner(params, args, seeds):
    cols_cache = {}

    def run(Xtr, ytr, Xs):
        cols = cols_cache.setdefault('c', tree_columns(Xtr))
        cats = [c for c in CATS if c in cols]
        outs = [np.zeros(len(X)) for X in Xs]
        for seed in seeds:
            fit_i, es_i = early_stop_split(len(Xtr), seed)
            common = dict(loss_function='RMSE', nan_mode='Min', random_seed=seed,
                          thread_count=args.threads, allow_writing_files=False, **params)
            m = CatBoostRegressor(iterations=args.max_iterations, **common)
            m.fit(Xtr[cols].iloc[fit_i], ytr.iloc[fit_i], cat_features=cats,
                  eval_set=(Xtr[cols].iloc[es_i], ytr.iloc[es_i]),
                  early_stopping_rounds=args.patience, use_best_model=True, verbose=False)
            trees = int((m.get_best_iteration() + 1) * 1.1)
            m = CatBoostRegressor(iterations=trees, **common)
            m.fit(Xtr[cols], ytr, cat_features=cats, verbose=False)
            for o, X in zip(outs, Xs):
                o += m.predict(X[cols]) / len(seeds)
        return outs
    return run


def lightgbm_runner(params, args, seeds):
    def prep(X, cols, template):
        X = X[cols].copy()
        for c in CATS:
            if c in X:
                X[c] = pd.Categorical(X[c], categories=template[c].cat.categories)
        return X

    def run(Xtr, ytr, Xs):
        cols = tree_columns(Xtr)
        template = Xtr[cols].copy()
        for c in CATS:
            template[c] = template[c].astype('category')
        # categories seen in training; unseen test categories become NaN (handled by LightGBM)
        Xtr_ = prep(Xtr, cols, template)
        Xs_ = [prep(X, cols, template) for X in Xs]
        outs = [np.zeros(len(X)) for X in Xs]
        for seed in seeds:
            fit_i, es_i = early_stop_split(len(Xtr_), seed)
            common = dict(random_state=seed, n_jobs=args.threads, verbose=-1, **params)
            m = lgb.LGBMRegressor(n_estimators=args.max_iterations, **common)
            m.fit(Xtr_.iloc[fit_i], ytr.iloc[fit_i],
                  eval_set=[(Xtr_.iloc[es_i], ytr.iloc[es_i])],
                  callbacks=[lgb.early_stopping(args.patience, verbose=False)])
            trees = max(50, int((m.best_iteration_ or 100) * 1.1))
            m = lgb.LGBMRegressor(n_estimators=trees, **common)
            m.fit(Xtr_, ytr)
            for o, X in zip(outs, Xs_):
                o += m.predict(X) / len(seeds)
        return outs
    return run


def ridge_runner():
    def run(Xtr, ytr, Xs):
        cat_cols = [c for c in CATS + LINEAR_ONLY if c in Xtr]
        spline_cols = [c for c in SPLINE_COLS if c in Xtr]
        other = [c for c in Xtr.columns if c not in cat_cols + spline_cols]
        pre = ColumnTransformer([
            ('spline', make_pipeline(SimpleImputer(strategy='median'),
                                     SplineTransformer(n_knots=8, degree=3,
                                                       extrapolation='constant')), spline_cols),
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), other),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
        ])
        model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 25)))
        model.fit(Xtr, ytr)
        return [model.predict(X) for X in Xs]
    return run


def tabpfn_runner(args):
    from tabpfn import TabPFNRegressor           # pip install tabpfn (downloads weights)

    def run(Xtr, ytr, Xs):
        cols = tree_columns(Xtr)
        cats = [c for c in CATS if c in cols]
        enc = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in cats}

        def prep(X):
            X = X[cols].copy()
            for c in cats:
                X[c] = X[c].map(enc[c]).fillna(-1)
            return X.to_numpy(dtype=float)
        idx = [cols.index(c) for c in cats]
        model = TabPFNRegressor(categorical_features_indices=idx, random_state=SEED)
        model.fit(prep(Xtr), ytr.to_numpy())
        return [model.predict(prep(X)) for X in Xs]
    return run


# --------------------------------------------------------------------------- driver
def out_of_fold(name, runner, X, y, X_test, folds):
    oof, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds, 1):
        p_va, p_te = runner(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                            [X.iloc[va].reset_index(drop=True), X_test])
        oof[va] = p_va
        test_pred += p_te / len(folds)
        print(f'  {name} fold {k}/{len(folds)}: RMSE={rmse(y.iloc[va], p_va):.4f}', flush=True)
    print(f'{name}: OOF RMSE = {rmse(y, oof):.4f}\n', flush=True)
    return oof, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, default=base.DATA_DIR / 'train.csv')
    ap.add_argument('--test', type=Path, default=base.DATA_DIR / 'test.csv')
    ap.add_argument('--params', type=Path,
                    default=Path(__file__).resolve().parent / 'catboost_optuna_results' / 'best_parameters.json')
    ap.add_argument('--output', type=Path,
                    default=Path(__file__).resolve().parent / 'stack_results')
    ap.add_argument('--seeds', type=int, default=3, help='seeds averaged per tree model')
    ap.add_argument('--max-iterations', type=int, default=5000)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--tabpfn', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [SEED + i for i in range(args.seeds)]

    train = base.load_data(args.train, labelled=True)
    test = base.load_data(args.test)
    y = train[TARGET].reset_index(drop=True)

    # ---- diagnostics worth reading before anything else -------------------------
    print('Target: mean=%.2f std=%.2f skew=%.2f' % (y.mean(), y.std(), y.skew()))
    print('Rows with missing inputs:', int(train[base.INPUT_COLUMNS].isna().any(axis=1).sum()))
    chain = train.previous_usage.round(3).isin(train[TARGET].round(3)).mean()
    print(f'Share of previous_usage values that equal some training target: {chain:.1%} '
          '(near 0% is normal; a large share means rows form time chains)\n')

    X, X_test = build_features(train, test)

    # ---- tuned CatBoost parameters from the Optuna run --------------------------
    try:
        cb_params = json.loads(args.params.read_text())['parameters']
        cb_params = {k: v for k, v in cb_params.items() if k not in ('use_cyclic', 'use_interactions')}
        print('Using tuned CatBoost parameters:', cb_params)
    except Exception as e:
        print('Could not read tuned parameters (%s); using defaults.' % e)
        cb_params = dict(learning_rate=0.03, depth=4, l2_leaf_reg=10, random_strength=1.0)

    lgb_params = dict(learning_rate=0.02, num_leaves=15, min_child_samples=30,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=10.0, cat_smooth=20, min_data_per_group=30)
    lgb_linear = dict(lgb_params, linear_tree=True, linear_lambda=1.0, num_leaves=7,
                      min_child_samples=60)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, X.building_id))
    models = {
        'catboost': catboost_runner(cb_params, args, seeds),
        'catboost_deep': catboost_runner(
            dict(cb_params, depth=6, learning_rate=0.03, l2_leaf_reg=5.0), args, seeds[:1]),
        'lightgbm': lightgbm_runner(lgb_params, args, seeds),
        'lightgbm_linear': lightgbm_runner(lgb_linear, args, seeds[:1]),
        'ridge_splines': ridge_runner(),
    }
    if args.tabpfn:
        models['tabpfn'] = tabpfn_runner(args)

    oofs, tests = {}, {}
    for name, runner in models.items():
        oofs[name], tests[name] = out_of_fold(name, runner, X, y, X_test, folds)

    # ---- blend ------------------------------------------------------------------
    names = list(models)
    O = np.column_stack([oofs[n] for n in names])
    T = np.column_stack([tests[n] for n in names])
    w, _ = nnls(O, y.to_numpy())
    w = w / w.sum()
    blend_oof = O @ w
    print('Model OOF RMSE:', {n: round(rmse(y, oofs[n]), 4) for n in names})
    print('Equal-weight blend OOF RMSE: %.4f' % rmse(y, O.mean(axis=1)))
    print('NNLS blend OOF RMSE:         %.4f' % rmse(y, blend_oof))
    print('Weights:', {n: round(float(v), 3) for n, v in zip(names, w)})
    print('(Blend weights are fitted on the same OOF rows, so expect ~0.01-0.03 optimism.)\n')

    # ---- where is the error? ----------------------------------------------------
    resid = pd.DataFrame({'abs_err': np.abs(y - blend_oof), 'sq': (y - blend_oof) ** 2,
                          'building_id': X.building_id, 'hour': train.hour, 'month': train.month,
                          'any_missing': train[base.INPUT_COLUMNS].isna().any(axis=1)})
    for col in ['building_id', 'hour', 'month', 'any_missing']:
        t = resid.groupby(col).sq.mean().pow(0.5).sort_values(ascending=False)
        print(f'RMSE by {col} (worst 5):', t.head(5).round(2).to_dict())

    pred = T @ w
    assert np.isfinite(pred).all() and len(pred) == len(test)
    pd.DataFrame({'id': test.id, TARGET: pred}).to_csv(args.output / 'predicted_energy_usage.csv', index=False)
    pd.DataFrame({'prediction': pred}).to_csv(args.output / 'submission.csv', index=False)
    pd.DataFrame(O, columns=names).assign(actual=y).to_csv(args.output / 'oof_predictions.csv', index=False)
    print('\nSaved:', args.output / 'submission.csv')


if __name__ == '__main__':
    main()