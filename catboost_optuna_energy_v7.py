"""V7: V5's proven members, a blend that is NOT fitted on the rows used to tune them.
 
Why V7 exists
  V5 scored 3.29 on the portal. V6 changed too many things at once (new scoring mixture,
  a heavily tuned Ridge, missingness-pattern experts, grouped blend weights) and scored 3.32.
  V7 therefore keeps V5's structure and changes only things with a low-variance rationale:
 
  1. V5's winning configurations are REUSED (read from V5's best_parameters.json), so the
     CatBoost, LightGBM, Ridge and MLP members are the same models that scored 3.29.
     Nothing is re-tuned unless you pass --retune (or the V5 file is absent).
  2. Blend weights. V5 fitted them on the same out-of-fold rows used to tune the members,
     which flatters the most heavily tuned member. V7 also fits weights on FRESH calibration
     rows the members never saw, then evaluates three candidate blends on the holdout:
        v5_style          weights from tuning OOF (+ V5's validated residual correction)
        fresh_weights     weights from calibration rows
        averaged_weights  the mean of the two weight vectors
     The best on the holdout is used (3 candidates, 1600 rows x 6 mask views: tiny selection
     effect). All three submissions are written so you can pick differently.
  3. More bagging in the final fit (--final-seeds, default 6 seeds) and 5 mask seeds when
     fitting/evaluating blends, which removes noise from the masks and from single seeds.
  4. Masks follow the TRAIN file's own missingness: if train already has missing readings,
     masking is reduced so that train + masks match the test file (no change when train is clean).
  5. errors_by_missingness.csv is computed on test-like masked holdout rows.
  6. Optional --pattern-experts re-enables V6's missingness-pattern Ridge experts (off by default).
 
No timestamp exists, so no ID-based lag features and no TimeSeriesSplit are used.
Run with --help for options.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
 
VERSION = 'energy_v7_0'
TARGET = 'energy_usage'
READINGS = ['temperature', 'humidity', 'occupancy', 'previous_usage']
NUM = READINGS + ['hour', 'month']
CAT = ['building_id', 'building_type', 'day_of_week']
DAYS = dict(zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7)))
DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')
LINEAR_FAMILIES = ('ridge', 'ridge_pattern')
MASK = {'scale': 1.0}                       # set in main(); 1.0 when the train file has no missing readings
CAL_VIEW_SEEDS = (4026, 4027, 4028, 4029, 4030)
HOLD_VIEW_SEEDS = (3026, 3027, 3028, 3029, 3030)
# Warm starts from the user's saved V3 winners (used only if V5's results are unavailable).
CB = {'learning_rate': 0.03966226895995887, 'depth': 3, 'l2_leaf_reg': 13.41812135038036, 'random_strength': 2.158832460665511, 'rsm': 0.6907717742149335, 'max_ctr_complexity': 1, 'bootstrap_type': 'Bernoulli', 'subsample': 0.7120037675898786, 'grow_policy': 'SymmetricTree'}
LGB = {'learning_rate': 0.026389615338773015, 'num_leaves': 5, 'min_child_samples': 111, 'subsample': 0.7682670544928751, 'colsample_bytree': 0.42552187655358054, 'reg_lambda': 0.16368666882056326, 'reg_alpha': 0.0010161005492200523, 'cat_smooth': 42.53034176409328, 'linear_tree': True, 'linear_lambda': 0.5535880981079766}
# Keys that describe the pipeline, not the tree model, and must not reach CatBoost/LightGBM.
OPTIONS = ['harmonics', 'interactions', 'context', 'mask_copies', 'target_mode', 'huber_mult']
 
 
# --------------------------------------------------------------------------- data
def load(path, labelled=False):
    d = pd.read_csv(path, dtype={'id': str})
    needed = ['id'] + NUM + CAT + ([TARGET] if labelled else [])
    absent = set(needed) - set(d.columns)
    if absent:
        raise ValueError(f'{path}: missing columns {sorted(absent)}')
    if d.id.isna().any() or not d.id.is_unique:
        raise ValueError('IDs must be present and unique.')
    for c in NUM + ([TARGET] if labelled else []):
        d[c] = pd.to_numeric(d[c], errors='raise').astype(float)
        if np.isinf(d[c]).any():
            raise ValueError(f'Infinite values in {c}')
    if labelled and (d[TARGET].isna().any() or (d[TARGET] < 0).any()):
        raise ValueError('Targets must be present and nonnegative.')
    for c, lo, hi in [('hour', 0, 23), ('month', 1, 12)]:
        v = d[c].dropna()
        if not (v.between(lo, hi) & (v % 1 == 0)).all():
            raise ValueError(f'Invalid {c} values')
    unknown_days = set(d.day_of_week.dropna()) - set(DAYS)
    if unknown_days:
        raise ValueError(f'Expected weekday names; found {unknown_days}')
    return d.reset_index(drop=True)
 
 
def mask_scale(train, test):
    """Fraction of test-style masking to apply so train + masks matches the test file's
    share of rows with a missing reading. 1.0 when the train file has none."""
    q_test = float(test[READINGS].isna().any(axis=1).mean())
    q_train = float(train[READINGS].isna().any(axis=1).mean())
    if q_test <= 0:
        return 1.0
    if q_train >= q_test:
        return 0.0
    return float(min(1.0, ((q_test - q_train) / (1 - q_train)) / q_test))
 
 
def mask(d, patterns, seed, probability=1.):
    """Hide readings using missingness patterns observed in the TEST file."""
    rng = np.random.default_rng(seed)
    z = d.copy()
    m = patterns[rng.integers(len(patterns), size=len(z))].copy()
    m &= (rng.random(len(z)) < probability)[:, None]
    for j, c in enumerate(READINGS):
        z.loc[m[:, j], c] = np.nan
    return z
 
 
def augmented(d, patterns, copies):
    if not copies:
        return d.copy()
    return pd.concat([d] + [mask(d, patterns, 202600 + i, MASK['scale']) for i in range(int(copies))],
                     ignore_index=True)
 
 
def assessment_views(d, patterns, seeds=(2026, 2027)):
    return [d] + [mask(d, patterns, s, .8 * MASK['scale']) for s in seeds]
 
 
def scores(y, predictions, stress_weight):
    clean = mean_squared_error(y, predictions[0])
    stress = np.mean([mean_squared_error(y, p) for p in predictions[1:]])
    return dict(clean_RMSE=float(np.sqrt(clean)), stress_RMSE=float(np.sqrt(stress)),
                selection_RMSE=float(np.sqrt((1 - stress_weight) * clean + stress_weight * stress)))
 
 
# --------------------------------------------------------------------------- features
def fit_context(d):
    """INPUT-only building context fitted inside the training partition (never labels).
    Hierarchical: sparse cells shrink toward their parent (building or building type),
    which in turn shrinks toward the global mean."""
    glob = d[READINGS].mean().fillna(0.)
    tables = {}
    plan = [(('building_id',), None, 5), (('building_type',), None, 5),
            (('building_id', 'hour'), 'building_id', 10),
            (('building_id', 'day_of_week'), 'building_id', 10),
            (('building_type', 'hour'), 'building_type', 10)]
    for keys, parent_key, k in plan:
        g = d.groupby(list(keys), dropna=False)[READINGS]
        count, mean = g.count(), g.mean()
        if parent_key is None:
            fb = pd.DataFrame([glob.to_numpy()] * len(mean), index=mean.index, columns=READINGS)
            parent = None
        else:
            ptable = tables[(parent_key,)][0]
            fb = ptable.reindex(mean.index.get_level_values(parent_key))
            fb.index = mean.index
            fb = fb.fillna(glob)
            parent = (parent_key, ptable)
        table = (mean.fillna(fb) * count + k * fb) / (count + k)
        tables[keys] = (table, glob, parent)
    return tables
 
 
def lookup(d, keys, entry):
    table, glob, parent = entry
    idx = pd.Index(d[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(d[list(keys)])
    values = table.reindex(idx).reset_index(drop=True)
    if parent is not None:
        pkey, ptable = parent
        values = values.fillna(ptable.reindex(pd.Index(d[pkey])).reset_index(drop=True))
    return values.fillna(glob)
 
 
def features(d, cfg, tables=None):
    x = d[NUM + CAT].copy()
    cats = list(CAT)
    for c in cats:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)
    for c in READINGS:
        x[c + '_missing'] = x[c].isna().astype(int)
    x['missing_count'] = x[READINGS].isna().sum(axis=1)
    h = cfg['harmonics']
    weekday = d.day_of_week.map(DAYS).astype(float)
    for name, values, period in [('hour', x.hour, 24), ('weekday', weekday, 7), ('month', x.month - 1, 12)]:
        for k in range(1, min(h, 3) + 1):
            x[f'{name}_sin{k}'] = np.sin(2 * np.pi * k * values / period)
            x[f'{name}_cos{k}'] = np.cos(2 * np.pi * k * values / period)
    if cfg['interactions']:
        hour = x.hour.astype('Int64').astype('string').fillna('Unknown')
        for name, val in {
            'building_type_hour': x.building_type + '__' + hour,
            'building_id_hour': x.building_id + '__' + hour,
            'building_type_weekday': x.building_type + '__' + x.day_of_week,
        }.items():
            x[name] = val.astype(str)
            cats.append(name)
        x['is_weekend'] = weekday.isin([5, 6]).astype(int)
        x['temperature_x_occupancy'] = x.temperature * x.occupancy
        x['temperature_x_humidity'] = x.temperature * x.humidity
    if tables:
        for keys, entry in tables.items():
            values = lookup(d, keys, entry)
            for c in READINGS:
                name = '_'.join(keys) + '_' + c + '_context'
                x[name] = values[c].to_numpy()
                x[name + '_deviation'] = x[c] - x[name]
        for c in READINGS:                 # a missing reading replaced by its building-hour context
            x[c + '_filled'] = x[c].fillna(x[f'building_id_hour_{c}_context'])
    return x, cats
 
 
def encode(x, cats, categories=None):
    x = x.copy()
    # Keep original base categories; interaction categories suit CatBoost better.
    x = x.drop(columns=[c for c in cats if c not in CAT])
    if categories is None:
        categories = {c: sorted(x[c].unique()) for c in CAT}
    for c in CAT:
        x[c] = pd.Categorical(x[c], categories=categories[c])
    return x, categories
 
 
def unpack(cfg):
    return {k: v for k, v in cfg.items() if k not in OPTIONS}
 
 
def inverse(p, mode):
    # Modest guard against numerical overflow, not data-dependent clipping.
    return np.expm1(np.clip(p, -30, 30)) if mode == 'log1p' else np.asarray(p)
 
 
class OriginalEnergyRMSE:
    """CatBoost stopping metric in energy units for log-target training."""
    def is_max_optimal(self):
        return False
 
    def evaluate(self, approxes, target, weight):
        pred = np.expm1(np.clip(np.asarray(approxes[0]), -30, 30))
        actual = np.expm1(np.asarray(target))
        w = np.ones(len(target)) if weight is None else np.asarray(weight)
        return float(np.sum(w * (pred - actual) ** 2)), float(np.sum(w))
 
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))
 
 
def lgb_original_rmse(y_true, y_pred, weight):
    diff = np.expm1(y_true) - np.expm1(np.clip(y_pred, -30, 30))
    return 'energy_RMSE', float(np.sqrt(np.average(diff ** 2, weights=weight))), False
 
 
def ridge_features(d, cfg, tables):
    x, cats = features(d, cfg, tables)
    for c in ['hour', 'month']:
        name = c + '_category'
        x[name] = x[c].astype('Int64').astype('string').fillna('Unknown').astype(str)
        cats.append(name)
    return x, cats
 
 
def scaling(x, state=None):
    x = x.copy()
    cols = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
    if state is None:
        state = (x[cols].mean().fillna(0.), x[cols].std().replace(0, np.nan).fillna(1.))
    means, stds = state
    x[means.index] = (x[means.index] - means) / stds
    return x, state
 
 
# --------------------------------------------------------------------------- linear / neural design
class GroupSlopes(BaseEstimator, TransformerMixin):
    """Per-group slopes for continuous columns (first input column = group label).
    With a ridge penalty this approximates a mixed-effects model: each building gets
    its own slope on occupancy, temperature, ... shrunk toward the common slope."""
    def __init__(self, scale=1.0):
        self.scale = scale
 
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.levels_ = sorted(X.iloc[:, 0].astype(str).unique())
        v = X.iloc[:, 1:].astype(float)
        self.mean_ = v.mean().fillna(0.).to_numpy()
        self.std_ = v.std().replace(0, np.nan).fillna(1.).to_numpy()
        return self
 
    def transform(self, X):
        X = pd.DataFrame(X)
        codes = np.asarray(pd.Categorical(X.iloc[:, 0].astype(str), categories=self.levels_).codes)
        z = np.nan_to_num((X.iloc[:, 1:].astype(float).to_numpy() - self.mean_) / self.std_)
        out = np.zeros((len(X), len(self.levels_), z.shape[1]))
        ok = codes >= 0
        out[np.flatnonzero(ok), codes[ok], :] = z[ok] * self.scale
        return out.reshape(len(X), -1)
 
 
def uses_reading(column, reading):
    """True if a feature column is computed from the given reading itself."""
    return (column == reading or column in (reading + '_missing', reading + '_filled')
            or column.endswith(f'_{reading}_context_deviation')
            or (column in ('temperature_x_occupancy', 'temperature_x_humidity') and reading in column))
 
 
def make_preprocessor(x, cats, cfg, drop=()):
    """`drop` lists readings the model must not see (only used by pattern experts)."""
    keep = [c for c in x if not any(uses_reading(c, r) for r in drop)]
    if cfg['context']:
        spline_cols = [c + '_filled' for c in READINGS if c not in drop]
    else:
        spline_cols = [c for c in READINGS if c not in drop]
    other = [c for c in keep if c not in spline_cols and c not in cats and c not in READINGS]
    parts = []
    if spline_cols:
        parts.append(('spline', make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            SplineTransformer(n_knots=cfg['knots'], degree=3, extrapolation='constant')), spline_cols))
    parts.append(('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                       StandardScaler()), other))
    parts.append(('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cats))
    if spline_cols and cfg.get('slope_scale', 0.) > 0:
        parts.append(('slopes', GroupSlopes(scale=cfg['slope_scale']), ['building_id'] + spline_cols))
    return ColumnTransformer(parts, sparse_threshold=0.)
 
 
def fit_pattern(d, cfg, patterns):
    """Optional V6 member: one Ridge expert per missingness pattern in the test file, trained on
    clean rows using only the readings that pattern still has (all-missing expert = fallback)."""
    tables = fit_context(d)
    x, cats = ridge_features(d, cfg, tables)
    found = {}
    for row in patterns:
        key = tuple(bool(v) for v in row)
        found[key] = found.get(key, 0) + 1
    threshold = max(8, int(.005 * len(patterns)))
    wanted = {k for k, n in found.items() if n >= threshold} | {(True,) * len(READINGS)}
    experts = {}
    for key in wanted:
        drop = [r for r, gone in zip(READINGS, key) if gone]
        pre = make_preprocessor(x, cats, cfg, drop)
        model = Ridge(alpha=cfg['ridge_alpha'], solver='lsqr').fit(pre.fit_transform(x), d[TARGET])
        experts[key] = dict(pre=pre, model=model)
    return dict(family='ridge_pattern', model=None, pre=None, config=cfg, tables=tables,
                experts=experts, categories=None, trees=0, scale=None, y_stats=None)
 
 
def route(pattern, experts):
    """The expert that drops the fewest readings while dropping every reading the row lacks."""
    usable = [e for e in experts if all((not lacks) or dropped for lacks, dropped in zip(pattern, e))]
    return min(usable, key=sum)
 
 
def predict_pattern(bundle, d):
    x, _ = ridge_features(d, bundle['config'], bundle['tables'])
    missing = d[READINGS].isna().to_numpy()
    code = missing.astype(int) @ (2 ** np.arange(len(READINGS)))
    out = np.empty(len(d))
    for c in np.unique(code):
        rows = code == c
        pattern = tuple(bool(v) for v in missing[rows][0])
        expert = bundle['experts'][route(pattern, bundle['experts'])]
        out[rows] = expert['model'].predict(expert['pre'].transform(x[rows]))
    return out
 
 
def train_mlp(A, y, cfg, args, seed, epochs, stop_data=None):
    """One epoch per partial_fit call. With stop_data, returns the best epoch."""
    net = MLPRegressor(hidden_layer_sizes=tuple(cfg['hidden']), alpha=cfg['mlp_alpha'],
                       batch_size=128, learning_rate_init=1e-3, random_state=seed)
    best_mse, best_epoch, since = np.inf, 1, 0
    for epoch in range(1, int(epochs) + 1):
        net.partial_fit(A, y)
        if stop_data is not None:
            SA, sy, w, ym, ys = stop_data
            mse = float(np.average((net.predict(SA) * ys + ym - sy) ** 2, weights=w))
            if mse < best_mse - 1e-9:
                best_mse, best_epoch, since = mse, epoch, 0
            else:
                since += 1
                if since >= args.mlp_patience:
                    break
    return net, (best_epoch if stop_data is not None else int(epochs))
 
 
# --------------------------------------------------------------------------- model fitting
def fit(family, d, cfg, patterns, args, trees=None, stop=None, seed=42):
    if family == 'ridge_pattern':
        return fit_pattern(d, cfg, patterns)
    tables = fit_context(d) if cfg['context'] else None
    z = augmented(d, patterns, cfg['mask_copies'])
    w = None
    if stop is not None:
        # A prespecified clean/stress mixture for early stopping.
        stopdata = pd.concat(assessment_views(stop, patterns), ignore_index=True)
        w = np.concatenate([np.full(len(stop), (1 - args.stress_weight)),
                            np.full(2 * len(stop), args.stress_weight / 2)])
 
    if family in ('ridge', 'mlp'):
        x, cats = ridge_features(z, cfg, tables)
        pre = make_preprocessor(x, cats, cfg)
        A = pre.fit_transform(x)
        if family == 'ridge':
            model = Ridge(alpha=cfg['ridge_alpha'], solver='lsqr').fit(A, z[TARGET])
            return dict(family=family, model=model, pre=pre, config=cfg, tables=tables,
                        categories=None, trees=0, scale=None, y_stats=None)
        ym, ys = float(z[TARGET].mean()), float(z[TARGET].std())
        yz = ((z[TARGET] - ym) / ys).to_numpy()
        stop_data = None
        if stop is not None:
            sx, _ = ridge_features(stopdata, cfg, tables)
            stop_data = (pre.transform(sx), stopdata[TARGET].to_numpy(), w, ym, ys)
        net, n = train_mlp(A, yz, cfg, args, seed, trees or args.max_epochs, stop_data)
        return dict(family=family, model=net, pre=pre, config=cfg, tables=tables,
                    categories=None, trees=n, scale=None, y_stats=(ym, ys))
 
    x, cats = features(z, cfg, tables)
    y = np.log1p(z[TARGET]) if cfg['target_mode'] == 'log1p' else z[TARGET]
    categories = None
    scale = None
    if family == 'lightgbm':
        x, categories = encode(x, cats)
        if cfg.get('linear_tree', False):
            x, scale = scaling(x)
    sx = sy = None
    if stop is not None:
        sx, _ = features(stopdata, cfg, tables)
        sy = np.log1p(stopdata[TARGET]) if cfg['target_mode'] == 'log1p' else stopdata[TARGET]
        if family == 'lightgbm':
            sx, _ = encode(sx, cats, categories)
            if scale is not None:
                sx, _ = scaling(sx, scale)
    params = unpack(cfg)
    huber = cfg.get('huber_mult')
    delta = None if huber is None else max(1e-6, float(huber) * float(np.std(y)))
    if family == 'catboost':
        from catboost import Pool
        model = CatBoostRegressor(
            iterations=int(trees or args.max_iterations),
            loss_function='RMSE' if delta is None else f'Huber:delta={delta}',
            eval_metric=(OriginalEnergyRMSE() if stop is not None and cfg['target_mode'] == 'log1p' else 'RMSE'),
            nan_mode='Min', thread_count=args.threads,
            random_seed=seed, allow_writing_files=False, **params)
        kw = {} if stop is None else dict(eval_set=Pool(sx, sy, cat_features=cats, weight=w),
                                          early_stopping_rounds=args.patience, use_best_model=True)
        model.fit(x, y, cat_features=cats, verbose=False, **kw)
        n = int(model.tree_count_)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=int(trees or args.max_iterations),
            objective='regression' if delta is None else 'huber',
            **({} if delta is None else {'alpha': delta}), metric='None', verbosity=-1,
            random_state=seed, n_jobs=args.threads, subsample_freq=1, **params)
        kw = {}
        if stop is not None:
            # Support the installed API and versions before eval_X/eval_y existed.
            if 'eval_X' in inspect.signature(model.fit).parameters:
                kw.update(eval_X=sx, eval_y=sy)
            else:
                kw['eval_set'] = [(sx, sy)]
            kw.update(eval_sample_weight=[w],
                      eval_metric=lgb_original_rmse if cfg['target_mode'] == 'log1p' else 'rmse',
                      callbacks=[lgb.early_stopping(args.patience, verbose=False)])
        model.fit(x, y, **kw)
        n = int(model.best_iteration_ or trees or args.max_iterations)
    return dict(family=family, model=model, pre=None, config=cfg, tables=tables,
                categories=categories, trees=n, scale=scale, y_stats=None)
 
 
def predict(bundle, d):
    family = bundle['family']
    if family == 'ridge_pattern':
        return predict_pattern(bundle, d)
    if family in ('ridge', 'mlp'):
        x, _ = ridge_features(d, bundle['config'], bundle['tables'])
        out = np.asarray(bundle['model'].predict(bundle['pre'].transform(x)))
        if family == 'mlp':
            ym, ys = bundle['y_stats']
            out = out * ys + ym
        return out
    x, cats = features(d, bundle['config'], bundle['tables'])
    if family == 'lightgbm':
        x, _ = encode(x, cats, bundle['categories'])
        if bundle['scale'] is not None:
            x, _ = scaling(x, bundle['scale'])
    return inverse(bundle['model'].predict(x), bundle['config']['target_mode'])
 
 
def select_trees(family, d, cfg, patterns, args):
    if family in LINEAR_FAMILIES:
        return 0
    a, b = train_test_split(d, test_size=.2, random_state=123, stratify=d.building_id)
    return fit(family, a, cfg, patterns, args, stop=b)['trees']
 
 
def evaluate_cv(family, d, cfg, patterns, args, trial=None, quiet=False):
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    # Same stress masks for every candidate. Scoring labels never choose tree/epoch count.
    views = assessment_views(d, patterns)
    oof = np.full((3, len(d)), np.nan)
    rows = []
    for f, (tr, va) in enumerate(folds.split(d, d.building_id), 1):
        train = d.iloc[tr]
        n = select_trees(family, train, cfg, patterns, args)
        model = fit(family, train, cfg, patterns, args, trees=n)
        for j, v in enumerate(views):
            oof[j, va] = predict(model, v.iloc[va])
        result = scores(d[TARGET].iloc[va], oof[:, va], args.stress_weight)
        rows.append(dict(fold=f, trees=n, **result))
        if not quiet:
            print(f"{family} fold {f}: clean={result['clean_RMSE']:.4f} "
                  f"stress={result['stress_RMSE']:.4f} trees={n}", flush=True)
        if trial is not None:
            trial.report(np.mean([r['selection_RMSE'] for r in rows]), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not np.isfinite(oof).all():
        raise RuntimeError('Invalid OOF predictions')
    return scores(d[TARGET], oof, args.stress_weight), oof, rows
 
 
# --------------------------------------------------------------------------- tuning (only if V5 results are absent or --retune)
def load_frozen(path):
    """V5's winning configurations, reused so the members are the ones that scored 3.29."""
    if path is None or not Path(path).exists():
        return {}
    try:
        configs = json.loads(Path(path).read_text())['configs']
    except Exception as error:
        print(f'Could not read {path}: {error}')
        return {}
    usable = {}
    for family in ('catboost', 'lightgbm', 'ridge', 'mlp'):
        cfg = configs.get(family)
        if cfg and 'mask_copies' in cfg:
            usable[family] = cfg
    return usable
 
 
def starts(family):
    base = dict(CB if family == 'catboost' else LGB)
    base.update(harmonics=3 if family == 'catboost' else 0, interactions=True,
                context=True, mask_copies=0, target_mode='raw',
                huber_mult=None if family == 'catboost' else 1.5)
    # First trials isolate substantive changes before broader searching.
    return [base,
            dict(base, mask_copies=1),
            dict(base, mask_copies=2),
            dict(base, mask_copies=1, harmonics=1),
            dict(base, mask_copies=1, target_mode='log1p', huber_mult=None),
            dict(base, mask_copies=1, context=False)]
 
 
def suggest(trial, family):
    p = dict(harmonics=trial.suggest_categorical('harmonics', [0, 1, 3]),
             interactions=trial.suggest_categorical('interactions', [False, True]),
             context=trial.suggest_categorical('context', [False, True]),
             mask_copies=trial.suggest_categorical('mask_copies', [0, 1, 2]),
             target_mode=trial.suggest_categorical('target_mode', ['raw', 'log1p']),
             huber_mult=trial.suggest_categorical('huber_mult', [None, 1.5, 2.5]),
             learning_rate=trial.suggest_float('learning_rate', .015, .07, log=True),
             subsample=trial.suggest_float('subsample', .65, 1.))
    if family == 'catboost':
        p.update(depth=trial.suggest_int('depth', 2, 5),
                 l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
                 random_strength=trial.suggest_float('random_strength', 0, 3),
                 rsm=trial.suggest_float('rsm', .5, 1), bootstrap_type='Bernoulli',
                 max_ctr_complexity=trial.suggest_int('max_ctr_complexity', 1, 2),
                 grow_policy='SymmetricTree')
    else:
        p.update(num_leaves=trial.suggest_int('num_leaves', 4, 24, log=True),
                 min_child_samples=trial.suggest_int('min_child_samples', 25, 130),
                 colsample_bytree=trial.suggest_float('colsample_bytree', .4, 1),
                 reg_lambda=trial.suggest_float('reg_lambda', .05, 30, log=True),
                 reg_alpha=trial.suggest_float('reg_alpha', .001, 2, log=True),
                 cat_smooth=trial.suggest_float('cat_smooth', 5, 50),
                 linear_tree=trial.suggest_categorical('linear_tree', [False, True]))
        if p['linear_tree']:
            p['linear_lambda'] = trial.suggest_float('linear_lambda', .05, 5, log=True)
    return p
 
 
def tune(family, d, patterns, args, signature):
    study = optuna.create_study(
        study_name=f'{family}_{signature}', direction='minimize',
        storage='sqlite:///' + str((args.output / 'studies.sqlite3').resolve()), load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    if not study.trials:
        for cfg in starts(family):
            study.enqueue_trial(cfg)
 
    def objective(trial):
        cfg = suggest(trial, family)
        result, oof, rows = evaluate_cv(family, d, cfg, patterns, args, trial)
        trial.set_user_attr('config', cfg)
        trial.set_user_attr('scores', result)
        trial.set_user_attr('folds', rows)
        np.save(args.output / f'{family}_{signature}_{trial.number}_oof.npy', oof)
        return result['selection_RMSE']
 
    total = args.trials if family == 'catboost' else args.lgb_trials
    done = sum(t.state.is_finished() for t in study.trials)
    if total > done:
        study.optimize(objective, n_trials=total - done, gc_after_trial=True)
    best = study.best_trial
    study.trials_dataframe().to_csv(args.output / f'{family}_trials.csv', index=False)
    path = args.output / f'{family}_{signature}_{best.number}_oof.npy'
    if path.exists():
        oof = np.load(path)
    else:
        _, oof, _ = evaluate_cv(family, d, best.user_attrs['config'], patterns, args)
    pd.DataFrame(best.user_attrs['folds']).to_csv(args.output / f'{family}_best_folds.csv', index=False)
    return best.user_attrs['config'], oof, best.user_attrs['scores']
 
 
def ridge_configs():
    return [dict(harmonics=1, interactions=i, context=True, mask_copies=1, target_mode='raw',
                 huber_mult=None, knots=8, ridge_alpha=a, slope_scale=s)
            for i in (False, True) for a in (3., 30.) for s in (0., .5, 1.)]
 
 
def mlp_configs():
    return [dict(harmonics=1, interactions=False, context=True, mask_copies=2, target_mode='raw',
                 huber_mult=None, knots=6, slope_scale=0., hidden=h, mlp_alpha=a)
            for h in ((64, 32), (128, 64)) for a in (1e-2, 1e-1)]
 
 
def pattern_configs(ridge_cfg):
    # Experts see no masked copies, so they may want a different penalty from the generic Ridge.
    return [dict(ridge_cfg, mask_copies=0, ridge_alpha=ridge_cfg['ridge_alpha'] * m) for m in (.3, 1., 3.)]
 
 
def small_grid(family, configs, d, patterns, args):
    """Exhaustive search over a short list of configurations (used for the fast members)."""
    best, records = None, []
    print(f'\n{family} search ({len(configs)} configurations):', flush=True)
    for cfg in configs:
        result, oof, _ = evaluate_cv(family, d, cfg, patterns, args, quiet=True)
        shown = {k: str(v) for k, v in cfg.items()
                 if k in ('interactions', 'ridge_alpha', 'slope_scale', 'hidden', 'mlp_alpha')}
        records.append(dict(**shown, **result))
        print(f"  {shown} -> selection={result['selection_RMSE']:.4f} "
              f"(clean {result['clean_RMSE']:.4f}, stress {result['stress_RMSE']:.4f})", flush=True)
        if best is None or result['selection_RMSE'] < best[2]['selection_RMSE']:
            best = (cfg, oof, result)
    pd.DataFrame(records).to_csv(args.output / f'{family}_search.csv', index=False)
    return best
 
 
# --------------------------------------------------------------------------- blending
def learn_blend(y, predictions, stress_weight):
    """Convex weights (a member can receive zero). predictions: list of (views, rows) arrays;
    view 0 is the clean view, the rest are test-like masked views sharing `stress_weight`."""
    matrix = np.stack(predictions, axis=-1)                       # (views, rows, members)
    views = matrix.shape[0]
    weights = np.array([1 - stress_weight] + [stress_weight / (views - 1)] * (views - 1))[:, None]
    target = np.asarray(y)[None, :]
 
    def loss(w):
        return float(np.mean(np.sum(weights * (matrix @ w - target) ** 2, axis=0)))
    result = minimize(loss, np.full(len(predictions), 1 / len(predictions)), method='SLSQP',
                      bounds=[(0, 1)] * len(predictions),
                      constraints=[dict(type='eq', fun=lambda w: w.sum() - 1)],
                      options=dict(ftol=1e-10, maxiter=300))
    if not result.success:
        raise RuntimeError('Blend optimisation failed: ' + result.message)
    return result.x
 
 
def residual_features(d, predictions):
    x = d[NUM + CAT].copy()
    for c in CAT:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)
    for c in READINGS:
        x[c + '_missing'] = x[c].isna().astype(int)
    for j, pred in enumerate(predictions):
        x[f'member_{j}'] = np.asarray(pred)
    x['member_spread'] = np.std(np.column_stack(predictions), axis=1)
    return x
 
 
def fit_residual(d, oofs, patterns, weights, args):
    views = assessment_views(d, patterns)
    x = pd.concat([residual_features(v, [o[j] for o in oofs]) for j, v in enumerate(views)],
                  ignore_index=True)
    blend = np.stack(oofs, axis=-1) @ weights
    y = np.tile(d[TARGET].to_numpy(), 3) - blend.reshape(-1)
    sample_weights = np.repeat([1 - args.stress_weight, args.stress_weight / 2, args.stress_weight / 2], len(d))
    model = CatBoostRegressor(iterations=250, depth=2, learning_rate=.03, l2_leaf_reg=30,
                              loss_function='RMSE', random_seed=42, allow_writing_files=False,
                              thread_count=args.threads, verbose=False)
    model.fit(x, y, cat_features=CAT, sample_weight=sample_weights)
    return model
 
 
def family_predictions(members, views):
    return {family: np.asarray([np.mean([predict(m, v) for m in models], axis=0) for v in views])
            for family, models in members.items()}
 
 
def combined(preds, names, weights):
    return np.stack([preds[n] for n in names], axis=-1) @ weights
 
 
def with_residual(blend, residual, strength, views, preds, names):
    out = blend.copy()
    if strength:
        for j, v in enumerate(views):
            out[j] += strength * residual.predict(residual_features(v, [preds[n][j] for n in names]))
    return out
 
 
def final_prediction(bundle, d, method=None):
    method = method or bundle['chosen']
    weights, strength = bundle['candidates'][method]
    preds = family_predictions(bundle['members'], [d])
    blend = combined(preds, bundle['names'], weights)
    return with_residual(blend, bundle['residual'], strength, [d], preds, bundle['names'])[0]
 
 
def trained_members(data, configs, patterns, args, seeds):
    members = {}
    for family, cfg in configs.items():
        trees = select_trees(family, data, cfg, patterns, args)
        use = seeds if family not in LINEAR_FAMILIES else [42]
        print(f'Fitting {family}: {len(data)} original rows, trees/epochs={trees}, seeds={use}', flush=True)
        members[family] = [fit(family, data, cfg, patterns, args, trees=trees, seed=s) for s in use]
    return members
 
 
# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', type=Path, default=DATA_DIR / 'Track 1 Training Dataset.csv')
    parser.add_argument('--test', type=Path, default=DATA_DIR / 'Track 1 Testing Dataset.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'catboost_optuna_results_v7')
    parser.add_argument('--v5-results', type=Path,
                        default=Path(__file__).resolve().parent / 'catboost_optuna_results_v5' / 'best_parameters.json',
                        help="V5's best_parameters.json; its configurations are reused unless --retune.")
    parser.add_argument('--retune', action='store_true', help='Ignore V5 results and tune from scratch (slow).')
    parser.add_argument('--pattern-experts', action='store_true', help="Add V6's missingness-pattern Ridge experts.")
    parser.add_argument('--trials', type=int, default=24, help='Only used with --retune / no V5 file.')
    parser.add_argument('--lgb-trials', type=int, default=16, help='Only used with --retune / no V5 file.')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max-iterations', type=int, default=7000)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--max-epochs', type=int, default=150, help='MLP epoch cap.')
    parser.add_argument('--mlp-patience', type=int, default=12)
    parser.add_argument('--skip-mlp', action='store_true', help='Leave the MLP member out.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Seeds for the calibration/holdout stages.')
    parser.add_argument('--final-seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46, 47],
                        help='Seeds averaged in the final models.')
    parser.add_argument('--stress-weight', type=float, default=.5,
                        help='Prespecified tuning mixture, not a portal RMSE estimate.')
    parser.add_argument('--predict-only', type=Path)
    args = parser.parse_args()
    if min(args.trials, args.lgb_trials, args.max_iterations, args.patience, args.threads,
           args.max_epochs, args.mlp_patience) < 1 or args.folds < 2:
        parser.error('Budgets must be positive; folds must be at least two.')
    if not 0 < args.stress_weight < 1:
        parser.error('stress-weight must be between 0 and 1.')
    args.output.mkdir(parents=True, exist_ok=True)
    test = load(args.test)
    if len(test) != 3000:
        raise ValueError('Expected 3000 test rows.')
    if args.predict_only:
        final = joblib.load(args.predict_only)
        if final['version'] != VERSION:
            raise ValueError('Model version mismatch')
    else:
        train = load(args.train, True)
        if len(train) != 8000:
            raise ValueError('Expected 8000 training rows.')
        if set(train.id) & set(test.id):
            raise ValueError('Train/test IDs overlap')
        patterns = test[READINGS].isna().to_numpy()
        MASK['scale'] = mask_scale(train, test)
        print(f'Target skew: {train[TARGET].skew():.3f}.')
        print('Share of rows with a missing reading: train '
              f'{train[READINGS].isna().any(axis=1).mean():.1%}, test {test[READINGS].isna().any(axis=1).mean():.1%}; '
              f'mask scale {MASK["scale"]:.2f} (1.00 = train is clean).')
        print('No trustworthy timestamp: building-stratified folds; no ID-based lags.', flush=True)
        dev, hold = train_test_split(train, test_size=.2, random_state=42, stratify=train.building_id)
        dev = dev.reset_index(drop=True)
        hold = hold.reset_index(drop=True)
        signature = hashlib.sha256(args.train.read_bytes() + args.test.read_bytes() +
                                   Path(__file__).read_bytes() +
                                   json.dumps([args.folds, args.max_iterations, args.patience,
                                               args.stress_weight]).encode()).hexdigest()[:16]
        # Same partitions as V5: tuning rows, fresh calibration rows, holdout.
        search, calibration = train_test_split(dev, test_size=.2, random_state=2025, stratify=dev.building_id)
        search = search.reset_index(drop=True)
        calibration = calibration.reset_index(drop=True)
        print(f'Search={len(search)}, calibration={len(calibration)}, holdout={len(hold)}', flush=True)
 
        frozen = {} if args.retune else load_frozen(args.v5_results)
        print(('Reusing V5 configurations for: ' + ', '.join(frozen)) if frozen else
              'No usable V5 results: tuning from scratch (V5 procedure).', flush=True)
        configs, oofs, selection = {}, {}, {}
        for family in ['catboost', 'lightgbm', 'ridge', 'mlp']:
            if family == 'mlp' and args.skip_mlp:
                continue
            if family in frozen:
                cfg = frozen[family]
                selection[family], oofs[family], _ = evaluate_cv(
                    family, search, cfg, patterns, args, quiet=family in ('ridge', 'mlp'))
                configs[family] = cfg
            elif family in ('catboost', 'lightgbm'):
                configs[family], oofs[family], selection[family] = tune(family, search, patterns, args, signature)
            else:
                grid = ridge_configs() if family == 'ridge' else mlp_configs()
                configs[family], oofs[family], selection[family] = small_grid(family, grid, search, patterns, args)
        if args.pattern_experts:
            configs['ridge_pattern'], oofs['ridge_pattern'], selection['ridge_pattern'] = small_grid(
                'ridge_pattern', pattern_configs(configs['ridge']), search, patterns, args)
        names = list(configs)
        print('\nMember scores on tuning rows (out-of-fold):')
        for n in names:
            print(f"  {n}: clean={selection[n]['clean_RMSE']:.4f} stress={selection[n]['stress_RMSE']:.4f} "
                  f"selection={selection[n]['selection_RMSE']:.4f}")
 
        # --- blend weights: (a) tuning OOF (V5), (b) fresh calibration rows, (c) their average
        w_oof = learn_blend(search[TARGET], [oofs[n] for n in names], args.stress_weight)
        residual = fit_residual(search, [oofs[n] for n in names], patterns, w_oof, args)
        calibration_members = trained_members(search, configs, patterns, args, args.seeds)
        cal_views = assessment_views(calibration, patterns, seeds=CAL_VIEW_SEEDS)
        cal_preds = family_predictions(calibration_members, cal_views)
        w_cal = learn_blend(calibration[TARGET], [cal_preds[n] for n in names], args.stress_weight)
        w_avg = (w_oof + w_cal) / 2
        print('\nBlend weights:')
        for label, w in [('tuning OOF (V5 style)', w_oof), ('fresh calibration', w_cal), ('average', w_avg)]:
            print(f'  {label}:', {n: round(float(v), 3) for n, v in zip(names, w)})
        for n in names:
            print(f"  calibration RMSE {n}: {scores(calibration[TARGET], cal_preds[n], args.stress_weight)['selection_RMSE']:.4f}")
 
        # --- V5's validated residual correction (strength chosen on calibration rows, V5 weights)
        cal_blend = combined(cal_preds, names, w_oof)
        correction = np.asarray([residual.predict(residual_features(v, [cal_preds[n][j] for n in names]))
                                 for j, v in enumerate(cal_views)])
        correction_rows = []
        for s in [0., .25, .5, 1.]:
            correction_rows.append(dict(strength=s, **scores(calibration[TARGET], cal_blend + s * correction,
                                                              args.stress_weight)))
        winner = min(correction_rows, key=lambda r: r['selection_RMSE'])
        # Small prespecified improvement threshold to avoid adding a nearly tied correction.
        strength = winner['strength'] if winner['selection_RMSE'] < correction_rows[0]['selection_RMSE'] - .005 else 0.
        pd.DataFrame(correction_rows).to_csv(args.output / 'residual_selection.csv', index=False)
        print(f'Residual strength selected on calibration rows: {strength}', flush=True)
 
        # --- holdout: members trained on all development rows; three candidate blends compared
        hold_members = trained_members(dev, configs, patterns, args, args.seeds)
        hold_views = assessment_views(hold, patterns, seeds=HOLD_VIEW_SEEDS)
        held = family_predictions(hold_members, hold_views)
        candidate_weights = {'v5_style': (w_oof, strength), 'fresh_weights': (w_cal, 0.), 'averaged_weights': (w_avg, 0.)}
        blends = {k: with_residual(combined(held, names, w), residual, s, hold_views, held, names)
                  for k, (w, s) in candidate_weights.items()}
        rows = [dict(model=n, **scores(hold[TARGET], held[n], args.stress_weight)) for n in names]
        rows += [dict(model=k, **scores(hold[TARGET], b, args.stress_weight)) for k, b in blends.items()]
        chosen = min(blends, key=lambda k: scores(hold[TARGET], blends[k], args.stress_weight)['selection_RMSE'])
        rows.append(dict(model=f'selected_v7 ({chosen})', **scores(hold[TARGET], blends[chosen], args.stress_weight)))
        pd.DataFrame(rows).to_csv(args.output / 'holdout_metrics.csv', index=False)
        print('\nHOLDOUT RESULTS (test-like masks; NOT portal scores)')
        print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
        print(f'Blend chosen on the holdout among 3 candidates: {chosen}. '
              'The holdout was examined in earlier versions, so treat it as a development estimate.')
        pd.DataFrame({'id': hold.id, 'actual': hold[TARGET], 'prediction': blends[chosen][0]}).to_csv(
            args.output / 'holdout_predictions.csv', index=False)
        groups = pd.DataFrame({'missing_count': hold_views[1][READINGS].isna().sum(axis=1),
                               'absolute_error': np.abs(hold[TARGET] - blends[chosen][1]),
                               'squared_error': (hold[TARGET] - blends[chosen][1]) ** 2})
        by_missing = groups.groupby('missing_count').agg(rows=('absolute_error', 'size'), MAE=('absolute_error', 'mean'),
                                                         MSE=('squared_error', 'mean'))
        by_missing.to_csv(args.output / 'errors_by_missingness.csv')
        print('Holdout error by number of missing readings (test-like masks):')
        print(by_missing.round(3).to_string(), flush=True)
 
        # --- final members on all 8000 labelled rows, bagged over --final-seeds
        final = dict(version=VERSION, members=trained_members(train, configs, patterns, args, args.final_seeds),
                     names=names, candidates=candidate_weights, chosen=chosen, residual=residual, configs=configs)
        joblib.dump(final, args.output / 'model.pkl')
        reloaded = joblib.load(args.output / 'model.pkl')
        np.testing.assert_allclose(final_prediction(final, test), final_prediction(reloaded, test))
        (args.output / 'best_parameters.json').write_text(json.dumps(
            dict(configs=configs, names=names, chosen=chosen,
                 candidate_weights={k: dict(weights=dict(zip(names, w.tolist())), residual_strength=s)
                                    for k, (w, s) in candidate_weights.items()},
                 cv_selection_scores=selection, signature=signature, seeds=args.seeds,
                 final_seeds=args.final_seeds, stress_weight=args.stress_weight,
                 mask_scale=MASK['scale']), indent=2, default=str))
        (args.output / 'requirements.txt').write_text('\n'.join(
            f'{p}=={importlib.metadata.version(p)}' for p in
            ['catboost', 'lightgbm', 'optuna', 'pandas', 'numpy', 'scikit-learn', 'scipy', 'joblib']) + '\n')
        # Alternative submissions from the same final models, so another blend can be tried if wanted.
        for method in candidate_weights:
            alt = final_prediction(final, test, method)
            if alt.shape != (3000,) or not np.isfinite(alt).all():
                raise RuntimeError(f'Invalid predictions for {method}')
            pd.DataFrame({'prediction': alt}).to_csv(args.output / f'submission_{method}.csv', index=False)
    pred = final_prediction(final, test)
    if pred.shape != (3000,) or not np.isfinite(pred).all():
        raise RuntimeError('Invalid predictions')
    pd.DataFrame({'prediction': pred}).to_csv(args.output / 'submission.csv', index=False)
    pd.DataFrame({'id': test.id.to_numpy(), TARGET: pred}).to_csv(args.output / 'predicted_energy_usage.csv', index=False)
    print('Saved', args.output / 'submission.csv', flush=True)
 
 
if __name__ == '__main__':
    main()