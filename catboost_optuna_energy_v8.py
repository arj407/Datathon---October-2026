"""Energy V8: preserve V7's blend; improve spline-Ridge missing-value estimates.

Install: python3 -m pip install numpy pandas scikit-learn catboost lightgbm scipy joblib
Run: python3 energy_v8.py --train "/path/train.csv" --test "/path/test.csv"

Default: train all members from scratch, using the frozen winning V7 settings.
Optional --v7-model reuses your trusted V7 model.pkl (same original training file).
Imputers never receive energy_usage. Training imputation is four-fold cross-fitted.
Calendar fields do not identify a chronological sequence: no fabricated target lags.
No promise is made about the private leaderboard score.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
from sklearn.model_selection import KFold
from pathlib import Path

import argparse

import hashlib

import importlib.metadata

import inspect

import json

import joblib

import numpy as np

import pandas as pd

import lightgbm as lgb

from catboost import CatBoostRegressor

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

TARGET = 'energy_usage'

READINGS = ['temperature', 'humidity', 'occupancy', 'previous_usage']

NUM = READINGS + ['hour', 'month']

CAT = ['building_id', 'building_type', 'day_of_week']

DAYS = dict(zip(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7)))

DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')

LINEAR_FAMILIES = ('ridge', 'ridge_pattern')

MASK = {'scale': 1.0}

OPTIONS = ['harmonics', 'interactions', 'context', 'mask_copies', 'target_mode', 'huber_mult']

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

def fit(family, d, cfg, patterns, args, trees=None, stop=None, seed=42):
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

def trained_members(data, configs, patterns, args, seeds):
    members = {}
    for family, cfg in configs.items():
        trees = select_trees(family, data, cfg, patterns, args)
        use = seeds if family not in LINEAR_FAMILIES else [42]
        print(f'Fitting {family}: {len(data)} original rows, trees/epochs={trees}, seeds={use}', flush=True)
        members[family] = [fit(family, data, cfg, patterns, args, trees=trees, seed=s) for s in use]
    return members

def auxfit(d):
    """Predict readings using other inputs only; energy_usage is excluded."""
    models={}
    for r in READINGS:
        cols=[c for c in NUM+CAT if c!=r]
        ok=d[r].notna()
        if ok.sum() < 20:
            raise ValueError(f'Too few observed values to learn {r}')
        m=CatBoostRegressor(iterations=700,depth=3,learning_rate=.05,l2_leaf_reg=10,loss_function='RMSE',verbose=False,thread_count=4,allow_writing_files=False,random_seed=42)
        m.fit(d.loc[ok,cols],d.loc[ok,r],cat_features=CAT)
        models[r]=(m,cols)
    return models

def auxpredict(models,d):
    return np.column_stack([models[r][0].predict(d[models[r][1]]) for r in READINGS])

def newfit(d,cfg,patterns):
    """Keep original rows and all their masked copies in the same imputation fold."""
    d=d.reset_index(drop=True);z=augmented(d,patterns,cfg['mask_copies']);p=np.zeros((len(z),4))
    for a,b in KFold(4,shuffle=True,random_state=741).split(d):
        models=auxfit(d.iloc[a]);idx=np.concatenate([b+i*len(d) for i in range(cfg['mask_copies']+1)])
        p[idx]=auxpredict(models,z.iloc[idx])
    tables=fit_context(d);x,cats=ridge_features(z,cfg,tables)
    for j,r in enumerate(READINGS):x.loc[z[r].isna(),r+'_filled']=p[z[r].isna(),j]
    pre=make_preprocessor(x,cats,cfg);model=Ridge(alpha=cfg['ridge_alpha'],solver='lsqr').fit(pre.fit_transform(x),z[TARGET])
    return dict(model=model,pre=pre,tables=tables,config=cfg,aux=auxfit(d))

def newpredict(b,d):
    x,_=ridge_features(d,b['config'],b['tables']);p=auxpredict(b['aux'],d)
    for j,r in enumerate(READINGS):x.loc[d[r].isna(),r+'_filled']=p[d[r].isna(),j]
    return b['model'].predict(b['pre'].transform(x))

CONFIGS = {'catboost': {'harmonics': 3, 'interactions': True, 'context': True, 'mask_copies': 2, 'target_mode': 'raw', 'huber_mult': None, 'learning_rate': 0.018222152446466338, 'subsample': 0.6976430885613888, 'depth': 5, 'l2_leaf_reg': 23.14275781834203, 'random_strength': 2.893117383850708, 'rsm': 0.8829928757392587, 'bootstrap_type': 'Bernoulli', 'max_ctr_complexity': 1, 'grow_policy': 'SymmetricTree'}, 'lightgbm': {'harmonics': 0, 'interactions': False, 'context': True, 'mask_copies': 2, 'target_mode': 'raw', 'huber_mult': 2.5, 'learning_rate': 0.03273033378287702, 'subsample': 0.7601824593768162, 'num_leaves': 4, 'min_child_samples': 73, 'colsample_bytree': 0.48804698442955685, 'reg_lambda': 0.20948106722459173, 'reg_alpha': 0.3677582194002555, 'cat_smooth': 42.261427201487585, 'linear_tree': True, 'linear_lambda': 0.20386044885168195}, 'ridge': {'harmonics': 1, 'interactions': True, 'context': True, 'mask_copies': 1, 'target_mode': 'raw', 'huber_mult': None, 'knots': 8, 'ridge_alpha': 30.0, 'slope_scale': 1.0}, 'mlp': {'harmonics': 1, 'interactions': False, 'context': True, 'mask_copies': 2, 'target_mode': 'raw', 'huber_mult': None, 'knots': 6, 'slope_scale': 0.0, 'hidden': [64, 32], 'mlp_alpha': 0.1}}
WEIGHTS = {'catboost': 0.16648400909166414, 'lightgbm': 0.23781407845013355, 'ridge': 0.4944741096085642, 'mlp': 0.10122780284963817}

def predict_v8(bundle, data):
    result = np.zeros(len(data))
    for name, weight in WEIGHTS.items():
        if name == 'ridge':
            p = newpredict(bundle['learned_ridge'], data)
        else:
            p = np.mean([predict(m, data) for m in bundle['members'][name]], axis=0)
        result += weight * p
    return result

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=DATA_DIR/'train.csv')
    p.add_argument('--test', type=Path, default=DATA_DIR/'test.csv')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'energy_v8_results')
    p.add_argument('--v7-model', type=Path, help='Optional trusted V7 model trained on the same train.csv.')
    p.add_argument('--predict-only', type=Path, help='Previously saved V8 model.pkl; no retraining.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = load(args.test)
    if len(test) != 3000:
        raise ValueError('Expected 3000 test rows.')
    if args.predict_only:
        bundle = joblib.load(args.predict_only)
    else:
        train = load(args.train, labelled=True)
        if len(train) != 8000:
            raise ValueError('Expected 8000 training rows.')
        if set(train.id) & set(test.id):
            raise ValueError('Training and test IDs overlap.')
        patterns = test[READINGS].isna().to_numpy()
        MASK['scale'] = mask_scale(train, test)
        args.max_iterations = 7000
        args.patience = 250
        args.max_epochs = 150
        args.mlp_patience = 12
        args.threads = 4
        args.stress_weight = .5
        if args.v7_model:
            old = joblib.load(args.v7_model)
            if old['configs'] != CONFIGS:
                raise ValueError('This V7 checkpoint does not use the expected winning settings.')
            members = {name: old['members'][name] for name in WEIGHTS if name != 'ridge'}
            print('Reusing your V7 CatBoost, LightGBM and MLP members.', flush=True)
        else:
            members = trained_members(train, {k:v for k,v in CONFIGS.items() if k != 'ridge'},
                                      patterns, args, list(range(42,48)))
        print('Fitting cross-fitted reading estimators and improved spline-Ridge...', flush=True)
        learned = newfit(train, CONFIGS['ridge'], patterns)
        bundle = dict(version='energy_v8', members=members, learned_ridge=learned,
                      train_sha256=file_hash(args.train), test_sha256=file_hash(args.test),
                      configs=CONFIGS, weights=WEIGHTS, mask_scale=MASK['scale'])
        original = predict_v8(bundle, test)
        joblib.dump(bundle, args.output/'model.pkl')
        bundle = joblib.load(args.output/'model.pkl')
        np.testing.assert_allclose(original, predict_v8(bundle, test), rtol=1e-10, atol=1e-10)
        report = {k:bundle[k] for k in ['version','train_sha256','test_sha256','configs','weights','mask_scale']}
        (args.output/'settings.json').write_text(json.dumps(report, indent=2))
        (args.output/'requirements.txt').write_text('\n'.join(
            f'{name}=={importlib.metadata.version(name)}' for name in
            ['numpy','pandas','scikit-learn','catboost','lightgbm','scipy','joblib'])+'\n')
    predictions = predict_v8(bundle, test)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError('Invalid predictions.')
    # Preserve the test file's row order and the proven V7 submission header.
    pd.DataFrame({'prediction':predictions}).to_csv(args.output/'submission.csv', index=False)
    pd.DataFrame({'id':test.id.to_numpy(), TARGET:predictions}).to_csv(
        args.output/'predicted_energy_usage.csv', index=False)
    print('Saved:', args.output/'submission.csv', flush=True)
    print('Test labels are unavailable: portal RMSE cannot be measured locally.', flush=True)

if __name__ == '__main__':
    main()