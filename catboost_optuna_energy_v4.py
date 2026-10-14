"""V4: CatBoost + LightGBM linear leaves + spline-Ridge + validated residual correction.
No timestamp is present: no ID-based history or TimeSeriesSplit is fabricated.
Run --help for options. Default: 24 CatBoost + 16 LightGBM attempted trials.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error

VERSION = 'energy_v4_1'
TARGET = 'energy_usage'
READINGS = ['temperature', 'humidity', 'occupancy', 'previous_usage']
NUM = READINGS + ['hour', 'month']
CAT = ['building_id', 'building_type', 'day_of_week']
DAYS = dict(zip(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], range(7)))
DATA_DIR = Path('/Users/ArjunMittal/Downloads/Track 1 Dataset')
# Warm starts from the user's saved V3 winners.
CB = {'learning_rate': 0.03966226895995887, 'depth': 3, 'l2_leaf_reg': 13.41812135038036, 'random_strength': 2.158832460665511, 'rsm': 0.6907717742149335, 'max_ctr_complexity': 1, 'bootstrap_type': 'Bernoulli', 'subsample': 0.7120037675898786, 'grow_policy': 'SymmetricTree'}
LGB = {'learning_rate': 0.026389615338773015, 'num_leaves': 5, 'min_child_samples': 111, 'subsample': 0.7682670544928751, 'colsample_bytree': 0.42552187655358054, 'reg_lambda': 0.16368666882056326, 'reg_alpha': 0.0010161005492200523, 'cat_smooth': 42.53034176409328, 'linear_tree': True, 'linear_lambda': 0.5535880981079766}
OPTIONS = ['harmonics','interactions','context','mask_copy','target_mode','huber_mult']


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
    for c, lo, hi in [('hour',0,23), ('month',1,12)]:
        v = d[c].dropna()
        if not (v.between(lo,hi) & (v % 1 == 0)).all():
            raise ValueError(f'Invalid {c} values')
    unknown_days = set(d.day_of_week.dropna()) - set(DAYS)
    if unknown_days:
        raise ValueError(f'Expected weekday names; found {unknown_days}')
    return d.reset_index(drop=True)


def mask(d, patterns, seed, probability=1.):
    rng = np.random.default_rng(seed)
    z = d.copy()
    m = patterns[rng.integers(len(patterns), size=len(z))].copy()
    m &= (rng.random(len(z)) < probability)[:, None]
    for j,c in enumerate(READINGS):
        z.loc[m[:,j], c] = np.nan
    return z


def augmented(d, patterns, enabled):
    return pd.concat([d,mask(d,patterns,202600)],ignore_index=True) if enabled else d.copy()


def assessment_views(d, patterns, seeds=(2026,2027)):
    return [d] + [mask(d,patterns,s,.8) for s in seeds]


def rmse(y,p):
    return float(np.sqrt(mean_squared_error(y,p)))


def scores(y, predictions, stress_weight):
    clean = mean_squared_error(y,predictions[0])
    stress = np.mean([mean_squared_error(y,p) for p in predictions[1:]])
    return dict(clean_RMSE=float(np.sqrt(clean)), stress_RMSE=float(np.sqrt(stress)),
                selection_RMSE=float(np.sqrt((1-stress_weight)*clean+stress_weight*stress)))


def fit_context(d):
    # INPUT-only summaries, fitted exclusively within the training partition.
    # These are building context, NOT historical features or target encodings.
    tables = {}
    for keys in [('building_id',), ('building_id','hour'), ('building_id','day_of_week'), ('building_type','hour')]:
        g = d.groupby(list(keys), dropna=False)[READINGS]
        count, mean = g.count(), g.mean()
        fallback = d[READINGS].mean().fillna(0.)
        smoothed = (mean.fillna(fallback)*count + 20*fallback)/(count+20)
        tables[keys] = (smoothed, fallback)
    return tables


def features(d, cfg, tables=None):
    x = d[NUM+CAT].copy()
    cats = list(CAT)
    for c in cats:
        x[c] = x[c].astype('string').fillna('Unknown').astype(str)
    for c in READINGS:
        x[c+'_missing'] = x[c].isna().astype(int)
    x['missing_count'] = x[READINGS].isna().sum(axis=1)
    h = cfg['harmonics']
    weekday = d.day_of_week.map(DAYS).astype(float)
    for name, values, period in [('hour',x.hour,24),('weekday',weekday,7),('month',x.month-1,12)]:
        for k in range(1, min(h,3)+1):
            x[f'{name}_sin{k}'] = np.sin(2*np.pi*k*values/period)
            x[f'{name}_cos{k}'] = np.cos(2*np.pi*k*values/period)
    if cfg['interactions']:
        hour = x.hour.astype('Int64').astype('string').fillna('Unknown')
        for name, val in {
            'building_type_hour':x.building_type+'__'+hour,
            'building_id_hour':x.building_id+'__'+hour,
            'building_type_weekday':x.building_type+'__'+x.day_of_week,
        }.items():
            x[name] = val.astype(str); cats.append(name)
        x['is_weekend'] = weekday.isin([5,6]).astype(int)
        x['temperature_x_occupancy'] = x.temperature*x.occupancy
        x['temperature_x_humidity'] = x.temperature*x.humidity
    if tables:
        for keys,(table,fallback) in tables.items():
            idx = (pd.Index(d[keys[0]]) if len(keys)==1 else pd.MultiIndex.from_frame(d[list(keys)]))
            values = table.reindex(idx).reset_index(drop=True).fillna(fallback)
            for c in READINGS:
                name = '_'.join(keys)+'_'+c+'_context'
                x[name] = values[c].to_numpy()
                x[name+'_deviation'] = x[c]-x[name]
    return x,cats


def encode(x,cats,categories=None):
    x=x.copy()
    # Keep original base categories; interaction categories suit CatBoost better.
    x=x.drop(columns=[c for c in cats if c not in CAT])
    if categories is None:
        categories={c:sorted(x[c].unique()) for c in CAT}
    for c in CAT:
        x[c]=pd.Categorical(x[c],categories=categories[c])
    return x,categories


def unpack(cfg):
    return {k:v for k,v in cfg.items() if k not in OPTIONS}


def inverse(p, mode):
    # Modest guard against numerical overflow, not data-dependent clipping.
    return np.expm1(np.clip(p,-30,30)) if mode=='log1p' else np.asarray(p)


class OriginalEnergyRMSE:
    """CatBoost stopping metric in energy units for log-target training."""
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        pred = np.expm1(np.clip(np.asarray(approxes[0]), -30, 30))
        actual = np.expm1(np.asarray(target))
        w = np.ones(len(target)) if weight is None else np.asarray(weight)
        return float(np.sum(w*(pred-actual)**2)), float(np.sum(w))

    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))


def lgb_original_rmse(y_true, y_pred, weight):
    diff = np.expm1(y_true) - np.expm1(np.clip(y_pred, -30, 30))
    return 'energy_RMSE', float(np.sqrt(np.average(diff**2, weights=weight))), False


def ridge_features(d,cfg,tables):
    x,cats=features(d,cfg,tables)
    for c in ['hour','month']:
        name=c+'_category'
        x[name]=x[c].astype('Int64').astype('string').fillna('Unknown').astype(str)
        cats.append(name)
    return x,cats


def scaling(x,state=None):
    x=x.copy()
    cols=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]
    if state is None:
        state=(x[cols].mean().fillna(0.), x[cols].std().replace(0,np.nan).fillna(1.))
    means,stds=state
    x[means.index]=(x[means.index]-means)/stds
    return x,state


def fit(family,d,cfg,patterns,args,trees=None,stop=None,seed=42):
    tables=fit_context(d) if cfg['context'] else None
    z=augmented(d,patterns,cfg['mask_copy'])
    if family=='ridge':
        x,cats=ridge_features(z,cfg,tables)
        other=[c for c in x if c not in READINGS+cats]
        pre=ColumnTransformer([
            ('spline',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                 SplineTransformer(n_knots=cfg['knots'],degree=3,extrapolation='constant')),READINGS),
            ('num',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),other),
            ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=5),cats)])
        model=make_pipeline(pre,Ridge(alpha=cfg['ridge_alpha'],solver='lsqr'))
        model.fit(x,z[TARGET])
        return dict(family=family,model=model,config=cfg,tables=tables,categories=None,trees=0,scale=None)
    x,cats=features(z,cfg,tables)
    y=np.log1p(z[TARGET]) if cfg['target_mode']=='log1p' else z[TARGET]
    categories=None
    scale=None
    if family=='lightgbm':
        x,categories=encode(x,cats)
        if cfg.get('linear_tree',False): x,scale=scaling(x)
    sx=sy=None
    if stop is not None:
        # A prespecified clean/stress mixture for early stopping.
        views=assessment_views(stop,patterns)
        stopdata=pd.concat(views,ignore_index=True)
        sx,_=features(stopdata,cfg,tables)
        sy=np.log1p(stopdata[TARGET]) if cfg['target_mode']=='log1p' else stopdata[TARGET]
        if family=='lightgbm':
            sx,_=encode(sx,cats,categories)
            if scale is not None:sx,_=scaling(sx,scale)
        w=np.concatenate([np.full(len(stop),(1-args.stress_weight)),
                          np.full(2*len(stop),args.stress_weight/2)])
    params=unpack(cfg)
    huber=cfg.get('huber_mult')
    delta=None if huber is None else max(1e-6,float(huber)*float(np.std(y)))
    if family=='catboost':
        from catboost import Pool
        model=CatBoostRegressor(iterations=int(trees or args.max_iterations),
            loss_function='RMSE' if delta is None else f'Huber:delta={delta}',
            eval_metric=(OriginalEnergyRMSE() if stop is not None and cfg['target_mode']=='log1p' else 'RMSE'),
            nan_mode='Min',thread_count=args.threads,
            random_seed=seed,allow_writing_files=False,**params)
        kw={} if stop is None else dict(eval_set=Pool(sx,sy,cat_features=cats,weight=w),
                                       early_stopping_rounds=args.patience,use_best_model=True)
        model.fit(x,y,cat_features=cats,verbose=False,**kw)
        n=int(model.tree_count_)
    else:
        model=lgb.LGBMRegressor(n_estimators=int(trees or args.max_iterations),
            objective='regression' if delta is None else 'huber',
            **({} if delta is None else {'alpha':delta}),metric='None',verbosity=-1,random_state=seed,n_jobs=args.threads,
            subsample_freq=1,**params)
        kw={}
        if stop is not None:
            # Support the installed API and versions before eval_X/eval_y existed.
            if 'eval_X' in inspect.signature(model.fit).parameters:
                kw.update(eval_X=sx,eval_y=sy)
            else: kw['eval_set']=[(sx,sy)]
            kw.update(eval_sample_weight=[w],
                      eval_metric=lgb_original_rmse if cfg['target_mode']=='log1p' else 'rmse',
                      callbacks=[lgb.early_stopping(args.patience,verbose=False)])
        model.fit(x,y,**kw)
        n=int(model.best_iteration_ or trees or args.max_iterations)
    return dict(family=family,model=model,config=cfg,tables=tables,categories=categories,trees=n,scale=scale)


def predict(bundle,d):
    if bundle['family']=='ridge':
        x,_=ridge_features(d,bundle['config'],bundle['tables'])
        return np.asarray(bundle['model'].predict(x))
    x,cats=features(d,bundle['config'],bundle['tables'])
    if bundle['family']=='lightgbm':
        x,_=encode(x,cats,bundle['categories'])
        if bundle['scale'] is not None:x,_=scaling(x,bundle['scale'])
    return inverse(bundle['model'].predict(x),bundle['config']['target_mode'])


def select_trees(family,d,cfg,patterns,args):
    if family=='ridge':return 0
    a,b=train_test_split(d,test_size=.2,random_state=123,stratify=d.building_id)
    selector=fit(family,a,cfg,patterns,args,stop=b)
    return selector['trees']


def evaluate_cv(family,d,cfg,patterns,args,trial=None):
    folds=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=42)
    # Same stress masks for every candidate. Scoring labels never choose tree count.
    views=assessment_views(d,patterns)
    oof=np.full((3,len(d)),np.nan)
    rows=[]
    for f,(tr,va) in enumerate(folds.split(d,d.building_id),1):
        train=d.iloc[tr]
        n=select_trees(family,train,cfg,patterns,args)
        model=fit(family,train,cfg,patterns,args,trees=n)
        for j,v in enumerate(views):oof[j,va]=predict(model,v.iloc[va])
        result=scores(d[TARGET].iloc[va],oof[:,va],args.stress_weight)
        rows.append(dict(fold=f,trees=n,**result))
        print(f"{family} fold {f}: clean={result['clean_RMSE']:.4f} "
              f"stress={result['stress_RMSE']:.4f} trees={n}",flush=True)
        if trial is not None:
            trial.report(np.mean([r['selection_RMSE'] for r in rows]),f)
            if trial.should_prune():raise optuna.TrialPruned()
    if not np.isfinite(oof).all():raise RuntimeError('Invalid OOF predictions')
    return scores(d[TARGET],oof,args.stress_weight),oof,rows


def starts(family):
    base=dict(CB if family=='catboost' else LGB)
    base.update(harmonics=3 if family=='catboost' else 0,interactions=True,
                context=True,mask_copy=False,target_mode='raw',
                huber_mult=None if family=='catboost' else 1.5)
    # First five trials isolate substantive changes before broader searching.
    variants=[base,dict(base,mask_copy=True),
              dict(base,mask_copy=True,harmonics=1),
              dict(base,mask_copy=True,target_mode='log1p',huber_mult=None),
              dict(base,mask_copy=True,context=False)]
    return variants


def suggest(trial,family):
    p=dict(harmonics=trial.suggest_categorical('harmonics',[0,1,3]),
           interactions=trial.suggest_categorical('interactions',[False,True]),
           context=trial.suggest_categorical('context',[False,True]),
           mask_copy=trial.suggest_categorical('mask_copy',[False,True]),
           target_mode=trial.suggest_categorical('target_mode',['raw','log1p']),
           huber_mult=trial.suggest_categorical('huber_mult',[None,1.5,2.5]),
           learning_rate=trial.suggest_float('learning_rate',.015,.07,log=True),
           subsample=trial.suggest_float('subsample',.65,1.))
    if family=='catboost':
        p.update(depth=trial.suggest_int('depth',2,5),
                 l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,30,log=True),
                 random_strength=trial.suggest_float('random_strength',0,3),
                 rsm=trial.suggest_float('rsm',.5,1),bootstrap_type='Bernoulli',
                 max_ctr_complexity=trial.suggest_int('max_ctr_complexity',1,2),
                 grow_policy='SymmetricTree')
    else:
        p.update(num_leaves=trial.suggest_int('num_leaves',4,24,log=True),
                 min_child_samples=trial.suggest_int('min_child_samples',25,130),
                 colsample_bytree=trial.suggest_float('colsample_bytree',.4,1),
                 reg_lambda=trial.suggest_float('reg_lambda',.05,30,log=True),
                 reg_alpha=trial.suggest_float('reg_alpha',.001,2,log=True),
                 cat_smooth=trial.suggest_float('cat_smooth',5,50),
                 linear_tree=trial.suggest_categorical('linear_tree',[False,True]))
        if p['linear_tree']:
            p['linear_lambda']=trial.suggest_float('linear_lambda',.05,5,log=True)
    return p


def tune(family,d,patterns,args,signature):
    study=optuna.create_study(study_name=f'{family}_{signature}',direction='minimize',
        storage='sqlite:///'+str((args.output/'studies.sqlite3').resolve()),load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=3))
    if not study.trials:
        for cfg in starts(family): study.enqueue_trial(cfg)
    def objective(trial):
        cfg=suggest(trial,family)
        result,oof,rows=evaluate_cv(family,d,cfg,patterns,args,trial)
        trial.set_user_attr('config',cfg);trial.set_user_attr('scores',result)
        trial.set_user_attr('folds',rows)
        np.save(args.output/f'{family}_{signature}_{trial.number}_oof.npy',oof)
        return result['selection_RMSE']
    total=args.trials if family=='catboost' else args.lgb_trials
    done=sum(t.state.is_finished() for t in study.trials)
    if total>done:study.optimize(objective,n_trials=total-done,gc_after_trial=True)
    best=study.best_trial
    study.trials_dataframe().to_csv(args.output/f'{family}_trials.csv',index=False)
    path=args.output/f'{family}_{signature}_{best.number}_oof.npy'
    if path.exists():oof=np.load(path)
    else:_,oof,_=evaluate_cv(family,d,best.user_attrs['config'],patterns,args)
    pd.DataFrame(best.user_attrs['folds']).to_csv(args.output/f'{family}_best_folds.csv',index=False)
    return best.user_attrs['config'],oof,best.user_attrs['scores']


def learn_blend(y, predictions, stress_weight):
    # Three-dimensional convex stacking; a member can receive zero weight.
    matrix=np.stack(predictions,axis=-1)
    weights=np.array([1-stress_weight,stress_weight/2,stress_weight/2])[:,None]
    def loss(w):
        return float(np.mean(np.sum(weights*(matrix@w-np.asarray(y)[None,:])**2,axis=0)))
    result=minimize(loss,np.full(len(predictions),1/len(predictions)),method='SLSQP',
                    bounds=[(0,1)]*len(predictions),constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],
                    options=dict(ftol=1e-10,maxiter=300))
    if not result.success:raise RuntimeError('Blend optimisation failed: '+result.message)
    return result.x


def residual_features(d,predictions):
    x=d[NUM+CAT].copy()
    for c in CAT:x[c]=x[c].astype('string').fillna('Unknown').astype(str)
    for c in READINGS:x[c+'_missing']=x[c].isna().astype(int)
    for j,pred in enumerate(predictions):x[f'member_{j}']=np.asarray(pred)
    x['member_spread']=np.std(np.column_stack(predictions),axis=1)
    return x


def fit_residual(d,oofs,patterns,weights,args):
    views=assessment_views(d,patterns)
    x=pd.concat([residual_features(v,[o[j] for o in oofs]) for j,v in enumerate(views)],ignore_index=True)
    blend=np.stack(oofs,axis=-1)@weights
    y=np.tile(d[TARGET].to_numpy(),3)-blend.reshape(-1)
    sample_weights=np.repeat([1-args.stress_weight,args.stress_weight/2,args.stress_weight/2],len(d))
    model=CatBoostRegressor(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=30,
                           loss_function='RMSE',random_seed=42,allow_writing_files=False,
                           thread_count=args.threads,verbose=False)
    model.fit(x,y,cat_features=CAT,sample_weight=sample_weights)
    return model


def family_predictions(members,views):
    return {family:np.asarray([np.mean([predict(m,v) for m in models],axis=0) for v in views])
            for family,models in members.items()}


def combined(preds,names,weights):
    return np.stack([preds[n] for n in names],axis=-1)@weights


def final_prediction(bundle,d):
    preds=family_predictions(bundle['members'],[d])
    blend=combined(preds,bundle['names'],bundle['weights'])[0]
    if bundle['residual_strength']:
        x=residual_features(d,[preds[n][0] for n in bundle['names']])
        blend=blend+bundle['residual_strength']*bundle['residual'].predict(x)
    return blend


def ridge_search(d,patterns,args):
    best=None;records=[]
    # Alpha is assessed by original-row CV, not leave-one-out over duplicated rows.
    for alpha in [1.,10.,100.]:
        cfg=dict(harmonics=1,interactions=False,context=True,mask_copy=True,
                 target_mode='raw',huber_mult=None,knots=8,ridge_alpha=alpha)
        result,oof,folds=evaluate_cv('ridge',d,cfg,patterns,args)
        records.append(dict(alpha=alpha,**result))
        if best is None or result['selection_RMSE']<best[2]['selection_RMSE']:
            best=(cfg,oof,result)
    pd.DataFrame(records).to_csv(args.output/'ridge_search.csv',index=False)
    return best


def trained_members(data,configs,patterns,args):
    members={}
    for family,cfg in configs.items():
        trees=select_trees(family,data,cfg,patterns,args)
        seeds=args.seeds if family!='ridge' else [42]
        print(f'Fitting {family}: {len(data)} original rows, trees={trees}, seeds={seeds}',flush=True)
        members[family]=[fit(family,data,cfg,patterns,args,trees=trees,seed=s) for s in seeds]
    return members


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=DATA_DIR/'train.csv')
    parser.add_argument('--test',type=Path,default=DATA_DIR/'test.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'catboost_optuna_results_v4')
    parser.add_argument('--trials',type=int,default=24)
    parser.add_argument('--lgb-trials',type=int,default=16)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--max-iterations',type=int,default=7000)
    parser.add_argument('--patience',type=int,default=250)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--stress-weight',type=float,default=.5,
                        help='Prespecified tuning mixture, not a portal RMSE estimate.')
    parser.add_argument('--predict-only',type=Path)
    args=parser.parse_args()
    if min(args.trials,args.lgb_trials,args.max_iterations,args.patience,args.threads)<1 or args.folds<2:
        parser.error('Budgets must be positive; folds must be at least two.')
    if not 0<args.stress_weight<1:parser.error('stress-weight must be between 0 and 1.')
    args.output.mkdir(parents=True,exist_ok=True)
    test=load(args.test)
    if len(test)!=3000:raise ValueError('Expected 3000 test rows.')
    if args.predict_only:
        final=joblib.load(args.predict_only)
        if final['version']!=VERSION:raise ValueError('Model version mismatch')
    else:
        train=load(args.train,True)
        if len(train)!=8000:raise ValueError('Expected 8000 training rows.')
        if set(train.id)&set(test.id):raise ValueError('Train/test IDs overlap')
        patterns=test[READINGS].isna().to_numpy()
        print(f'Target skew: {train[TARGET].skew():.3f}. Log transform is a candidate, not assumed better.')
        print('No trustworthy timestamp: building-stratified folds; no ID-based lags.',flush=True)
        dev,hold=train_test_split(train,test_size=.2,random_state=42,stratify=train.building_id)
        dev=dev.reset_index(drop=True);hold=hold.reset_index(drop=True)
        signature=hashlib.sha256(args.train.read_bytes()+args.test.read_bytes()+
            Path(__file__).read_bytes()+json.dumps([args.folds,args.max_iterations,args.patience,
                                                  args.stress_weight]).encode()).hexdigest()[:16]
        # Independent meta-selection partition: not used by base tuning or OOF fitting.
        search,calibration=train_test_split(dev,test_size=.2,random_state=2025,stratify=dev.building_id)
        search=search.reset_index(drop=True);calibration=calibration.reset_index(drop=True)
        print(f'Search={len(search)}, residual selection={len(calibration)}, holdout={len(hold)}',flush=True)
        configs={};oofs={};selection={}
        for family in ['catboost','lightgbm']:
            configs[family],oofs[family],selection[family]=tune(family,search,patterns,args,signature)
        configs['ridge'],oofs['ridge'],selection['ridge']=ridge_search(search,patterns,args)
        names=list(configs)
        weights=learn_blend(search[TARGET],[oofs[n] for n in names],args.stress_weight)
        residual=fit_residual(search,[oofs[n] for n in names],patterns,weights,args)
        print('OOF-fitted blend weights:',dict(zip(names,weights.tolist())),flush=True)
        print('OOF scores are selection scores, not independent test scores.')
        # No resplitting of the OOF matrix to claim independent nonlinear-meta CV.
        calibration_members=trained_members(search,configs,patterns,args)
        cal_views=assessment_views(calibration,patterns,seeds=(4026,4027))
        cal_preds=family_predictions(calibration_members,cal_views)
        cal_blend=combined(cal_preds,names,weights)
        correction=np.asarray([residual.predict(residual_features(v,[cal_preds[n][j] for n in names]))
                               for j,v in enumerate(cal_views)])
        correction_rows=[]
        for strength in [0.,.25,.5,1.]:
            result=scores(calibration[TARGET],cal_blend+strength*correction,args.stress_weight)
            correction_rows.append(dict(strength=strength,**result))
        winner=min(correction_rows,key=lambda r:r['selection_RMSE'])
        # Small prespecified improvement threshold to avoid adding a nearly tied correction.
        strength=winner['strength'] if winner['selection_RMSE']<correction_rows[0]['selection_RMSE']-.005 else 0.
        pd.DataFrame(correction_rows).to_csv(args.output/'residual_selection.csv',index=False)
        print(f'Residual strength selected on separate calibration rows: {strength}',flush=True)
        # Configs, weights, correction and strength now frozen before examining holdout.
        hold_members=trained_members(dev,configs,patterns,args)
        holdviews=assessment_views(hold,patterns,seeds=(3026,3027))
        held=family_predictions(hold_members,holdviews)
        blend=combined(held,names,weights)
        ensemble=blend.copy()
        if strength:
            for j,v in enumerate(holdviews):
                ensemble[j]+=strength*residual.predict(residual_features(v,[held[n][j] for n in names]))
        rows=[dict(model=n,**scores(hold[TARGET],held[n],args.stress_weight)) for n in names]
        rows += [dict(model='convex_stack',**scores(hold[TARGET],blend,args.stress_weight)),
                 dict(model='selected_v4',**scores(hold[TARGET],ensemble,args.stress_weight))]
        pd.DataFrame(rows).to_csv(args.output/'holdout_metrics.csv',index=False)
        print(pd.DataFrame(rows).round(4).to_string(index=False),flush=True)
        print('Holdout rows were examined in prior experiments; these are NOT portal scores.')
        pd.DataFrame({'id':hold.id,'actual':hold[TARGET],'prediction':ensemble[0]}).to_csv(
            args.output/'holdout_predictions.csv',index=False)
        groups=pd.DataFrame({'missing_count':hold[READINGS].isna().sum(axis=1),
                             'absolute_error':np.abs(hold[TARGET]-ensemble[0]),
                             'squared_error':(hold[TARGET]-ensemble[0])**2})
        groups.groupby('missing_count').agg(rows=('absolute_error','size'),MAE=('absolute_error','mean'),
                                          MSE=('squared_error','mean')).to_csv(args.output/'errors_by_missingness.csv')
        # All 8000 labels train final base members. Meta-model retains its OOF training.
        final=dict(version=VERSION,members=trained_members(train,configs,patterns,args),
                   names=names,weights=weights,residual=residual,residual_strength=strength,configs=configs)
        joblib.dump(final,args.output/'model.pkl')
        reloaded=joblib.load(args.output/'model.pkl')
        np.testing.assert_allclose(final_prediction(final,test),final_prediction(reloaded,test))
        (args.output/'best_parameters.json').write_text(json.dumps(dict(configs=configs,
             blend_weights=dict(zip(names,weights.tolist())),residual_strength=strength,
             cv_selection_scores=selection,signature=signature,seeds=args.seeds,
             stress_weight=args.stress_weight),indent=2))
        (args.output/'requirements.txt').write_text('\n'.join(
            f'{p}=={importlib.metadata.version(p)}' for p in
            ['catboost','lightgbm','optuna','pandas','numpy','scikit-learn','scipy','joblib'])+'\n')
    pred=final_prediction(final,test)
    if pred.shape!=(3000,) or not np.isfinite(pred).all():raise RuntimeError('Invalid predictions')
    pd.DataFrame({'prediction':pred}).to_csv(args.output/'submission.csv',index=False)
    pd.DataFrame({'id':test.id.to_numpy(),TARGET:pred}).to_csv(args.output/'predicted_energy_usage.csv',index=False)
    print('Saved',args.output/'submission.csv',flush=True)


if __name__=='__main__':main()
