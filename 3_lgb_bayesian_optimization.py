import warnings
warnings.simplefilter('ignore')

import gc

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.6f' % x)

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

from bayes_opt import BayesianOptimization

train = pd.read_pickle('train.pickle')

mapping_dict = {21:0, 1:1, 4:2, 5:3}
train['label'] = train.label.map(mapping_dict)
use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

def run_lgb(df_train, df_test, use_features, params, is_predict=False):
    
    target = 'label'
    oof_pred = np.zeros((len(df_train), 4))
    if is_predict:
        y_pred = np.zeros((len(df_test), 4))
    
    folds = GroupKFold(n_splits=5)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'], train['event_id'])):
        if is_predict:
            print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        if is_predict:
            verbose_eval = 100
        else:
            verbose_eval = 0
        
        model = lgb.train(params, 
                          train_set, 
                          num_boost_round=500,
                          early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          verbose_eval=verbose_eval)
        oof_pred[val_ind] = model.predict(x_val)
        
        if is_predict:
            y_pred += model.predict(df_test[use_features]) / folds.n_splits
                
        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
        
    y_one_hot = label_binarize(df_train['label'], np.arange(4)) 
    oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
    score = roc_auc_score(y_one_hot, oof_one_hot) 
        
    return score

def LGB_bayesian(learning_rate,
                 feature_fraction,
                 max_depth,
                 num_leaves,
                 lambda_l1,
                 lambda_l2,
                 bagging_fraction,
                 bagging_freq):
    
    params = {
        'metric': 'multiclass',
        'objective': 'multiclass',
        'num_classes': 4,
        'n_jobs': -1,
        'seed': 1029,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(bagging_freq),
        'verbose': 0
    }
    
    return run_lgb(train, test, use_features, params, is_predict=False)

init_points = 10
n_iter = 10

bounds_LGB = {
    'learning_rate': (0.08, 0.2),
    'feature_fraction': (0.4, 0.98),
    'max_depth': (8, 11),
    'num_leaves': (64, 128),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 5),
    'bagging_fraction': (0.4, 0.6),
    'bagging_freq': (1, 10)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1222)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


print(LGB_BO.max['params'])