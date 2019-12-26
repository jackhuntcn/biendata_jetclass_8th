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

from catboost import CatBoostClassifier

from bayes_opt import BayesianOptimization

train = pd.read_pickle('train.pickle')
test = pd.read_pickle('test.pickle')

mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train.label.map(mapping_dict)

use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

def run_cat(df_train, df_test, use_features, params, is_predict=False):

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
        
        if is_predict:
            verbose_eval = 100
        else:
            verbose_eval = 0
        
        model = CatBoostClassifier(**params)
        model.fit(x_train,
                  y_train,
                  eval_set=(x_val, y_val),
                  verbose=verbose_eval)
        
        oof_pred[val_ind] = model.predict_proba(x_val)
        
        if is_predict:
            y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits
                
        del x_train, x_val, y_train, y_val, model
        gc.collect()
        
    y_one_hot = label_binarize(df_train['label'], np.arange(4)) 
    oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
    score = roc_auc_score(y_one_hot, oof_one_hot) 
        
    return score

def CAT_bayesian(learning_rate,
                 max_depth,
#                  max_leaves,
                 reg_lambda):
    
    params = {
        'task_type': 'GPU',
        'learning_rate': learning_rate,
        'eval_metric': 'MultiClass',
        'loss_function': 'MultiClass',
        'iterations': 6000,
        'classes_count': 4,
        'random_seed': 1029,
        'max_depth': int(max_depth),
#         'max_leaves': int(max_leaves),
        'reg_lambda': reg_lambda,
        'early_stopping_rounds': 100
    }
    
    return run_cat(train, test, use_features, params, is_predict=False)

init_points = 10
n_iter = 10

bounds_CAT = {
    'learning_rate': (0.08, 0.2),
    'max_depth': (8, 12),
#     'max_leaves': (32, 64),
    'reg_lambda': (0, 5)
}

CAT_BO = BayesianOptimization(CAT_bayesian, bounds_CAT, random_state=1222)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    CAT_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

print(CAT_BO.max['params'])