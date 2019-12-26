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

train = pd.read_pickle('train.pickle')
test = pd.read_pickle('test.pickle')

mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train.label.map(mapping_dict)

use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

def run_cat(df_train, df_test, use_features):
    
    target = 'label'
    oof_pred = np.zeros((len(df_train), 4))
    y_pred = np.zeros((len(df_test), 4))
    
    folds = GroupKFold(n_splits=5)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'], train['event_id'])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        
        params = {
            'task_type': 'GPU',
            'learning_rate': 0.1,
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'classes_count': 4,
            'iterations': 6000,
            'random_seed': 1029,
            'max_depth': 8,
            'max_leaves': 64,
            'reg_lambda': 0.5,
            'early_stopping_rounds': 100
        }
        
        model = CatBoostClassifier(**params)
        
        model.fit(x_train,
                  y_train,
                  eval_set=(x_val, y_val),
                  verbose=100)
        oof_pred[val_ind] = model.predict_proba(x_val)
        y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits
        
        y_one_hot = label_binarize(y_val, np.arange(4)) 
        oof_one_hot = label_binarize(oof_pred[val_ind].argmax(axis=1), np.arange(4)) 
        score = roc_auc_score(y_one_hot, oof_one_hot) 
        print('auc: ', score)
        
        del x_train, x_val, y_train, y_val
        gc.collect()
        
    return y_pred, oof_pred

y_pred, oof_pred = run_cat(train, test, use_features)

y_one_hot = label_binarize(train['label'], np.arange(4)) 
oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y_one_hot, oof_one_hot) 
print('auc: ', score)

submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
submission.label = y_pred.argmax(axis=1)
submission.label = submission.label.map(mapping_dict_inv)
submission.head()

submission.to_csv('submission_cat.csv', index=False)

np.save('y_pred_cat', y_pred)
np.save('oof_pred_cat', oof_pred)