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

train = pd.read_pickle('train.pickle')
test = pd.read_pickle('test.pickle')

mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train.label.map(mapping_dict)

use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

def run_lgb(df_train, df_test, use_features):
    
    target = 'label'
    oof_pred = np.zeros((len(df_train), 4))
    y_pred = np.zeros((len(df_test), 4))
    
    folds = GroupKFold(n_splits=5)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'], train['event_id'])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        
        params = {
            'learning_rate': 0.08020673463424263,
            'metric': 'multiclass',
            'objective': 'multiclass',
            'num_classes': 4,
            'feature_fraction': 0.6413169921718676,
            'bagging_fraction': 0.7633258873630561,
            'bagging_freq': 9,
            'n_jobs': 8,
            'seed': 1029,
            'max_depth': 8,
            'num_leaves': 64,
            'lambda_l1': 4.612904421522525,
            'lambda_l2': 1.7265948031817602
        }
        
        model = lgb.train(params, 
                          train_set, 
                          num_boost_round=500,
                          early_stopping_rounds=100,
                          valid_sets=[train_set, val_set],
                          verbose_eval=100)
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits
        
        y_one_hot = label_binarize(y_val, np.arange(4)) 
        oof_one_hot = label_binarize(oof_pred[val_ind].argmax(axis=1), np.arange(4)) 
        score = roc_auc_score(y_one_hot, oof_one_hot) 
        print('auc: ', score)
        
        print("Features importance...")
        gain = model.feature_importance('gain')
        feat_imp = pd.DataFrame({'feature': model.feature_name(), 
                         'split': model.feature_importance('split'), 
                         'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        print('Top 50 features:\n', feat_imp.head(50))
        
        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
        
    return y_pred, oof_pred


y_pred, oof_pred = run_lgb(train, test, use_features)

y_one_hot = label_binarize(train['label'], np.arange(4)) 
oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y_one_hot, oof_one_hot) 
print('auc: ', score)

submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
submission.label = y_pred.argmax(axis=1)
submission.label = submission.label.map(mapping_dict_inv)
submission.head()

submission.to_csv('submission_lgb.csv', index=False)

np.save('y_pred_lgb', y_pred)
np.save('oof_pred_lgb', oof_pred)