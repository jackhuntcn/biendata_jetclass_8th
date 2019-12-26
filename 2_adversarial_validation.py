import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import gc

train_df = pd.read_pickle('train.pickle')
test_df = pd.read_pickle('test.pickle')

train_df.drop(['jet_id', 'event_id', 'label'], axis=1, inplace=True)
test_df.drop(['jet_id', 'event_id'], axis=1, inplace=True)

train_df['target'] = 0
test_df['target'] = 1

features = train_df.columns.tolist()

train_ = train_df.copy()
test_ = test_df.copy()

train_test = pd.concat([train_df, test_df], axis =0)

target = train_test['target'].values

del train_df, test_df
gc.collect()

train_df, test_df = train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)

del train_test
gc.collect()

feature_imp = train_df['target'].values
train_y = train_df['target'].values
test_y = test_df['target'].values

del train_df['target'], test_df['target']
gc.collect()

train = lgb.Dataset(train_df, label=train_y)
test = lgb.Dataset(test_df, label=test_y)

param = {
    'num_leaves': 50,
    'min_data_in_leaf': 30, 
    'objective':'binary',
    'max_depth': 5,
    'learning_rate': 0.2,
    "min_child_samples": 20,
    "boosting": "gbdt",
    "feature_fraction": 0.9,
    "bagging_freq": 1,
    "bagging_fraction": 0.9 ,
    "bagging_seed": 44,
    "metric": 'auc',
    "verbosity": -1
}

num_round = 100
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds=50)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance('gain'),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 100))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(500))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
# plt.savefig('lgbm_importances-01.png')

feature_imp[feature_imp.Value > 0]

for col in feature_imp[feature_imp.Value > 100]['Feature'].values.tolist():
    try:
        plt.figure(figsize=(20, 6))
        sns.distplot(train_[col], hist=False)
        sns.distplot(test_[col], hist=False)
        plt.show()
    except:
        pass