import numpy as np
import pandas as pd

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.6f' % x)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats

lgb_oof_pred = np.load('oof_pred_v15_lgb.npy')
xgb_oof_pred = np.load('oof_pred_v15_xgb.npy')
cat_oof_pred = np.load('oof_pred_v15_cat.npy')
nn_oof_pred = np.load('oof_pred_v15_nn.npy')

lgb_y_pred = np.load('y_pred_v15_lgb.npy')
xgb_y_pred = np.load('y_pred_v15_xgb.npy')
cat_y_pred = np.load('y_pred_v15_cat.npy')
nn_y_pred = np.load('y_pred_v15_nn.npy')

df1 = pd.DataFrame(lgb_oof_pred)
df1.columns = ['c'+str(i) for i in range(0,4)]
df2 = pd.DataFrame(xgb_oof_pred)
df2.columns = ['c'+str(i) for i in range(4,8)]
df3 = pd.DataFrame(cat_oof_pred)
df3.columns = ['c'+str(i) for i in range(8,12)]
df4 = pd.DataFrame(nn_oof_pred)
df4.columns = ['c'+str(i) for i in range(12,16)]

train = pd.concat([df1, df2, df3, df4], axis=1)
train.head()

y = pd.read_csv('./jet_simple_data/simple_train_R04_jet.csv')['label']

sc = StandardScaler()
sc.fit(train)
X_train_std = sc.transform(train)

lr = LogisticRegression(C=0.01, random_state=0)
lr.fit(X_train_std, y)

df1 = pd.DataFrame(lgb_y_pred)
df1.columns = ['c'+str(i) for i in range(0,4)]
df2 = pd.DataFrame(xgb_y_pred)
df2.columns = ['c'+str(i) for i in range(4,8)]
df3 = pd.DataFrame(cat_y_pred)
df3.columns = ['c'+str(i) for i in range(8,12)]
df4 = pd.DataFrame(nn_y_pred)
df4.columns = ['c'+str(i) for i in range(12,16)]

test = pd.concat([df1, df2, df3, df4], axis=1)
test.head()

X_test_std = sc.transform(test)
sub = pd.DataFrame()
sub['label'] = lr.predict(X_test_std)

sub2 = pd.read_csv('./jet_simple_data/simple_test_R04_jet.csv')[['jet_id','event_id']]

sub['id'] = sub2['jet_id']
sub['event_id'] = sub2['event_id']
sub['label'] = sub.groupby(['event_id'])['label'].transform(lambda x:stats.mode(x)[0][0])
sub.head()

y_one_hot = label_binarize(y, np.arange(4)) 
oof_pred = lr.predict_proba(X_train_std)
oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y_one_hot, oof_one_hot) 
print('auc: ', score)

sub[['id','label']].to_csv('submission_lgb_xgb_cat_nn_stacking.csv', index=False)

y_prob = lr.predict_proba(X_test_std)
df = pd.DataFrame(y_prob)
df.columns = ['p1', 'p2', 'p3', 'p4']
df.to_csv('stacking_prob.csv', index=False)