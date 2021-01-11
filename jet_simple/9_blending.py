import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

lgb = pd.read_csv('./submission_lgb.csv')
xgb = pd.read_csv('./submission_xgb.csv')
cat = pd.read_csv('./submission_cat.csv')
nn = pd.read_csv('./submission_nn.csv')

pd.DataFrame({'lgb':lgb.label, 'xgb':xgb.label, 'cat':cat.label, 'nn':nn.label}).corr()

lgb_oof_pred = np.load('oof_pred_lgb.npy')
xgb_oof_pred = np.load('oof_pred_xgb.npy')
cat_oof_pred = np.load('oof_pred_cat.npy')
nn_oof_pred = np.load('oof_pred_nn.npy')

lgb_y_pred = np.load('y_pred_lgb.npy')
xgb_y_pred = np.load('y_pred_xgb.npy')
cat_y_pred = np.load('y_pred_cat.npy')
nn_y_pred = np.load('y_pred_nn.npy')

DATA_DIR = './jet_simple_data/'

train = pd.read_csv(DATA_DIR+'simple_train_R04_jet.csv')
mapping_dict = {21:0, 1:1, 4:2, 5:3}
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

train['label'] = train.label.map(mapping_dict)
y_one_hot = label_binarize(train['label'], np.arange(4)) 

best_score = 0.

for a in np.arange(0.05, 1, 0.05):
    for b in np.arange(0.05, 1, 0.05):
        for c in np.arange(0.05, 1, 0.05):
            for d in np.arange(0.05, 1, 0.05):
                if a + b + c + d != 1:
                    continue
                else:
                    oof_pred_blend = a*lgb_oof_pred + b*xgb_oof_pred + c*cat_oof_pred + d*nn_oof_pred
                    oof_one_hot = label_binarize(oof_pred_blend.argmax(axis=1), np.arange(4))
                    score = roc_auc_score(y_one_hot, oof_one_hot)
                    print(f'a={a} b={b} c={c} d={d} score: {score}')
                    if best_score < score:
                        best_score = score
                    
print(f'best_score: {best_score}')

submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
y_pred_blend = 0.4*lgb_y_pred + 0.25*xgb_y_pred + 0.15*cat_y_pred + 0.2*nn_y_pred
submission.label = y_pred_blend.argmax(axis=1)
submission.label = submission.label.map(mapping_dict_inv)
submission.head()

submission.to_csv('submission_lgb_xgb_cat_nn_blending.csv', index=False)