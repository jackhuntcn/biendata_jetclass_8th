import warnings
warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
pd.set_option('float_format', lambda x: '%.6f' % x)

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler

from keras.layers import Dense, Input, Activation, concatenate
from keras.layers import BatchNormalization, Add, Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K

train = pd.read_pickle('train.pickle')
test = pd.read_pickle('test.pickle')

data = pd.concat([train, test], axis=0)
data = data.fillna(0)

use_features = [col for col in train.columns if col not in ['jet_id', 'event_id', 'label']]

X = data[use_features].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

train_X = X[:len(train)]
test_X = X[len(train):]

y = label_binarize(train['label'].map({21:0, 1:1, 4:2, 5:3}), np.arange(4))

del test, data, X
gc.collect()

def build_nn_model_bak3(input_shape):
    inp = Input(shape=(input_shape,))
    binl = BatchNormalization()(inp)    

    x_1 = Dense(1024, activation="relu", kernel_initializer='normal')(binl)
    x_1 = Dropout(0.2)(x_1)
    x_1 = BatchNormalization()(x_1)    
    x_2 = Dense(512, activation="relu", kernel_initializer='normal')(x_1)
    x_2 = Dropout(0.2)(x_2)
    x_2 = BatchNormalization()(x_2)    
    x_3 = Dense(256, activation="relu", kernel_initializer='normal')(x_2)
    x_3 = Dropout(0.2)(x_3)
    x_3 = BatchNormalization()(x_3)
    x_4 = Dense(128, activation="relu", kernel_initializer='normal')(x_3)
    x_4 = Dropout(0.2)(x_4)
    x_4 = BatchNormalization()(x_4)

    x = concatenate([x_1, x_2, x_3, x_4])
    
    x = Dense(80, activation="relu", kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu", kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu", kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    model = Model(inputs=inp, outputs=out)

    return model

class roc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)),str(round(roc_val, 4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def run_nn(train_x, test_x, y):
    
    oof_pred = np.zeros((len(train_x), 4))
    y_pred = np.zeros((len(test_x), 4))
    
    folds = GroupKFold(n_splits=5)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train['label'], train['event_id'])):
        print(f'Fold {fold + 1}')
        
        x_tr, x_val = train_x[tr_ind], train_x[val_ind]
        y_tr, y_val = y[tr_ind], y[val_ind]
        
        model = build_nn_model(x_tr.shape[1])
        model.compile(optimizer=Adam(3e-4), loss='categorical_crossentropy', metrics=['acc'])
        
        checkpoint = ModelCheckpoint(f'nn_{fold}.h5', 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     save_weights_only = True)
        
        model.fit(x_tr, 
                  y_tr, 
                  batch_size=1024, 
                  epochs=6,
                  validation_data=(x_val, y_val),
                  callbacks=[checkpoint, roc_callback((x_tr, y_tr), (x_val, y_val))])
        
        model.load_weights(f'nn_{fold}.h5')
        
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(test_x) / folds.n_splits
        
        y_one_hot = label_binarize(y_val, np.arange(4)) 
        oof_one_hot = label_binarize(oof_pred[val_ind].argmax(axis=1), np.arange(4)) 
        score = roc_auc_score(y_one_hot, oof_one_hot) 
        print('auc: ', score)
        
        del x_tr, x_val, y_tr, y_val, model
        K.clear_session()
        gc.collect()
        
    return y_pred, oof_pred

y_pred, oof_pred = run_nn(train_X, test_X, y)

oof_one_hot = label_binarize(oof_pred.argmax(axis=1), np.arange(4)) 
score = roc_auc_score(y, oof_one_hot) 
print('auc: ', score)

DATA_DIR = './jet_simple_data/'
mapping_dict_inv = {0:21, 1:1, 2:4, 3:5}

submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
submission.label = y_pred.argmax(axis=1)
submission.label = submission.label.map(mapping_dict_inv)
submission.head()

submission.to_csv('submission_nn.csv', index=False)

np.save('y_pred_nn', y_pred)
np.save('oof_pred_nn', oof_pred)