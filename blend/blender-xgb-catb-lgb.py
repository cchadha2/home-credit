
# coding: utf-8

# # Universal Blender: XGB+CatB+LGB
# ## Part 1. Staging.
# This is an attempt to build a universal blender frame, that collects holdout, crossval and train predictions, so that more advanced stacking and blending techinques can be used. 
# 
# This realisation prepares stage to blend Ridge, DNN, XGB, CatBoost and LGBM, but you can add or drop any models you want. The code could have been more elegant with a estimator class, but it is not :). If you know how to make it better, please share.
# 
# It also includes Data Builder function with memory optimisation that decreases memory usage by 70%. The optimized pickle dump can be used instead of building the dataframe from scratch.
# The kernel is submitted in debug mode. State debag = False in oof_regression_stacker to get real results. Chose number of folds carefully as XGBoost and CatBoost take forever to train.
# 
# **Note:** Dataloader applies MinMaxscaler on the dataset that includes train and test data, therefore a leak from test to train occures. It is beneficial for leaderboard score, but should be avoided in real projects. To get a real life example how it can mess up your results please watch Caltech lecture, Puzzle 4, 52:00 : https://www.youtube.com/watch?v=EZBUDG12Nr0&index=17&list=PLD63A284B7615313A

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import sklearn.linear_model
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import gc
import os
import time

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


def oof_regression_stacker(train_x, train_y, test_x,
                           estimators, 
                           pred_cols, 
                           train_eval_metric, 
                           compare_eval_metric,
                           n_folds = 3,
                           holdout_x=False,
                           debug = False):
    
    """
    Original script:
        Jovan Sardinha
        https://medium.com/weightsandbiases/an-introduction-to-model-ensembling-63effc2ca4b3
        
    Args:
        train_x, train_y, test_x (DataFrame).
        n_folds (int): The number of folds for crossvalidation.
        esdtimators (list): The list of estimator functions.
        pred_cols (list): The estimator related names of prediction columns.
        train_eval_metric (class): Fucntion for the train eval metric.
        compare_eval_metric (class): Fucntion for the crossvalidation eval metric.
        holdout_x (DataFrame): Holdout dataframe if you intend to stack/blend using holdout.
        
    Returns:
        train_blend, test_blend, model
    """
    
    if debug == True:
        train_x = train_x.sample(n=1000, random_state=seed_val)
        train_y = train_y.sample(n=1000, random_state=seed_val)
        
    # Start timer:
    start_time = time.time()
    
    # List to save models:
    model_list = []
    
    # Initializing blending data frames:
    with_holdout = isinstance(holdout_x, pd.DataFrame)
    if with_holdout: holdout_blend = pd.DataFrame(holdout_x.index)
    
    train_blend = pd.DataFrame(train_x.index)
    val_blend = pd.DataFrame(train_x.index)
    test_blend = pd.DataFrame(test_x.index)

    # Arrays to hold estimators' predictions:
    test_len = test_x.shape[0]
    train_len = train_x.shape[0]

    dataset_blend_train = np.zeros((train_len, len(estimators))) # Mean train prediction holder
    dataset_blend_val = np.zeros((train_len, len(estimators))) # Validfation prediction holder                   
    dataset_blend_test = np.zeros((test_len, len(estimators))) # Mean test prediction holder
    if with_holdout: dataset_blend_holdout = np.zeros((holdout_x.shape[0], len(estimators))) # Same for holdout
        
    # Note: StratifiedKFold splits into roughly 66% train 33% test  
    folds = StratifiedShuffleSplit(n_splits= n_folds, random_state=seed_val,
                                  test_size = 1/n_folds, train_size = 1-(1/n_folds))
        
    # For every estimator:
    for j, estimator in enumerate(estimators):
        
        # Array to hold folds number of predictions on test:
        dataset_blend_train_j = np.zeros((train_len, n_folds))
        dataset_blend_test_j = np.zeros((test_len, n_folds))
        if with_holdout: dataset_blend_holdout_j = np.zeros((holdout_x.shape[0], n_folds))
        
        # For every fold:
        for i, (train, test) in enumerate(folds.split(train_x, train_y)):
            trn_x = train_x.iloc[train, :] 
            trn_y = train_y.iloc[train].values.ravel()
            val_x = train_x.iloc[test, :] 
            val_y = train_y.iloc[test].values.ravel()
            
            # Estimators conditional training:
            if estimator == 'lgb':
                model = kfold_lightgbm(trn_x, trn_y)
                pred_val = model.predict(val_x)
                pred_test = model.predict(test_x)
                pred_train = model.predict(train_x)
                if with_holdout:
                    pred_holdout = model.predict(holdout_x)                
            elif estimator == 'xgb':
                model = kfold_xgb(trn_x, trn_y)
                pred_val = xgb_predict(val_x, model)
                pred_test = xgb_predict(test_x, model)
                pred_train = xgb_predict(train_x, model)
                if with_holdout:
                    pred_holdout = xgb_predict(holdout_x, model)
            elif estimator == 'f10_dnn':
                model = f10_dnn(trn_x, trn_y)
                pred_val = model.predict(val_x).ravel()
                pred_test = model.predict(test_x).ravel()
                pred_train = model.predict(train_x).ravel()
                if with_holdout:
                    pred_holdout = model.predict(holdout_x).ravel()
                #print(pred_val.shape, pred_test.shape, pred_train.shape)             
            elif estimator == 'ridge':
                model = ridge(trn_x, trn_y)
                pred_val = model.predict(val_x)
                pred_test = model.predict(test_x)
                pred_train = model.predict(train_x)
                if with_holdout:
                    pred_holdout = model.predict(holdout_x)                         
            else:
                model = kfold_cat(trn_x, trn_y)
                pred_val = model.predict_proba(val_x)[:,1]
                pred_test = model.predict_proba(test_x)[:,1]
                pred_train = model.predict_proba(train_x)[:,1]
                if with_holdout:
                    pred_holdout = model.predict_proba(holdout_x)[:,1]         
            
            dataset_blend_val[test, j] = pred_val
            dataset_blend_test_j[:, i] = pred_test
            dataset_blend_train_j[:, i] = pred_train
            if with_holdout: 
                dataset_blend_holdout_j[:, i] = pred_holdout
            
            print('fold:', i+1, '/', n_folds,
                  '; estimator:',  j+1, '/', len(estimators),
                  ' -> oof cv score:', compare_eval_metric(val_y, pred_val))

            del trn_x, trn_y, val_x, val_y
            gc.collect()
    
        # Save curent estimator's mean prediction for test, train and holdout:
        dataset_blend_test[:, j] = np.mean(dataset_blend_test_j, axis=1)
        dataset_blend_train[:, j] = np.mean(dataset_blend_train_j, axis=1)
        if with_holdout: dataset_blend_holdout[:, j] = np.mean(dataset_blend_holdout_j, axis=1)
        
        model_list += [model]
        
    #print('--- comparing models ---')
    for i in range(dataset_blend_val.shape[1]):
        print('model', i+1, ':', compare_eval_metric(train_y, dataset_blend_val[:,i]))
        
    for i, j in enumerate(estimators):
        val_blend[pred_cols[i]] = dataset_blend_val[:,i]
        test_blend[pred_cols[i]] = dataset_blend_test[:,i]
        train_blend[pred_cols[i]] = dataset_blend_train[:,i]
        if with_holdout: 
            holdout_blend[pred_cols[i]] = dataset_blend_holdout[:,i]
        else:
            holdout_blend = False
    
    end_time = time.time()
    print("Total Time usage: " + str(int(round(end_time - start_time))))
    return train_blend, val_blend, test_blend, holdout_blend, model_list


# ## Blending:
# ### Estimators:
# #### Ridge regression

# In[ ]:


from sklearn.linear_model import Ridge
import sklearn.linear_model

def ridge(trn_x, trn_y):
    clf = Ridge(alpha=20, 
                copy_X=True, 
                fit_intercept=True, 
                solver='auto',max_iter=10000,
                normalize=False, 
                random_state=0,  
                tol=0.0025)
    clf.fit(trn_x, trn_y)
    return clf


# #### Simple DNN
# Please thank its author:
# https://www.kaggle.com/tottenham/10-fold-simple-dnn-with-rank-gauss
# 
# works surprisingly fast.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from sklearn.model_selection import KFold

import gc
import os

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
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
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def f10_dnn(X_train, Y_train, nn_num_folds=10):
    
    folds = KFold(n_splits=nn_num_folds, shuffle=True, random_state=seed_val)

    for n_fold, (nn_trn_idx, nn_val_idx) in enumerate(folds.split(X_train)):
        nn_trn_x, nn_trn_y = X_train.iloc[nn_trn_idx,:], Y_train[nn_trn_idx]
        nn_val_x, nn_val_y = X_train.iloc[nn_val_idx,:], Y_train[nn_val_idx]

        print( 'Setting up neural network...' )
        nn = Sequential()
        nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = X_train.shape[1]))
        nn.add(PReLU())
        nn.add(Dropout(.3))
        nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
        nn.add(PReLU())
        nn.add(BatchNormalization())
        nn.add(Dropout(.3))
        nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
        nn.add(PReLU())
        nn.add(BatchNormalization())
        nn.add(Dropout(.3))
        nn.add(Dense(units = 26, kernel_initializer = 'normal'))
        nn.add(PReLU())
        nn.add(BatchNormalization())
        nn.add(Dropout(.3))
        nn.add(Dense(units = 12, kernel_initializer = 'normal'))
        nn.add(PReLU())
        nn.add(BatchNormalization())
        nn.add(Dropout(.3))
        nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer='adam')

        print( 'Fitting neural network...' )
        nn.fit(nn_trn_x, nn_trn_y, validation_data = (nn_val_x, nn_val_y), epochs=10, verbose=2,
              callbacks=[roc_callback(training_data=(nn_trn_x, nn_trn_y),validation_data=(nn_val_x, nn_val_y))])
        
        #print( 'Predicting...' )
        #sub_preds += nn.predict(X_test).flatten().clip(0,1) / folds.n_splits
    
        gc.collect()
        
        return nn


# #### LightGBM
# the best of the batch, fast and convenient to use.

# In[ ]:


def kfold_lightgbm(trn_x, trn_y, num_folds=5):
       
    # Cross validation model
    in_folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
        
    # Create arrays and dataframes to store results
    for train_idx, valid_idx in in_folds.split(trn_x, trn_y):
        dtrain = lgb.Dataset(data=trn_x.values[train_idx], 
                             label=trn_y[train_idx], 
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=trn_x.values[valid_idx], 
                             label=trn_y[valid_idx], 
                             free_raw_data=False, silent=True)

        # LightGBM parameters found by Bayesian optimization
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 20,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60, # 39.3259775,
            'seed': seed_val,
            'verbose': -1,
            'metric': 'auc',
        }
        
        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        del dtrain, dvalid
        gc.collect()
    
    return clf


# #### XGBoost 
# has a nasty feature that it takes only DMatrix as arguments therefore predict method has to be wrapend into a function.

# In[ ]:


def xgb_predict(X, model):
    xgb_X = xgb.DMatrix(X.values)
    return model.predict(xgb_X)


# In[ ]:


def kfold_xgb(trn_x, trn_y, num_folds=3):
    
    # Cross validation model
    folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)
        
    # Create arrays and dataframes to store results
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(trn_x, trn_y)):
        dtrain = xgb.DMatrix(trn_x.values[train_idx], 
                             trn_y[train_idx])
        dvalid = xgb.DMatrix(trn_x.values[valid_idx], 
                             trn_y[valid_idx])

        # LightGBM parameters found by Bayesian optimization
        n_rounds = 2000
        
        xgb_params = {'eta': 0.05,
                      'max_depth': 6, 
                      'subsample': 0.85, 
                      'colsample_bytree': 0.85,
                      'colsample_bylevel': 0.632,
                      'min_child_weight' : 30,
                      'objective': 'binary:logistic', 
                      'eval_metric': 'auc', 
                      'seed': seed_val,
                      'lambda': 0,
                      'alpha': 0,
                      'silent': 1
                     }
        
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        xgb_model = xgb.train(xgb_params, 
                              dtrain, 
                              n_rounds, 
                              watchlist, 
                              verbose_eval=False,
                              early_stopping_rounds=200)

        del dtrain, dvalid
        gc.collect()
    
    return xgb_model


# #### Catboost
# watch out, it has predict and predict_proba methods. predict_proba should be used; it returns 2d array, that has to be flattened.

# In[ ]:


def kfold_cat(trn_x, trn_y, num_folds=3):
    
    # Cross validation model
    folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)
        
    # Create arrays and dataframes to store results
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(trn_x, trn_y)):
        cat_X_train, cat_y_train = trn_x.values[train_idx], trn_y[train_idx]
        cat_X_valid, cat_y_valid = trn_x.values[valid_idx], trn_y[valid_idx]

        # Catboost:
        #cb_model = CatBoostClassifier(iterations=1000,
                              #learning_rate=0.1,
                              #depth=7,
                              #l2_leaf_reg=40,
                              #bootstrap_type='Bernoulli',
                              #subsample=0.7,
                              #scale_pos_weight=5,
                              #eval_metric='AUC',
                              #metric_period=50,
                              #od_type='Iter',
                              #od_wait=45,
                              #random_seed=17,
                              #allow_writing_files=False)
        
        cb_model = CatBoostClassifier(iterations=2000,
                                      learning_rate=0.02,
                                      depth=6,
                                      l2_leaf_reg=40,
                                      bootstrap_type='Bernoulli',
                                      subsample=0.8715623,
                                      scale_pos_weight=5,
                                      eval_metric='AUC',
                                      metric_period=50,
                                      od_type='Iter',
                                      od_wait=45,
                                      random_seed=seed_val,
                                     allow_writing_files=False)
        
        cb_model.fit(cat_X_train, cat_y_train,
                     eval_set=(cat_X_valid, cat_y_valid),
                     use_best_model=True,
                     verbose=False)

        del cat_X_train, cat_y_train, cat_X_valid, cat_y_valid 
        gc.collect()
    
    return cb_model


def data_loader(to_load=False):
    
    if not to_load:
        
        df = pd.read_csv('../data/processed_data_3.8.csv', compression = 'zip')
        df.set_index('SK_ID_CURR', inplace=True, drop=False)
        feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
        # Split to train and test:
        y = df['TARGET']
        X = df[feats]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean()).clip(-1e11,1e11)
        X = X.drop(list(X.loc[:, X.isnull().any()].columns), axis = 1)

        print("X shape: ", X.shape, "    y shape:", y.shape)
        print("\nPreparing data...")

        training = y.notnull()
        testing = y.isnull()
        
        X_train = X.loc[training,:]
        X_test = X.loc[testing,:]
        y_train = y.loc[training]
        
        # Scale:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_train.loc[:, X_train.columns] = scaler.transform(X_train[X_train.columns])
        X_test.loc[:, X_test.columns] = scaler.transform(X_test[X_test.columns])
        
        print(X_train.shape, X_test.shape, y_train.shape)
        
        del df, X, y, training, testing
        gc.collect()
    
    return X_train, X_test, y_train


# Fix random seed:
seed_val = 42

# Load data:
train_x, test_x, train_y = data_loader()


# estimators = ['cat','lgb', 'xgb','ridge','f10_dnn']
estimators = ['ridge','f10_dnn']

# pred_cols = ['pred_cat','pred_lgb','pred_xgb','ridge','f10_dnn']
pred_cols = ['ridge','f10_dnn']


#Holdout
from sklearn.model_selection import train_test_split
x_train, x_hold, y_train, y_hold = train_test_split(train_x, train_y, test_size=0.1, random_state=seed_val)


n_folds = 2
tr_blend, va_blend, tst_blend, hold_blend, m_list = oof_regression_stacker(x_train, y_train, test_x, 
                                                                           n_folds = 5, 
                                                                           estimators=estimators, 
                                                                           pred_cols = pred_cols,
                                                                           train_eval_metric=roc_auc_score,
                                                                           compare_eval_metric=roc_auc_score,
                                                                           debug = False, holdout_x = x_hold)

tr_blend.to_csv('../predictions/tr_blend_nn.csv')
va_blend.to_csv('../predictions/va_blend_nn.csv')
tst_blend.to_csv('../predictions/tst_blend_nn.csv')
hold_blend.to_csv('../predictions/hold_blend_nn.csv')
# m_list.to_csv('../output/m_list.pkl')

