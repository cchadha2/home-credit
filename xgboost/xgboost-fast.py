
# coding: utf-8

# In[1]:


import os
import math
import time

from contextlib import contextmanager
import numpy as np
import pandas as pd
from IPython.display import display
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[6]:


def xgboost(df, num_folds, stratified = False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    del df
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1000)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1000)
    # Create arrays and dataframes to store results
    xgb_val_preds = np.zeros(train_df.shape[0])
    xgb_preds = np.zeros(test_df.shape[0])
    
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        
        data_train, data_valid = train_df[feats].iloc[train_idx], train_df[feats].iloc[valid_idx]
        label_train, label_valid = train_df['TARGET'].iloc[train_idx], train_df['TARGET'].iloc[valid_idx]
                
        dtrain_xgb = xgb.DMatrix(data_train, label_train)
        dvalid_xgb = xgb.DMatrix(data_valid, label_valid)
        dtest_xgb = xgb.DMatrix(test_df[feats])
        
        params_xgb = {
            'objective': 'binary:logistic',
            'boosting_type': 'gbtree',
            'nthread': 20,
            'learning_rate': 0.02,  # 02,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60, # 39.3259775,
            'seed': 0,
            'eval_metric': 'auc',
            'verbose': 100
        }
        
        xgb_clf = xgb.train(
            params=params_xgb,
            dtrain=dtrain_xgb,
            num_boost_round=10000,
            evals=[(dtrain_xgb, 'train'), (dvalid_xgb, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        xgb_val_preds[valid_idx] = xgb_clf.predict(dvalid_xgb)
        xgb_preds += xgb_clf.predict(dtest_xgb) / folds.n_splits
        
        print('XGB fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid_xgb.get_label(), xgb_val_preds[valid_idx])))
        del xgb_clf, dtrain_xgb, dvalid_xgb, data_train, data_valid, label_train, label_valid


    print('XGB Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], xgb_val_preds))

    # Write submission file
    pred_df = test_df[['SK_ID_CURR']].copy()
    pred_df['TARGET'] = xgb_preds
    pred_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)


# In[7]:


# Thanks You Guillaume Martin for the Awesome Memory Optimizer!
# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


def main():
    df = pd.read_csv('../data/processed_data_2.2.csv')
    df = reduce_mem_usage(df)
    with timer("Ran model blend with kfold"):
        xgboost(df, num_folds= 5, stratified = True) 

if __name__ == "__main__":
    submission_file_name = "../predictions/xgb_pred.csv"
    with timer("Full model run"):
        main()

