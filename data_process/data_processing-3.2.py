
# coding: utf-8

# In[1]:


import time

from contextlib import contextmanager
import numpy as np
import pandas as pd


# In[2]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[3]:


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[4]:


def application_train_and_test():
    
    application_train = pd.read_csv('../data/application_train.csv')
    application_test = pd.read_csv('../data/application_test.csv')
    
    application_train = application_train.sort_values(by = 'SK_ID_CURR')
    application_test = application_test.sort_values(by = 'SK_ID_CURR')
    df = application_train.append(application_test).reset_index()

    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    df['CODE_GENDER'].replace({'XNA': np.nan}, inplace = True)
    df['ORGANIZATION_TYPE'].replace({'XNA': np.nan}, inplace = True)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    useless_features = ['FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 
                        'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 
                        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    
    df = df.drop(useless_features, axis = 1)
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
#     df['NEW_EXT_SOURCES_MEAN_AMT_ANNUITY_RATIO'] = df['NEW_EXT_SOURCES_MEAN']/df['AMT_ANNUITY']
#     df['NEW_EXT_SOURCES_MEAN_AMT_CREDIT_RATIO'] = df['NEW_EXT_SOURCES_MEAN']/df['AMT_CREDIT']
#     df['NEW_EXT_SOURCES_MEAN_AMT_GOODS_PRICE_RATIO'] = df['NEW_EXT_SOURCES_MEAN']/df['AMT_GOODS_PRICE']
#     df['NEW_EXT_SOURCES_MEAN_AMT_INCOME_TOTAL_PROD'] = df['NEW_EXT_SOURCES_MEAN']*df['AMT_INCOME_TOTAL']
#     df['NEW_EXT_SOURCES_MEAN_DAYS_BIRTH_PROD'] = df['NEW_EXT_SOURCES_MEAN']*df['DAYS_BIRTH']
#     df['NEW_EXT_SOURCES_MEAN_DAYS_EMPLOYED_PROD'] = df['NEW_EXT_SOURCES_MEAN']*df['DAYS_EMPLOYED']
#     df['DAYS_REGISTRATION_ID'] = df['DAYS_REGISTRATION']*df['DAYS_ID_PUBLISH']
#     df['EXT_SOURCES_PAYMENT_RATE'] = df['NEW_EXT_SOURCES_MEAN']*df['PAYMENT_RATE']
#     df['AGE_TO_CAR_AGE_RATIO'] = df['DAYS_BIRTH']/df['OWN_CAR_AGE']
#     df['AMT_REQ'] = (df['AMT_REQ_CREDIT_BUREAU_HOUR'] + df['AMT_REQ_CREDIT_BUREAU_DAY'] + 
#                      df['AMT_REQ_CREDIT_BUREAU_WEEK'] + df['AMT_REQ_CREDIT_BUREAU_MON'] + 
#                      df['AMT_REQ_CREDIT_BUREAU_QRT'] + df['AMT_REQ_CREDIT_BUREAU_YEAR'])
#     df['CREDIT_PER_CHILD'] = df['AMT_CREDIT']/(1 + df['CNT_CHILDREN'])
#     df['ANNUITY_PER_CHILD'] = df['AMT_ANNUITY']/(1 + df['CNT_CHILDREN'])
#     df['GOODS_PER_CHILD'] = df['AMT_GOODS_PRICE']/(1 + df['CNT_CHILDREN'])
#     df['CREDIT_PER_FAMILY_MEMBER'] = df['AMT_CREDIT']/df['CNT_FAM_MEMBERS']
#     df['ANNUITY_PER_FAMILY_MEMBER'] = df['AMT_ANNUITY']/df['CNT_FAM_MEMBERS']
#     df['GOODS_PER_FAMILY_MEMBER'] = df['AMT_GOODS_PRICE']/df['CNT_FAM_MEMBERS']
#     df['FAM_SIZE_PER_POPULATION'] = df['CNT_FAM_MEMBERS']/df['REGION_POPULATION_RELATIVE']
#     df['CHILDREN_PER_POPULATION'] = df['CNT_CHILDREN']/df['REGION_POPULATION_RELATIVE']

    
#     accom_avg_list = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
#                       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
#                       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
#                       'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']
#     accom_mode_list = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
#                       'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
#                       'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
#                       'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
#                       'TOTALAREA_MODE']
#     accom_medi_list = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
#                       'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
#                       'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
#                       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']
    
#     df['ACCOM_SCORE_AVG'] = df[accom_avg_list].mean(axis=1)
#     df['ACCOM_SCORE_MODE'] = df[accom_mode_list].mean(axis=1)
#     df['ACCOM_SCORE_MEDI'] = df[accom_medi_list].mean(axis=1)
#     df['DEF_AVE_CNT_SOCIAL_CIRCLE'] = df[['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].mean(axis=1)
#     df['OBS_AVE_CNT_SOCIAL_CIRCLE'] = df[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']].mean(axis=1)
#     df['30_CNT_SOCIAL_CIRCLE_RATIO'] = df['DEF_30_CNT_SOCIAL_CIRCLE']/df['OBS_30_CNT_SOCIAL_CIRCLE']
#     df['60_CNT_SOCIAL_CIRCLE_RATIO'] = df['DEF_60_CNT_SOCIAL_CIRCLE']/df['OBS_60_CNT_SOCIAL_CIRCLE']
    
    # Feature interactions
    df['EXT_SOURCE_1_1'] = df['EXT_SOURCE_1']**2
    df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_DAYS_BIRTH'] = df['EXT_SOURCE_1'] * df['DAYS_BIRTH']
    df['EXT_SOURCE_2_2'] = df['EXT_SOURCE_2']**2
    df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_DAYS_BIRTH'] = df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
    df['EXT_SOURCE_3_3'] = df['EXT_SOURCE_3']**2
    df['EXT_SOURCE_3_DAYS_BIRTH'] = df['EXT_SOURCE_3'] * df['DAYS_BIRTH']
    df['DAYS_BIRTH_DAYS_BIRTH'] = df['DAYS_BIRTH']**2
    df['DAYS_EMPLOYED_DAYS_BIRTH'] = df['DAYS_EMPLOYED'] * df['DAYS_BIRTH']
    df['DAYS_EMPLOYED_DAYS_EMPLOYED'] = df['DAYS_EMPLOYED']**2
    df['AMT_CREDIT_AMT_ANNUITY'] = df['AMT_CREDIT'] * df['AMT_ANNUITY']
    df['AMT_CREDIT_AMT_CREDIT'] = df['AMT_CREDIT']**2
    df['AMT_ANNUITY_AMT_ANNUITY'] = df['AMT_ANNUITY']**2
    
    categorical_features = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
                            'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
                            'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
                            'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                            'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11', 
                            'FLAG_DOCUMENT_18', 'CODE_GENDER', 'NAME_CONTRACT_TYPE',
                            'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EMERGENCYSTATE_MODE',
                            'HOUSETYPE_MODE', 'FONDKAPREMONT_MODE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                            'NAME_HOUSING_TYPE', 'NAME_TYPE_SUITE', 'WALLSMATERIAL_MODE','WEEKDAY_APPR_PROCESS_START',
                            'HOUR_APPR_PROCESS_START', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    
    for feature in categorical_features:
        df[feature], uniques = pd.factorize(df[feature]) 
    
    df = df.drop('index', axis=1)
    
    del application_train
    del application_test
    del categorical_features
    
    return df


# In[5]:


def bureau_and_balance(df):
    
    bureau = pd.read_csv('../data/bureau.csv')
    bureau_balance = pd.read_csv('../data/bureau_balance.csv')
    
    previous_loans = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loans'})
    df = df.merge(previous_loans, on = 'SK_ID_CURR', how = 'left')
    df['previous_loans'] = df['previous_loans'].fillna(0)
    
    closed_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    closed_loans = closed_loans.groupby('SK_ID_CURR', as_index=False)['CREDIT_ACTIVE'].count().rename(columns = {'CREDIT_ACTIVE': 'closed_loans'})
    df = df.merge(closed_loans, on = 'SK_ID_CURR', how = 'left')
    df['closed_loans'] = df['closed_loans'].fillna(0)
    active_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_loans = active_loans.groupby('SK_ID_CURR', as_index=False)['CREDIT_ACTIVE'].count().rename(columns = {'CREDIT_ACTIVE': 'active_loans'})
    df = df.merge(active_loans, on = 'SK_ID_CURR', how = 'left')
    df['active_loans'] = df['active_loans'].fillna(0)
        
    bureau, bureau_cat_cols = one_hot_encoder(bureau)
    bureau_balance, bureau_balance_cat_cols = one_hot_encoder(bureau_balance)


    bureau_aggregations = {'DAYS_CREDIT':['min', 'max', 'mean', 'var'],
                           'CREDIT_DAY_OVERDUE':['max', 'mean'],
                           'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
                           'DAYS_ENDDATE_FACT': ['mean'],
                           'AMT_CREDIT_MAX_OVERDUE': ['mean'],
                           'CNT_CREDIT_PROLONG': ['count', 'sum'],
                           'AMT_CREDIT_SUM': ['min', 'max', 'mean'],
                           'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
                           'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean'],
                           'AMT_CREDIT_SUM_OVERDUE': ['mean'],
                           'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
                           'AMT_ANNUITY': ['max', 'mean'],
                           'MONTHS_BALANCE_MIN': ['min'],
                           'MONTHS_BALANCE_MAX': ['max'],
                           'MONTHS_BALANCE_SIZE': ['mean', 'sum']}

    bureau_balance_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    
    for col in bureau_cat_cols:
        bureau_aggregations[col] = ['mean']
    for col in bureau_balance_cat_cols:
        bureau_balance_aggregations[col] = ['mean']
        
    bureau_balance_aggregations = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_aggregations)
    bureau_balance_aggregations.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance_aggregations.columns.tolist()])
    bureau = bureau.join(bureau_balance_aggregations, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_aggregations)
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(bureau_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(bureau_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]
    
    df = df.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
            
    del bureau
    del bureau_balance
    del bureau_balance_aggregations
    del bureau_agg
    del previous_loans
    del closed_loans
    del active_loans
    del closed
    del closed_agg
    
    return df


# In[6]:


def cc_balance(df):
    
    credit_card_balance = pd.read_csv('../data/credit_card_balance.csv')
    
    current_loan_status = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']].sort_values(by = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).drop_duplicates()
    current_loan_status = current_loan_status[current_loan_status['NAME_CONTRACT_STATUS'].isin(['Active', 'Completed'])]
    
    prev_credit_completed = current_loan_status[current_loan_status['NAME_CONTRACT_STATUS'] == 'Completed']
    prev_credit_completed = pd.get_dummies(prev_credit_completed)
    del prev_credit_completed['SK_ID_PREV']
    prev_credit_completed = prev_credit_completed.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed'].count()
    df = df.merge(prev_credit_completed, on = 'SK_ID_CURR', how = 'left')
    df['NAME_CONTRACT_STATUS_Completed'] = df['NAME_CONTRACT_STATUS_Completed'].fillna(0)
    
    prev_credit_active = current_loan_status.drop_duplicates(subset = ['SK_ID_CURR', 'SK_ID_PREV'], keep = False)
    prev_credit_active = prev_credit_active[prev_credit_active['NAME_CONTRACT_STATUS'] == 'Active']
    prev_credit_active = pd.get_dummies(prev_credit_active)
    del prev_credit_active['SK_ID_PREV']
    prev_credit_active = prev_credit_active.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active'].count()
    df = df.merge(prev_credit_active, on = 'SK_ID_CURR', how = 'left')
    df['NAME_CONTRACT_STATUS_Active'] = df['NAME_CONTRACT_STATUS_Active'].fillna(0)
        
    credit_card_balance.drop(columns = ['NAME_CONTRACT_STATUS', 'SK_ID_PREV'], inplace = True)
    
    cc_agg = credit_card_balance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = credit_card_balance.groupby('SK_ID_CURR').size()
    
    df = df.merge(cc_agg, on = 'SK_ID_CURR', how = 'left')
    df['CC_COUNT'] = df['CC_COUNT'].fillna(0)
    
    del credit_card_balance
    del current_loan_status
    del prev_credit_completed
    del prev_credit_active
    del cc_agg
    
    return df


# In[7]:


def installments(df):
    
    ins = pd.read_csv('../data/installments_payments.csv')

    ins['INSTALLMENTS_DAY_DIFF'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['INSTALLMENTS_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Feature interactions
    ins['INSTALLMENTS_DIFF_INSTALLMENTS_DAY_DIFF'] = ins['INSTALLMENTS_DIFF'] * ins['INSTALLMENTS_DAY_DIFF']
    ins['INSTALLMENTS_DIFF_INSTALLMENTS_DIFF'] = ins['INSTALLMENTS_DIFF']**2
    ins['INSTALLMENTS_DAY_DIFF_INSTALLMENTS_DAY_DIFF'] = ins['INSTALLMENTS_DAY_DIFF']**2
    
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'], 
        'INSTALLMENTS_DAY_DIFF': ['max', 'mean', 'sum', 'var'], 
        'INSTALLMENTS_DIFF': ['max', 'mean', 'sum', 'var'], 
        'INSTALLMENTS_DIFF_INSTALLMENTS_DAY_DIFF': ['max', 'mean', 'sum', 'var'], 
        'INSTALLMENTS_DIFF_INSTALLMENTS_DIFF': ['max', 'mean', 'sum', 'var'], 
        'INSTALLMENTS_DAY_DIFF_INSTALLMENTS_DAY_DIFF': ['max', 'mean', 'sum', 'var']
    }
    
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    df = df.merge(ins_agg, on = 'SK_ID_CURR', how = 'left')
    
    del ins
    del ins_agg
    
    return df


# In[8]:


def pos_cash(df):
    
    POS_CASH_balance = pd.read_csv('../data/POS_CASH_balance.csv')

    current_POS_status = POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']].sort_values(by = ['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).drop_duplicates()
    current_POS_status = current_POS_status[current_POS_status['NAME_CONTRACT_STATUS'].isin(['Active', 'Completed'])]

    prev_POS_completed = current_POS_status[current_POS_status['NAME_CONTRACT_STATUS'] == 'Completed']
    prev_POS_completed = pd.get_dummies(prev_POS_completed)
    del prev_POS_completed['SK_ID_PREV']
    prev_POS_completed = prev_POS_completed.rename(columns = {'NAME_CONTRACT_STATUS_Completed': 'NAME_CONTRACT_STATUS_Completed_POS'})
    prev_POS_completed = prev_POS_completed.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed_POS'].count()
    df = df.merge(prev_POS_completed, on = 'SK_ID_CURR', how = 'left')
    df['NAME_CONTRACT_STATUS_Completed_POS'] = df['NAME_CONTRACT_STATUS_Completed_POS'].fillna(0)

    prev_POS_active = current_POS_status.drop_duplicates(subset = ['SK_ID_CURR', 'SK_ID_PREV'], keep = False)
    prev_POS_active = prev_POS_active[prev_POS_active['NAME_CONTRACT_STATUS'] == 'Active']
    prev_POS_active = pd.get_dummies(prev_POS_active)
    del prev_POS_active['SK_ID_PREV']
    prev_POS_active = prev_POS_active.rename(columns = {'NAME_CONTRACT_STATUS_Active': 'NAME_CONTRACT_STATUS_Active_POS'})
    prev_POS_active = prev_POS_active.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active_POS'].count()
    df = df.merge(prev_POS_active, on = 'SK_ID_CURR', how = 'left')
    df['NAME_CONTRACT_STATUS_Active_POS'] = df['NAME_CONTRACT_STATUS_Active_POS'].fillna(0)
    
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'], 
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean']
    }

    POS_CASH_balance.drop(columns = ['NAME_CONTRACT_STATUS', 'SK_ID_PREV'], inplace = True) 
    POS_CASH_balance_agg = POS_CASH_balance.groupby('SK_ID_CURR').agg(aggregations)
    POS_CASH_balance_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in POS_CASH_balance_agg.columns.tolist()])
    POS_CASH_balance_agg['POS_COUNT'] = POS_CASH_balance.groupby('SK_ID_CURR').size()
    
    df = df.merge(POS_CASH_balance_agg, on = 'SK_ID_CURR', how = 'left')
    
    del current_POS_status
    del prev_POS_completed
    del prev_POS_active
    del POS_CASH_balance
    del POS_CASH_balance_agg
    
    return df


# In[9]:


def prev_app(df):

    previous_application = pd.read_csv('../data/previous_application.csv')

    previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']
    
    previous_application, cat_cols = one_hot_encoder(previous_application)
    
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = previous_application.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused = previous_application[previous_application['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    

    df = df.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')
    
    del previous_application
    del prev_agg
    del approved
    del approved_agg
    del refused
    del refused_agg
    
    return df


# In[10]:


def remove_features(df):
    
    # Remove all features that weren't split on above some threshold times in most recent run
    importance = pd.read_csv('../output/importance.csv')
    importance_threshold = 2
    importance_red = importance[importance['importance']<=importance_threshold]
    importance_red.reset_index(inplace = True)
    importance_list = list(importance_red['feature'])
    df = df.drop(importance_list, axis = 1)
    
    del importance
    del importance_red
    del importance_list
    
    return df


# In[11]:


def main(importance_prune = False):
    with timer("Process application_train and _test:"):
        df = application_train_and_test()
        print("df shape:", df.shape)
    with timer("Process bureau and bureau_balance"):
        df = bureau_and_balance(df)
        print("df shape:", df.shape)
    with timer("Process credit card balance"):
        df = cc_balance(df)
        print("df shape:", df.shape)
    with timer("Process installments payments"):
        df = installments(df)
        print("df shape:", df.shape)
    with timer("Process POS_CASH_balance"):
        df = pos_cash(df)
        print("df shape:", df.shape)
    with timer("Process previous_applications"):
        df = prev_app(df)
        print("df shape:", df.shape)
    if importance_prune == True:
        with timer("Post-processing"):
            df = remove_features(df)
            print("df shape:", df.shape)
        
    # Save processed data to csv    
    df.to_csv('../data/processed_data_3.2.csv')
        

if __name__ == "__main__":
    with timer("Processing pipeline run"):
        main(importance_prune = True)

