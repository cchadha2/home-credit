import time

from contextlib import contextmanager
import numpy as np
import pandas as pd



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def create_features(df):
    
    # Interactions between similarly distributed features in train and test
    df['AMT_REQ'] = (df['AMT_REQ_CREDIT_BUREAU_DAY'] + 
                     df['AMT_REQ_CREDIT_BUREAU_WEEK'] +
                     df['AMT_REQ_CREDIT_BUREAU_MON'] + 
                     df['AMT_REQ_CREDIT_BUREAU_QRT'] + 
                     df['AMT_REQ_CREDIT_BUREAU_YEAR'])
    
    accom_avg_list = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                      'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                      'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
                      'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']
    accom_mode_list = ['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
                      'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
                      'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
                      'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
                      'TOTALAREA_MODE']
    accom_medi_list = ['APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
                      'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
                      'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                      'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']
    
    df['ACCOM_SCORE_AVG'] = df[accom_avg_list].mean(axis=1)
    df['ACCOM_SCORE_MODE'] = df[accom_mode_list].mean(axis=1)
    df['ACCOM_SCORE_MEDI'] = df[accom_medi_list].mean(axis=1)
    
    
    return(df)


def remove_on_missing(df):
    
    # Removing features with high percentage of missing values (> 95§% missing in both train and test)
    missing_list = ['NEW_RATIO_BURO_CNT_CREDIT_PROLONG_SUM', 'NEW_RATIO_BURO_CREDIT_TYPE_Mortgage_MEAN',
                    'NEW_RATIO_BURO_CREDIT_TYPE_Microloan_MEAN', 'ACTIVE_DAYS_ENDDATE_FACT_MEAN',
                    'NEW_RATIO_BURO_DAYS_ENDDATE_FACT_MEAN', 'Unnamed: 0']
    
    df = df.drop(missing_list, axis=1)
    
    return(df)


def remove_on_distributions(df):
    
    # Remove features that are dissimilarly distributed between train and test
    dist_list = ['BURO_MONTHS_BALANCE_SIZE_MEAN', 'CLOSED_MONTHS_BALANCE_SIZE_MEAN']
    
    df = df.drop(dist_list, axis=1)
    
    return(df)


def main(
        interactions = False,
        remove_missing = False, 
        remove_dissimilar = False
        ):
    
    df = pd.read_csv('../data/processed_data_3.2.csv')

    if interactions == True:
        with timer('Created feature interactions'):
            df = create_features(df)
            print("df shape:", df.shape)

    if remove_missing == True:
        with timer('Removed features with high number of missing values'):
            df = remove_on_missing(df)
            print("df shape:", df.shape)

    if remove_dissimilar == True:
        with timer('Removed dissimilarly distributed features'):
            df = remove_on_distributions(df)
            print("df shape:", df.shape)

        
    # Save processed data to csv    
    df.to_csv('../data/processed_data_3.3.csv')  

if __name__ == "__main__":
    with timer("Processing pipeline run"):
        main(
            interactions = True,
            remove_missing = True,
            remove_dissimilar = True
            )

