#!/usr/bin/env python
"""
coding=utf-8

Code supporting question 3 of Uber DS challenge (see docs/instructions)

"""
# Imports
import logging

from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import statsmodels.api as sm

from patsy.highlevel import dmatrices




# Functions
def main():
    """
    Main method for question 3 of Uber DS challenge.

    This method reads, cleans and enriches the data set, and then runs models on that dataset.

    Model output is printed to screen, though the accompanying write up should be the primary reference for reviewing
    these results.
    :return: None
    :rtype: None
    """

    # Create enriched data set, save to file for external validation
    signup_df = create_signup_df('../data/input/ds_challenge_v2_1_data.csv')
    signup_df.to_csv('../data/output/signups_enriched.csv')

    drive_percentage = signup_df['drove'].astype(int).mean()
    print 'Q1: Percentage of drivers w/ completed ride: %s ' % drive_percentage
    # Prepare model for statsmodels consumption
    stats_df = statsmodel_prep_df(signup_df)

    # Train test split
    train = stats_df.sample(frac=0.8, random_state=200)
    test = stats_df.drop(train.index)
    # Run logistic regression models
    run_statsmodels_models(train, test, 'drove ~ signup_channel_referral + city_Berton + signup_weekday + '
                                        'vehicle_inspection_known')
    run_statsmodels_models(train, test, 'drove ~ signup_channel_referral + city_Berton  + '
                                        'signup_to_vehicle_add + signup_to_bgc')


def extract_days(input_delta):
    """
    Helper function to extract the number of days from a time delta. Returns:
     - Number of days, if valid time delta
     - np.NaN if time delta is null or invalid
    :param input_delta:
    :return: number of days in time delta
    :rtype: float
    """

    # Attempt to coerce into Pandas time delta
    delta = pd.Timedelta(input_delta)

    # Attempt to extract number of days
    days = np.NaN
    if pd.notnull(delta):
        days = delta.days

    # Return result
    return days


def create_signup_df(data_path):
    """
    Read signup data in from path, clean, enrich and return
    :param data_path: path to signup data
    :type data_path: str
    :return: Pandas data frame, containing original and enriched data
    :rtype: pd.DataFrame
    """
    logging.info('Beginning import of signup data, from path: %s' % (data_path))

    # Read in data
    logging.info('Reading in data')
    signup_df = pd.read_csv(data_path)
    isinstance(signup_df, pd.DataFrame)
    logging.info('Data shape: %s rows, %s columns' % signup_df.shape)
    logging.info('Data columns: %s' % (list(signup_df.columns)))

    # Create drove field
    logging.info('Creating drove field')
    signup_df['drove'] = signup_df['first_completed_date'].notnull()

    # Data cleaning
    logging.info('Cleaning data')
    # Replace invalid vehicle_year value w/ NaN
    signup_df['vehicle_year'] = signup_df['vehicle_year'].replace(to_replace=[0], value=np.NaN)

    # Convert data types
    logging.info('Converting data types')
    signup_df['signup_date'] = pd.to_datetime(signup_df['signup_date'], format='%m/%d/%y')
    signup_df['bgc_date'] = pd.to_datetime(signup_df['bgc_date'], format='%m/%d/%y')
    signup_df['vehicle_added_date'] = pd.to_datetime(signup_df['vehicle_added_date'], format='%m/%d/%y')

    # Create simple enriched features
    logging.info('Enriching data with simple derived features')
    signup_df['bgc_known'] = signup_df['bgc_date'].notnull()
    signup_df['vehicle_inspection_known'] = signup_df['vehicle_added_date'].notnull()
    signup_df['signup_os_known'] = signup_df['signup_os'].notnull()
    signup_df['vehicle_make_known'] = signup_df['vehicle_make'].notnull()

    # Create time series based enriched features
    logging.info('Enriching data with timeseries derived features')
    signup_df['signup_to_bgc'] = (signup_df['bgc_date'] - signup_df['signup_date']).apply(extract_days)
    signup_df['signup_to_vehicle_add'] = (signup_df['bgc_date'] - signup_df['vehicle_added_date']).apply(extract_days)

    # Clip at 0 to prevent events before signups
    signup_df['signup_to_bgc'] = signup_df['signup_to_bgc'].clip(lower=0)
    signup_df['signup_to_vehicle_add'] = signup_df['signup_to_vehicle_add'].clip(lower=0)

    # For weekday() description, see
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.weekday.html
    signup_df['signup_weekday'] = signup_df['bgc_date'].apply(lambda x: x.weekday() <= 4)

    # Re-centering
    signup_df['vehicle_year_past_2000'] = signup_df['vehicle_year'] - 2000

    # Return enriched DF
    logging.debug('Enriched data frame description: \n%s' % signup_df.describe())
    return signup_df


def statsmodel_prep_df(input_df):
    """
    Prepare the input data frame to be consumed by statsmodels. This process includes:
     - Zero filling columns where NaN logically means 0
     - Smarter null filling
    :param input_df: Raw data frame to be prepared
    :type input_df: pd.DataFrame
    :return: Prepared data frame
    :rtype: pd.DataFrame
    """

    prepped_df = input_df.copy(deep=True)
    isinstance(prepped_df, pd.DataFrame)

    # Appropriately deal w/ NaN values
    zero_fill_list = ['bgc_known', 'signup_os_known', 'vehicle_make_known', 'signup_weekday']
    for col in zero_fill_list:
        num_null = prepped_df[col].isnull().sum()
        if num_null > 0:
            logging.warn('%s null values found in column: %s' % (num_null, col))
            prepped_df[col] = prepped_df[col].fillna(0)

    isinstance(prepped_df, pd.DataFrame)

    # Smarter null filling
    prepped_df['signup_channel'] = prepped_df['signup_channel'].fillna('not_known')
    prepped_df['city_name'] = prepped_df['city_name'].fillna('not_known')

    # Manual dummy'ing
    prepped_df['city_Berton'] = prepped_df['city_name'] == 'Berton'
    prepped_df['signup_channel_organic'] = prepped_df['signup_channel'] == 'Organic'
    prepped_df['signup_channel_referral'] = prepped_df['signup_channel'] == 'Referral'

    # Remove instances missing drove field
    prepped_df = prepped_df[prepped_df['drove'].notnull()]

    # Convert drove to int
    prepped_df['drove'] = prepped_df['drove'].astype(int)

    # Return formatted DF
    logging.debug('Statsmodel prepped df: \n%s' % prepped_df.describe())
    return prepped_df


def run_statsmodels_models(train, test, model_description):
    """
    Run logistic regression model to predict whether a signed up driver ever actually drove.
    :param input_df: Data frame prepared for statsmodels regression
    :type input_df: pd.DataFrame
    :return: AUC for model generated
    :rtype: float
    """
    # Run model on all observations
    # Use dmatrices to format data
    logging.info('Running model w/ description: %s' %model_description)
    logging.debug('Train df: \n%s' % train.describe())
    logging.debug('Test df: \n%s' % test.describe())
    y_train, X_train = dmatrices(model_description, data=train, return_type='dataframe', NA_action='drop')
    y_test, X_test = dmatrices(model_description, data=test, return_type='dataframe', NA_action='drop')

    # Create, fit model
    mod = sm.Logit(endog=y_train, exog=X_train)
    res = mod.fit(method='bfgs', maxiter=100)

    # Output model summary
    print train['city_name'].value_counts()
    print train['signup_channel'].value_counts()
    print res.summary()

    # Create, output AUC
    predicted = res.predict(X_test)
    auc = roc_auc_score(y_true=y_test, y_score=predicted)
    print 'AUC for 20%% holdout: %s' %auc

    # Return AUC for model generated
    return auc



# Main section
if __name__ == '__main__':
    main()
