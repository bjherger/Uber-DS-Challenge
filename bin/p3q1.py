#!/usr/bin/env python
"""
coding=utf-8
"""
import pandas as pd
import sklearn
import numpy as np
import logging
import statsmodels.api as sm
import sys
from patsy.highlevel import dmatrices

logging.basicConfig(level=logging.DEBUG)


def extract_days(input_delta):
    delta = pd.Timedelta(input_delta)
    days = np.NaN
    if pd.notnull(delta):
        days = delta.days
    return days


def create_signup_df(data_path):
    """
    Read signup data in from path, clean, enrich and return
    :param data_path: path to signup data
    :type data_path: str
    :return: Pandas dataframe, containing original and enriched data
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

    # For weekday() description, see
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.weekday.html
    signup_df['signup_weekday'] = signup_df['bgc_date'].apply(lambda x: x.weekday() <= 4)
    logging.debug('Enriched dataframe description: \n%s' % signup_df.describe())
    return signup_df

def statsmodel_prep_df(input_df):
    # TODO remove once replaced w/ :type :
    isinstance(input_df, pd.DataFrame)

    prepped_df = input_df.copy(deep=True)
    isinstance(prepped_df, pd.DataFrame)

    # Add constant column
    # prepped_df['constant'] = 1

    # Appropriately deal w/ NaN values
    zero_fill_list = ['bgc_known', 'signup_os_known', 'vehicle_make_known', 'signup_weekday']
    for col in zero_fill_list:
        num_null = prepped_df[col].isnull().sum()
        if num_null >0:
            logging.warn('%s null values found in column: %s' %(num_null, col))
            prepped_df[col] = prepped_df[col].fillna(0)

    prepped_df['signup_channel'] = prepped_df['signup_channel'].fillna('not_known')
    prepped_df['city_name'] = prepped_df['city_name'].fillna('not_known')
    prepped_df['vehicle_year_past_2000'] = prepped_df['vehicle_year'] - 2000
    prepped_df['city_Berton'] = prepped_df['city_name'] == 'Berton'
    prepped_df['signup_channel_organic'] = prepped_df['signup_channel'] == 'Organic'
    prepped_df['signup_channel_referral'] = prepped_df['signup_channel'] == 'Referral'
    # Remove instances missing drove field
    prepped_df = prepped_df[prepped_df['drove'].notnull()]
    prepped_df['drove'] = prepped_df['drove'].astype(int)
    logging.debug('Statsmodel prepped df: \n%s' % prepped_df.describe())
    return prepped_df


def run_statsmodel_models(input_df):
    print input_df['city_name'].value_counts()
    print input_df['signup_channel'].value_counts()
    y, X = dmatrices('drove ~ signup_channel_referral + city_Berton+'
                     'signup_weekday + vehicle_year_past_2000 + vehicle_inspection_known',
                     data=input_df, return_type='dataframe', NA_action='drop')

    mod = sm.Logit(endog=y, exog=X)
    res = mod.fit(method='bfgs', maxiter=100)
    print res.summary()
def main():
    """
    Description
    :return: 
    """

    # Create enriched data set, save to file for external validation
    signup_df = create_signup_df('../data/input/ds_challenge_v2_1_data.csv')
    signup_df.to_csv('../data/output/signups_enriched.csv')

    stats_df = statsmodel_prep_df(signup_df)
    run_statsmodel_models(stats_df)


if __name__ == '__main__':
    main()
