#!/usr/bin/env python
"""
coding=utf-8
"""
import pandas as pd
import sklearn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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
    signup_df['signup_os_known'] = signup_df['signup_os'].notnull()
    signup_df['vehicle_make_known'] = signup_df['vehicle_make'].notnull()

    # Create timeseries enriched features
    logging.info('Enriching data with timeseries derived features')
    signup_df['signup_to_bgc'] = (signup_df['bgc_date'] - signup_df['signup_date']).apply(extract_days)
    signup_df['signup_to_vehicle_add'] = (signup_df['bgc_date'] - signup_df['vehicle_added_date']).apply(extract_days)

    # For weekday() description, see
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.weekday.html
    signup_df['signup_weekday'] = signup_df['bgc_date'].apply(lambda x: x.weekday() <= 4)
    logging.info('Enriched dataframe description: \n%s' % signup_df.describe())
    return signup_df


def main():
    """
    Description
    :return: 
    """

    # Create enriched data set, save to file for external validation
    signup_df = create_signup_df('../data/input/ds_challenge_v2_1_data.csv')
    signup_df.to_csv('../data/output/signups_enriched.csv')


if __name__ == '__main__':
    main()
