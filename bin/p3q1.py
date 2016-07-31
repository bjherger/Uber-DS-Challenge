#!/usr/bin/env python
"""
coding=utf-8
"""
import pandas as pd
import sklearn
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def create_signup_df(data_path):
    logging.info('Beginning import of signup data, from path: %s' %(data_path))

    # Read in data
    logging.info('Reading in data')
    signup_df = pd.read_csv(data_path)
    isinstance(signup_df, pd.DataFrame)
    logging.info('Data shape: %s rows, %s columns' % signup_df.shape)
    logging.info('Data columns: %s' % (list(signup_df.columns)))

    # Create drove field
    logging.info('Creating drove field')
    signup_df['drove'] = signup_df['first_completed_date'].notnull()

    # Convert data types
    logging.info('Converting data types')
    signup_df['signup_date'] = pd.to_datetime(signup_df['signup_date'])
    signup_df['bgc_date'] = pd.to_datetime(signup_df['bgc_date'])
    signup_df['vehicle_added_date'] = pd.to_datetime(signup_df['vehicle_added_date'])

    # Create a enriched features
    logging.info('Enriching data with derived features')
    signup_df['bgc_known'] = signup_df['bgc_date'].notnull()
    signup_df['signup_os_known'] = signup_df['signup_os'].notnull()
    signup_df['vehicle_make_known'] = signup_df['vehicle_make'].notnull()

    signup_df['signup_to_bgc'] = (signup_df['bgc_date'] - signup_df['signup_date']).days
    print signup_df['signup_to_bgc']
    pd.tslib.Timedelta.days
    # TODO add signup weekday or weekend
    # TODO add time to vehicle added from signup
    # TODO add time to bgc from signup
    # TODO add vehicle added before bgc?
    return signup_df

def main():
    """
    Description
    :return: 
    """

    signup_df = create_signup_df('../data/input/ds_challenge_v2_1_data.csv')
    # print signup_df
    signup_df.to_csv('../data/output/signups_enriched.csv')


if __name__ == '__main__':
    main()
