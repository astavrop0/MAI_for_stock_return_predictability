#RAW TO INTERIM

import os
import pandas as pd
import numpy as np
import warnings

# Get the path to the data folder from the environment variable
DATA_PATH = os.environ.get("RESEARCH_DATA_PATH")

# Check if the DATA_PATH is not None
if DATA_PATH is not None:

    mai_daily_data = pd.read_csv(f'{DATA_PATH}/raw/mai_daily_data_raw.csv')
    mai_monthly_data = pd.read_csv(f'{DATA_PATH}/raw/mai_monthly_data_raw.csv')
    mef_daily_data = pd.read_csv(f'{DATA_PATH}/raw/mef_daily_data_raw.csv')
    mef_monthly_data = pd.read_csv(f'{DATA_PATH}/raw/mef_monthly_data_raw.csv')
    mkt_daily_data = pd.read_csv(f'{DATA_PATH}/raw/mkt_daily_data_raw.csv')
    mkt_monthly_data = pd.read_csv(f'{DATA_PATH}/raw/mkt_monthly_data_raw.csv')

    print("Data loaded successfully!")
else:
    print("ERROR: The RESEARCH_DATA_PATH environment variable is not set.")


#Set date as index column
mai_daily_data.set_index('date', inplace=True)
mai_monthly_data.set_index('date', inplace=True)
mef_daily_data.set_index('date', inplace=True)
mef_monthly_data.set_index('date', inplace=True)
mkt_daily_data.set_index('date', inplace=True)
mkt_monthly_data.set_index('date', inplace=True)


column_bases = ['credit_rating', 'gdp', 'house_mkt', 'inflation', 'monetary', 'oil', 'unemp', 'usd']

for base in column_bases:
    wi_col = f'{base}_wi'
    ni_col = f'{base}_ni'

    # Replace 0 in wi_col with value from ni_col if ni_col is not 0
    mask = (mai_daily_data[wi_col] == 0) & (mai_daily_data[ni_col] != 0)
    mai_daily_data.loc[mask, wi_col] = mai_daily_data.loc[mask, ni_col]

    # Replace 0 in ni_col with value from wi_col if wi_col is not 0
    mask = (mai_daily_data[ni_col] == 0) & (mai_daily_data[wi_col] != 0)
    mai_daily_data.loc[mask, ni_col] = mai_daily_data.loc[mask, wi_col]


column_bases = ['credit_rating', 'gdp', 'house_mkt', 'inflation', 'monetary', 'oil', 'unemp', 'usd']

for base in column_bases:
    wi_col = f'{base}_wi'
    ni_col = f'{base}_ni'

    # Replace 0 in wi_col with value from ni_col if ni_col is not 0
    mask = (mai_monthly_data[wi_col] == 0) & (mai_monthly_data[ni_col] != 0)
    mai_monthly_data.loc[mask, wi_col] = mai_monthly_data.loc[mask, ni_col]

    # Replace 0 in ni_col with value from wi_col if wi_col is not 0
    mask = (mai_monthly_data[ni_col] == 0) & (mai_monthly_data[wi_col] != 0)
    mai_monthly_data.loc[mask, ni_col] = mai_monthly_data.loc[mask, wi_col]


# Replace 0 with NaN and then forward fill
mai_daily_data.replace(0, pd.NA, inplace=True)
mai_daily_data.fillna(method='ffill', inplace=True)

# Replace 0 with NaN and then forward fill
mai_monthly_data.replace(0, pd.NA, inplace=True)
mai_monthly_data.fillna(method='ffill', inplace=True)


mai_daily_data.to_csv(f'{DATA_PATH}/interim/mai_daily_data_interim.csv', index=True)
mai_monthly_data.to_csv(f'{DATA_PATH}/interim/mai_monthly_data_interim.csv', index=True)


mef_daily_data.to_csv(f'{DATA_PATH}/interim/mef_daily_data_interim.csv', index=True)
mef_monthly_data.to_csv(f'{DATA_PATH}/interim/mef_monthly_data_interim.csv', index=True)


# Drop the specified columns in the daily market dataset
columns_to_drop_daily = ['lag_GSPC_1', 'lead_GSPC_2', 'lag_rfr_1', 'lag_date_1', 'lead_date_2']
mkt_daily_data = mkt_daily_data.drop(columns=columns_to_drop_daily)

# Drop the specified columns in the daily market dataset
columns_to_drop_monthly = ['lag_GSPC_1', 'lag_rfr_1', 'lag_date_1']
mkt_monthly_data = mkt_monthly_data.drop(columns=columns_to_drop_monthly)


mkt_daily_data.to_csv(f'{DATA_PATH}/interim/mkt_daily_data_interim.csv', index=True)
mkt_monthly_data.to_csv(f'{DATA_PATH}/interim/mkt_monthly_data_interim.csv', index=True)


#INTERIM TO PROCESSED

#Transforms MAI data from interim to processed
def mai_interim_to_processed(input_file, output_file):
    # Load the original CSV file
    df = pd.read_csv(input_file, index_col='date')

    # Calculate outputs
    df['credit_rating'] = df[['credit_rating_ni', 'credit_rating_wi']].mean(axis=1)
    df['gdp'] = df[['gdp_ni', 'gdp_wi']].mean(axis=1)
    df['house_mkt'] = df[['house_mkt_ni', 'house_mkt_wi']].mean(axis=1)
    df['inflation'] = df[['inflation_ni', 'inflation_wi']].mean(axis=1)
    df['monetary'] = df[['monetary_ni', 'monetary_wi']].mean(axis=1)
    df['oil'] = df[['oil_ni', 'oil_wi']].mean(axis=1)
    df['unemp'] = df[['unemp_ni', 'unemp_wi']].mean(axis=1)
    df['usd'] = df[['usd_ni', 'usd_wi']].mean(axis=1)

    # Create a new DataFrame with the desired columns
    new_df = df[['credit_rating', 'gdp', 'house_mkt', 'inflation', 'monetary', 'oil', 'unemp', 'usd']]

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_file, index=True)

mai_interim_to_processed(f'{DATA_PATH}/interim/mai_daily_data_interim.csv', f'{DATA_PATH}/processed/mai_daily_data_processed.csv')
mai_interim_to_processed(f'{DATA_PATH}/interim/mai_monthly_data_interim.csv', f'{DATA_PATH}/processed/mai_monthly_data_processed.csv')


#Transforms MEF data from interim to processed
def mef_interim_to_processed(input_file, output_file):
    # Load the original CSV file
    df = pd.read_csv(input_file, index_col='date')

    # Calculate outputs
    df['dp'] = np.log(df['d12']) - np.log(df['index'])
    df['dy'] = np.log(df['d12']) - np.log(df['lag_index_1'])
    df['ep'] = np.log(df['e12']) - np.log(df['index'])
    df['de'] = np.log(df['d12']) - np.log(df['e12'])
    df['rvol'] = df['svar']
    df['tms']= df['lty'] - df['tbl']
    df['dfy']= df['baa'] - df['aaa']
    df['dfr']= df['corpr'] - df['ltr']

    # Create a new DataFrame with the desired columns
    new_df = df[['dp','dy','ep','de', 'rvol','bm','ntis','tbl','lty','ltr','tms','dfy','dfr','infl']]

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_file, index=True)


mef_interim_to_processed(f'{DATA_PATH}/interim/mef_monthly_data_interim.csv', f'{DATA_PATH}/processed/mef_monthly_data_processed.csv')
mef_interim_to_processed(f'{DATA_PATH}/interim/mef_daily_data_interim.csv', f'{DATA_PATH}/processed/mef_daily_data_processed.csv')

def mkt_interim_to_processed(input_file, output_file):
    # Load the original CSV file
    df = pd.read_csv(input_file)

    # Convert date columns to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    df['lead_date_1'] = pd.to_datetime(df['lead_date_1'])

    # Calculate annualised excess return
    df['GSPCprem'] = (df['lead_GSPC_1'] / df['GSPC'] - 1) * 12 * 100 - df['rfr']

    # Create a new DataFrame with the desired columns
    new_df = df[['date', 'GSPCprem']]
    
    new_df.set_index('date', inplace=True)

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_file, index=True)


mkt_interim_to_processed(f'{DATA_PATH}/interim/mkt_daily_data_interim.csv', f'{DATA_PATH}/processed/mkt_daily_data_processed.csv')
mkt_interim_to_processed(f'{DATA_PATH}/interim/mkt_monthly_data_interim.csv', f'{DATA_PATH}/processed/mkt_monthly_data_processed.csv')    


