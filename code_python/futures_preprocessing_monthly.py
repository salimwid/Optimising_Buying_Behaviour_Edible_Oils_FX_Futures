# Compress daily data to get the prices by week
import datetime
import pandas as pd
import numpy as np
import datetime

def get_price_by_date(data):
    
    df = data.copy()
    
    # Change dtype of tradingday
    df['tradingday'] = pd.DatetimeIndex(df['tradingday'])
    
    # Get mean close price, sum of weekly volume and mean open interest by week
    aggregations = {"close": np.mean, 'volume': np.sum, 'openinterest': np.mean}
    df = df.groupby(['symbol', 'delivery_month', 'delivery_year',
                           pd.Grouper(key='tradingday', freq='M')]).agg(aggregations).reset_index().sort_values(['symbol', 'tradingday'])
    
    # Construct a key
    df['key'] = df['tradingday'].astype(str) + '//' +df['delivery_month'].astype(str).str.pad(2, fillchar = '0') + df['delivery_year'].astype(str)
    
    return df


def get_ema(data, cols_to_calculate, ema_lookback:list, pct_lookback:list):
  """
  Calculates the exponential moving average for the desired columns in the dataframe. Adds additional columns to df.
  cols_to_calculate: Which columns to calculate the EMA
  lookback_days: Lookback periods to include in EMA calculation.
  """
  group_cols = ['delivery_year', 'delivery_month', 'symbol', 'tradingday']
  data = data.sort_values(by=group_cols)
  
  new_cols = []

  for col in cols_to_calculate:
    for lookback in ema_lookback:
    # grouping by symbol also groups by delivery month by default because symbol is tied to year and month  
        data['{}_ema_{}m'.format(col, lookback)] = data.groupby(['delivery_year', 'delivery_month', 'symbol'])[col].ewm(lookback).mean().values # Calculate exponential moving average
        new_cols.append('{}_ema_{}m'.format(col, lookback))

  for col in cols_to_calculate:
    for lookback in pct_lookback:
        data['{}_pctgrowth_{}m'.format(col, lookback)] = data.groupby(['symbol'])[col].pct_change(lookback).values # Calculate percentage change
        new_cols.append('{}_pctgrowth_{}m'.format(col, lookback))
  
  return data, new_cols


def add_futures_data(adv_months_list, price_df, data, additional_cols_to_merge=[]):
    
    df = data.copy()
    
    for adv_months in adv_months_list:
        df['delivery_month'] = df['buy_month'] + adv_months
        df['delivery_year'] = df['buy_year']

        # if this crosses into a new year (delivery_month > 12):
        df['delivery_year'] = np.where(df['delivery_month'] > 12, df['delivery_year'] +1, df['delivery_year']) # increase year by 1
        df['delivery_month'] = np.where(df['delivery_month'] > 12, df['delivery_month'] - 12, df['delivery_month']) # subtract 12 from the month

        # create a key to join the price, specific to that 'months_in_advance'
        df['{}m_key'.format(adv_months)] = df['date'].astype(str) + '//' +df['delivery_month'].astype(str).str.pad(2, fillchar = '0') + df['delivery_year'].astype(str)
        
        # join data from price df to main df on that key
        cols_to_merge = ['key', 'close', 'volume', 'openinterest']+additional_cols_to_merge
        df = df.merge(price_df[cols_to_merge], how = 'left', left_on = '{}m_key'.format(adv_months), right_on = 'key')

        # rename columns to have the month of the futures
        col_dict = {}
        for col in cols_to_merge:
            col_dict[col] = '{}m_{}'.format(adv_months, col)

        df = df.rename(columns=col_dict)

        # drop the joining column, can uncheck this if you want to debug things
        cols_to_drop = ['delivery_month', 'delivery_year']
        df.drop(cols_to_drop, inplace = True, axis = 1)
    
    return df

def impute_data(df):
    #Get columns with NA and clean the list
    col_with_na_clean = df.columns[df.isna().any()].tolist()
    
    for col in col_with_na_clean:
    #Get index of all missing values
        na_index = df[df[col].isna()].index

        #Reiterate to all missing values and impute accordingly
        #Either take the first 5 closest values if date difference is acceptable, or take the mean
        #Closest values for most rows are forward looking, but for the rows at the end will be backward looking as future data might not be available
        for index in na_index:
            selected_rows = df.iloc[index:]
            if selected_rows[~selected_rows[col].isna()].shape[0] != 0:
                date_difference = (selected_rows[~selected_rows[col].isna()].iloc[0]['date'] - df.iloc[index]['date']).days

                if date_difference <= 10:
                    moving_mean = selected_rows[~selected_rows[col].isna()].iloc[index:index + 5][col].mean()
                    df.at[index, col] = moving_mean

                else:
                    df.at[index, col] = df[col].mean()
                    #print(df[col].mean())

            else:
                date_difference = (df.iloc[index - 1]['date'] - df.iloc[index]['date']).days

                if date_difference <= 10:
                    df.at[index, col] = df[col][index - 1]

                else:
                    df.at[index, col] = df[col].mean()
    
    return df

def lag_variables(data, lag_dict, name):
    df = data.copy()
    for variable, lag_list in lag_dict.items():
        for lag_period in lag_list:
            df['{}(t-{})_{}'.format(variable,lag_period, name)] = df[variable].shift(periods = lag_period)
        
        # Drop the original column once done
        df = df.drop(variable, axis = 1)
    return df

def preprocess(raw_data, ema_lookback, pct_lookback, adv_months_list, start_date, end_date):
    
  #--------------------------------------------------------#
  price_by_week = get_price_by_date(data = raw_data)
  
  #----------------------------------------------------#
  # Calculate exponential moving average
  cols_to_calculate = ['close','volume','openinterest']
  price_by_week, output_col = get_ema(data = price_by_week, 
                                      cols_to_calculate = cols_to_calculate, 
                                      ema_lookback = ema_lookback, 
                                      pct_lookback = pct_lookback)

  #---------------------------------------------------#
  # Generate a list of month-end closes
  start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
  end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
  date_generated = pd.date_range(start, end, freq='M')

  # Format date columns 
  df = pd.DataFrame(date_generated)
  df.columns = ['date'] # rename column
  df['buy_month'] = pd.DatetimeIndex(df['date']).month
  df['buy_year'] = pd.DatetimeIndex(df['date']).year

  #----------------------------------------------------#
  # Join n month futures
  df = add_futures_data(adv_months_list = adv_months_list, price_df = price_by_week, data = df, additional_cols_to_merge=output_col)

  #----------------------------------------------------#
  # Join n month futures
  # short script to remove redundant key columns that are only used for debugging
  cols = df.columns
  remove_cols = [v for v in cols if 'key' in v] + ['buy_month', 'buy_year'] # remove because its already captured in susan's
  df = df.drop(remove_cols, axis = 1)
  
  #df = impute_data(df) #--> throwing an error
  df = df.fillna(method = 'ffill')

  return df