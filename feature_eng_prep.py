# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:54:20 2018
This program builds additional features and targets from the ohlcv data. The 
targets built are the times at which extraordinary movements to the up and downside
are present. These times are located by comparing the percentage of returns at each
time step and identifying the overall distribution of these changes. With such a 
distribution, arbitrarily significant movements can then be identified and 
characterized as the target variable of a binary classifier. 
@author: Çağkan Acarbay
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import feature_eng_TA as ta
import math

def build_targets(prices, target, train_begin, valid_begin, test_begin, test_end = -1):
    """ Computes the percentage change in prices, returns train/test/validation sets
    as a True/False array for the given target variable where:
            - for positive target the output is 1 for values greater than target
            - for negative target the output is 1 for values less than target
    """
    percentage_change = (prices/prices.shift()-1)*100
    percentage_change.fillna(value = 0, inplace = True)
    if target >= 0:
        outliers = percentage_change.map(lambda x: 0 if x <= target else 1)
    else:
        outliers = percentage_change.map(lambda x: 0 if x > target else 1)
    # Split   
    Y_train = outliers[train_begin:valid_begin]
    Y_train.index = range(len(Y_train))
    Y_valid = outliers[valid_begin:test_begin]
    Y_valid.index = range(len(Y_valid))
    Y_test = outliers[test_begin:test_end]
    Y_test.index = range(len(Y_test))
    return Y_train, Y_valid, Y_test

def build_targets_full(prices, target):
    """ A basic version of the function build_targets that returns the entire
    target list for the given list of prices as one array."""
    percentage_change = (prices/prices.shift()-1)*100
    percentage_change.fillna(value = 0, inplace = True)
    if target >= 0:
        outliers = percentage_change.map(lambda x: 0 if x <= target else 1)
    else:
        outliers = percentage_change.map(lambda x: 0 if x > target else 1)
    return outliers

def get_features(df, sliding_window_size = 24, ma_size = [10, 20, 50, 200]):
    """ Computes technical indicator values from given ohlcv data. Returns single 
    dataframe containing the original dataframe and the computed features.
        Parameters:
            df - ohlcv data
            sliding_window_size - number of lags to add to the dataframe for each
                time point
            ma_size - list of n values for the technical indicators of
            MA, EMA, TRIX, MOM, ROC, Vortex and RSI.
    """
    # Run TA algorithms to prepare features
    n = ma_size 
    for i in range(len(n)):
        #df = ta.MA(df,n[i])
        #df = ta.EMA(df,n[i])
        df = ta.TRIX(df,n[i])
        df = ta.MOM(df,n[i])
        df = ta.ROC(df,n[i])
        df = ta.Vortex(df,n[i])
        df = ta.RSI(df,n[i])
    # For these TA algorithms the most commonly used values of parameters are employed
    df = ta.MACD(df)
    df = ta.KST(df)
    df = ta.STOCH_OSC(df,nS = 3*24)
    
    df1 = df.drop(columns = ['time', 'date', 'open', 'low', 'high'])
    df2 = df1
    # Another feature apart from the TA results are the shifted version of the closing price
    # For this, a sliding window method is used to provide the closing prices for
    # the past 24 hours.    
    for i in range(1, sliding_window_size+1):
        temp = df1.shift(i)
        temp = temp.add_suffix('_'+str(i))
        df2 = df2.join(temp)
        
    # Check if any NANs exist beyond the initial calculations
    if df[2000:].isnull().values.any() == True:
        print('There are NANs in the data. Clean data before proceeding.')
    else:
        print('No NANs in the features')
        
    return df2   

def set_features(X, train_begin, valid_begin, test_begin, test_end = -1):
    """ Takes dataframe with prepared features, beginning index for train, validation 
    and test sets and prepares the data by scaling and removing unnecessary ohlcv
    data. Returns seperate X_train, X_valid and X_test data.
    """
    X_train = X[train_begin-1:valid_begin-1]
    X_valid = X[valid_begin-1:test_begin-1]
    X_test = X[test_begin-1:test_end-1]
    # MinMax Scaler built on X_train
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test

def set_features_basic(X, train_begin = -1, valid_begin = -1):
    """ A basic version of the function set_features that returns the training
    set and the rest of the features as two multi-dimensional arrays. If
    train_begin isn't set, scales according to train_begin, but returns the entire 
    set as one.
    """
    if train_begin == -1:
        X_train = X[:valid_begin-1] 
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X_train)
        X_fit = scaler.transform(X[:-1]) # drop last point
        return X_fit
    else:
        X_train = X[train_begin-1:valid_begin-1]
        X_rest = X[valid_begin-1:-1]
        # MinMax Scaler built on X_train
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_rest = scaler.transform(X_rest)
    return X_train, X_rest

def get_splits(df, data_size = [0.3, 0.5, 0.75, 1], train_ratios = 
               [0.8, 0.7, 0.6], valid_ratios = [0.05, 0.1, 0.2]):
    """ This function produces a variety of train/validation/test splits of the data
    based on the given ratios of total data size and train/valid set sizes. The
    function returns a list of series denoting the start of each train/valid/test
    split, which may then be used in conjunction with the set_features function to
    get the resulting sets."""
    splits = []
    for data_ratio in data_size:
        begin = round(len(df) - data_ratio*len(df)) + 1 
        size = len(df) - begin
        for train_ratio in train_ratios:
            for valid_ratio in valid_ratios:
                if train_ratio + valid_ratio < 1:
                    split = pd.Series(name = 'Size:' + str(size) + ' Ratio:' 
                                         + str(train_ratio) + ':' + str(valid_ratio) 
                                         + ':' + str(round(1-train_ratio-valid_ratio,2)))
                    split['train_begin'] = begin 
                    split['valid_begin'] = round(begin + (len(df) - begin) * train_ratio)
                    split['test_begin'] = round(begin + (len(df) - begin) * (train_ratio + valid_ratio))
                    split['ratio'] = [train_ratio,valid_ratio,round(1-train_ratio-valid_ratio,2)]
                    splits.append(split)
    return splits

def get_rolling_splits(df, start = 2000, delta = 1000, split_size = [10000, 1000, 2000]):
    """ This function produces a rolling train/validation/test splits of the data
    based on the fixed train/validation/test dataset sizes and a fixed delta. The
    list output may then be used with the set_features function to get the resulting
    sets.
    """
    no_of_splits = math.floor((len(df)-sum(split_size)-start)/delta)
    begin = int(start)
    splits = []
    for i in range(no_of_splits):
        split = pd.Series(name = 'Begin:' + str(begin) + ' Sizes:' 
                                          + str(int(split_size[0])) + ':' 
                                          + str(int(split_size[1])) + ':'
                                          + str(int(split_size[2])))
        split['train_begin'] = begin 
        split['valid_begin'] = begin + int(split_size[0])
        split['test_begin'] = split['valid_begin'] + int(split_size[1])
        split['test_end'] = split['test_begin'] + int(split_size[2])
        begin = begin + delta
        splits.append(split)
    return splits
    
## 

#df_org = pd.read_pickle("./btc_hourly")
## Get features 
#df = get_features(df_org, sliding_window_size = 72)
#import pickle
#df.to_pickle("./hourly_with_features")
