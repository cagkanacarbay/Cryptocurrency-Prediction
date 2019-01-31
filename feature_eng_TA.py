# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:26:57 2018

@author: Acarbay

Credit to Bruno Franca and Peter Bakker. The original code belongs to them and
was taken from https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
and modified as needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


""" TA indicators are seperated into two groups: Moving averages and Oscillators.
With moving averages, the resulting signal is on the same scale as the prices, 
while with oscillators, the resulting signal(s) are usually on a 0-100 or -100 to 100
scale. The aim is to normalize each of these signals to the same scale prior to
applying them to machine learning algorithms. This will be accomplished by normalizing
the outputs of TA algorithms on a case-by-case basis. In addition to the signals themselves
most TA algorithms also output buy/sell signals. These will be turned into continous
signals denoted by signal strength, where the definition of signal strength is
dependent on the TA algorithm. 
PS. The default values for each algorithm, if set, are set to the most commonly used
daily values adjusted for hourly data.
"""
# Moving Average
def MA(df, n):
    MA = pd.Series(df['close'].rolling(n).mean(), name = 'MA_' + str(n))
    df1 = df.join(MA)
    return df1

# Exponential Moving Average
def EMA(df, n):  
    EMA = df['close'].ewm(span = n, min_periods = n-1).mean()
    EMA.rename('EMA_' + str(n), inplace = True)
    df1 = df.join(EMA)  
    return df1 

# Triple Exponential Moving Average
def TRIX(df, n):  
    EX = df['close'].ewm(span = n, min_periods = n-1).mean()  
    EX = EX.ewm(span = n, min_periods = n-1).mean()  
    EX = EX.ewm(span = n, min_periods = n-1).mean()  
    ROC = [0]  
    for i in range (len(df)-1):  
        ROC_temp = (EX[i+1]-EX[i])/EX[i]  
        ROC.append(ROC_temp)  
    Trix = pd.Series(ROC, name = 'Trix_' + str(n))  
    df1 = df.join(Trix)  
    return df1

# Momentum  
def MOM(df, n):  
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))  
    df1 = df.join(M)  
    return df1

# Rate of Change - Output range: -1 and 1
def ROC(df, n):  
    M = df['close'].diff(n-1)  
    N = df['close'].shift(n-1)  
    ROC = pd.Series(M/N, name = 'ROC_' + str(n))  
    df1 = df.join(ROC)  
    return df1

# Stochastic oscillator with optional smoothing - Default is with no smoothing
def STOCH_OSC(df, nK = 14*24, nD = 3*24, nS = 1*24, smoothing_type = 0, join = True):  
    """ Stochastic oscillator with optional smoothing. Leave default values as is
    if not using smoothing.
    Parameters:
        df - ohcl/ohclv dataframe
        nK - Window size for %K
        nD - Window size for %D
        nS - Smoothing window size
        smoothing_type - Takes value 0 or 1. 
            0: Simple moving average
            1: Exponential moving average
        join - returns the dataframe + results if True, only results otherwise
    """
    K = pd.Series((df['close'] - df['low'].rolling(nK).min())/
                  (df['high'].rolling(nK).max() - df['low'].rolling(nK).min()),
                  name = 'Stoch-K_%k'+ str(nK))
    # Due to exchange maintanance or other issues, some time slots have constant
    # and equal close,low, high and open values that lead to NANs in the calculation
    # These can be removed by imputing the previous non-NAN value
    K = fix_nan(K)
    if smoothing_type == 0:
        # No smoothing case
        D = pd.Series(K.rolling(nD).mean(), name = 'Stoch-D_%D' + str(nD))
        # SMA smoothing case
        if nS != 1:
            D.rename('StochSMA-D_%d' + str(nD), inplace = True)
            K = K.rolling(nS).mean()  
            D = D.rolling(nS).mean()  
    elif smoothing_type == 1: 
        # EMA smoothing case
        D = pd.Series(K.ewm(span = nD, min_periods = nD-1).mean(), name = 'StochEMA-%D' + str(nD))
        K = K.ewm(span = nS, min_periods = nS-1).mean()
        D = K.ewm(span = nS, min_periods = nS-1).mean()
    # Signal Strength: Distance between K&D normalized to 0-1
    STOCHdiff = pd.Series(D-K, name = 'Stoch_ss-%D' + str(nD))
    df1 = pd.concat([K,D,STOCHdiff],axis = 1)
    if join == True:
        df1 = df.join(df1)
    return df1

# Moving Average Convergence Divergence (MACD) 
def MACD(df, n_fast = 12*24, n_slow = 26*24, n_signal = 9*24):  
    MACD = df['close'].ewm(span = n_fast, min_periods = n_slow-1).mean() -\
              df['close'].ewm(span = n_slow, min_periods = n_slow-1).mean() 
    MACD.rename('MACD_' + str(n_fast) + '_' + str(n_slow), inplace = True)
 
    MACDsign = MACD.ewm(span = n_signal, min_periods = n_signal-1).mean()
    MACDsign.rename('MACDsign_' + str(n_fast) + '_' + str(n_slow), inplace = True)
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df1 = df.join(MACD)  
    df1 = df1.join(MACDsign)  
    df1 = df1.join(MACDdiff)  
    return df1

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    TR = [0]  
    # True Range 
    for i in range(1, len(df)):  
        TR.append(max(df.at[i, 'high'], df.at[i-1, 'close']) - 
                  min(df.at[i, 'low'], df.at[i-1, 'close']))  
    VM_pos = [0]  
    VM_neg = [0] 
    # VM+ and VM-
    for i in range(1, len(df)):  
        VM_pos.append(abs(df.at[i, 'high'] - df.at[i-1, 'low']))
        VM_neg.append( abs(df.at[i, 'low'] - df.at[i-1, 'high'])) 
    # VI+ and VI-     
    VIpos = pd.Series(pd.Series(VM_pos).rolling(n).sum()/pd.Series(TR).rolling(n).sum(), name = 'Vortex+_' + str(n))
    VIneg = pd.Series(pd.Series(VM_neg).rolling(n).sum()/pd.Series(TR).rolling(n).sum(), name = 'Vortex-_' + str(n))
    df1 = df.join(VIpos)
    df1 = df1.join(VIneg) 
    return df1

# KST Oscillator  
def KST(df, r1 = 10*24, r2 = 15*24, r3 = 20*24, r4 = 30*24, n1 = 10*24, 
        n2 = 10*24, n3 = 10*24, n4 = 15*24, n_signal = 9*24):  
    M = df['close'].diff(r1 - 1)  
    N = df['close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['close'].diff(r2 - 1)  
    N = df['close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['close'].diff(r3 - 1)  
    N = df['close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['close'].diff(r4 - 1)  
    N = df['close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(ROC1.rolling(n1).sum()+ ROC2.rolling(n2).sum()*2 + 
                    ROC3.rolling(n3).sum()*3 + ROC4.rolling(n4).sum()*4,
                    name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' 
                    + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    KST_SMA = pd.Series(KST.rolling(n_signal).mean(), name = 'KST_signal_' + str(n_signal))
    df1 = df.join(KST)  
    df1 = df1.join(KST_SMA)
    return df1

# Relative Strength Index  
def RSI(df, n):  
    diff = np.diff(df['close'])
    # RSI values for the first n time slots
    first_n = diff[:n+1]
    U = first_n[first_n >= 0].sum()/n
    D = -first_n[first_n < 0].sum()/n
    RS = U/D
    RSI = [0] * (n-1)
    RSI.append(100-100/(1+RS))
    # RSI values for the rest of the data
    for i in range(n-1, len(df)-1):
        if diff[i] >= 0:
            Uv = diff[i]
            Dv = 0
        else:
            Uv = 0
            Dv = -diff[i]
        U = (U*(n-1) + Uv)/n
        D = (D*(n-1) + Dv)/n
        RSI.append(100 - 100/(1+(U/D)))
    RSI = pd.Series(RSI, name = 'RSI_' + str(n))
    df1 = df.join(RSI)
    return df1

""" Utility functions """
# Imputes the previous non-NAN value when a previous non-NAN value is 
# present and leaves as is for NAN values at the start of the data.
def fix_nan(data):
    nan_index = np.argwhere(np.isnan(data))
    # The 0th item will always be NAN, so we don't need to check it
    for i in range(1,len(nan_index)):
        data[nan_index[i]] = data[nan_index[i]-1]
    return data

# Cleans data from NANs, prepares array input for scaler, returns scaled values
def scaler(data, feature_range = (0,1)):
    # First take note of number of NANs and drop
    no_of_nans = np.argwhere(np.isnan(data))[-1][0] + 1 # index of final NAN values
    scaled_data = data.dropna()
    # Data is in the form of series. They need to be converted into an array and reshaped
    scaled_data = scaled_data.values.reshape(-1,1)
    # Perform scaling
    scaler = MinMaxScaler(feature_range)
    scaler = scaler.fit(scaled_data)
    scaled_data = scaler.transform(scaled_data)
    # NAN values must be put back in to preserve data integrity
    nans = np.empty([no_of_nans,1])
    nans[:] = np.nan
    scaled_data = np.insert(scaled_data, 0, nans, axis = 0)
    scaled_data = scaled_data.flatten()
    return scaled_data



#
#
#plt.figure(figsize = (14,6))
#plt.plot(a['date'][9800:],a['RSI_14'][9800:])
#plt.figure(figsize = (14,6))
#plt.plot(a['date'][9800:],a['close'][9800:])
#plt.figure(figsize = (14,8))
#plt.plot(b['date'][44900:],b['Vortex+_21'][44900:],
#        b['date'][44900:],b['Vortex-_21'][44900:])    
#
#plt.figure(figsize = (14,8))
#plt.plot(a['date'][44900:],a['Vortex_21'][44900:])
#    
#
#    
#import time
#v1 = time.time()
#a = Vortex(df,n)
#v2 = time.time()
#print(v2-v1)

