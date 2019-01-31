# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:02:13 2018
This program produces a variety of train/validation/test splits of the data. The
aim of employing multiple splits are as follows:
    - To check if the models built are not highly specific to the split
    - To provide information on the effects of the split on model accuracy
    - To provide information on the effects of total data size on model accuracy
@author: Acarbay
"""

import pandas as pd
import numpy as np
import time

import feature_eng_prep as prep
import model_builder as mb

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

import lightgbm as lgb


# Get features 
df = pd.read_pickle("./hourly_with_features")
df = df[2000:] # TA algorithms return NA for the first few datapoints so we discard them
df = df.reset_index(drop = True)


# LightGBM parameters
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss', 'binary_error'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 50
params['max_depth'] = 1000
params['max_bin'] = 63
params['device'] = 'gpu'


# Search grid for probability tuning with the validation set
pUp = np.linspace(0.5025, 0.55, num = 39)
pDown = np.linspace(0.45, 0.4975, num = 39)
probs = [(a,b) for a in pUp for b in pDown]


df1 = df.drop(columns = 'close')
# Get the data splits
train_size = np.linspace(10000,30000,11)
valid_size = [500, 1000, 2000, 3000]
test_size = np.linspace(2000,20000,10)
# These sets denote various split-sizes to test the model on
sets = [(a,b,c) for a in train_size for b in valid_size for c in test_size]
    
# splits list gives all train/valid/test splits that the model will train upon
# be validated with and to be tested on. 
splits = []
for split_size in sets:
    splits = splits + prep.get_rolling_splits(df, start = 2000, delta = 1000, split_size = split_size)
""" There is some overlap between these splits where a training set is followed by 
different sized validation and test sets. Training the model on the same set is
thus redundant. To remove this redundancy we first identify the unique training
sets, then we identify the combination of each validation and test set that is
available to each of the unique training sets.
"""
train_sets = []
for split in splits:
    current_train_set = [split['train_begin'],split['valid_begin']]
    if current_train_set not in train_sets:
        train_sets.append(current_train_set)
# After the unique train sets are identified, the end of the validation set for 
# each case is then also identified
test_begin = [[] for _ in train_sets]
for split in splits:
    current_train_set = [split['train_begin'],split['valid_begin']]
    ind = train_sets.index(current_train_set)
    # Check if the end of the validation set/begining of the test is already 
    # appended to the list, append if not
    if split['test_begin'] not in test_begin[ind]:
        test_begin[ind].append(split['test_begin'])
# For each validation case, there are multiple test cases available, so finally
# those are identified
test_end = []
for i in range(len(train_sets)):
    temp = [[]for _ in test_begin[i]] 
    test_end.append(temp)
for split in splits:
    current_train_set = [split['train_begin'],split['valid_begin']]
    ind1 = train_sets.index(current_train_set)
    ind2 = test_begin[ind1].index(split['test_begin'])
    test_end[ind1][ind2].append(split['test_end'])
""" The code above creates three lists train_sets, test_begin and test_end, which
are bound to each other by their indexes. Each pair in train_sets denote the 
beginning and end of a unique training set. Each value in test_begin then denote
the end of one validation set for the training set with the same index. Each list
within the same index of test_end then denote different size test sets for
each validation set denoted by test_begin. By differentiating each set such as
this, the redundancy of training and testing different models is entirely removed.
"""
prices = df['close']
Y = prep.build_targets_full(df['close'],0)

s = time.time()
# Master results lists. The columns are in order:
# train_begin, valid_begin, test_begin, test_end, prob, validation % of calls,
# valid_acc, test_acc, test % of calls
res_up_full = np.empty([0,9])
res_down_full = np.empty([0,9])
for i, train_set in enumerate(train_sets[159:]):
    s1 = time.time()
    print('Loop:',i ,' Training Set:', train_set)
    X_train, X_rest = prep.set_features_basic(df1, train_set[0], train_set[1])
    Y_train = Y[train_set[0]:train_set[1]]
    # Train the model
    s2 = time.time()
    lgb_train = lgb.Dataset(X_train, Y_train)
    clf = lgb.train(params, lgb_train, 100, verbose_eval = 10)
    e2 = time.time()
    print('Loop:',i, 'train time:', str(round((e2-s2),2)), 'seconds')
    # loop for each validation set for the given training set
    res_up_train = np.empty([0,9])
    res_down_train = np.empty([0,9])
    for j, valid_ind in enumerate(test_begin[i]):
        X_valid = X_rest[:valid_ind-train_set[1]]
        Y_valid = Y[train_set[1]:valid_ind]
        # Probabilities for the validation set
        clf_prob = clf.predict(X_valid)
        cls, up, down = mb.mass_classify_from_prob(pd.Series(clf_prob), pDown, pUp, Y_valid)
        # Select best classification levels based on validation results
        # Scrap models with accuracy below 52%
        best_up = up.loc[up['Accuracy(%)'] > 52]
        best_down = down.loc[down['Accuracy(%)'] > 52]
        # Quantize the % of calls column into parts of 5%, and select the model that
        # provides the highest accuracy in each bin
        best_up['bins'] = pd.np.digitize(best_up['% of calls'], bins = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5])
        best_down['bins'] = pd.np.digitize(best_down['% of calls'], bins = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5])
        # The values proposed in bin 0 (those with % of calls below 2.5%) are quite
        # unreliable. So we scrap them and then recover the highest accuracy 
        # classification level in each bin.
        best_up = best_up[best_up['bins'] != 0] # remove bin 0
        best_down = best_down[best_down['bins'] != 0] # remove bin 0
        best_up = best_up.loc[best_up.groupby(by = 'bins')['Accuracy(%)'].idxmax()]
        best_down =  best_down.loc[best_down.groupby(by = 'bins')['Accuracy(%)'].idxmax()]
        # In some cases there are no good classifiers availiable for the particular dataset
        # yet a classification is still necessary so we feed it prob level with the
        # lowest number of calls made that is non-zero
        if best_down.empty:
            best_down = down.loc[down['Accuracy(%)'] >= 50]
            best_down = best_down.loc[best_down['# of calls'] == best_down['# of calls'].max()]     
            best_down['bins'] = pd.np.digitize(best_down['% of calls'], bins = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5])
            # if the set is still empty grab the highest prob available
            if best_down.empty:
                best_down = down.iloc[[-1]]
                best_down['bins'] = 0
            elif len(best_down) > 1:
                best_down = best_down.iloc[[-1]]
                best_down['bins'] = 0
            
        if best_up.empty:
            best_up = up.loc[up['Accuracy(%)'] >= 50]
            best_up = best_up.loc[best_up['# of calls'] == best_up['# of calls'].max()]
            best_up['bins'] = pd.np.digitize(best_up['% of calls'], bins = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5])
            # if the set is still empty grab the lowest prob available
            if best_up.empty:
                best_up = up.iloc[[0]]
                best_up['bins'] = 0
            elif len(best_down) > 1:
                best_up = best_up.iloc[[0]]
                best_up['bins'] = 0
                
        """ Now that the best classification levels are identified, we proceed to test the
        model on the dataset. To do so we test on the largest possible test set for the
        given validation set(to conserve time), and then feed the results to a function that
        returns classification statistics based on the specific test sets.
        """
        largest_test_end = test_end[i][j][-1] 
        X_test = X_rest[valid_ind-train_set[1]:largest_test_end-train_set[1]]
        Y_test = Y[valid_ind:largest_test_end]
        # Predict for the test set
        clf_prob = clf.predict(X_test)
        # convert selected probabilities to float
        p_up = [float(prob) for prob in best_up.index] 
        p_down = [float(prob) for prob in best_down.index]
        
        # State test set sizes
        levels = test_end[i][j] - valid_ind
        # Get classification results
        cls,up,down = mb.mass_classify_from_prob(pd.Series(clf_prob), p_down, p_up, Y_test, levels)
        # Prepare results
        res_up_valid = np.empty([0,9])
        res_down_valid = np.empty([0,9])
        res_up = np.empty([len(best_up),9])
        res_up[:,0] = train_set[0]              # train_begin
        res_up[:,1] = train_set[1]              # valid_begin
        res_up[:,2] = valid_ind                 # test_begin
        res_up[:,5] = best_up['% of calls']     # validation % of calls
        res_up[:,6] = best_up['Accuracy(%)']    # valid accuracy
        res_down = np.empty([len(best_down),9])
        res_down[:,0] = train_set[0]              # train_begin
        res_down[:,1] = train_set[1]              # valid_begin
        res_down[:,2] = valid_ind                 # test_begin
        res_down[:,5] = best_down['% of calls']     # validation % of calls
        res_down[:,6] = best_down['Accuracy(%)']  # valid accuracy
        for k, level in enumerate(levels):
            res_up[:,3] = valid_ind + level     # test_end
            res_up[:,4] = up[k].index           # pUp
            res_up[:,7] = up[k]['Accuracy(%)']  # Test Accuracy
            res_up[:,8] = up[k]['% of calls']   # test % of calls
            res_down[:,3] = valid_ind + level     # test_end
            res_down[:,4] = down[k].index           # pDown
            res_down[:,7] = down[k]['Accuracy(%)']  # Test Accuracy
            res_down[:,8] = down[k]['% of calls']   # test % of calls
            # Join results
            res_up_valid = np.concatenate([res_up_valid,res_up])
            res_down_valid = np.concatenate([res_down_valid,res_down])
        # Join results with the loop above
        res_up_train = np.concatenate([res_up_train,res_up_valid])
        res_down_train = np.concatenate([res_down_train,res_down_valid])
    # Join results with the master results lists
    res_up_full = np.concatenate([res_up_full,res_up_train])
    res_down_full = np.concatenate([res_down_full,res_down_train])
    e1 = time.time()
    print('Loop:',i, 'total time:', str(round((e1-s1),2)), 'seconds')
e = time.time()
print('Total time:', str(round((e-s),2)), 'seconds')       

        
results_up = pd.DataFrame(data = res_up_full, columns = ['train_begin', 
                          'valid_begin', 'test_begin', 'test_end', 'pUp', 
                          'valid (%) of calls', 'valid_acc', 'test_acc',
                          'test (%) of calls'])
results_down = pd.DataFrame(data = res_down_full, columns = ['train_begin',
                            'valid_begin', 'test_begin', 'test_end', 'pDown', 
                            'valid (%) of calls', 'valid_acc', 'test_acc', 
                            'test (%) of calls'])

import pickle
results_up.to_pickle("./results_up2")
results_down.to_pickle("./results_down2")


