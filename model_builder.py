# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:26:37 2018

@author: Acarbay

Functions that aid in building and analyzing ML models.
"""
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools


def performance_measures(cm):
    """ Computes accuracy, precision, specificity and recall statistics for given 
    classification results. The function is primarily built to be used within
    classify_from_prob function, yet it can be used elsewhere with the correct inputs.
        Parameters:
        cm: Series of classification results with indices:
            - True Up, False Down, False Up, True Down
    """
    perf = pd.Series()
    # Accuracy: % of correct classifications
    perf['accuracy'] = (cm['True Up']+cm['True Down'])/cm.sum()
    # Precision: True Up/Predicted Up
    perf['precision'] = cm['True Up']/(cm['True Up']+cm['False Up'])
    # Recall/sensitivity(True Positive Rate): True Positive/Actual Positive
    perf['recall'] = cm['True Up']/(cm['True Up']+cm['False Down'])
    # Specificity(True Negative Rate): True Negative/Actual Negative
    perf['specificity'] = cm['True Down']/(cm['True Down']+cm['False Up'])
    # False Positive Rate: 1-Specificity
    perf['false positive rate'] = 1-perf['specificity']
    # False Negative Rate: 1-Recall
    perf['false negative rate'] = 1-perf['recall']
    return perf

def classify_from_prob(prob, pDown, pUp, Y_test, prob_tuning = False):
    """ This function takes the probabilities computed from a model, the probability levels
    for classification and returns the classification results, a confusion matrix and
    performance measures for classification accuracy. If prob_tuning is set to True, 
    only return accuracy and the number of total calls made. 
    """
    # probs = pd.DataFrame({'Down': prob[:,0].flatten(), 'Up': prob[:,1].flatten()})
    # pred = probs['Down'].map(lambda x: 0 if x > pUp else(1 if x < pDown else -1))
    pred = prob.map(lambda x: 1 if x > pUp else(0 if x < pDown else -1))
    # Identify indices of up/down classes
    ind_up = pred[pred == 1].index.values
    ind_down = pred[pred == 0].index.values
    cm = pd.Series()
    cm['True Up'] = 0
    cm['True Down'] = 0
    for i in range(len(ind_up)):
        if Y_test[ind_up[i]] == pred[ind_up[i]]:
            cm['True Up'] += 1
    for i in range(len(ind_down)):
        if Y_test[ind_down[i]] == pred[ind_down[i]]:
            cm['True Down'] += 1 
    total_correct = cm['True Up'] + cm['True Down']
    percentage_correct = (total_correct/(len(ind_up)+len(ind_down)))*100
    print('Accuracy (%):' + str(round(percentage_correct,2)))
    cm['False Up'] = (len(ind_up))-cm['True Up']
    cm['False Down'] = (len(ind_down))-cm['True Down']
    cm = cm.reindex(index = ['True Up', 'False Down', 'False Up', 'True Down'])
    if prob_tuning == False:
        perf = performance_measures(cm)
        return pred, cm, perf
    else:
        total_calls = len(pred)+1-pd.value_counts(pred)[-1] # total number of calls
        return percentage_correct, total_calls, cm

def mass_classify_from_prob(prob, pDown, pUp, targets, levels = math.nan, 
                            result_per_level = True, join_cm = False):
    """ This function takes the probabilities computed from a model and two lists
    denoting the classification levels for buy/sell cases and computes the 
    predictions at each combination of probabilities and accuracy statistics for each
    case. If join_cm is set to True, forms a confusion matrix for each probability
    pair, if not, returns two seperate dataframes one for the accuracy of each
    pUp and one for the accuracy of pDown. If levels is set to varying test set sizes,
    the function instead returns a list of the same outputs, each regarding one test
    size. If result per_level is set to False, the overall results displaying 
    each level will be returned instead."""
    
    targets.reset_index(drop = True, inplace = True)
    if levels is not math.nan:
        down_ind = []
        true_neg = []
        false_neg = []
        for p in pDown:
            down = prob.map(lambda x: 0 if x < p else -1)
            ind = down[down == 0].index.values
            down_ind.append(ind)
            true = []
            false = []
            for level in levels:
                if len(ind) > 0:
                    if ind[0] < level:
                        true.append((targets[:level][ind] == down[:level][ind]).sum())
                        false.append(len(targets[:level][ind].dropna()) - true[-1])
            true_neg.append([0]*(len(levels)-len(true))+true)
            false_neg.append([0]*(len(levels)-len(false))+false)

                
        up_ind = []
        true_pos = []
        false_pos = []
        for p in pUp:
            up = prob.map(lambda x: 1 if x > p else -1)
            ind = up[up == 1].index.values
            up_ind.append(ind)
            true = []
            false = []
            for level in levels:
                if len(ind) > 0:
                    if ind[0] < level:
                        true.append((targets[:level][ind] == up[:level][ind]).sum())
                        false.append(len(targets[:level][ind].dropna()) - true[-1])      
            true_pos.append([0]*(len(levels)-len(true))+true)
            false_pos.append([0]*(len(levels)-len(false))+false)

        
        true_neg = pd.DataFrame(data = np.array(true_neg), columns = levels, index = pDown)
        false_neg = pd.DataFrame(data = np.array(false_neg), columns = levels, index = pDown)
        true_pos = pd.DataFrame(data = np.array(true_pos), columns = levels, index = pUp)
        false_pos = pd.DataFrame(data = np.array(false_pos), columns = levels, index = pUp)
        
        accuracy_neg = (true_neg/(true_neg+false_neg))*100
        accuracy_pos = (true_pos/(true_pos+false_pos))*100
        
        no_of_calls_up = true_pos+false_pos
        no_of_calls_down = true_neg+false_neg
        
        perc_of_calls_up = 100*(true_pos+false_pos)/levels
        perc_of_calls_down = 100*(true_neg+false_neg)/levels
        
        if result_per_level:
        # Right now each accuracy statistic is in a dataframe of its own. Instead, we'd 
        # like to merge these statistic for each test set.
            up = []
            down = []
            for i in range(len(levels)):
                neg = pd.concat([true_neg[levels[i]], false_neg[levels[i]], 
                                 accuracy_neg[levels[i]], no_of_calls_down[levels[i]], 
                                 perc_of_calls_down[levels[i]]], axis = 1)
                neg.columns = ['True Down', 'False Down', 'Accuracy(%)', '# of calls', '% of calls']
                pos = pd.concat([true_pos[levels[i]], false_pos[levels[i]], 
                                 accuracy_pos[levels[i]], no_of_calls_up[levels[i]], 
                                 perc_of_calls_up[levels[i]]], axis = 1)
                pos.columns = ['True Down', 'False Down', 'Accuracy(%)', '# of calls', '% of calls']
                up.append(pos)
                down.append(neg)
        else:
            up, down = {}, {}
            up["True"] = true_pos
            up["False"] =  false_pos
            up["Accuracy(%)"] = accuracy_pos
            up["# of calls"] = no_of_calls_up
            up["% of calls"] = perc_of_calls_up  
            down["True"] = true_neg
            down["False"] =  false_neg
            down["Accuracy(%)"] = accuracy_neg
            down["# of calls"] = no_of_calls_down
            down["% of calls"] = perc_of_calls_down  
            
        
        # Join the predictions for pUp and pDown into pairs
        classifications = pd.DataFrame()
        for i, pD in enumerate(pDown):
            for j, pU in enumerate(pUp):
                temp = np.full((len(targets),),-1)
                temp[list(down_ind[i])] = 0
                temp[list(up_ind[j])] = 1
                classifications[str(round(pD,5))+':'+str(round(pU,5))] = temp
        cls = []
        for level in levels:
            cls.append(classifications[:level])
        
        return cls, up, down
    else:
        # Classify each case according to the given probabilities and calculate
        # the number of true positives&negatives
        down_ind = []
        true_neg = pd.Series(name = 'True Down')
        false_neg = pd.Series(name = 'False Down')
        for p in pDown:
            down = prob.map(lambda x: 0 if x < p else -1)
            ind = down[down == 0].index.values
            true_neg[str(round(p,5))] = (targets[ind] == down[ind]).sum()
            false_neg[str(round(p,5))] = (len(ind)) - true_neg[str(round(p,5))] 
            down_ind.append(ind)
        
        up_ind = []
        true_pos = pd.Series(name = 'True Up')
        false_pos = pd.Series(name = 'False Up')
        for p in pUp:
            up = prob.map(lambda x: 1 if x > p else -1)
            ind = up[up == 1].index.values
            true_pos[str(round(p,5))] = (targets[ind] == up[ind]).sum()
            false_pos[str(round(p,5))] = (len(ind)) - true_pos[str(round(p,5))] 
            up_ind.append(ind)
            
        # Join the predictions for pUp and pDown into pairs
        classifications = pd.DataFrame()
        for i, pD in enumerate(pDown):
            for j, pU in enumerate(pUp):
                temp = np.full((len(targets),),-1)
                temp[list(down_ind[i])] = 0
                temp[list(up_ind[j])] = 1
                classifications[str(round(pD,5))+':'+str(round(pU,5))] = temp
                
        # Produce confusion matrix for each pair
        if join_cm:
            true_up = pd.Series(name = 'True Up')
            false_down = pd.Series(name = 'False Down')
            false_up = pd.Series(name = 'False Up')
            true_down = pd.Series(name = 'True Down')
            for i, pD in enumerate(pDown):
                for j, pU in enumerate(pUp): 
                    true_up[str(round(pD,5))+':'+str(round(pU,5))] =  true_pos[str(round(pU,5))]
                    false_down[str(round(pD,5))+':'+str(round(pU,5))] =  false_neg[str(round(pD,5))]
                    false_up[str(round(pD,5))+':'+str(round(pU,5))] =  false_pos[str(round(pU,5))]
                    true_down[str(round(pD,5))+':'+str(round(pU,5))] =  true_neg[str(round(pD,5))]
            cm = pd.concat([true_up,false_down,false_up,true_down], axis = 1)
            return classifications, cm
        else:
            cm_up = pd.concat([true_pos, false_pos], axis = 1)
            cm_up['Accuracy(%)'] = (true_pos/(true_pos+false_pos))*100
            cm_up['# of calls'] = true_pos+false_pos
            cm_up['% of calls'] = 100*(true_pos+false_pos)/len(targets)
            cm_down = pd.concat([true_neg, false_neg], axis = 1)
            cm_down['Accuracy(%)'] = (true_neg/(true_neg+false_neg))*100
            cm_down['# of calls'] = true_neg+false_neg
            cm_down['% of calls'] = 100*(true_neg+false_neg)/len(targets)
            return classifications, cm_up, cm_down
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken from the sci-kit learn website:
        http://scikit-learn.org/stable/auto_examples/model_selection/
        plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() + cm.min())/ 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_results(accuracy, normalize = False):
    # Plot the accuracy results for all models
    plt.figure(figsize = (12,36))
    # Compute subplot size - we want 3 plots in each row
    row_no = math.ceil(len(accuracy.columns)/3)
    for i in range(len(accuracy.columns)):
        plt.subplot(row_no,3,i+1)
        plot_confusion_matrix(accuracy[accuracy.columns[i]].values.reshape([2,2]), 
                              classes = ['up','down'],
                              normalize=normalize,
                              title=accuracy.columns[i],
                              cmap=plt.cm.Blues)