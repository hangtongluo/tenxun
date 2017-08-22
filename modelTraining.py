# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:12:00 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from imblearn import under_sampling
from evaluation import self_logloss
from features_make import features_poll



def process_train_test_data():
    data = pd.read_csv(r'F:\data\tenxun\data\original_features_add.csv', low_memory=False) 
    data = data.drop(['marriageStatus','haveBaby','education','hometown','residence','telecomsOperator'], axis=1)  
    data = data.fillna(0)     
    train = data[data['instanceID'] == -1]
    test = data[data['label'] == -1]     
    train.to_csv(r'E:\competition\tenxun\data\train_process.csv', index=False)
    test.to_csv(r'E:\competition\tenxun\data\test_process.csv', index=False)
    print(train.head())           
    print(test.head())   
    return train, test


def trainModel(clf):
    print('=========================分割数据集================================')
    train, test = process_train_test_data()
    train = train[(train['clickTime']>= 200000)] #提升
    target = np.ravel(train['label'])
    train = np.asarray(train.drop(['instanceID','userID','clickTime','label'], axis=1)) #
    test = pd.read_csv(r'E:\competition\tenxun\data\test_process.csv', low_memory=False)
    test = test.drop(['clickTime','userID','label'], axis=1) #
    submission = pd.DataFrame(test['instanceID'])
    test = np.asarray(test.drop('instanceID', axis=1))

    train_roc_auc_score = []
    test_roc_auc_score = []
    
    train_self_logloss = []
    test_self_logloss = []

    sss = StratifiedShuffleSplit(target, 3, test_size=0.25, random_state=2017)
    for train_index, test_index in sss:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        clf.fit(X_train, y_train)

        train_proba = clf.predict_proba(X_train)[:,1]
        test_proba = clf.predict_proba(X_test)[:,1]
        
        print('train roc_auc_score：', metrics.roc_auc_score(y_train, train_proba))
        print('test roc_auc_score：', metrics.roc_auc_score(y_test, test_proba))
        train_roc_auc_score.append(metrics.roc_auc_score(y_train, train_proba))
        test_roc_auc_score.append(metrics.roc_auc_score(y_test, test_proba))
        
        print('train self_logloss：', self_logloss(y_train, train_proba))
        print('test self_logloss：', self_logloss(y_test, test_proba))
        train_self_logloss.append(self_logloss(y_train, train_proba))
        test_self_logloss.append(self_logloss(y_test, test_proba))
        
        
    proba = clf.predict_proba(test)[:,1]
    print(Counter(np.asarray(proba > 0.5)))
    
    print("mean train_roc_auc_score", np.mean(train_roc_auc_score))
    print("mean test_roc_auc_score", np.mean(test_roc_auc_score))
    print("mean train_self_logloss", np.mean(train_self_logloss))
    print("mean test_self_logloss", np.mean(test_self_logloss))
    
    submission['prob'] = proba
    submission.to_csv(r'E:\competition\tenxun\submit\submission.csv', index=False)
        
    
print('=============================开始==================================')    
clf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=300,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    seed=2017)

#clf= GradientBoostingClassifier(n_estimators=100, random_state=2017)


print('=============================特征提取==================================')   
features_poll()
print('=============================模型训练==================================')  
trainModel(clf)















































