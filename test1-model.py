# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:26:13 2017

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn import under_sampling
from evaluation import self_logloss

#import scipy as sp
#def logloss(act, pred):
#  epsilon = 1e-15
#  pred = sp.maximum(epsilon, pred)
#  pred = sp.minimum(1-epsilon, pred)
#  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
#  ll = ll * -1.0/len(act)
#  return ll

def process_train_test_data():

    ad = pd.read_csv(r'F:\data\tenxun\pre\ad.csv', low_memory=False) 
#    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False) 
#    user_installedapps = pd.read_csv(r'F:\data\tenxun\pre\user_installedapps.csv', low_memory=False) 
    app_categories = pd.read_csv(r'F:\data\tenxun\pre\app_categories.csv', low_memory=False) 
#    app_categories_process = pd.read_csv(r'F:\data\tenxun\data\app_categories_process.csv', low_memory=False) 
    position = pd.read_csv(r'F:\data\tenxun\pre\position.csv', low_memory=False) 
    user = pd.read_csv(r'F:\data\tenxun\pre\user.csv', low_memory=False) 
    
    train = pd.read_csv(r'F:\data\tenxun\pre\train.csv', low_memory=False) 
    train = train.drop('conversionTime', axis=1)
    test = pd.read_csv(r'F:\data\tenxun\pre\test.csv', low_memory=False) 
    
#    train['clickTime_day'] = train['clickTime'].apply(lambda x: int(x / 10000))
#    train['clickTime_hour'] = train['clickTime'].apply(lambda x: int(x % 100))
#    train['clickTime_hour'] = train['clickTime'].apply(lambda x: int((x / 100)  % 100))
#    
#    test['clickTime_day'] = test['clickTime'].apply(lambda x: int(x / 10000))
#    test['clickTime_hour'] = test['clickTime'].apply(lambda x: int(x % 100))
#    test['clickTime_hour'] = test['clickTime'].apply(lambda x: int((x / 100)  % 100))
#    

    print('====================================基本处理=============================================')
    
    train = pd.merge(train, user, on='userID', how='left')
    test = pd.merge(test, user, on='userID', how='left')
    
    train = pd.merge(train, position, on='positionID', how='left')
    test = pd.merge(test, position, on='positionID', how='left')
    
    train = pd.merge(train, ad, on='creativeID', how='left')
    test = pd.merge(test, ad, on='creativeID', how='left')
    
    train = pd.merge(train, app_categories, on='appID', how='left')
    test = pd.merge(test, app_categories, on='appID', how='left')
    
    


    train.to_csv(r'E:\competition\tenxun\data\train_process.csv', index=False)

    test.to_csv(r'E:\competition\tenxun\data\test_process.csv', index=False)

    return train, test


def model1_test(clf):
    train = pd.read_csv(r'E:\competition\tenxun\data\train_process.csv', low_memory=False) 
    target = np.ravel(train['label'])
    train = np.asarray(train.drop(['label','clickTime','userID'], axis=1))
    test = pd.read_csv(r'E:\competition\tenxun\data\test_process.csv', low_memory=False) 
    test = test.drop(['clickTime','userID','label'], axis=1)
    submission = pd.DataFrame(test['instanceID'])
    test = np.asarray(test.drop('instanceID', axis=1))

#    rus = under_sampling.RandomUnderSampler(ratio=0.1,random_state=2017)
#    X_res, y_res = rus.fit_sample(train, target)
    
    train_roc_auc_score = []
    test_roc_auc_score = []
    
    train_self_logloss = []
    test_self_logloss = []
    
    train_log_loss = []
    test_log_loss = []
    
    sss = StratifiedShuffleSplit(target, 3, test_size=0.25, random_state=2017)
    for train_index, test_index in sss:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        clf.fit(X_train, y_train)
        
        train_pre = clf.predict(X_train)
        test_pre = clf.predict(X_test)

        train_proba = clf.predict_proba(X_train)[:,1]
        test_proba = clf.predict_proba(X_test)[:,1]
        
        
        print('train roc_auc_score：', metrics.roc_auc_score(y_train, train_proba))
        print('test roc_auc_score：', metrics.roc_auc_score(y_test, test_proba))
        
        train_roc_auc_score.append(metrics.roc_auc_score(y_train, train_proba))
        test_roc_auc_score.append(metrics.roc_auc_score(y_test, test_proba))
        
        print('train accuracy_score', metrics.accuracy_score(y_train, train_pre))
        print('test accuracy_score', metrics.accuracy_score(y_test, test_pre))
        
        print('train recall_score', metrics.recall_score(y_train, train_pre))
        print('test recall_score', metrics.recall_score(y_test, test_pre))
        
        print('train f1_score', metrics.f1_score(y_train, train_pre))
        print('test f1_score', metrics.f1_score(y_test, test_pre))
        
        print('train self_logloss：', self_logloss(y_train, train_proba))
        print('test self_logloss：', self_logloss(y_test, test_proba))
        
        train_self_logloss.append(self_logloss(y_train, train_proba))
        test_self_logloss.append(self_logloss(y_test, test_proba))
        
        print('train logloss：', metrics.log_loss(y_train, train_proba))
        print('test logloss：', metrics.log_loss(y_test, test_proba))
    
        train_log_loss.append(metrics.log_loss(y_train, train_proba))
        test_log_loss.append(metrics.log_loss(y_test, test_proba))
        
        
    proba = clf.predict_proba(test)[:,1]
    print(Counter(np.asarray(proba > 0.5)))
    
    print("mean train_roc_auc_score", np.mean(train_roc_auc_score))
    print("mean test_roc_auc_score", np.mean(test_roc_auc_score))
    
    print("mean train_self_logloss", np.mean(train_self_logloss))
    print("mean test_self_logloss", np.mean(test_self_logloss))
    
    print("mean train_log_loss", np.mean(train_log_loss))
    print("mean test_log_loss", np.mean(test_log_loss))
    
    submission['prob'] = proba
    submission.to_csv(r'E:\competition\tenxun\submit\submission.csv', index=False)
        
    
print('=============================开始==================================')    
clf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=100,
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


print('=============================数据处理==================================')   
process_train_test_data()
print('=============================模型训练==================================')  
model1_test(clf)    
    
    
#mean train_roc_auc_score 0.771633103562
#mean test_roc_auc_score 0.767735964201    

#mean train_self_logloss 0.10339918835
#mean test_self_logloss 0.103792903675  
    
#实际：0.10518    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





































