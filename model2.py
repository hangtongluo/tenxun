# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:41:46 2017

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
from make_features import original_features


def process_train_test_data():
    data = pd.read_csv(r'F:\data\tenxun\data\original_features.csv', low_memory=False) #原始特征数据
#    data = pd.read_csv(r'F:\data\tenxun\data\original_features_addnew.csv', low_memory=False) 
#    data = data.drop(['marriageStatus','haveBaby','education','hometown','residence','telecomsOperator'], axis=1)  
#    data = data.fillna(0)     
#    data['user_app_count'] = data['user_app_actions_count'] + data['user_installedapps_features']
    
    train = data[data['instanceID'] == -1]
    test = data[data['label'] == -1]     
    
    train.to_csv(r'E:\competition\tenxun\data\train_process.csv', index=False)
    test.to_csv(r'E:\competition\tenxun\data\test_process.csv', index=False)
    print(train.head())           
    print(test.head())   
    
    
    return train, test

def model1_test(clf):
    train = pd.read_csv(r'E:\competition\tenxun\data\train_process.csv', low_memory=False)
    
    train_data = train[(train['clickTime'] >= 200000) & (train['clickTime'] < 280000)]    #20-27训练（20/21/22/23/24/25/26/27）
    test_data = train[(train['clickTime'] >= 280000) & (train['clickTime'] < 290000)]    #28测试                  
          
    train_target = np.ravel(train_data['label'])
    test_target = np.ravel(test_data['label'])
    train_data = np.asarray(train_data.drop(['instanceID','clickTime','userID','label'], axis=1))
    test_data = np.asarray(test_data.drop(['instanceID','clickTime','userID','label'], axis=1))
    
    test_train_data = train[(train['clickTime'] >= 230000) & (train['clickTime'] < 310000)]  #23-30训练 （23/24/25/26/27/28/29/30）  
    test_train_target = np.ravel(test_train_data['label'])
    test_train_data = np.asarray(test_train_data.drop(['instanceID','clickTime','userID','label'], axis=1))
    
    
    test = pd.read_csv(r'E:\competition\tenxun\data\test_process.csv', low_memory=False)
    test = test.drop(['clickTime','userID','label'], axis=1)
    submission = pd.DataFrame(test['instanceID'])
    test = np.asarray(test.drop('instanceID', axis=1))
    
    
    '''==========================开始训练================================'''
    clf.fit(train_data, train_target)
        
    train_pre = clf.predict(train_data)
    test_pre = clf.predict(test_data)

    train_proba = clf.predict_proba(train_data)[:,1]
    test_proba = clf.predict_proba(test_data)[:,1]
    
    
    print('train roc_auc_score：', metrics.roc_auc_score(train_target, train_proba))
    print('test roc_auc_score：', metrics.roc_auc_score(test_target, test_proba))
    
    print('train accuracy_score', metrics.accuracy_score(train_target, train_pre))
    print('test accuracy_score', metrics.accuracy_score(test_target, test_pre))
    
    print('train recall_score', metrics.recall_score(train_target, train_pre))
    print('test recall_score', metrics.recall_score(test_target, test_pre))
    
    print('train f1_score', metrics.f1_score(train_target, train_pre))
    print('test f1_score', metrics.f1_score(test_target, test_pre))
    
    print('train self_logloss：', self_logloss(train_target, train_proba))
    print('test self_logloss：', self_logloss(test_target, test_proba))
    
    print('train logloss：', metrics.log_loss(train_target, train_proba))
    print('test logloss：', metrics.log_loss(test_target, test_proba))
    
    
    
    clf.fit(test_train_data, test_train_target)   
    proba = clf.predict_proba(test)[:,1]
    print(Counter(np.asarray(proba > 0.5)))

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


print('=============================数据处理==================================')   
original_features(add_features=False) #对原始特征进行组合
#original_features(add_features=True) #对原始特征进行组合
process_train_test_data()
print('=============================模型训练==================================')  
model1_test(clf)    
    
    
#train roc_auc_score： 0.788709332715(原始特征)
#test roc_auc_score： 0.781861826126
#train self_logloss： 0.10254778779
#test self_logloss： 0.107019142605


#train roc_auc_score： 0.79316870942（增加特征，昨天旭旭说的）
#test roc_auc_score： 0.788066842296
#train self_logloss： 0.102027176241
#test self_logloss： 0.106099900499

#train roc_auc_score： 0.788857460541（增加特征，用户今天的点击统计）
#test roc_auc_score： 0.782005584454
#train self_logloss： 0.102535308859
#test self_logloss： 0.106975484862


















































