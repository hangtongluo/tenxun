# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:31:21 2017

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
#    data = pd.read_csv(r'F:\data\tenxun\data\original_features.csv', low_memory=False) #原始特征数据
    data = pd.read_csv(r'F:\data\tenxun\data\original_features_addnew.csv', low_memory=False) 
#    data = data.drop(['marriageStatus','haveBaby','education','hometown','residence','telecomsOperator'], axis=1)  
#    data = data.fillna(0)     
    
    train = data[data['instanceID'] == -1]
    test = data[data['label'] == -1]     
    
    train.to_csv(r'E:\competition\tenxun\data\train_process.csv', index=False)
    test.to_csv(r'E:\competition\tenxun\data\test_process.csv', index=False)
    print(train.head())           
    print(test.head())   
    return train, test

def model1_test(clf):
    train = pd.read_csv(r'E:\competition\tenxun\data\train_process.csv', low_memory=False)
#    train = train[(train['clickTime']<180000) | (train['clickTime']>= 200000)] #提升
    train = train[(train['clickTime']>= 200000)] #提升
    target = np.ravel(train['label'])
    train = np.asarray(train.drop(['instanceID','userID','clickTime','label'], axis=1)) #
    test = pd.read_csv(r'E:\competition\tenxun\data\test_process.csv', low_memory=False)
    test = test.drop(['clickTime','userID','label'], axis=1) #
    submission = pd.DataFrame(test['instanceID'])
    test = np.asarray(test.drop('instanceID', axis=1))

#    rus = under_sampling.RandomUnderSampler(ratio=0.1,random_state=2017)
#    X_res, y_res = rus.fit_sample(train, target)
    
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


print('=============================数据处理==================================')   
#original_features(add_features=False) #对原始特征进行组合
original_features(add_features=True) #对原始特征进行组合
process_train_test_data()
print('=============================模型训练==================================')  
model1_test(clf)    
    
    
#mean train_roc_auc_score 0.771633103562
#mean test_roc_auc_score 0.767735964201    
#mean train_self_logloss 0.10339918835
#mean test_self_logloss 0.103792903675  
    
#实际：0.10518    

#mean train_roc_auc_score 0.768477515018（去掉17、18、19数据）
#mean test_roc_auc_score 0.763510188529
#mean train_self_logloss 0.102810375943
#mean test_self_logloss 0.103173862493

#实际：0.104871    

#mean train_roc_auc_score 0.767530925013 （去掉17、18、19数据）(对user的age进行离散化，对省份和城市处理)
#mean test_roc_auc_score 0.762851742981
#mean train_self_logloss 0.102888154187
#mean test_self_logloss 0.103229104945

#实际：0.104983 

#mean train_roc_auc_score 0.790560565908 (增加n_estimators=300，去掉17、18、19数据)
#mean test_roc_auc_score 0.778169111379
#mean train_self_logloss 0.100440222989
#mean test_self_logloss 0.10164723687

#实际：0.103325

#mean train_roc_auc_score 0.788222787969 (增加n_estimators=300，去掉无用特征)
#mean test_roc_auc_score 0.779355142924
#mean train_self_logloss 0.100704385033
#mean test_self_logloss 0.101546978987



#mean train_roc_auc_score 0.796224937092 (增加n_estimators=300，去掉无用特征, 增加app统计数据)
#mean test_roc_auc_score 0.785902303781
#mean train_self_logloss 0.0996426296104
#mean test_self_logloss 0.100733323418
#
#
#实际：0.103282   #基本线上和线下相差0.0025

#mean train_roc_auc_score 0.798410443174 (增加n_estimators=300, 增加app统计数据（徐徐给的）)
#mean test_roc_auc_score 0.785784743037
#mean train_self_logloss 0.0994267636495
#mean test_self_logloss 0.100762336919
#
#实际：0.103386   #基本线上和线下相差0.0025


#mean train_roc_auc_score 0.790414449372
#mean test_roc_auc_score 0.779086928233
#mean train_self_logloss 0.100465477026
#mean test_self_logloss 0.101587328972



#mean train_roc_auc_score 0.794069472514
#mean test_roc_auc_score 0.782192308005
#mean train_self_logloss 0.100004754093
#mean test_self_logloss 0.101219762471





#基本线上和线下相差0.0017


#train roc_auc_score： 0.794279465214(去掉29/30天)
#test roc_auc_score： 0.785051889701
#train self_logloss： 0.102996220171
#test self_logloss： 0.104028658248


#train roc_auc_score： 0.793923703142(去掉30天)
#test roc_auc_score： 0.782981639561
#train self_logloss： 0.10281797934
#test self_logloss： 0.103951870801


#mean train_roc_auc_score 0.798827002094(全数据，增加用户当天点击次数)（不能复现了?）
#mean test_roc_auc_score 0.789517538936
#mean train_self_logloss 0.100457803314
#mean test_self_logloss 0.101512784771

#实际：0.102779   #基本线上和线下相差0.0013


#mean train_roc_auc_score 0.793400966714（不知为啥，）
#mean test_roc_auc_score 0.783619075949
#mean train_self_logloss 0.101024662545
#mean test_self_logloss 0.10209932295









#mean train_roc_auc_score 0.797697212142
#mean test_roc_auc_score 0.786223549001
#mean train_self_logloss 0.100610271551
#mean test_self_logloss 0.101877608256

#实际：0.108



#全数据、增加用户当天点击次数、增加增加两个app统计数据
#mean train_roc_auc_score 0.800951654836
#mean test_roc_auc_score 0.790981772121
#mean train_self_logloss 0.0999812471232
#mean test_self_logloss 0.101126903706
#mean train_log_loss 0.0999812471232
#mean test_log_loss 0.101126903706


#全数据、增加用户当天点击次数、增加增加两个app统计数据、user_data_process、app_categories_process











####################################################################################################################
#mean train_roc_auc_score 0.7936987025
#mean test_roc_auc_score 0.783506231513
#mean train_self_logloss 0.100994545569
#mean test_self_logloss 0.102091328157
#mean train_log_loss 0.100994545569
#mean test_log_loss 0.102091328157


#mean train_roc_auc_score 0.793387702359 # 增加user_today_clickNum()
#mean test_roc_auc_score 0.783669362503
#mean train_self_logloss 0.101032056809
#mean test_self_logloss 0.102080702433
#mean train_log_loss 0.101032056809
#mean test_log_loss 0.102080702433



#mean train_roc_auc_score 0.793485912401 # 增加user_today_clickNum() userID
#mean test_roc_auc_score 0.782735855187
#mean train_self_logloss 0.101029245477
#mean test_self_logloss 0.102181715214
#mean train_log_loss 0.101029245477
#mean test_log_loss 0.102181715214



























