# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:13:36 2017

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
from sklearn.externals import joblib

def process_train_test_data():
    data = pd.read_csv(r'F:\data\tenxun\data\df_data_1.csv', low_memory=False) 
#    data = data.drop(['marriageStatus','haveBaby','education','hometown','residence','telecomsOperator'], axis=1)  
#    data = data.fillna(0)     
    cols = ['appID_avg_cvr', 'appID_age', 'appID_gender', \
            'appID_education', 'appID_age_avg_cvr', 'appID_gender_avg_cvr', \
            'appID_education_avg_cvr', 'positionID_avg_cvr']
    data = data.drop(cols, axis=1)
#    data = data[(data['clickTime']>= 200000)]
    
#    print("======增加特征user_before_return_ratio======")
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_before_return_ratio.csv', \
#                       low_memory=False, usecols=['user_before_return_ratio']) 
#    data['user_before_return_ratio'] = temp['user_before_return_ratio']
    
#    print("======增加特征before_threeDay_clickNum======")#还可以
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_before_clickNum.csv', \
#                       low_memory=False, usecols=['before_threeDay_clickNum']) 
#    data['before_threeDay_clickNum'] = temp['before_threeDay_clickNum']
#
#    print("======增加特征user_clickNum_last_is1======")#还可以
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_clickNum_last_is1.csv', low_memory=False, usecols=['last_is1'])  
#    data['last_is1'] = temp['last_is1']
##    
    
#    print("======增加特征user_install_app_not_return======")
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_install_app_not_return.csv', low_memory=False, usecols=['user_install_app_not_return'])  
#    data['user_install_app_not_return'] = temp['user_install_app_not_return']
#    print(data.columns)
    
#    print("======增加特征original_features_user_before_threeDay_cvr======")#效果一般
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_user_before_threeDay_cvr.csv', low_memory=False, usecols=['user_before_threeDay_cvr'])  
#    data['user_before_threeDay_cvr'] = temp['user_before_threeDay_cvr']
#    print(data.columns)
#   
#    print("======增加特征user_in_applist======")#还可以
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_in_applist.csv', low_memory=False, usecols=['user_in_applist'])  
#    print(data.columns)
#    data['user_in_applist'] = temp['user_in_applist']

#    print("======增加特征user_in_applist======")#
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_user_before_clickNum.csv', low_memory=False, usecols=['user_before_clickNum'])  
#    data['user_before_clickNum'] = temp['user_before_clickNum']


#    print("======增加特征user_before_clickNum_three======")#
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_user_before_clickNum_three.csv', low_memory=False, usecols=['user_before_clickNum_three'])  
#    data['user_before_clickNum_three'] = temp['user_before_clickNum_three']


#    print("======增加特征groupbyuserID_last_is1======")#还可以
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_addnew_5-29.csv', low_memory=False, usecols=['groupbyuserID_last_is1'])  
#    data['groupbyuserID_last_is1'] = temp['groupbyuserID_last_is1']

#    print("======增加特征groupbyuserID_fisrt_is1======")#
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_addnew_5-29.csv', low_memory=False, usecols=['groupbyuserID_fisrt_is1'])  
#    data['groupbyuserID_fisrt_is1'] = temp['groupbyuserID_fisrt_is1']

#    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))
#    appID_hot = data.groupby(['clickTime_day','appID']).apply(lambda df:len(df['userID'].unique())).reset_index()
#    appID_hot.columns = ['clickTime_day','appID','appID_hot']
##    print(appID_hot.head())
#    data = pd.merge(data, appID_hot, how="left", on=['clickTime_day','appID'])
#    data = data.drop('clickTime_day', axis=1)
##    print(data.head())



#    print("======增加特征original_features_user_before_clickNum_three======")#
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_user_before_clickNum_three.csv', low_memory=False, usecols=['user_before_clickNum_three'])  
#    data['user_before_clickNum_three'] = temp['user_before_clickNum_three']
    

#    print("======增加特征original_features_user_before_cvr_three======")#
#    temp = pd.read_csv(r'F:\data\tenxun\data\original_features_user_before_cvr_three.csv', low_memory=False, usecols=['user_before_cvr_three'])  
#    data['user_before_cvr_three'] = temp['user_before_cvr_three']
#    
#    cols = ['appCategory','sitesetID','appID','positionType','connectionType',\
#        'advertiserID','positionID','adID', 'creativeID','camgaignID']
#    def one_ID_d_h_count(data, cols, key_target='clickTime_d_h'):
#        data['clickTime_d_h_prev'] = data['clickTime'].apply(lambda x: int(x/100)-1)
#        data['clickTime_d_h'] = data['clickTime'].apply(lambda x: int(x/100))
#        data['clickTime_d_h_next'] = data['clickTime'].apply(lambda x: int(x/100)+1)
#        
#        if key_target == 'clickTime_d_h':
#            for col in cols:
#                data['grp_key'] = data['clickTime_d_h'].astype(str) + data[col].astype(str) 
#                temp = data.groupby('grp_key', as_index=False)[col].count()
#                temp.columns = ['grp_key',col+'_d_h_count_now']
#                data = pd.merge(data, temp, on='grp_key', how='left')
#                data = data.drop('grp_key', axis=1)
#                data = data.fillna(0)
#            
#        if key_target == 'clickTime_d_h_prev':
#            for col in cols:
#                data['grp_key'] = data['clickTime_d_h'].astype(str) + data[col].astype(str) 
#                temp = data.groupby('grp_key', as_index=False)[col].count()
#                temp.columns = ['grp_key',col+'_d_h_count_prev']
#                data = pd.merge(data, temp, on='grp_key', how='left')
#                data = data.drop('grp_key', axis=1)
#                data = data.fillna(0)       
#            
#        if key_target == 'clickTime_d_h_next':
#            for col in cols:
#                data['grp_key'] = data['clickTime_d_h'].astype(str) + data[col].astype(str) 
#                temp = data.groupby('grp_key', as_index=False)[col].count()
#                temp.columns = ['grp_key',col+'_d_h_count_next']
#                data = pd.merge(data, temp, on='grp_key', how='left')
#                data = data.drop('grp_key', axis=1)
#                data = data.fillna(0) 
#            
#        data = data.drop(['clickTime_d_h','clickTime_d_h_prev','clickTime_d_h_next'], axis=1)
#            
#        return data
#    
#    data = one_ID_d_h_count(data, cols, key_target='clickTime_d_h')     
#    data = one_ID_d_h_count(data, cols, key_target='clickTime_d_h_prev')     
#    data = one_ID_d_h_count(data, cols, key_target='clickTime_d_h_next')     
#    print('=====================data over=====================')
        
#    data['positionID_connectionType'] = data['positionID'].astype(str) + data['connectionType'].astype(str) 
#    merge_feature_train = data[data['label'] != -1]
#    #计算历史的positionID_connectionType点击量
#    app_Cvr = merge_feature_train.groupby('positionID_connectionType').apply(lambda df: np.size(df["label"])).reset_index()
#    app_Cvr.columns = ['positionID_connectionType', "positionID_connectionType_count"]
#    data = pd.merge(data, app_Cvr, how="left", on='positionID_connectionType')

    
    
#    data['grp_key'] = data['advertiserID'].astype(str) + data['positionID'].astype(str) 
#    merge_feature_train = data[data['label'] != -1]
#    #计算历史的positionID_advertiserID点击率
#    app_Cvr = merge_feature_train.groupby('grp_key').apply(lambda df: np.size(df["label"])).reset_index()
#    app_Cvr.columns = ['grp_key', "ad_app_Cvr"]
#    data = pd.merge(data, app_Cvr, how="left", on='grp_key')
#    data = data.drop('grp_key', axis=1)
#
#    
##    data['grp_key'] = data['adID'].astype(str) + data['appID'].astype(str) 
#    merge_feature_train = data[data['label'] != -1]
#    #计算历史的positionID转化量
#    app_Cvr = merge_feature_train.groupby('appID').apply(lambda df: np.sum(df["label"])).reset_index()
#    app_Cvr.columns = ['appID', "ad_app_Cvr"]
#    data = pd.merge(data, app_Cvr, how="left", on='appID')
##    data = data.drop('positionID', axis=1)    
#
##    data['grp_key'] = data['adID'].astype(str) + data['appID'].astype(str) 
#    merge_feature_train = data[data['label'] != -1]
#    #计算历史的positionID点击量
#    app_Cvr = merge_feature_train.groupby('appID').apply(lambda df: np.size(df["label"])).reset_index()
#    app_Cvr.columns = ['appID', "ad_app_Cvr"]
#    data = pd.merge(data, app_Cvr, how="left", on='appID')
##    data = data.drop('positionID', axis=1)    
    


#    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False)#, usecols=['userID','appID']) #原始特征数据          
#    user_app_actions['mylabel'] = 1
#    temp = user_app_actions.groupby('userID')['mylabel'].first().reset_index()
#    data = pd.merge(data, temp, on='userID', how='left').fillna(0)


#    temp_count = data.groupby('userID')['label'].count().reset_index()
#    temp_count.columns = ['userID','count']
#    temp_count['count'] = temp_count['count'].apply(lambda x: 0 if x>1 else 1)
#    data = pd.merge(data, temp_count, on='userID', how='left')

#    temp = pd.read_csv(r'F:\data\tenxun\data\near_time_lab.csv', low_memory=False)
#    data['near_time_lab'] = temp['near_time_lab']

#    temp = pd.read_csv(r'F:\data\tenxun\data\df_positionID_creativeID_cnt.csv', low_memory=False)
#    data['positionID_cnt'] = temp['positionID_cnt']
#    data['creativeID_cnt'] = temp['creativeID_cnt']


#    temp = pd.read_csv(r'F:\data\tenxun\data\today_first_is1.csv', low_memory=False)
#    data['today_first_is1'] = temp['today_first_is1']
    
    
#    data['fisrt_is1'] = data['fisrt_is1'] + data['clickNum']
#    data['fisrt_is1'] = data['fisrt_is1'].apply(lambda x: 1 if x>1 else 0)


#    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))
#    appID_hot = data.groupby(['clickTime_day','appID']).apply(lambda df:len(df['userID'].unique())).reset_index()
#    appID_hot.columns = ['clickTime_day','appID','appID_hot']
#    data = pd.merge(data, appID_hot, how="left", on=['clickTime_day','appID'])
#    data = data.drop('clickTime_day', axis=1)
    
#    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))
#    appID_hot = data.groupby(['clickTime_day','positionID']).apply(lambda df:len(df['userID'].unique())).reset_index()
#    appID_hot.columns = ['clickTime_day','positionID','positionID_hot']
#    data = pd.merge(data, appID_hot, how="left", on=['clickTime_day','positionID'])
#    data = data.drop('clickTime_day', axis=1)
    
    
#    temp = pd.read_csv(r'F:\data\tenxun\data\clickNum_positionID.csv', low_memory=False)
#    data['clickNum_positionID'] = temp['clickNum_positionID']    
    
#    temp = pd.read_csv(r'F:\data\tenxun\data\clickNum_adID.csv', low_memory=False)
#    data['clickNum_adID'] = temp['clickNum_adID']   
    

#    temp = pd.read_csv(r'F:\data\tenxun\data\clickNum.csv', low_memory=False)
#    data['clickNum'] = temp['clickNum']       
    
    temp = pd.read_csv(r'F:\data\tenxun\data\user_click_time_flag.csv', low_memory=False)
    data['user_click_time_flag'] = temp['user_click_time_flag']       
#    data['fisrt_is1'] = temp['user_click_time_flag']    
    
#    def user_app_installcount_data():
#        user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False)#, usecols=['userID','appID']) #原始特征数据          
#        data = pd.read_csv(r'F:/data/tenxun/data/data.csv', low_memory=False)#, usecols=['userID','appID']) #原始特征数据     
#        data_temp = pd.merge(data, user_app_actions, on=['userID'], how='inner')
#        data_temp['diff_time'] = data_temp['clickTime'] - data_temp['installTime'] 
#        data_temp = data_temp[data_temp['diff_time'] > 0]
#    
#        #用户在这个时间之前安装了多少个APP
#        user_app_installcount = data_temp.groupby(['userID','clickTime'], as_index=False)['appID'].count()
#        user_app_installcount.columns = ['userID','clickTime','user_app_installcount']
#        
#        data_temp = pd.merge(data, user_app_installcount, on=['userID','clickTime'], how='left')
#        data_temp['user_app_installcount'] = data_temp['user_app_installcount'].fillna(-1)
#        
#        return pd.DataFrame(data_temp['user_app_installcount'])    
#    temp = user_app_installcount_data()
#    data['user_app_installcount'] = temp['user_app_installcount']
    
    
    

#    
#    temp = pd.read_csv(r'F:\data\tenxun\data\df_clickNum_ad_appID.csv', low_memory=False)
#    data['df_clickNum_ad_appID'] = temp
    
    
#    temp = pd.read_csv(r'F:\data\tenxun\data\user_installedapps_num.csv', low_memory=False)
#    data['user_installedapps_num'] = temp   
    
    temp = pd.read_csv(r'F:\data\tenxun\data\user_installedapps_userID_num.csv', low_memory=False)
    data['user_installedapps_userID_num'] = temp  
        
    
    train = data[data['label'] != -1]
    test = data[data['label'] == -1]     
    train.to_csv(r'E:\competition\tenxun\data\train_process.csv', index=False)
    test.to_csv(r'E:\competition\tenxun\data\test_process.csv', index=False)
    print(train.head())           
    print(test.head())   
    return train, test

#process_train_test_data()


def trainModel(clf):
    print('=========================分割数据集================================')
    train, test = process_train_test_data()
#    temp = train[(train['clickTime']< 200000) & (train['clickTime']>= 180000) & (train['label'] == 1)] 
    
    pre = train[(train['clickTime']< 180000)] 
    pre_target = np.ravel(pre['label']) 
    pre_train = np.asarray(pre.drop(['userID','clickTime','label'], axis=1)) # 
    
    train = train[(train['clickTime']>= 200000)] #提升
#    train = train[(train['clickTime']>= 180000)] #提升

    
#    train = pd.concat([temp, train], ignore_index=True)
    
#    train = train[(train['clickTime']< 300000)] #提升
#    target = np.ravel(train['label'])
#    train = np.asarray(train.drop(['userID','clickTime','label'], axis=1)) #
    
                      
    target = np.ravel(train['label']) 
    train = np.asarray(train.drop(['userID','clickTime','label'], axis=1)) #       
    
    
    test = pd.read_csv(r'E:\competition\tenxun\data\test_process.csv', low_memory=False)
    test = test.drop(['clickTime','userID','label'], axis=1) #
    test['instanceID'] = test.index + 1
    submission = pd.DataFrame(test['instanceID'])
    test = np.asarray(test.drop('instanceID', axis=1))

    train_roc_auc_score = []
    test_roc_auc_score = []
    proba_roc_auc_score = []
    
    train_self_logloss = []
    test_self_logloss = []
    proba_self_logloss = []
    

    sss = StratifiedShuffleSplit(target, 1, test_size=0.25, random_state=2017)
    for train_index, test_index in sss:
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, 'model1-sample.pkl') 

        train_proba = clf.predict_proba(X_train)[:,1] 
        test_proba = clf.predict_proba(X_test)[:,1] 
        proba = clf.predict_proba(pre_train)[:,1] 
        
#                
#        tra = pd.DataFrame(train_proba)
#        tra['label'] = y_train
#        tra.to_csv(r'E:\competition\tenxun\submit\train_proba.csv', index=False)
#        tes = pd.DataFrame(test_proba)
#        tes['label'] = y_test
#        tes.to_csv(r'E:\competition\tenxun\submit\test_proba.csv', index=False)
            
        print('train roc_auc_score：', metrics.roc_auc_score(y_train, train_proba))
        print('test roc_auc_score：', metrics.roc_auc_score(y_test, test_proba))
        print('proba roc_auc_score：', metrics.roc_auc_score(pre_target, proba))
        
        train_roc_auc_score.append(metrics.roc_auc_score(y_train, train_proba))
        test_roc_auc_score.append(metrics.roc_auc_score(y_test, test_proba))
        proba_roc_auc_score.append(metrics.roc_auc_score(pre_target, proba))
        
        print('train self_logloss：', self_logloss(y_train, train_proba))
        print('test self_logloss：', self_logloss(y_test, test_proba))
        print('proba self_logloss：', self_logloss(pre_target, proba))
        train_self_logloss.append(self_logloss(y_train, train_proba))
        test_self_logloss.append(self_logloss(y_test, test_proba))
        proba_self_logloss.append(self_logloss(pre_target, proba))
        
        
    proba = clf.predict_proba(test)[:,1]
    print(Counter(np.asarray(proba > 0.5)))
    
    print("mean train_roc_auc_score", np.mean(train_roc_auc_score))
    print("mean test_roc_auc_score", np.mean(test_roc_auc_score))
    print("mean test_roc_auc_score", np.mean(proba_roc_auc_score))
    
    print("mean train_self_logloss", np.mean(train_self_logloss))
    print("mean test_self_logloss", np.mean(test_self_logloss))
    print("mean proba_self_logloss", np.mean(proba_self_logloss))
    
    
    submission['prob'] = proba
    submission.to_csv(r'E:\competition\tenxun\submit\submission.csv', index=False)
    
    
print('=============================开始==================================')    
clf1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=300,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=2,
    scale_pos_weight=1,
    seed=2017)

clf2 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=300,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective= 'binary:logistic',
    nthread=2,
    scale_pos_weight=1,
    seed=1991)

clf3 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=300,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=1,
    colsample_bytree=1,
    objective= 'binary:logistic',
    nthread=2,
    scale_pos_weight=1,
    seed=0)


from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
#eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])
lr = LogisticRegression()
#sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
sclf = StackingClassifier(classifiers=[clf1], meta_classifier=lr, use_probas=True)

#trainModel(clf)
#trainModel(eclf)
trainModel(sclf)



#mean train_roc_auc_score 0.802353189833
#mean test_roc_auc_score 0.79285696693
#mean train_self_logloss 0.098896006086
#mean test_self_logloss 0.0998569910878

#实际：0.1018

#Counter({False: 338437, True: 52})
#mean train_roc_auc_score 0.812655444538#原始结果
#mean test_roc_auc_score 0.803470321709
#mean train_self_logloss 0.0973234999844
#mean test_self_logloss 0.0983930717938

#实际：0.1008大概 （实际：0.1021）

#Counter({False: 338436, True: 53})增加特征user_before_return_ratio
#mean train_roc_auc_score 0.812680093505
#mean test_roc_auc_score 0.802967907546
#mean train_self_logloss 0.0972671682786
#mean test_self_logloss 0.0983778594993

#实际：0.1005

#Counter({False: 338440, True: 49})增加特征before_threeDay_clickNum
#mean train_roc_auc_score 0.814192009099
#mean test_roc_auc_score 0.805310523386
#mean train_self_logloss 0.097137302356
#mean test_self_logloss 0.0981809686746

#实际：0.100265

#Counter({False: 338407, True: 82})增加特征user_clickNum_last_is1
#mean train_roc_auc_score 0.816516214325
#mean test_roc_auc_score 0.80720427225
#mean train_self_logloss 0.0968962620197
#mean test_self_logloss 0.0979762160376

#实际：0.099978


#Counter({False: 338424, True: 65})增加特征user_install_app_not_return
#mean train_roc_auc_score 0.812639171775
#mean test_roc_auc_score 0.803117739522
#mean train_self_logloss 0.0972961482083
#mean test_self_logloss 0.0983755156744

#实际：


#Counter({False: 338464, True: 25})
#mean train_roc_auc_score 0.813156137713
#mean test_roc_auc_score 0.803328209997
#mean train_self_logloss 0.0972763334015
#mean test_self_logloss 0.098365323547



#Counter({False: 338435, True: 54})增加特征user_in_applist
#mean train_roc_auc_score 0.813563650987
#mean test_roc_auc_score 0.804455472391
#mean train_self_logloss 0.0971208151181
#mean test_self_logloss 0.0981297256606
#实际：0.100378



#Counter({False: 338412, True: 77})增加特征original_features_user_before_threeDay_cvr
#mean train_roc_auc_score 0.812927657479
#mean test_roc_auc_score 0.803407462689
#mean train_self_logloss 0.0972947913259
#mean test_self_logloss 0.0983770047616


#Counter({False: 338432, True: 57})增加特征user_before_clickNum_three 过拟合
#mean train_roc_auc_score 0.81414391478
#mean test_roc_auc_score 0.80446169726
#mean train_self_logloss 0.0971530245201
#mean test_self_logloss 0.0982437886441

#实际：0.10645



#Counter({False: 338400, True: 89})增加特征groupbyuserID_last_is1
#mean train_roc_auc_score 0.814942749497
#mean test_roc_auc_score 0.80553154796
#mean train_self_logloss 0.0970516096765
#mean test_self_logloss 0.0981240804636

#实际：0.10004


#Counter({False: 338388, True: 101})增加特征groupbyuserID_fisrt_is1
#mean train_roc_auc_score 0.812388883198
#mean test_roc_auc_score 0.802630144637
#mean train_self_logloss 0.097320816194
#mean test_self_logloss 0.0984413865337

#实际：





#Counter({False: 338372, True: 117})所有的
#mean train_roc_auc_score 0.824732292943
#mean test_roc_auc_score 0.81552035769
#mean train_self_logloss 0.0957055993662
#mean test_self_logloss 0.0969182306067

#实际：0.099309





#Counter({False: 338399, True: 90})original_features_user_before_clickNum_three
#mean train_roc_auc_score 0.813776592235
#mean test_roc_auc_score 0.804192280295
#mean train_self_logloss 0.0971956411725
#mean test_self_logloss 0.0982427555394


#Counter({False: 338443, True: 46})
#mean train_roc_auc_score 0.816767193601
#mean test_roc_auc_score 0.803579680171
#mean train_self_logloss 0.0967913994365
#mean test_self_logloss 0.0983147760495

#实际：0.112（严重过拟合）



#Counter({False: 338424, True: 65})
#mean train_roc_auc_score 0.814329119205
#mean test_roc_auc_score 0.805048597944
#mean train_self_logloss 0.0971481417209
#mean test_self_logloss 0.0981993265713




#===================================================================================================
#Counter({False: 338412, True: 77})计算历史的positionID转化量
#mean train_roc_auc_score 0.816662504539
#mean test_roc_auc_score 0.806958529508
#mean train_self_logloss 0.0969065557558
#mean test_self_logloss 0.0980028144585


#Counter({False: 338429, True: 60})计算历史的positionID点击量
#mean train_roc_auc_score 0.814503365824
#mean test_roc_auc_score 0.804589089347
#mean train_self_logloss 0.0971210532358
#mean test_self_logloss 0.0982585805098



#Counter({False: 338424, True: 65})计算历史的appID转化量
#mean train_roc_auc_score 0.813146167333
#mean test_roc_auc_score 0.803534225333
#mean train_self_logloss 0.0972477871598
#mean test_self_logloss 0.0983352724789


#Counter({False: 338417, True: 72})计算历史的appID点击量
#mean train_roc_auc_score 0.812774937386
#mean test_roc_auc_score 0.80333593298
#mean train_self_logloss 0.0972872559043
#mean test_self_logloss 0.0983719816426




#Counter({False: 338360, True: 129})
#mean train_roc_auc_score 0.815223720448
#mean test_roc_auc_score 0.806067191046
#mean train_self_logloss 0.0969216779907
#mean test_self_logloss 0.097986454791

#实际：0.1022



#Counter({False: 338443, True: 46})
#mean train_roc_auc_score 0.815336712928
#mean test_roc_auc_score 0.804977152218
#mean train_self_logloss 0.0970486344131
#mean test_self_logloss 0.0982451214426
#实际：0.101763

#Counter({False: 338350, True: 139})
#mean train_roc_auc_score 0.827424546318
#mean test_roc_auc_score 0.820310430508
#mean train_self_logloss 0.0952443287952
#mean test_self_logloss 0.0963126091883

#实际：0.103464


#Counter({False: 338458, True: 31})计算历史的positionID_advertiserID点击率
#mean train_roc_auc_score 0.814018901185
#mean test_roc_auc_score 0.803493607816
#mean train_self_logloss 0.0971992901259
#mean test_self_logloss 0.0984061317197


#Counter({False: 338458, True: 31})计算历史的advertiserID_positionID点击率
#mean train_roc_auc_score 0.814394322656
#mean test_roc_auc_score 0.804212860964
#mean train_self_logloss 0.097139855361
#mean test_self_logloss 0.0983034152188




#Counter({False: 338444, True: 45})计算历史的advertiserID_positionID点击量
#mean train_roc_auc_score 0.814513098435
#mean test_roc_auc_score 0.804774303174
#mean train_self_logloss 0.0971180693315
#mean test_self_logloss 0.0982360946223

#Counter({False: 338438, True: 51}) data = data.drop('grp_key', axis=1)
#mean train_roc_auc_score 0.814810945258
#mean test_roc_auc_score 0.804572806505
#mean train_self_logloss 0.0971037761438
#mean test_self_logloss 0.0982539191808


#Counter({False: 338441, True: 48})temp_count
#mean train_roc_auc_score 0.818265349228
#mean test_roc_auc_score 0.808360857856
#mean train_self_logloss 0.0966426336038
#mean test_self_logloss 0.0978353603024

#实际：0.10163







#Counter({False: 338442, True: 47})
#mean train_roc_auc_score 0.815324795699
#mean test_roc_auc_score 0.805908984813
#mean test_roc_auc_score 0.808828840566
#mean train_self_logloss 0.096960575548
#mean test_self_logloss 0.0980523095788
#mean proba_self_logloss 0.100426728805


#Counter({False: 338476, True: 13})
#mean train_roc_auc_score 0.815768503381
#mean test_roc_auc_score 0.806080136061
#mean test_roc_auc_score 0.806276649472
#mean train_self_logloss 0.0915661959077
#mean test_self_logloss 0.0926836910557
#mean proba_self_logloss 0.100920812144



#Counter({False: 338346, True: 143})
#mean train_roc_auc_score 0.821735573529
#mean test_roc_auc_score 0.812922125186
#mean test_roc_auc_score 0.809396135543
#mean train_self_logloss 0.102284726304
#mean test_self_logloss 0.103535909814
#mean proba_self_logloss 0.100922945245




#Counter({False: 338420, True: 69})near_time_lab
#mean train_roc_auc_score 0.817560436065
#mean test_roc_auc_score 0.808287436738
#mean test_roc_auc_score 0.807380100734
#mean train_self_logloss 0.0967225302327
#mean test_self_logloss 0.0978043852852
#mean proba_self_logloss 0.100691305513

#实际：0.10169

#Counter({False: 338413, True: 76})
#mean train_roc_auc_score 0.815405201867
#mean test_roc_auc_score 0.805502513214
#mean test_roc_auc_score 0.807896793758
#mean train_self_logloss 0.096969229606
#mean test_self_logloss 0.0981283672531
#mean proba_self_logloss 0.100540258471

#Counter({False: 338423, True: 66})df_positionID_creativeID_cnt
#mean train_roc_auc_score 0.820138212454
#mean test_roc_auc_score 0.811295950913
#mean test_roc_auc_score 0.810238201907
#mean train_self_logloss 0.097125414318
#mean test_self_logloss 0.0981932758387
#mean proba_self_logloss 0.100297208036

#实际：0.101808


#Counter({False: 338432, True: 57})
#mean train_roc_auc_score 0.814548931955
#mean test_roc_auc_score 0.804533241687
#mean test_roc_auc_score 0.805853257496
#mean train_self_logloss 0.0970623243859
#mean test_self_logloss 0.0981744059184
#mean proba_self_logloss 0.100766943785


#Counter({False: 338412, True: 77})
#mean train_roc_auc_score 0.816330156781
#mean test_roc_auc_score 0.805855081601
#mean test_roc_auc_score 0.804655006936
#mean train_self_logloss 0.0968768825867
#mean test_self_logloss 0.0980463941571
#mean proba_self_logloss 0.1009575113




#Counter({False: 338433, True: 56})clickNum_positionID
#mean train_roc_auc_score 0.812560039343
#mean test_roc_auc_score 0.802826108359
#mean test_roc_auc_score 0.806269063718
#mean train_self_logloss 0.0973316114388
#mean test_self_logloss 0.0984249453605
#mean proba_self_logloss 0.100722291221



#Counter({False: 338441, True: 48})clickNum_adID
#mean train_roc_auc_score 0.81269778862
#mean test_roc_auc_score 0.80310304848
#mean test_roc_auc_score 0.806192805533
#mean train_self_logloss 0.0972854544459
#mean test_self_logloss 0.0983802732737
#mean proba_self_logloss 0.100719551497





#Counter({False: 338387, True: 102})
#mean train_roc_auc_score 0.815970908525
#mean test_roc_auc_score 0.806431160668
#mean test_roc_auc_score 0.808286873048
#mean train_self_logloss 0.0968983127568
#mean test_self_logloss 0.097994653259
#mean proba_self_logloss 0.10044637668


#实际：0.10153




















