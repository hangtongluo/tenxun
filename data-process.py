# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:38:27 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

'''==================================数据的基本处理====================================='''
#组合训练数据和测试数据
def train_test_data_combin():
    train = pd.read_csv(r'F:\data\tenxun\pre\train.csv', low_memory=False) 
    train = train.drop('conversionTime', axis=1)
    train['instanceID'] = -1
    print(len(train.userID.unique())) #2595627
    print(train.shape)
#    print(train.columns)
    test = pd.read_csv(r'F:\data\tenxun\pre\test.csv', low_memory=False) 
#    print(test.columns)    
    print(len(test.userID.unique())) #297466
    print(test.shape)
    data = pd.concat([train, test])
    data = data[['instanceID','label','userID','clickTime','positionID','connectionType','telecomsOperator','creativeID']] 
    print(len(data.userID.unique())) #2805118
    print(data.shape)
    data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)
    
#    2595627 + 297466 - 2805118 = 87975 说明有87975用户重复出现
    return data

#训练数据基本处理
def data_process():
    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False) 
    print(len(data.connectionType.unique()))
    print(len(data.telecomsOperator.unique()))  
    print(data.columns)
    print(data.shape)    
    data = pd.get_dummies(data, columns=['connectionType','telecomsOperator'])
    
    data.to_csv(r'F:\data\tenxun\data\data_process.csv', index=False)
    return data


#处理展示位置数据
def position_data_process():
    position = pd.read_csv(r'F:\data\tenxun\pre\position.csv', low_memory=False) 
#    print(position.columns)
    print(position.shape)
    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False, usecols=['userID','positionID']) 
    print(data.columns)
    print(data.shape)

    position_data = pd.merge(data, position, on='positionID', how='left')
#    position_data = pd.get_dummies(position_data, columns=['positionID','sitesetID','positionType'])
    position_data = pd.get_dummies(position_data, columns=['sitesetID','positionType'])
#    print(position_data.columns)
    print(position_data.shape)
    print(position_data.head())
    
    position_data.to_csv(r'F:\data\tenxun\data\position_process.csv', index=False)
    
    return position_data
    
#处理用户基本信息数据
def user_data_process():    
    user = pd.read_csv(r'F:\data\tenxun\pre\user.csv', low_memory=False)
    #切分省份和城市
    user['hometown_province'] = np.ravel(user['hometown'] / 100).astype(int)
    user['hometown_city'] =  np.ravel(user['hometown'] % 100)  
    user['residence_province'] = np.ravel(user['residence'] / 100).astype(int)
    user['residence_city'] = np.ravel(user['residence'] % 100)    
    
    age_map_values = {'(-1, 0]':0,
                 '(0, 10]':1,
                 '(10, 20]':2,
                 '(20, 30]':3,
                 '(30, 40]':4,
                 '(40, 50]':5,
                 '(50, 60]':6,
                 '(60, 70]':7,
                 '(70, 80]':8,}
    
    user['age_discretize'] = pd.cut(user['age'], bins=[-1,0,10,20,30,40,50,60,70,80]).map(age_map_values)
    
#    print(len(user.gender.unique())) #3
#    print(len(user.education.unique())) #8
#    print(len(user.marriageStatus.unique())) #4
#    print(len(user.haveBaby.unique())) #7
#    print(len(user.hometown_province.unique())) #35
#    print(len(user.hometown_city.unique())) # 22
#    print(len(user.residence_province.unique())) #35
#    print(len(user.residence_city.unique())) #22
    
    '''      
#    temp1 = pd.DataFrame(user['hometown'].unique())
#    temp1.columns = ['hometown']
#    print(temp1.head()) 
#    print(temp1.shape) #365
#    temp2 = pd.DataFrame(user['residence'].unique())
#    temp2.columns = ['hometown']
#    print(temp2.head())
#    print(temp2.shape) #400
#    print(pd.merge(temp1, temp2, on='hometown', how='inner').shape) #362
    hometown和residence中有362相同的values
    '''
#    temp1 = pd.DataFrame(user['hometown'].unique())
#    temp1.columns = ['hometown']
#    print(temp1.head()) 
#    print(temp1.shape) #365
#    temp2 = pd.DataFrame(user['residence'].unique())
#    temp2.columns = ['hometown']
#    print(temp2.head())
#    print(temp2.shape) #400
#    print(pd.merge(temp1, temp2, on='hometown', how='inner').shape) #362    
    
    #是否居住地没变过
    user['same_town'] = user['hometown'] - user['residence']
    user['same_town'] = user['same_town'].apply(lambda x: 1 if x==0 else 0)    
    
    #是否居住省份没变过
    user['same_province'] = user['hometown_province'] - user['residence_province']
    user['same_province'] = user['same_province'].apply(lambda x: 1 if x==0 else 0)   
    
#==============================================================================
#     #onthot编码
#     cols = ['gender','education','marriageStatus','haveBaby','hometown_province',\
#             'residence_province','hometown_city','residence_city','age_discretize',\
#             'same_town','same_province']
#     user = pd.get_dummies(user, columns=cols)
#==============================================================================
    cols = ['age','hometown','residence']
    user = user.drop(cols, axis=1)
    print(user.head())
    
    user.to_csv(r'F:\data\tenxun\data\user_process.csv', index=False)
    
    return user      
    

def app_categories_process():
    app_categories = pd.read_csv(r'F:\data\tenxun\pre\app_categories.csv', low_memory=False)
    print(app_categories.shape)
    
    #一级目录
    app_categories['appCategory_one_level'] = app_categories['appCategory'].map(lambda x: x if x < 10 else int(x/100))
    #二级目录
    app_categories['appCategory_two_level'] = app_categories['appCategory'].map(lambda x: 0 if x < 10 else int(x%100))
    
    app_categories.to_csv(r'F:\data\tenxun\data\app_categories_process.csv', index=False)
    return app_categories



#用户历史的APP安装文件处理
def user_installedapps_process():
    user_installedapps = pd.read_csv(r'F:\data\tenxun\pre\user_installedapps.csv', low_memory=False)
    print(user_installedapps.shape)
    
    app_categories = pd.read_csv(r'F:\data\tenxun\data\app_categories_process.csv', low_memory=False)
    print(app_categories.shape)
    
    user_installedapps = pd.merge(user_installedapps, app_categories, on='appID', how='left')
    print(user_installedapps.shape)
    
    #用户安装app的数量
    features = user_installedapps[['userID','appID']].groupby('userID').count().reset_index()
    features.columns = ['userID','app_count']
    
    
    #用户安装每类APP的数量(细分)
    feature_temp = user_installedapps[['userID','appCategory','appID']].groupby(['userID','appCategory']).count().reset_index().pivot(index='userID', columns='appCategory', values='appID')              
    
    feature_temp.columns = ['appCategory_'+str(x)  for x in feature_temp.columns]
    feature_temp = feature_temp.reset_index()
    
    features = pd.merge(features, feature_temp, on='userID', how='left')
    features = features.fillna(0)
    
    #计算用户安装APP类别的占比
    for col in features.columns[2:]:
        features[col] = features[col] / features['app_count']
    
        
    #用户安装每类APP的数量(粗分——一级目录)
    feature_temp = user_installedapps[['userID','appCategory_one_level','appID']].groupby(['userID','appCategory_one_level']).count().reset_index().pivot(index='userID', columns='appCategory_one_level', values='appID')              
    
    feature_temp.columns = ['appCategory_one_level_'+str(x)  for x in feature_temp.columns]
    feature_temp = feature_temp.reset_index()
    
    features = pd.merge(features, feature_temp, on='userID', how='left')
    features = features.fillna(0)
    
    #计算用户安装APP类别的占比
    for col in features.columns[2:]:
        features[col] = features[col] / features['app_count']     
        

    
    features.to_csv(r'F:\data\tenxun\data\user_installedapps_process.csv', index=False)
    return features



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(重要)
#用户APP近期流水基本统计信息
def user_app_actions_process():
#    train[['creativeID','clickTime']、user_app_actions、app_categories
    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False)
    print(user_app_actions.shape)
    print(user_app_actions.head())
    
    
           
           
    pass
    
    


#广告基本信息
def ad_process():
    ad = pd.read_csv(r'F:\data\tenxun\pre\ad.csv', low_memory=False)
    
    #暂时不动
    
    ad.to_csv(r'F:\data\tenxun\data\ad_process.csv', index=False)
    return ad
    
    
    
    
    
    
    
    
    
    



if __name__ == '__main__':
#    train_test_data_combin()   
#    data_process() 
#    position_data_process()
#    user_data_process()
#    app_categories_process()
#    user_installedapps_process()
#    ad_process()
    user_app_actions_process()
































