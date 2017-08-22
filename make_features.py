# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:31:20 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter


#组合训练数据和测试数据
def train_test_data_combin():
    train = pd.read_csv(r'F:\data\tenxun\pre\train.csv', low_memory=False) 
    train = train.drop('conversionTime', axis=1)   #（===========================暂时=======================）
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
    data = data[(data['clickTime'] < 290000) & (data['clickTime'] <= 270000)] #提升
#    data['clickTime_d'] = data['clickTime'].apply(lambda x: int(x/10000))
#    data['clickTime_h'] = data['clickTime'].apply(lambda x: int(x/100%100))
#    data['clickTime_m'] = data['clickTime'].apply(lambda x: int(x%100))
#    
#    app_count = pd.read_csv(r'F:\data\tenxun\data\df_app_count.csv', low_memory=False) 
#    data['app_count'] = app_count['app_count'] 
#    
#    print(len(data.userID.unique())) #2805118
#    print(data.shape)
    
    
    data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)
    
#    2595627 + 297466 - 2805118 = 87975 说明有87975用户重复出现
    return data


def user_data_process(user):
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

    #是否居住地没变过
    user['same_town'] = user['hometown'] - user['residence']
    user['same_town'] = user['same_town'].apply(lambda x: 1 if x==0 else 0)    
    
    #是否居住省份没变过
    user['same_province'] = user['hometown_province'] - user['residence_province']
    user['same_province'] = user['same_province'].apply(lambda x: 1 if x==0 else 0)   
    
    cols = ['age','hometown','residence']
    user = user.drop(cols, axis=1)
    print(user.head())
    
#    user.to_csv(r'F:\data\tenxun\data\user_process.csv', index=False)
    
    return user

def app_categories_process(app_categories):    
    #一级目录
    app_categories['appCategory_one_level'] = app_categories['appCategory'].map(lambda x: x if x < 10 else int(x/100))
    #二级目录
    app_categories['appCategory_two_level'] = app_categories['appCategory'].map(lambda x: 0 if x < 10 else int(x%100))
    
    app_categories = app_categories.drop(['appCategory'], axis=1)
    
#    app_categories.to_csv(r'F:\data\tenxun\data\app_categories_process.csv', index=False)
    return app_categories



def user_app_actions_features():
    '''用户1-30号的APP流水记录'''
    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False, usecols=['userID','appID'])    
    user_app_actions = user_app_actions.groupby('userID').count().reset_index()
    user_app_actions.columns = ['userID','user_app_actions_count']
    
    user_app_actions.to_csv(r'F:\data\tenxun\data\user_app_actions_features.csv', index=False)
    
    return user_app_actions

#def user_app_actions_features_time_before():
#    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False)
#    data = pd.read_csv(r'F:\data\tenxun\data\original_features_addnew.csv', low_memory=False, usecols=['clickTime','userID','appID']) 
#    
##    data['appID_install_time_before_count'] = 
#    temp = []
#    for dataitem in data:
        
    
    



def user_installedapps_features():
    '''用户1号之前的手机APP安装列表'''
    user_installedapps = pd.read_csv(r'F:\data\tenxun\pre\user_installedapps.csv', low_memory=False)  
    user_installedapps = user_installedapps.groupby('userID').count().reset_index()
    user_installedapps.columns = ['userID','user_installedapps_count']
    
    user_installedapps.to_csv(r'F:\data\tenxun\data\user_installedapps_features.csv', index=False)
    
    return user_installedapps




def user_app_information():
    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False, usecols=['userID','appID']) 
    user_installedapps = pd.read_csv(r'F:\data\tenxun\pre\user_installedapps.csv', low_memory=False)   
    user_app_actions = pd.concat([user_app_actions, user_installedapps])
    print(user_app_actions.shape)
    print(user_app_actions.drop_duplicates().shape)
    
    user_installedapps = user_app_actions.groupby('userID').count().reset_index()
    
    print(user_installedapps.shape)
    print(len(user_installedapps.userID.unique()))
    print(len(user_installedapps.appID.unique()))
    print(user_installedapps.shape)
    
    
def user_age_process(merge_feature):
    #对age进行分组填充
    merge_feature['age'] = merge_feature['age'].apply(lambda x: np.nan if x<10 else x)
    fill_mean = lambda g: g.fillna(g.mean())

    merge_feature = merge_feature.groupby(['appCategory','positionType','adID'], as_index=False).apply(fill_mean)
    merge_feature['age'] = merge_feature['age'].values
    merge_feature['age'] = merge_feature['age'].fillna(int(np.mean(merge_feature['age'])))
    merge_feature['age'] = merge_feature['age'].values.astype(int)
    
    return merge_feature    

def user_gender_process(merge_feature):
    #对gender进行分组填充    
    merge_feature['gender'] = merge_feature['gender'].apply(lambda x: np.nan if x==0 else x)
    fill_median = lambda g: g.fillna(g.median())
    merge_feature = merge_feature.groupby(['appCategory','positionType','adID'], as_index=False).apply(fill_median)
    merge_feature['gender'] = merge_feature['gender'].fillna(1) #1性别的多
    merge_feature['gender'] = merge_feature['gender'].values.astype(int)

    print(Counter(merge_feature['gender']))


    return merge_feature 

#def user_gender_process(merge_feature):
#    #对gender进行分组填充    
#    merge_feature['gender'] = merge_feature['gender'].apply(lambda x: np.nan if x==0 else x)
#    fill_median = lambda g: g.fillna(g.median())
#    merge_feature = merge_feature.groupby(['appCategory','positionType','adID'], as_index=False).apply(fill_median)
#    merge_feature['gender'] = merge_feature['gender'].fillna(1) #1性别的多
#    merge_feature['gender'] = merge_feature['gender'].values.astype(int)
#
#    print(Counter(merge_feature['gender']))
#
#
#    return merge_feature 


def cross_features_user_class(merge_feature): #制造交叉特征
    '''Index([u'instanceID', u'label', u'userID', u'clickTime', u'positionID',
   u'connectionType', u'telecomsOperator', u'creativeID', u'clickNum',
   u'age', u'gender', u'education', u'marriageStatus', u'haveBaby',
   u'hometown', u'residence', u'sitesetID', u'positionType', u'adID',
   u'camgaignID', u'advertiserID', u'appID', u'appPlatform',
   u'appCategory', u'fisrt_is1'],
  dtype='object')'''
    '''用户类和其他'''
    #userID 和 其他
    merge_feature['userID_positionID'] = merge_feature['userID'] * merge_feature['positionID']
    merge_feature['userID_connectionType'] = merge_feature['userID'] * merge_feature['connectionType']
    merge_feature['userID_creativeID'] = merge_feature['userID'] * merge_feature['creativeID']
    merge_feature['userID_sitesetID'] = merge_feature['userID'] * merge_feature['sitesetID']
    merge_feature['userID_positionType'] = merge_feature['userID'] * merge_feature['positionType']
    merge_feature['userID_adID'] = merge_feature['userID'] * merge_feature['adID']
    merge_feature['userID_camgaignID'] = merge_feature['userID'] * merge_feature['camgaignID']
    merge_feature['userID_advertiserID'] = merge_feature['userID'] * merge_feature['advertiserID']
    merge_feature['userID_appID'] = merge_feature['userID'] * merge_feature['appID']
    merge_feature['userID_appCategory'] = merge_feature['userID'] * merge_feature['appCategory']
    
    #gender 和 其他
    merge_feature['gender_positionID'] = merge_feature['gender'] * merge_feature['positionID']
    merge_feature['gender_connectionType'] = merge_feature['gender'] * merge_feature['connectionType']
    merge_feature['gender_creativeID'] = merge_feature['gender'] * merge_feature['creativeID']
    merge_feature['gender_sitesetID'] = merge_feature['gender'] * merge_feature['sitesetID']
    merge_feature['gender_positionType'] = merge_feature['gender'] * merge_feature['positionType']
    merge_feature['gender_adID'] = merge_feature['gender'] * merge_feature['adID']
    merge_feature['gender_camgaignID'] = merge_feature['gender'] * merge_feature['camgaignID']
    merge_feature['gender_advertiserID'] = merge_feature['gender'] * merge_feature['advertiserID']
    merge_feature['gender_appID'] = merge_feature['gender'] * merge_feature['appID']
    merge_feature['gender_appCategory'] = merge_feature['gender'] * merge_feature['appCategory']
    
    #education 和 其他
    merge_feature['education_positionID'] = merge_feature['education'] * merge_feature['positionID']
    merge_feature['education_connectionType'] = merge_feature['education'] * merge_feature['connectionType']
    merge_feature['education_creativeID'] = merge_feature['education'] * merge_feature['creativeID']
    merge_feature['education_sitesetID'] = merge_feature['education'] * merge_feature['sitesetID']
    merge_feature['education_positionType'] = merge_feature['education'] * merge_feature['positionType']
    merge_feature['education_adID'] = merge_feature['education'] * merge_feature['adID']
    merge_feature['education_camgaignID'] = merge_feature['education'] * merge_feature['camgaignID']
    merge_feature['education_advertiserID'] = merge_feature['education'] * merge_feature['advertiserID']
    merge_feature['education_appID'] = merge_feature['education'] * merge_feature['appID']
    merge_feature['education_appCategory'] = merge_feature['education'] * merge_feature['appCategory'] 

    #marriageStatus 和 其他
    merge_feature['marriageStatus_positionID'] = merge_feature['marriageStatus'] * merge_feature['positionID']
    merge_feature['marriageStatus_connectionType'] = merge_feature['marriageStatus'] * merge_feature['connectionType']
    merge_feature['marriageStatus_creativeID'] = merge_feature['marriageStatus'] * merge_feature['creativeID']
    merge_feature['marriageStatus_sitesetID'] = merge_feature['marriageStatus'] * merge_feature['sitesetID']
    merge_feature['marriageStatus_positionType'] = merge_feature['marriageStatus'] * merge_feature['positionType']
    merge_feature['marriageStatus_adID'] = merge_feature['marriageStatus'] * merge_feature['adID']
    merge_feature['marriageStatus_camgaignID'] = merge_feature['marriageStatus'] * merge_feature['camgaignID']
    merge_feature['marriageStatus_advertiserID'] = merge_feature['marriageStatus'] * merge_feature['advertiserID']
    merge_feature['marriageStatus_appID'] = merge_feature['marriageStatus'] * merge_feature['appID']
    merge_feature['marriageStatus_appCategory'] = merge_feature['marriageStatus'] * merge_feature['appCategory']     
    
    #marriageStatus 和 其他
    merge_feature['haveBaby_positionID'] = merge_feature['haveBaby'] * merge_feature['positionID']
    merge_feature['haveBaby_connectionType'] = merge_feature['haveBaby'] * merge_feature['connectionType']
    merge_feature['haveBaby_creativeID'] = merge_feature['haveBaby'] * merge_feature['creativeID']
    merge_feature['haveBaby_sitesetID'] = merge_feature['haveBaby'] * merge_feature['sitesetID']
    merge_feature['haveBaby_positionType'] = merge_feature['haveBaby'] * merge_feature['positionType']
    merge_feature['haveBaby_adID'] = merge_feature['haveBaby'] * merge_feature['adID']
    merge_feature['haveBaby_camgaignID'] = merge_feature['haveBaby'] * merge_feature['camgaignID']
    merge_feature['haveBaby_advertiserID'] = merge_feature['haveBaby'] * merge_feature['advertiserID']
    merge_feature['haveBaby_appID'] = merge_feature['haveBaby'] * merge_feature['appID']
    merge_feature['haveBaby_appCategory'] = merge_feature['haveBaby'] * merge_feature['appCategory']   
    
    return merge_feature
        
   


def cross_features_ad_class(merge_feature): #制造交叉特征
    '''Index([u'instanceID', u'label', u'userID', u'clickTime', u'positionID',
   u'connectionType', u'telecomsOperator', u'creativeID', u'clickNum',
   u'age', u'gender', u'education', u'marriageStatus', u'haveBaby',
   u'hometown', u'residence', u'sitesetID', u'positionType', u'adID',
   u'camgaignID', u'advertiserID', u'appID', u'appPlatform',
   u'appCategory', u'fisrt_is1'],
  dtype='object')'''
    
    '''广告类特征和其他'''
    #           positionID 和 其他
    merge_feature['positionID_connectionType'] = merge_feature['positionID'] * merge_feature['connectionType']
    merge_feature['positionID_creativeID'] = merge_feature['positionID'] * merge_feature['creativeID']
    merge_feature['positionID_sitesetID'] = merge_feature['positionID'] * merge_feature['sitesetID']
    merge_feature['positionID_positionType'] = merge_feature['positionID'] * merge_feature['positionType']
    merge_feature['positionID_adID'] = merge_feature['positionID'] * merge_feature['adID']
    merge_feature['positionID_camgaignID'] = merge_feature['positionID'] * merge_feature['camgaignID']
    merge_feature['positionID_advertiserID'] = merge_feature['positionID'] * merge_feature['advertiserID']
    merge_feature['positionID_appID'] = merge_feature['positionID'] * merge_feature['appID']
    merge_feature['positionID_appCategory'] = merge_feature['positionID'] * merge_feature['appCategory']

    #           creativeID 和 其他
    merge_feature['creativeID_sitesetID'] = merge_feature['creativeID'] * merge_feature['sitesetID']
    merge_feature['creativeID_positionType'] = merge_feature['creativeID'] * merge_feature['positionType']
    merge_feature['creativeID_adID'] = merge_feature['creativeID'] * merge_feature['adID']
    merge_feature['creativeID_camgaignID'] = merge_feature['creativeID'] * merge_feature['camgaignID']
    merge_feature['creativeIDadvertiserID'] = merge_feature['creativeID'] * merge_feature['advertiserID']
    merge_feature['creativeID_appID'] = merge_feature['creativeID'] * merge_feature['appID']
    merge_feature['creativeID_appCategory'] = merge_feature['creativeID'] * merge_feature['appCategory']
    
    #           sitesetID 和 其他
    merge_feature['sitesetID_positionType'] = merge_feature['sitesetID'] * merge_feature['positionType']
    merge_feature['sitesetID_adID'] = merge_feature['sitesetID'] * merge_feature['adID']
    merge_feature['sitesetID_camgaignID'] = merge_feature['sitesetID'] * merge_feature['camgaignID']
    merge_feature['sitesetID_advertiserID'] = merge_feature['sitesetID'] * merge_feature['advertiserID']
    merge_feature['sitesetID_appID'] = merge_feature['sitesetID'] * merge_feature['appID']
    merge_feature['sitesetID_appCategory'] = merge_feature['sitesetID'] * merge_feature['appCategory']
    
    #           positionType 和 其他
    merge_feature['positionType_adID'] = merge_feature['positionType'] * merge_feature['adID']
    merge_feature['positionType_camgaignID'] = merge_feature['positionType'] * merge_feature['camgaignID']
    merge_feature['positionType_advertiserID'] = merge_feature['positionType'] * merge_feature['advertiserID']
    merge_feature['positionType_appID'] = merge_feature['positionType'] * merge_feature['appID']
    merge_feature['positionType_appCategory'] = merge_feature['positionType'] * merge_feature['appCategory']
    
    #           adID 和 其他
    merge_feature['adID_camgaignID'] = merge_feature['adID'] * merge_feature['camgaignID']
    merge_feature['adID_advertiserID'] = merge_feature['adID'] * merge_feature['advertiserID']
    merge_feature['adID_appID'] = merge_feature['adID'] * merge_feature['appID']
    merge_feature['adID_appCategory'] = merge_feature['adID'] * merge_feature['appCategory']
    
    #           camgaignID 和 其他
    merge_feature['camgaignID_advertiserID'] = merge_feature['camgaignID'] * merge_feature['advertiserID']
    merge_feature['camgaignID_appID'] = merge_feature['camgaignID'] * merge_feature['appID']
    merge_feature['camgaignID_appCategory'] = merge_feature['camgaignID'] * merge_feature['appCategory']
    
    
     #           advertiserID 和 其他
    merge_feature['advertiserID_advertiserID'] = merge_feature['advertiserID'] * merge_feature['appID']
    merge_feature['advertiserID_appCategory'] = merge_feature['advertiserID'] * merge_feature['appCategory']  
    
     #           appID和 其他
    merge_feature['appID_appCategory'] = merge_feature['appID'] * merge_feature['appCategory']
    
    
    return merge_feature
    

'''===================================================================================================='''

def user_today_clickNum(): #用户当天的点击量
    print('=================user_today_clickNum==========================')                   
    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False)
    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))
    
    features = data.groupby(['clickTime_day','userID'])['clickTime'].count().reset_index()
    features.columns = ['clickTime_day','userID','clickNum']
    features['clickNum'] = features['clickNum'].apply(lambda x:1 if x<10 else 0)
    data = pd.merge(data, features, on=['clickTime_day','userID'], how='left')
    data = data.drop(['clickTime_day'], axis=1)
    
    data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)
    
    return data

def user_before_clickNum(data):#用户之前的点击量（三天）
    #用户前三天内的点击量(某一类商品)
    print('=================user_before_clickNum==========================')                   
#    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False)
    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))
    new_data = pd.DataFrame(columns=data.columns)
    for day in data.clickTime_day.unique():
        look_day = data[data['clickTime_day'] == day]
        look_features = data[(data['clickTime_day'] >= day - 3) & (data['clickTime_day'] < day)]
        look_features = look_features.groupby(['userID','appCategory'], as_index=False)['clickTime'].count()
        look_features.columns = ['userID','appCategory','before_threeDay_clickNum']
        look_features = pd.merge(look_day, look_features, on=['userID','appCategory'], how='left')
        new_data = pd.concat([new_data, look_features])                    
        
    new_data = new_data.fillna(0)
#    new_data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)
    return new_data
    
    
    
    

def user_clickNum_fisrt_is1(merge_feature):#用户点击量大于两次的设第一次为1(可以增加细分到天)
    print('=================user_clickNum_fisrt_is1==========================')                   
    import numpy as np
    merge_feature['my_label1'] = np.zeros(merge_feature.shape[0], dtype=np.int)
    merge_feature['my_label2'] = np.zeros(merge_feature.shape[0], dtype=np.int)
    def df_state1(df):
        temp = np.zeros(df.shape[0], dtype=np.int)
        temp[0] = 1
        df['my_label1'] = temp
        return df
    merge_feature = merge_feature.groupby(['userID','appID'], as_index=False).apply(df_state1)#.sort_values(by=['userID','appID'])#[['userID','clickTime','appID','label','my_label1']]             
    
    def df_state2(df):
        df['my_label2'] = df.shape[0]
        return df
    merge_feature = merge_feature.groupby(['userID'], as_index=False).apply(df_state2)#.sort_values(by=['userID','appID'])#[['userID','clickTime','appID','label','my_label1','my_label2']]          
    
    merge_feature['my_label2'] = merge_feature['my_label2'].apply(lambda x: 0 if x>1 else 1 )
    merge_feature['fisrt_is1'] = merge_feature['my_label1'] - merge_feature['my_label2']                       
    merge_feature = merge_feature.drop(['my_label1','my_label2'], axis=1)
    
    return merge_feature


#def user_appPlatformNum(merge_feature):#统计用户拥有几个类型手机
#    print('=================user_appPlatformNum==========================')   
#    user_appPlatformNum = merge_feature.groupby(['userID','appPlatform'], as_index=False)['appID'].count()
#    user_appPlatformNum = user_appPlatformNum.groupby(['userID'])['appPlatform'].count().reset_index()#.sort_values('appPlatform')
#    user_appPlatformNum.columns = ['userID','appPlatformNum']
##    user_appPlatformNum
#    return user_appPlatformNum
    
#==============================================================================
# def user_before_return_ratio(merge_feature):#用户在同一个appCategory中的转化率
#     print('=================user_before_return_ratio==========================')   
#     merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
#     user_before_return_ratio = []
#     for userid,clicktime,appcategory in zip(merge_feature.userID, merge_feature.clickTime_day, merge_feature.appCategory):
#         
#         #同一用户，在同一appCategory上的以前时间的点击次数
#         temp_before = merge_feature[(merge_feature['userID'] == userid) & (merge_feature['clickTime_day'] < clicktime)]  #这个用户的数据
#         temp_before = temp_before[temp_before['appCategory'] == appcategory]
#         
#         #同一用户，在同一appCategory上的以前时间的转化次数
#         temp_before1 = temp_before[temp_before['label'] == 1]  #这个用户的数据    
#         
#         if temp_before.shape[0] == 0:
#             ratio = 0
#         else:
#             ratio = 1.0 * temp_before1.shape[0] / temp_before.shape[0]
#         user_before_return_ratio.append(ratio)
#     #     print(userid, clicktime, appCategory, label, temp_before.shape[0], temp_before1.shape[0], ratio)
#     merge_feature['user_before_return_ratio'] = user_before_return_ratio 
#     
#     return merge_feature
#==============================================================================

def user_before_return_ratio(merge_feature):#用户在同一个appCategory中的转化率(双表连接)
    merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
    temp_data = merge_feature[['userID','clickTime_day','appID','appCategory','label']]
    temp_merge = pd.merge(temp_data,temp_data, on=['userID'], how='left')
    temp_merge['time'] = temp_merge['clickTime_day_x'] - temp_merge['clickTime_day_y']
    temp_merge['appCate'] = temp_merge['appCategory_x'] - temp_merge['appCategory_y']
    
    def user_before_ret_ratio(df):
        temp1 = df[(df['time'] > 0) & (df['appCate'] == 0)]
        temp2 = temp1[temp1['label_x'] == 1]
        if temp1.shape[0] == 0:
            ratio = 0
        else:
            ratio = temp2.shape[0] / temp1.shape[0]
        df['user_before_ret_ratio'] = ratio
        return df
    ttt = temp_merge.groupby(['userID','appCategory_x','clickTime_day_x']).apply(user_before_ret_ratio).drop_duplicates(['userID','clickTime_x','appID_x'])
    ttt = ttt[['userID','clickTime_day_x','appID_x','user_before_ret_ratio']]
    ttt.columns = ['userID','clickTime_day','appID','user_before_ret_ratio']
    merge_feature = pd.merge(merge_feature, ttt, on=['userID','clickTime_day','appID'], how='left')
    merge_feature = merge_feature.drop('clickTime_day', axis=1)
                   
    return merge_feature


#==============================================================================
# def new_or_old_user(merge_feature):
#     print('=================new_or_old_user==========================')  
#     #对新用户没用，针对老用户
#     merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
#     before_clickNum = []
#     for userid,clickTime_day in zip(merge_feature.userID,merge_feature.clickTime_day):
#         #同一用户，在之前出现的次数(今天之前)
#         temp_before = merge_feature[(merge_feature['userID'] == userid) & (merge_feature['clickTime_day'] < clickTime_day)]                                    
#                                         
#         before_clickNum.append(temp_before.shape[0])
#         
#     #     print(userid, clicktime, appCategory, label, temp_before.shape[0],)
#     merge_feature['before_clickNum'] = before_clickNum   
#     merge_feature['before_clickNum'] = merge_feature['before_clickNum'].apply(lambda x: 1 if x>0 else 0)
#     merge_feature = merge_feature.drop(['clickTime_day'], axis=1)
#     return merge_feature  
#==============================================================================

def new_or_old_user(merge_feature):
    print('=================new_or_old_user==========================')  
    #对新用户没用，针对老用户(看当天用户是不是新用户)
    merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
    temp_merge = pd.merge(merge_feature[['userID','clickTime','label']], merge_feature[['userID','clickTime','label']], on=['userID'], how='left')
    temp_merge['new_or_old_user'] = temp_merge['clickTime_x'] - temp_merge['clickTime_y']
    
    ttt = temp_merge[temp_merge['new_or_old_user'] >=0].groupby(['userID','clickTime_x'], as_index=False).max()
    ttt['new_or_old_user'] = ttt['new_or_old_user'].apply(lambda x: 1 if x>0 else 0)
    ttt = ttt['userID','clickTime_x','new_or_old_user']
    ttt.columns = ['userID','clickTime_day','new_or_old_user']
    merge_feature = pd.merge(merge_feature, ttt, on=['userID','clickTime_day'], how='left')
    merge_feature = merge_feature.drop('clickTime_day', axis=1)
    
    return merge_feature  

        
def user_install_app_not_return1(merge_feature):#用户安装过得APP则不会安装
    print('=================user_install_app_not_return1==========================')
    merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
    user_app_actions = pd.read_csv(r'F:/data/tenxun/pre/user_app_actions.csv', low_memory=False)
    user_app_actions['installTime_day'] = user_app_actions['installTime'].apply(lambda x: int(x/10000))
    temp = pd.merge(merge_feature, user_app_actions, on=['userID','appID'], how='left')
    temp['time'] = temp['installTime_day'] - temp['clickTime_day']
    temp['user_install_app_not_return1'] = temp['time'].apply(lambda x: 0 if x<0 else 1)
    # temp[['userID','appID','installTime_day','clickTime_day','time','label','mylabel']]  
    temp = temp.drop(['installTime_day','clickTime_day','time','installTime'], axis=1)
    
    return temp    
    
def user_install_app_not_return2(merge_feature):#用户安装过得APP则不会安装
    print('=================user_install_app_not_return2==========================')
    merge_feature['clickTime_day'] = merge_feature['clickTime'].apply(lambda x: int(x/10000))
    user_installedapps = pd.read_csv(r'F:/data/tenxun/pre/user_installedapps.csv', low_memory=False)
    user_installedapps['installTime'] = 0
    user_installedapps['installTime_day'] = user_installedapps['installTime'].apply(lambda x: int(x/10000))
    temp = pd.merge(merge_feature, user_installedapps, on=['userID','appID'], how='left')
    temp['time'] = temp['installTime_day'] - temp['clickTime_day']
    temp['user_install_app_not_return2'] = temp['time'].apply(lambda x: 0 if x<0 else 1)
    # temp[['userID','appID','installTime_day','clickTime_day','time','label','mylabel']]  
    temp = temp.drop(['installTime_day','clickTime_day','time','installTime'], axis=1)
    
    return temp  

def user_install_app_not_return(merge_feature):#用户安装过得APP则不会安装
    merge_feature = user_install_app_not_return1(merge_feature)        
    merge_feature = user_install_app_not_return2(merge_feature)  
    merge_feature['user_install_app_not_return'] = merge_feature['user_install_app_not_return1'] + merge_feature['user_install_app_not_return2']                 
    merge_feature['user_install_app_not_return'] = merge_feature['user_install_app_not_return'].apply(lambda x: 1 if x==2 else 0)

    return merge_feature

'''==================================================================================================================='''
def id_cvr(merge_feature):
    #计算历史的APP转化率
    app_Cvr = merge_feature.groupby('appID').apply(lambda df: np.mean(df["label"])).reset_index()
    app_Cvr.columns = ['appID', "app_Cvr"]
    merge_feature = pd.merge(merge_feature, app_Cvr, how="left", on='appID')
    
    #计算历史的appCategory转化率
    appCategory_Cvr = merge_feature.groupby('appCategory').apply(lambda df: np.mean(df["label"])).reset_index()
    appCategory_Cvr.columns = ['appCategory', "appCategory_Cvr"]
    merge_feature = pd.merge(merge_feature, appCategory_Cvr, how="left", on='appCategory')
    
    #计算历史的positionID转化率
    positionID_Cvr = merge_feature.groupby('positionID').apply(lambda df: np.mean(df["label"])).reset_index()
    positionID_Cvr.columns = ['positionID', "positionID_Cvr"]
    merge_feature = pd.merge(merge_feature, positionID_Cvr, how="left", on='positionID')
    
    #计算历史的connectionType转化率
    connectionType_Cvr = merge_feature.groupby('connectionType').apply(lambda df: np.mean(df["label"])).reset_index()
    connectionType_Cvr.columns = ['connectionType', "connectionType_Cvr"]
    merge_feature = pd.merge(merge_feature, connectionType_Cvr, how="left", on='connectionType')
    
    #计算历史的positionType转化率
    positionType_Cvr = merge_feature.groupby('positionType').apply(lambda df: np.mean(df["label"])).reset_index()
    positionType_Cvr.columns = ['positionType', "positionType_Cvr"]
    merge_feature = pd.merge(merge_feature, positionType_Cvr, how="left", on='positionType')
    
    #计算历史的advertiserID转化率
    advertiserID_Cvr = merge_feature.groupby('advertiserID').apply(lambda df: np.mean(df["label"])).reset_index()
    advertiserID_Cvr.columns = ['advertiserID', "advertiserID_Cvr"]
    merge_feature = pd.merge(merge_feature, advertiserID_Cvr, how="left", on='advertiserID')
    
    #计算历史的adID转化率
    adID_Cvr = merge_feature.groupby('adID').apply(lambda df: np.mean(df["label"])).reset_index()
    adID_Cvr.columns = ['adID', "adID_Cvr"]
    merge_feature = pd.merge(merge_feature, adID_Cvr, how="left", on='adID')

    #计算历史的creativeID转化率
    creativeID_Cvr = merge_feature.groupby('creativeID').apply(lambda df: np.mean(df["label"])).reset_index()
    creativeID_Cvr.columns = ['creativeID', "creativeID_Cvr"]
    merge_feature = pd.merge(merge_feature, creativeID_Cvr, how="left", on='creativeID')
    
    #计算历史的camgaignID转化率
    camgaignID_Cvr = merge_feature.groupby('camgaignID').apply(lambda df: np.mean(df["label"])).reset_index()
    camgaignID_Cvr.columns = ['camgaignID', "camgaignID_Cvr"]
    merge_feature = pd.merge(merge_feature, camgaignID_Cvr, how="left", on='camgaignID')
    
    #计算历史的gender转化率
    gender_Cvr = merge_feature.groupby('gender').apply(lambda df: np.mean(df["label"])).reset_index()
    gender_Cvr.columns = ['gender', "gender_Cvr"]
    merge_feature = pd.merge(merge_feature, gender_Cvr, how="left", on='gender')
    
    #计算历史的education转化率
    education_Cvr = merge_feature.groupby('education').apply(lambda df: np.mean(df["label"])).reset_index()
    education_Cvr.columns = ['education', "education_Cvr"]
    merge_feature = pd.merge(merge_feature, education_Cvr, how="left", on='education')
    
    return merge_feature













def user_is_click_not_in_installedapp_is1(data_temp):
    data_temp['clickTime_day'] = data_temp['clickTime'].apply(lambda x: int(x/10000))
    user_app_actions = pd.read_csv(r'F:/data/tenxun/pre/user_app_actions.csv', low_memory=False)
    user_app_actions['installTime_day'] = user_app_actions['installTime'].apply(lambda x: int(x/10000))
    user_app_actions = user_app_actions.drop('installTime', axis=1)
    user_installedapps = pd.read_csv(r'F:/data/tenxun/pre/user_installedapps.csv', low_memory=False)
    user_installedapps['installTime_day'] = 0
    user_installedapps = pd.concat([user_app_actions, user_installedapps], ignore_index=True)
    user_is_click_not_in_installedapp = []
    
    for userid, clickTime_day, appid in zip(data_temp.userID, data_temp.clickTime_day, data_temp.appID):
        if userid not in user_installedapps.userID.unique():
            not_in_click = -1
            user_is_click_not_in_installedapp.append(not_in_click)
        else:    
            temp_user_installedapps = user_installedapps[(user_installedapps['userID'] == userid) & (user_installedapps['installTime_day'] < clickTime_day)]
            not_in_click = temp_user_installedapps[temp_user_installedapps['appID'] == appid].shape[0] #用户访问的APP出现过没有，出现过shape不为0，没有shape为0
            user_is_click_not_in_installedapp.append(not_in_click)

    data_temp['user_is_installedapp_not_in_click'] = user_is_click_not_in_installedapp
#     data_temp['user_is_installedapp_not_in_click'] = data_temp['user_is_installedapp_not_in_click']#.apply(lambda x:1 if x==0 else 0)
    data_temp = data_temp.drop(['clickTime_day'], axis=1)
    data_temp.to_csv(r'F:/data/tenxun/data/user_is_installedapp_not_in_click.csv', index=False)
    return data_temp
    

def original_features(add_features=False):
    train_test_data_combin()
    user_today_clickNum()
#    user_before_clickNum()
    
    
    
#    user_app_actions = pd.read_csv(r'F:\data\tenxun\pre\user_app_actions.csv', low_memory=False)
#    user_installedapps = pd.read_csv(r'F:\data\tenxun\pre\user_installedapps.csv', low_memory=False)
    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False)
    user = pd.read_csv(r'F:\data\tenxun\pre\user.csv', low_memory=False)
    ad = pd.read_csv(r'F:\data\tenxun\pre\ad.csv', low_memory=False)
    app_categories = pd.read_csv(r'F:\data\tenxun\pre\app_categories.csv', low_memory=False)
    position = pd.read_csv(r'F:\data\tenxun\pre\position.csv', low_memory=False)
    
    merge_feature = pd.merge(data, user, on='userID', how='left')
    merge_feature = pd.merge(merge_feature, position, on='positionID', how='left')
    merge_feature = pd.merge(merge_feature, ad, on='creativeID', how='left')
    merge_feature = pd.merge(merge_feature, app_categories, on='appID', how='left')
#    merge_feature.apply(lambda x: sum(pd.isnull(x))) #无缺失值
                       
                        
                        
#    merge_feature.to_csv(r'F:\data\tenxun\data\original_features.csv', index=False) #原始特征数据

    #对数据做基本的加工（age异常值填充）                  
#    merge_feature = user_age_process(merge_feature)
    #对数据做基本的加工（gender异常值填充）
#    merge_feature = user_gender_process(merge_feature)

    
    
                         
                         

    if add_features == True:
        print("======================增加特征========================")
#        merge_feature = user_data_process(merge_feature)
#        merge_feature = app_categories_process(merge_feature)
#        user_installedapps_features() 
#        user_installedapps_feature = pd.read_csv(r'F:\data\tenxun\data\user_installedapps_features.csv', low_memory=False)
#        merge_feature = pd.merge(merge_feature, user_installedapps_feature, on='userID', how='left')
#        merge_feature = merge_feature.fillna(0)   
#        
#        user_app_actions_features()
#        user_app_actions_feature = pd.read_csv(r'F:\data\tenxun\data\user_app_actions_features.csv', low_memory=False)
#        merge_feature = pd.merge(merge_feature, user_app_actions_feature, on='userID', how='left')
#        merge_feature = merge_feature.fillna(0) 
#        
#        merge_feature['user_app_count'] = merge_feature['user_app_actions_count'] + merge_feature['user_installedapps_count']

        merge_feature = user_before_clickNum(merge_feature)#用户之前的点击量（三天）
        
        #用户安装过得APP则不会安装   
#        merge_feature = user_install_app_not_return(merge_feature)        
        
        merge_feature = id_cvr(merge_feature)
        #用户点击量大于两次的设第一次为1（强特征）
#        merge_feature = user_clickNum_fisrt_is1(merge_feature)
        
        #统计用户拥有几个类型手机
#        appPlatformNum = user_appPlatformNum(merge_feature)
#        merge_feature = pd.merge(merge_feature, appPlatformNum, on='userID', how='left')
        
        
        

        merge_feature.to_csv(r'F:\data\tenxun\data\original_features_addnew.csv', index=False)
    else:
        merge_feature.to_csv(r'F:\data\tenxun\data\original_features.csv', index=False) #原始特征数据 
                               
    
    return merge_feature







if __name__ == '__main__':
#    original_features(add_features=False) #对原始特征进行组合
    original_features(add_features=True) #对原始特征进行组合
#    user_app_actions_features()
#    user_installedapps_features()
    
#    user_app_information()
#    train_test_data_combin()
#    user_today_clickNum()
                      










































