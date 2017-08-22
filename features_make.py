# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:40:25 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

#组合训练数据和测试数据
def train_test_data_combin():
    train = pd.read_csv(r'F:\data\tenxun\pre\train.csv', low_memory=False) 
    train = train.drop('conversionTime', axis=1)   
    train['instanceID'] = -1
    test = pd.read_csv(r'F:\data\tenxun\pre\test.csv', low_memory=False) 
    data = pd.concat([train, test])
    data = data[['instanceID','label','userID','clickTime','positionID','connectionType','telecomsOperator','creativeID']] 
    data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)

    return data


'''==============================用户类特征======================================='''
def user_today_clickNum(data): #用户当天的点击量
    print('=================user_today_clickNum==========================')                   
    data['clickTime_day'] = data['clickTime'].apply(lambda x: int(x/10000))  
    features = data.groupby(['clickTime_day','userID'])['clickTime'].count().reset_index()
    features.columns = ['clickTime_day','userID','clickNum']
    features['clickNum'] = features['clickNum'].apply(lambda x:1 if x<10 else 0)
    data = pd.merge(data, features, on=['clickTime_day','userID'], how='left')
    data = data.drop(['clickTime_day'], axis=1)
    
#    data.to_csv(r'F:\data\tenxun\data\data.csv', index=False)
    data.to_csv(r'F:\data\tenxun\data\original_features_user_today_clickNum.csv', index=False)
    return data

def user_before_clickNum(data):#用户之前的点击量（三天）
    #用户前三天内的点击量(某一类商品)
    print('=================user_before_clickNum==========================')                   
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
    data.to_csv(r'F:\data\tenxun\data\original_features_user_today_user_before_clickNum.csv', index=False)
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
    
    merge_feature.to_csv(r'F:\data\tenxun\data\original_features_user_clickNum_fisrt_is1.csv', index=False)

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

    merge_feature.to_csv(r'F:\data\tenxun\data\original_features_user_install_app_not_return.csv', index=False)
    return merge_feature


def original_features(add_features=False):
    data = pd.read_csv(r'F:\data\tenxun\data\data.csv', low_memory=False)
    user = pd.read_csv(r'F:\data\tenxun\pre\user.csv', low_memory=False)
    ad = pd.read_csv(r'F:\data\tenxun\pre\ad.csv', low_memory=False)
    app_categories = pd.read_csv(r'F:\data\tenxun\pre\app_categories.csv', low_memory=False)
    position = pd.read_csv(r'F:\data\tenxun\pre\position.csv', low_memory=False)
    
    merge_feature = pd.merge(data, user, on='userID', how='left')
    merge_feature = pd.merge(merge_feature, position, on='positionID', how='left')
    merge_feature = pd.merge(merge_feature, ad, on='creativeID', how='left')
    merge_feature = pd.merge(merge_feature, app_categories, on='appID', how='left')   

    merge_feature.to_csv(r'F:\data\tenxun\data\original_features.csv', index=False) #原始特征数据 
                               
    
    return merge_feature




'''==============================广告类特征======================================='''











'''==============================APP类特征======================================='''











'''==============================ID类统计特征======================================='''
def id_cvr(merge_feature):
    merge_feature_train = merge_feature[merge_feature['label'] != -1]
    
    #计算历史的APP转化率
    app_Cvr = merge_feature_train.groupby('appID').apply(lambda df: np.mean(df["label"])).reset_index()
    app_Cvr.columns = ['appID', "app_Cvr"]
    merge_feature = pd.merge(merge_feature, app_Cvr, how="left", on='appID')
    
    #计算历史的sitesetID转化率
    sitesetID_Cvr = merge_feature_train.groupby('sitesetID').apply(lambda df: np.mean(df["label"])).reset_index()
    sitesetID_Cvr.columns = ['sitesetID', "sitesetID_Cvr"]
    merge_feature = pd.merge(merge_feature, sitesetID_Cvr, how="left", on='sitesetID')
    
    #计算历史的appCategory转化率
    appCategory_Cvr = merge_feature_train.groupby('appCategory').apply(lambda df: np.mean(df["label"])).reset_index()
    appCategory_Cvr.columns = ['appCategory', "appCategory_Cvr"]
    merge_feature = pd.merge(merge_feature, appCategory_Cvr, how="left", on='appCategory')
    
    #计算历史的positionID转化率
    positionID_Cvr = merge_feature_train.groupby('positionID').apply(lambda df: np.mean(df["label"])).reset_index()
    positionID_Cvr.columns = ['positionID', "positionID_Cvr"]
    merge_feature = pd.merge(merge_feature, positionID_Cvr, how="left", on='positionID')
    
    #计算历史的connectionType转化率
    connectionType_Cvr = merge_feature_train.groupby('connectionType').apply(lambda df: np.mean(df["label"])).reset_index()
    connectionType_Cvr.columns = ['connectionType', "connectionType_Cvr"]
    merge_feature = pd.merge(merge_feature, connectionType_Cvr, how="left", on='connectionType')
    
    #计算历史的positionType转化率
    positionType_Cvr = merge_feature_train.groupby('positionType').apply(lambda df: np.mean(df["label"])).reset_index()
    positionType_Cvr.columns = ['positionType', "positionType_Cvr"]
    merge_feature = pd.merge(merge_feature, positionType_Cvr, how="left", on='positionType')
    
    #计算历史的advertiserID转化率
    advertiserID_Cvr = merge_feature_train.groupby('advertiserID').apply(lambda df: np.mean(df["label"])).reset_index()
    advertiserID_Cvr.columns = ['advertiserID', "advertiserID_Cvr"]
    merge_feature = pd.merge(merge_feature, advertiserID_Cvr, how="left", on='advertiserID')
    
    #计算历史的adID转化率
    adID_Cvr = merge_feature_train.groupby('adID').apply(lambda df: np.mean(df["label"])).reset_index()
    adID_Cvr.columns = ['adID', "adID_Cvr"]
    merge_feature = pd.merge(merge_feature, adID_Cvr, how="left", on='adID')

    #计算历史的creativeID转化率
    creativeID_Cvr = merge_feature_train.groupby('creativeID').apply(lambda df: np.mean(df["label"])).reset_index()
    creativeID_Cvr.columns = ['creativeID', "creativeID_Cvr"]
    merge_feature = pd.merge(merge_feature, creativeID_Cvr, how="left", on='creativeID')
    
    #计算历史的camgaignID转化率
    camgaignID_Cvr = merge_feature_train.groupby('camgaignID').apply(lambda df: np.mean(df["label"])).reset_index()
    camgaignID_Cvr.columns = ['camgaignID', "camgaignID_Cvr"]
    merge_feature = pd.merge(merge_feature, camgaignID_Cvr, how="left", on='camgaignID')
    
    #计算历史的gender转化率
    gender_Cvr = merge_feature_train.groupby('gender').apply(lambda df: np.mean(df["label"])).reset_index()
    gender_Cvr.columns = ['gender', "gender_Cvr"]
    merge_feature = pd.merge(merge_feature, gender_Cvr, how="left", on='gender')
    
    #计算历史的education转化率
    education_Cvr = merge_feature_train.groupby('education').apply(lambda df: np.mean(df["label"])).reset_index()
    education_Cvr.columns = ['education', "education_Cvr"]
    merge_feature = pd.merge(merge_feature, education_Cvr, how="left", on='education')
    
    merge_feature.to_csv(r'F:\data\tenxun\data\original_features_id_cvr.csv', index=False) #原始特征数据 
    
    return merge_feature



'''==============================ID类特征热度统计======================================='''
def id_hotValues(merge_feature):
#    dayNum = len(merge_feature['clickTime'].apply(lambda x: int(x/10000)).unique())
    #APP的平均热度（每一天）
    appID_hot = merge_feature.groupby('appID').apply(lambda df:len(df['userID'].unique())).reset_index()
    appID_hot.columns = ['appID','appID_hot']
    merge_feature = pd.merge(merge_feature, appID_hot, how="left", on='appID')
    
    #appCategory的平均热度（每一天）
    appCategory_hot = merge_feature.groupby('appCategory').apply(lambda df:len(df['userID'].unique())).reset_index()
    appCategory_hot.columns = ['appCategory','appCategory_hot']
    merge_feature = pd.merge(merge_feature, appCategory_hot, how="left", on='appCategory')
    
    #sitesetID的平均热度（每一天）
    sitesetID_hot = merge_feature.groupby('sitesetID').apply(lambda df:len(df['userID'].unique())).reset_index()
    sitesetID_hot.columns = ['sitesetID','sitesetID_hot']
    merge_feature = pd.merge(merge_feature, sitesetID_hot, how="left", on='sitesetID')
    
    #positionID的平均热度（每一天）
    positionID_hot = merge_feature.groupby('positionID').apply(lambda df:len(df['userID'].unique())).reset_index()
    positionID_hot.columns = ['positionID','positionID_hot']
    merge_feature = pd.merge(merge_feature, positionID_hot, how="left", on='positionID')

    #connectionType的平均热度（每一天）
    connectionType_hot = merge_feature.groupby('connectionType').apply(lambda df:len(df['userID'].unique())).reset_index()
    connectionType_hot.columns = ['connectionType','connectionType_hot']
    merge_feature = pd.merge(merge_feature, connectionType_hot, how="left", on='connectionType')

    #positionType的平均热度（每一天）
    positionType_hot = merge_feature.groupby('positionType').apply(lambda df:len(df['userID'].unique())).reset_index()
    positionType_hot.columns = ['positionType','positionType_hot']
    merge_feature = pd.merge(merge_feature, positionType_hot, how="left", on='positionType')

    #advertiserID的平均热度（每一天）
    advertiserID_hot = merge_feature.groupby('advertiserID').apply(lambda df:len(df['userID'].unique())).reset_index()
    advertiserID_hot.columns = ['advertiserID','advertiserID_hot']
    merge_feature = pd.merge(merge_feature, advertiserID_hot, how="left", on='advertiserID')

    #adID的平均热度（每一天）
    adID_hot = merge_feature.groupby('adID').apply(lambda df:len(df['userID'].unique())).reset_index()
    adID_hot.columns = ['adID','adID_hot']
    merge_feature = pd.merge(merge_feature, adID_hot, how="left", on='adID')

    #creativeID的平均热度（每一天）
    creativeID_hot = merge_feature.groupby('creativeID').apply(lambda df:len(df['userID'].unique())).reset_index()
    creativeID_hot.columns = ['creativeID','creativeID_hot']
    merge_feature = pd.merge(merge_feature, creativeID_hot, how="left", on='creativeID')

    #camgaignID的平均热度（每一天）
    camgaignID_hot = merge_feature.groupby('camgaignID').apply(lambda df:len(df['userID'].unique())).reset_index()
    camgaignID_hot.columns = ['camgaignID','camgaignID_hot']
    merge_feature = pd.merge(merge_feature, camgaignID_hot, how="left", on='camgaignID')

    #gender的平均热度（每一天）
    gender_hot = merge_feature.groupby('gender').apply(lambda df:len(df['userID'].unique())).reset_index()
    gender_hot.columns = ['gender','gender_hot']
    merge_feature = pd.merge(merge_feature, gender_hot, how="left", on='gender')
    
    #education的平均热度（每一天）
    education_hot = merge_feature.groupby('education').apply(lambda df:len(df['userID'].unique())).reset_index()
    education_hot.columns = ['education','education_hot']
    merge_feature = pd.merge(merge_feature, education_hot, how="left", on='education')

    merge_feature.to_csv(r'F:\data\tenxun\data\original_features_id_hotValues.csv', index=False) #原始特征数据 
    
    return merge_feature


def features_poll():
    #组合训练数据和测试数据
    print('=============组合训练数据和测试数据===========')
    train_test_data_combin()
        
    print('============原始ID类特征连接==================')
    #原始ID类特征连接
    merge_feature = original_features()
    
    #用户当天的点击量    
#    print('============用户当天的点击量==================')
#    merge_feature_add = user_today_clickNum(merge_feature)
#
#    print('============用户之前的点击量（三天）============')
#    #用户之前的点击量（三天）
#    merge_feature_add = user_before_clickNum(merge_feature)
#    
#    print('============用户点击量大于两次的设第一次为1=============')
#    #用户点击量大于两次的设第一次为1(可以增加细分到天)
#    merge_feature_add = user_clickNum_fisrt_is1(merge_feature)
#
#    print('============用户安装过得APP则不会安装===============')
#    #用户安装过得APP则不会安装
#    merge_feature_add = user_install_app_not_return1(merge_feature_add)
#    merge_feature_add = user_install_app_not_return2(merge_feature_add)
#    merge_feature_add = user_install_app_not_return(merge_feature_add)
#    
    
    
    
    
    
    
#    
#    print('============id_cvr：ID类转化率统计===============')
#    merge_feature = id_cvr(merge_feature)
#    
#    print('============id_hotValues：ID类热度统计===============')
#    merge_feature = id_hotValues(merge_feature)
#    
    
    
    
    
    merge_feature.to_csv(r'F:\data\tenxun\data\merge_feature_add.csv', index=False) #原始特征数据 

if __name__ == '__main__':
    features_poll()
























