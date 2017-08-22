# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:24:28 2017

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from evaluation import self_logloss

#定义模型训练评价体系
def model_estimate_tree(clf, X, y, feature_name=None, printFeatureImportance=False, performCV=True, cv_folds=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2017)
    clf.fit(X_train, y_train)
    
#    y_pred_train = clf.predict(X_train)
#    y_pred_test = clf.predict(X_test)
    #计算ROC时需要
    y_proba_train = clf.predict_proba(X_train)[:,1]
    y_proba_test = clf.predict_proba(X_test)[:,1]
    
    print(clf.__class__.__name__+"对训练集的预测log_loss得分：%.5f" % log_loss(y_train, y_proba_train))
    print(clf.__class__.__name__+"对测试集的预测log_loss得分：{0:.5f}".format(log_loss(y_test, y_proba_test))) 
    
    print(clf.__class__.__name__+"对训练集的预测self_logloss得分：%.5f" % self_logloss(y_train, y_proba_train))
    print(clf.__class__.__name__+"对测试集的预测self_logloss得分：{0:.5f}".format(self_logloss(y_test, y_proba_test))) 
    
    print(clf.__class__.__name__+"对训练集的预测roc_auc_score得分：%.5f" % roc_auc_score(y_train, y_proba_train))
    print(clf.__class__.__name__+"对测试集的预测roc_auc_score得分：{0:.5f}".format(roc_auc_score(y_test, y_proba_test))) 
    
    '''scoring value.['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro','roc_auc'
                    'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 
                    'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
                    'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted']'''
    if performCV:
        cv_score = cross_val_score(clf, X, y, cv=cv_folds, scoring='roc_auc')
        print(clf.__class__.__name__+"交叉验证得分 : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % \
                                    (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    #输出特征的重要性
    if printFeatureImportance:
        feature_importances = pd.Series(clf.feature_importances_, feature_name).sort_values(ascending=True)
        feature_importances.plot(kind='barh', title='Feature Importances')
        plt.xlabel('Feature Importance Score')
        

    return feature_importances


def model_KBest():
    pass


def select_user_data(clf):
    train = pd.read_csv(r'F:\data\tenxun\data\data_process.csv', low_memory=False, nrows=3749528, usecols=['label','userID'])
#    test = pd.read_csv(r'F:\data\tenxun\data\data_process.csv', low_memory=False, skiprows=3749529, header=None)
#    test.columns = train.columns
    user = pd.read_csv(r'F:\data\tenxun\data\user.csv', low_memory=False)
    print(user.columns)
    train = pd.merge(train, user, on='userID', how='left')
    
#    target = train['label']
#    train = train.drop(['instanceID','label','userID','clickTime','creativeID'], axis=1)
#    
#    feature_importances = model_estimate_tree(clf, np.asarray(train), np.ravel(target), feature_name=train.columns, printFeatureImportance=False, performCV=True, cv_folds=3)
#    
#    feature_importances.to_csv('F:\data\tenxun\data\feature_importances_position.csv', index=False)
#    return feature_importances
    
    

def select_position_data(clf):
#    train = pd.read_csv(r'F:\data\tenxun\data\data_process.csv', low_memory=False, nrows=3749528)
#    position = pd.read_csv(r'F:\data\tenxun\data\position_data.csv', low_memory=False)
#    train = pd.merge(train, position, on=['userID','positionID'], how='left')
#    
#    target = train['label']
#    train = train.drop(['instanceID','label','userID','clickTime','creativeID'], axis=1)

    train = pd.read_csv(r'E:\competition\tenxun\data\train_process.csv', low_memory=False)
    train = train[(train['clickTime']>= 200000)] #提升

    
    target = train['label']
    train = train.drop(['instanceID','clickTime','userID','label'], axis=1)
    
    feature_importances = model_estimate_tree(clf, np.asarray(train), np.ravel(target), feature_name=train.columns, printFeatureImportance=True, performCV=True, cv_folds=3)
    
    feature_importances.to_csv(r'F:\data\tenxun\data\feature_importances_position.csv', index=False)
    return feature_importances


if __name__ == '__main__':

    RF = RandomForestClassifier(n_estimators=50, max_depth=5, max_features=0.8, n_jobs=-1, random_state=2017)
#    select_user_data()
#    select_user_data(RF)
    select_position_data(RF)






















































