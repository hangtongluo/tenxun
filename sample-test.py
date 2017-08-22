# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:46:04 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib

clf = joblib.load('model1-sample.pkl') 

train = pd.read_csv(r'E:\competition\tenxun\data\train_process.csv', low_memory=False)
train = train[(train['clickTime']< 180000)] 

target = np.ravel(train['label']) 
train = np.asarray(train.drop(['userID','clickTime','label'], axis=1)) #   

#print('proba self_logloss：', self_logloss(pre_target, proba))
























