# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:22:25 2017

@author: Administrator
"""

import scipy as sp

def self_logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll








































