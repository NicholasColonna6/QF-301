# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:04:26 2018

@author: gcreamer       
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
import os

# In[37]:
path = "C:/Users/colon/Desktop/Fall 2018/QF 301 Advanced Time Series Analysis/Homework7"
os.chdir(path)
macrov = pd.read_csv("./USMacroG.csv") 
macrov = macrov.dropna()
macrov['y'] = (macrov['gdp'] > macrov['gdp'].median()) * 1
macrov.head()

# In[38]:
    
predictors = macrov.columns.tolist()
predictors.remove('y')
X = macrov[predictors].values
y = macrov['y']
    

#X_new = macrov.iloc[:, 1:7]    

# In[39]:
#logspace(start, stop, num): 10^start .... 10^stop (the exponent changes by (stop-start)/(num-1))  
for c in np.logspace(-4,1, 10):
    clf= SVC(kernel='linear', C=c)
    scores = cross_val_score(clf, X, y, n_jobs=-1, cv=5)
    print("Linear SVM with c={} has test accuracy of {}".format(round(c,4), round(scores.mean(), 3)))


# In[40]:

for c in np.logspace(-2,3, 10):
    for g in range(1,5):  
        gamma = g/100000
        clf = SVC(kernel='rbf', gamma=gamma, C=c)
        scores = cross_val_score(clf, X, y, n_jobs=-1, cv=5)
        print("RBF SVM with c={} and gamma = {} has test accuracy of {}".format(round(c,4), gamma, round(scores.mean(), 3)))


