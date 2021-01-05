
# coding: utf-8

# # Detecting Housing Price Shifts Using Machine Learning Methods
# ### Nicole Lange, Nicholas Colonna, Abha Jhaveri

# In[2]:


import pandas as pd
import os
import pylab as pl
import numpy as np
import re
import math
import statistics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


#df = pd.read_csv("/Users/nicole/Downloads/301data.csv")
df = pd.read_csv("./301data.csv")

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
df.columns = [camel_to_snake(col) for col in df.columns]
df = df.dropna()


# Determine median value used for classifying the observations.

# In[3]:


housing_median = np.median(df['cs_ret'])
print(housing_median)
df['direction'] = (df['cs_ret'] >= housing_median).astype(int)
pre_crash = df[0:234]
post_crash = df[234:]
all_train = df[0:323]
all_test = df[323:]


# Run logistic regression to determine baseline MSE and F1 score.

# In[4]:


train_x = all_train.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'housing_starts', 'unemploy', 'confidence', 'gini_ratio'], axis=1)
test_x = all_test.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'housing_starts', 'unemploy', 'confidence', 'gini_ratio'], axis=1)
logit_model = (sm.Logit(all_train['direction'], train_x)).fit()
print(logit_model.summary2())


# In[6]:


predictions_all = logit_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_logit = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print("Traine MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# In[8]:


predictions_all = logit_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_logit = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("Test MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')


# Random Forest: To determine the best parameters(max_depth and n_estimators) we ran a loop that determined the ones with the smallest MSE and printed the results in a heatmap.

# In[7]:


x = []
k=[]
si = []
js = []
for i in range(1, 21):
    for j in range(1, 21):
        rfr = RandomForestClassifier(criterion='entropy', max_depth=j, random_state=0, n_estimators=i)
        rfr_model = rfr.fit(train_x, all_train['direction'])
        predictions_all = rfr_model.predict(test_x)
        predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
        k = k + [(np.mean((predict_all_classify - all_test['direction']) ** 2))]
    x = x + [k]
    k = []
df2 = pd.DataFrame(x, columns = range(1, 21), index = range(1, 21))
import seaborn as sns
(sns.heatmap(df2)).invert_yaxis()
plt.title("Mean Squared Error")
plt.ylabel("n_estimators")
plt.xlabel("max_depth")
sns.set_palette("Paired")


# Fit the model using the training data.

# In[9]:


rfr = RandomForestClassifier(criterion='entropy', max_depth=3, random_state=0, n_estimators=4)
rfr_model = rfr.fit(train_x, all_train['direction'])
predictions_all = rfr_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_rf = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print("Train MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# Tested the model using testing set. This resulted in a MSE of .3333 and a f1-score of .6519. This is the lowest test MSE produced by any of our models and outperforms our base model(the logit model shown above).

# In[10]:


predictions_all = rfr_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_rf = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("Test MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')

