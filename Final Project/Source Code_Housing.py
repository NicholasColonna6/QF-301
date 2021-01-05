
# coding: utf-8

# In[55]:


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


#df = pd.read_csv("/Users/nicole/Downloads/301data.csv")
df = pd.read_csv("./301data.csv")

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
df.columns = [camel_to_snake(col) for col in df.columns]

print(df.columns)
df = df.dropna()


# In[56]:


housing_median = np.median(df['cs_ret'])
print(housing_median)
df['direction'] = (df['cs_ret'] >= housing_median).astype(int)

print(df.count())
print(df.head())


# In[57]:


num_up = 0
num_down = 0
for direction in df['direction']:
    if direction==1:
        num_up+=1
    else:
        num_down+=1
print("Number of Up Months: ",num_up," (",round(100*num_up/(num_up+num_down),2),"%)", sep='')
print("Number of Down Months: ",num_down," (",round(100*num_down/(num_up+num_down),2),"%)", sep='')


# In[58]:


plt.plot(df['date'], df['case_shiller'])
plt.show()

plt.plot(df['date'], df['cs_ret'])
plt.show()


# In[59]:


indep = df.drop(['case_shiller', 'date', 'cs_ret', 'direction'], axis=1)
OLS_model = (sm.OLS(df['case_shiller'], indep)).fit()
print(OLS_model.summary2())


# In[60]:


pre_crash = df[0:234]
post_crash = df[234:]
all_train = df[0:323]
all_test = df[323:]
plt.plot(pre_crash['date'], pre_crash['case_shiller'])
plt.show()


# In[61]:


plt.plot(all_train['date'], all_train['case_shiller'])
plt.show()
plt.plot(all_test['date'], all_test['case_shiller'])
plt.show()


# Pre Crash Logistic Model

# In[62]:


pre_x = pre_crash.drop(['date', 'case_shiller', 'cs_ret', 'direction'], axis=1)
post_x = post_crash.drop(['date', 'case_shiller', 'cs_ret', 'direction'], axis=1)
logit_model_pre = (sm.Logit(pre_crash['direction'], pre_x)).fit()
print(logit_model_pre.summary2())


# In[63]:


pre_x = pre_crash.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'confidence', 'housing_starts', 'cpi', 'income', 'unemploy'], axis=1)
post_x = post_crash.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'confidence', 'housing_starts', 'cpi', 'income', 'unemploy'], axis=1)
logit_model_pre = (sm.Logit(pre_crash['direction'], pre_x)).fit()
print(logit_model_pre.summary2())


# In[64]:


predictions_pre_crash = logit_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_crash]
conf_logit = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[65]:


predictions_post = logit_model_pre.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_logit = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# Post Crash Logistic Model

# In[66]:


logit_model_post = (sm.Logit(post_crash['direction'], post_x)).fit()
print(logit_model_post.summary2())


# In[67]:


predictions_post = logit_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_logit = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[68]:


predictions_pre = logit_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre]
conf_logit = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# All Logistic

# In[69]:


train_x = all_train.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'housing_starts', 'unemploy', 'confidence', 'gini_ratio'], axis=1)
test_x = all_test.drop(['date', 'case_shiller', 'cs_ret', 'direction', 'housing_starts', 'unemploy', 'confidence', 'gini_ratio'], axis=1)
logit_model = (sm.Logit(all_train['direction'], train_x)).fit()
print(logit_model.summary2())


# In[70]:


predictions_all = logit_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_logit = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))


# In[71]:


predictions_all = logit_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_logit = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_logit = conf_logit.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_logit)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))


# LDA Pre Crash

# In[72]:


lda = LDA()
lda_model_pre = lda.fit(pre_x, pre_crash['direction'])
predictions_pre_train = lda_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_train]
conf_lda = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_lda = conf_lda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[73]:


predictions_post_train = lda_model_pre.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_lda = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_lda = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# LDA Post Crash

# In[74]:


lda_model_post = lda.fit(post_x, post_crash['direction'])
predictions_post_train = lda_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_lda = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_lda = conf_lda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[75]:


predictions_pre_train = lda_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_train]
conf_lda = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_lda = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# LDA All

# In[76]:


lda_model = lda.fit(train_x, all_train['direction'])
predictions_all = lda_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_lda = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_lda = conf_lda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# In[77]:


predictions_all = lda_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_lda = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_lda = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_lda)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')


# QDA Pre Crash

# In[78]:


qda = QDA()
qda_model_pre = qda.fit(pre_x, pre_crash['direction'])
predictions_pre_train = qda_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_train]
conf_qda = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_qda = conf_qda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[79]:


predictions_post_train = qda_model_pre.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_qda = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_qda = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# QDA Post Crash

# In[80]:


qda_model_post = qda.fit(post_x, post_crash['direction'])
predictions_post_train = qda_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_qda = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_qda = conf_qda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[81]:


predictions_pre_train = qda_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_train]
conf_qda = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_qda = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# QDA All

# In[82]:


qda_model = qda.fit(train_x, all_train['direction'])
predictions_all = qda_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_qda = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_qda = conf_qda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# In[83]:


predictions_all = qda_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_qda = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_qda = conf_qda.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_qda)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')


# Pre Crash KNN Model

# In[84]:


neigh = KNeighborsClassifier(n_neighbors=2, weights='distance')
neigh_model_pre = neigh.fit(pre_x, pre_crash['direction']) 
predictions_pre = neigh_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre]
conf_neigh = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[85]:


predictions_post = neigh_model_pre.predict(post_x)
post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_neigh = pd.DataFrame(confusion_matrix(post_crash['direction'], post_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(post_crash['direction'], post_classify, digits=4))


# Post Crash KNN Model

# In[86]:


neigh = KNeighborsClassifier(n_neighbors=10)
neigh_model_post = neigh.fit(post_x, post_crash['direction']) 
predictions_post = neigh_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_neigh = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[87]:


predictions_pre = neigh_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre]
conf_neigh = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# KNN All

# In[88]:


neigh = KNeighborsClassifier(n_neighbors=9, weights='distance')
neigh_model = neigh.fit(train_x, all_train['direction']) 
predictions_all = neigh_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_neigh = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))


# In[89]:


predictions_all = neigh_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_neigh = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_neigh = conf_neigh.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_neigh)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))


# Pre Crash SVM Model

# In[90]:


#for c in np.logspace(-2,3, 10):
#    for g in range(1,5):  
#        gamma = g/100000
#        svm = SVC(kernel='rbf', gamma=gamma, C=c)
#        scores = cross_val_score(svm, pre_x, pre_crash['direction'], n_jobs=-1, cv=5)
#        print("RBF SVM with c={} and gamma = {} has test accuracy of {}".format(round(c,4), gamma, round(scores.mean(), 3)))
#best: RBF SVM with c=0.1292 and gamma = 1e-05 has test accuracy of 0.563

svm_pre = SVC(kernel='rbf', gamma=0.1, C=10)
svm_model_pre = svm_pre.fit(pre_x, pre_crash['direction'])
predictions_pre = svm_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre]
conf_svm = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[91]:


predictions_post = svm_model_pre.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_svm = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# Post Crash SVM Model

# In[92]:


#for c in np.logspace(-2,3, 10):
#    for g in range(5,11):  
#        gamma = g
#        svm = SVC(kernel='rbf', gamma=gamma, C=c)
#        scores = cross_val_score(svm, post_x, post_crash['direction'], n_jobs=-1, cv=5)
#        print("RBF SVM with c={} and gamma = {} has test accuracy of {}".format(round(c,4), gamma, round(scores.mean(), 3)))
# RBF SVM with c=0.01 and gamma = 1e-05 has test accuracy of 0.6

svm_post = SVC(kernel='rbf', gamma=1, C=50)
svm_model_post = svm_post.fit(post_x, post_crash['direction'])
predictions_post = svm_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post]
conf_svm = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[93]:


predictions_pre = svm_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre]
conf_svm = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# SVM All

# In[94]:


#for c in np.arange(.08, 1.3, .01):    
#    for g in np.arange(.1, .6, .1):
#        gamma = g/100000
#        svm = SVC(kernel='linear', gamma=gamma, C=c)
#        scores = cross_val_score(svm, train_x, all_train['direction'], n_jobs=-1, cv=5)
#        print("RBF SVM with c={} and gamma = {} has test accuracy of {}".format(round(c,4), gamma, round(scores.mean(), 3)))
# RBF SVM with c=0.1 and gamma = 1e-06 has test accuracy of 0.665

svm_all = SVC(kernel='rbf', gamma=.00000007, C=0.1)
svm_model = svm_all.fit(train_x, all_train['direction'])
predictions_all = svm_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_svm = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print()
print("MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# In[95]:


predictions_all = svm_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_svm = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_svm = conf_svm.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_svm)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')


# Random Forest Pre Model

# In[96]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=6)
rfr_model_pre = rfr.fit(pre_x, pre_crash['direction'])
predictions_pre_train = rfr_model_pre.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_train]
conf_rf = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# In[97]:


predictions_post_train = rfr.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_rf = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# Random Forest Post Model

# In[98]:


rfr = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=3)
rfr_model_post = rfr.fit(post_x, post_crash['direction'])
predictions_post_train = rfr_model_post.predict(post_x)
predict_post_classify = [1 if x>0.5 else 0 for x in predictions_post_train]
conf_rf = pd.DataFrame(confusion_matrix(post_crash['direction'], predict_post_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(post_crash['direction'], predict_post_classify, digits=4))


# In[99]:


predictions_pre_test = rfr_model_post.predict(pre_x)
predict_pre_classify = [1 if x>0.5 else 0 for x in predictions_pre_test]
conf_rf = pd.DataFrame(confusion_matrix(pre_crash['direction'], predict_pre_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(pre_crash['direction'], predict_pre_classify, digits=4))


# Random Forest All Model

# In[100]:


rfr = RandomForestClassifier(criterion='entropy', max_depth=3, random_state=0, n_estimators=4)
rfr_model = rfr.fit(train_x, all_train['direction'])
predictions_all = rfr_model.predict(train_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_rf = pd.DataFrame(confusion_matrix(all_train['direction'], predict_all_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(all_train['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_train['direction']) ** 2), sep='')


# In[101]:


predictions_all = rfr_model.predict(test_x)
predict_all_classify = [1 if x>0.5 else 0 for x in predictions_all]
conf_rf = pd.DataFrame(confusion_matrix(all_test['direction'], predict_all_classify))
conf_rf = conf_rf.rename(index={0:'Actual Down', 1:'Actual Up'}, columns={0:'Predicted Down', 1:'Predicted Up'})
print(conf_rf)
print()
print(classification_report(all_test['direction'], predict_all_classify, digits=4))
print("MSE = ", np.mean((predict_all_classify - all_test['direction']) ** 2), sep='')


# Random Forest Heatmap

# In[102]:


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


# K-fold Cross Validation Comparison

# In[103]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

kf = StratifiedKFold(n_splits=6)
svm_cv = SVC(kernel='rbf', gamma=.00000007, C=0.1)
rf_cv = RandomForestClassifier(criterion='entropy', max_depth=3, random_state=0, n_estimators=4)
svm_scores = cross_val_score(svm_cv, train_x, all_train['direction'], n_jobs=-1, cv=kf, scoring='f1_weighted')
rf_scores = cross_val_score(rf_cv, train_x, all_train['direction'], n_jobs=-1, cv=kf, scoring='f1_weighted')

print("SVM", svm_scores)
print("rf", rf_scores)

plt.plot(range(1,7), svm_scores)
plt.show()

plt.plot(range(1,7), rf_scores)
plt.show()

print(svm_scores.mean(), "  ", rf_scores.mean())


# Optimal Parameters

# In[114]:


from sklearn.metrics import roc_curve, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

clf = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1)

param_grid = {
    'n_estimators' : [3, 4, 5, 6],
    'max_depth': [3, 4, 5, 6],
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}

def grid_search_wrapper(refit_score):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(train_x, all_train['direction'])

    # make the predictions
    y_pred = grid_search.predict(test_x)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(all_test['direction'], y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

grid_search_clf = grid_search_wrapper(refit_score='f1_score')

