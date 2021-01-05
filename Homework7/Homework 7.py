
# coding: utf-8

# # Homework 7
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System."

# In[2]:


import pandas as pd
import numpy as np
import pylab as pl
import statsmodels as sm
from sklearn import tree, metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
import math
import statistics as stat
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

df = pd.read_csv("./USMacroG.csv")
df = df.dropna()
df.head()


# (a) Create a binary variable that takes on a 1 when GDP is equal or above the median, and a 0 otherwise.
# 
# First, I calculated the median of the gdp variable. Next, I created a new variable 'binGDP' that takes values as specified above.

# In[3]:


gdp_median = np.median(df['gdp'])
df['binGDP'] = (df['gdp'] >= gdp_median).astype(int)


# (b) Fit a support vector classifier to the data with various values of C (penalty parameter of the error term)  in order to predict whether a GDP is high or low. Report the cross-validation errors associated with different values of this parameter. Comment on your results.
# 
# I ran the test on C values from 3^-7 to 3^7 to get a wide range of possible values. I ran the SVC as linear and utilized the cross validation accuracy to score to calculate the error. As you can see, by varying the penalty parameter C from 3^-7 all the way to 3^7, the CV errors remained mostly the same. There may be extremely small variations, but overall the error rate stayed the same.

# In[12]:


X = df.drop(['gdp'], axis=1)
X = df.drop(['binGDP'], axis=1)
errors_cross_val = []
exp = [float(i) for i in np.arange(-7, 8, 1)]
C_vals = [3**i for i in exp]

for c in C_vals:
    model = SVC(C=c, kernel='linear')
    errors_cross_val.append(np.mean(1-cross_val_score(model, X, df['binGDP'], cv=10)))

results = {'C':C_vals, 'CV Error':errors_cross_val}  
results_table = pd.DataFrame(results)
print(results_table)


# (c) Now repeat (b), this time using SVMs with a radial basis kernel, with different values of C and gamma. Comment on your results.
# 
# I ran the test with the same C values as part (b), however, I also utilized 6 possible values of gamma with a nice spread between them. The SVC was also ran as rbf for radial basis kernel. As you can see from the results, there wasn't that much variation in the error terms. However, there are some variations at each level of C and gamma. The minimum error is 0.25 when C=1 and gamma=0.0001

# In[27]:


errors_cross_val2 = []
exp = [float(i) for i in np.arange(-7, 8, 1)]
C_vals = [3**i for i in exp]
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

for c in C_vals:
    for g in gammas:
        model = SVC(C=c, kernel='rbf', gamma=g)
        errors_cross_val2.append(np.mean(1-cross_val_score(model, X, df['binGDP'], cv=10)))

def chunks(l, n):
    #Makes n-sized chunks from a list.
    for i in range(0, len(l), n):
        yield l[i:i + n]        

splits = list(chunks(errors_cross_val2, 6))
split1 = errors_cross_val2[0:15]
split2 = errors_cross_val2[15:30]
split3 = errors_cross_val2[30:45]
split4 = errors_cross_val2[45:60]
split5 = errors_cross_val2[60:75]
split6 = errors_cross_val2[75:90]

results2 = {'Gamma / C':gammas, '3^-7':splits[0], '3^-6':splits[1], '3^-5':splits[2], '3^-4':splits[3], '3^-3':splits[4], '3^-2':splits[5], '3^-1':splits[6], '3^0':splits[7], '3^1':splits[8], '3^2':splits[9], '3^3':splits[10], '3^4':splits[11], '3^5':splits[12], '3^6':splits[13], '3^7':splits[14]}  
results2_table = pd.DataFrame(data=results2)
print('                           Cross Validation Errors:')
print(results2_table)

print()
print('Minimum CV Error: ', min(errors_cross_val2), " at C=", C_vals[errors_cross_val2.index(min(errors_cross_val2)) // 6], 
      ' and Gamma=', gammas[errors_cross_val2.index(min(errors_cross_val2)) % 6], sep='')

