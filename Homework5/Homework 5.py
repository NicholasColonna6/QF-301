
# coding: utf-8

# # Homework 5
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System."

# In[93]:


import pandas as pd
import numpy as np
import pylab as pl
import statsmodels as sm
from sklearn import tree, metrics
import matplotlib.pyplot as plt
import math
import statistics as stat
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn import metrics


df = pd.read_csv("./OJ.csv")
df = df.drop(['Unnamed: 0'], axis=1)
OJ = df
OJ['Store7'] = (OJ['Store7'] == "Yes").astype(int)
OJ.head()


# (a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
# 
# First, I generated a random sample of 800 numbers within the range of total observations for the training set. I then filtered out all indexes not in training set to get the test set. From there, I created the appropriate data frames for the train and test sets.

# In[94]:


total_observations = OJ['Purchase'].count()
indexes = list(range(0, total_observations))
train = random.sample(list(range(0,total_observations)), 800)
test = filter(lambda x: x not in train, indexes)

OJ_train = pd.DataFrame([OJ.iloc[i] for i in train])
OJ_test = pd.DataFrame([OJ.iloc[i] for i in test])


# (b) Fit a tree to the training data, with Purchase as the response and the other variables as predictors. Produce summary statistics about the tree, and describe the results obtained. What is the training error rate? How many terminal nodes does the tree have?
# 
# As you can see below, we have a very small training error rate for the decision tree, which means it does a very good job at predicting our test set. The tree has over 300 nodes and a large amount of terminal nodes.

# In[95]:


OJ_predictors = OJ_train.columns
OJ_predictors = np.array(OJ_predictors[1:])
X = OJ_train[OJ_predictors]
OJ_tree = tree.DecisionTreeClassifier()
OJ_tree.fit(X, OJ_train['Purchase'])

print('Number of Nodes: ', OJ_tree.tree_.node_count, sep='')

y_train_pred = OJ_tree.predict(OJ_train.drop(['Purchase'], axis=1))
print('Training Error Rate:', 1-metrics.accuracy_score(OJ_train['Purchase'], y_train_pred))
print()
conf_train = pd.DataFrame(confusion_matrix(OJ_train['Purchase'], y_train_pred))
conf_train = conf_train.rename(index={0:'Actual CH', 1:'Actual MM'}, columns={0:'Predicted CH', 1:'Predicted MM'})
print(conf_train)
print()
print(classification_report(OJ_train['Purchase'], y_train_pred, digits=4))


# (c) Type in the name of the tree object in order to get a detailed text output. Pick one of the terminal nodes, and interpret the information displayed.
# 
# The bottom nodes of a decision tree are called the terminal nodes, or leaves, of the tree. For each of these terminal nodes, there is a final decision made on how to classify our data. From this node, you can trace back up the path of the tree to see which decisions were made to achieve the classification of our terminal node. I was unable to get a readable test output of the tree and terminal nodes, so I cannot interpret a specific leaf.

# In[110]:


OJ_fit = OJ_tree.fit(X, OJ_train['Purchase'])
#print(tree.export_graphviz(OJ_fit, out_file=None))


# (d) Create a plot of the tree, and interpret the results.
# 
# As you can see from the plot, the there are many decisions being made in the tree. The far left branch appears to be the shortest, making a decision after just 3 splits to the left. This tree is very large and can be dificult to interpret and trace decsion paths.

# In[97]:


OJ_fit = OJ_tree.fit(X, OJ_train['Purchase'])
OJ_graphviz = tree.export_graphviz(OJ_fit, out_file = None)

pydot_graph = pydotplus.graph_from_dot_data(OJ_graphviz)

Image(pydot_graph.create_png())


# (e) Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels. What is the test error rate?
# 
# After predicting with the test data, I got a test error rate around 0.24. You can see from the confusion matrix that there is a little over 30 misclassifications for both CH and MM.

# In[99]:


y_test_pred = OJ_tree.predict(OJ_test.drop(['Purchase'], axis=1))
print('Test Error Rate:', 1-metrics.accuracy_score(OJ_test['Purchase'], y_test_pred))
print()
conf_test = pd.DataFrame(confusion_matrix(OJ_test['Purchase'], y_test_pred))
conf_test = conf_test.rename(index={0:'Actual CH', 1:'Actual MM'}, columns={0:'Predicted CH', 1:'Predicted MM'})
print(conf_test)
print()
print(classification_report(OJ_test['Purchase'], y_test_pred, digits=4))


# (f) Run a cross-validation experiment with the training set in order to determine the optimal tree size.
# 
# After running a validation test and searching for the minimized error rate, I have found 4 to be our optimal tree size because it corresponds to the lowest cross-validated classification error rate, which was 0.176.

# In[103]:


x_test = OJ_test[OJ_predictors]
errors_train = []
errors_test = []
errors_cross_val = []
depths = range(1, 21)

for md in depths:
    model = tree.DecisionTreeClassifier(max_depth=md)
    model.fit(X, OJ_train['Purchase'])
    
    errors_train.append(1 - metrics.accuracy_score(model.predict(X), OJ_train['Purchase']))
    errors_test.append(1 - metrics.accuracy_score(model.predict(x_test), OJ_test['Purchase']))
    errors_cross_val.append(np.mean(1-cross_val_score(model, X, OJ_train['Purchase'])))
    
minCV = errors_cross_val[0]
minCV_ind = 0
for i in range(0, len(errors_cross_val)):
    if(errors_cross_val[i] < minCV):
        minCV = errors_cross_val[i]
        minCV_ind = i
print('Minimum Cross Validation Classification Error:', minCV)
print('Optimal Tree Size:', 1+minCV_ind)


# (g) Produce a plot with tree size on the x-axis and cross-validated classification error rate on the y-axis.
# 
# Utilizing pyplot, I plotted the training, test, and cross-validated classifcation error rates. As you can see, cross-validated error rate is minimized when tree size is 4.

# In[105]:


plt.figure(figsize=[10,7])
plt.plot(depths, errors_cross_val, label="Cross Validation")
plt.plot(depths, errors_train, label="Train")
plt.plot(depths, errors_test, label="Test")
plt.title("Performance on train and test data")
plt.xlabel("Max depth")
plt.ylabel("Error Rate")
plt.ylim([0, 0.4])
plt.xlim([1,20])
plt.xticks(np.arange(1, 20, 1))
plt.legend()
plt.show()


# (h) Which tree size corresponds to the lowest cross-validated classification error rate?
# 
# As seen in the cross validation and plot, we choose 4 to be our optimal tree size because it corresponds to the lowest cross-validated classification error rate.

# (i) Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation. If cross-validation does not lead to selection of a pruned tree, then create a pruned tree with five terminal nodes.
# 
# Since my cross-validation led me to an optimal tree size of 4, I produced a pruned decision tree with a max depth of 4.

# In[107]:


pruned_tree = tree.DecisionTreeClassifier(max_depth = 1+minCV_ind)
pruned_tree.fit(X, OJ_train['Purchase'])

y_train_prune_pred = pruned_tree.predict(OJ_train.drop(['Purchase'], axis=1))
print('Pruned Training Error:', 1 - metrics.accuracy_score(OJ_train['Purchase'], y_train_prune_pred))

y_test_prune_pred = pruned_tree.predict(OJ_test.drop(['Purchase'], axis=1))
print('Pruned Test Error:', 1 - metrics.accuracy_score(OJ_test['Purchase'], y_test_prune_pred))


# (j) Compare the training and test error rates between the pruned and unpruned trees. Which is higher?
# 
# As you can see from the results below, by pruning the tree, we increased the training error significantly. However, our test error decreased by a significant amount too, which means the pruned model did a better job at predicting our test set.
# 
# Therefore, the unpruned tree has higher test error and the pruned tree has higher training error.

# In[109]:


print('Unpruned Training Error Rate:', 1-metrics.accuracy_score(OJ_train['Purchase'], y_train_pred))
print('Unpruned Test Error Rate:', 1-metrics.accuracy_score(OJ_test['Purchase'], y_test_pred))
print()


print('Pruned Training Error:', 1 - metrics.accuracy_score(OJ_train['Purchase'], y_train_prune_pred))
print('Pruned Test Error:', 1 - metrics.accuracy_score(OJ_test['Purchase'], y_test_prune_pred))

