
# coding: utf-8

# # Homework 8
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System."

# In[24]:


import pandas as pd
import numpy as np
import pylab as pl
import statsmodels as sm
from sklearn import tree, metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
import math
import statistics as stat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

yields = pd.read_csv("./yields2009.csv")
returns = pd.read_csv("./midCapD.ts.csv")

print(yields.head())
print(returns.head())


# 1) Perform a principal components (PC) analysis with the minimum number of principal components to capture at least 95% of the variability.
# 
# As you can see, I performed PCA with an increasing number of components for each dataset until the sum of explained variance surpassed the threshold of 95%. 

# In[25]:


#Yields data
x_y = yields.drop(['Date'], axis=1)
x_y = StandardScaler().fit_transform(x_y)

pca_y = PCA(n_components=1)
principalComponents_y = pca_y.fit(x_y).components_
help_y = {'PC #1':principalComponents_y[0]}
principalDf_y = pd.DataFrame(data = help_y)
variability_y = pca_y.explained_variance_ratio_
print('1 Principal Component')
print('Explained Variance of Each PC:', variability_y)
print('Sum of Variability:', sum(variability_y))
print()

pca_y = PCA(n_components=2)
principalComponents_y = pca_y.fit(x_y).components_
help_y = {'PC #1':principalComponents_y[0], 'PC #2':principalComponents_y[1]}
principalDf_y = pd.DataFrame(data = help_y)
variability_y = pca_y.explained_variance_ratio_
print('2 Principal Components')
print('Explained Variance of Each PC:', variability_y)
print('Sum of Variability:', sum(variability_y))
print()

pca_y = PCA(n_components=3)
principalComponents_y = pca_y.fit(x_y).components_
help_y = {'PC #1':principalComponents_y[0], 'PC #2':principalComponents_y[1],'PC #3':principalComponents_y[2]}
principalDf_y = pd.DataFrame(data = help_y)
variability_y = pca_y.explained_variance_ratio_
print('3 Principal Components')
print('Explained Variance of Each PC:', variability_y)
print('Sum of Variability:', sum(variability_y))


# In[26]:


#Returns data
x_r = returns.drop(['X.Y..m..d'], axis=1)
x_r = StandardScaler().fit_transform(x_r)

pca_r = PCA(n_components=1)
principalComponents_r = pca_r.fit(x_r).components_
help_r = {'PC #1':principalComponents_r[0]}
principalDf_r = pd.DataFrame(data = help_r)
variability_r = pca_r.explained_variance_ratio_
print('1 Principal Component')
print('Explained Variance of Each PC:', variability_r)
print('Sum of Variability:', sum(variability_r))
print()

pca_r = PCA(n_components=2)
principalComponents_r = pca_r.fit(x_r).components_
help_r = {'PC #1':principalComponents_r[0], 'PC #2':principalComponents_r[1]}
principalDf_r = pd.DataFrame(data = help_r)
variability_r = pca_r.explained_variance_ratio_
print('2 Principal Components')
print('Explained Variance of Each PC:', variability_r)
print('Sum of Variability:', sum(variability_r))
print()

print('...')
print('...')
print()

pca_r = PCA(n_components=17)
principalComponents_r = pca_r.fit(x_r).components_
help_r = {'PC #1':principalComponents_r[0], 'PC #2':principalComponents_r[1], 'PC #3':principalComponents_r[2], 'PC #4':principalComponents_r[3], 'PC #5':principalComponents_r[4], 'PC #6':principalComponents_r[5], 'PC #7':principalComponents_r[6], 'PC #8':principalComponents_r[7], 'PC #9':principalComponents_r[8], 'PC #10':principalComponents_r[9], 'PC #11':principalComponents_r[10], 'PC #12':principalComponents_r[11], 'PC #13':principalComponents_r[12], 'PC #14':principalComponents_r[13], 'PC #15':principalComponents_r[14], 'PC #16':principalComponents_r[15], 'PC #17':principalComponents_r[16]}
principalDf_r = pd.DataFrame(data = help_r)
variability_r = pca_r.explained_variance_ratio_
print('17 Principal Components')
print('Explained Variance of Each PC:', variability_r)
print('Sum of Variability:', sum(variability_r))
print()

pca_r = PCA(n_components=18)
principalComponents_r = pca_r.fit(x_r).components_
help_r = {'PC #1':principalComponents_r[0], 'PC #2':principalComponents_r[1], 'PC #3':principalComponents_r[2], 'PC #4':principalComponents_r[3], 'PC #5':principalComponents_r[4], 'PC #6':principalComponents_r[5], 'PC #7':principalComponents_r[6], 'PC #8':principalComponents_r[7], 'PC #9':principalComponents_r[8], 'PC #10':principalComponents_r[9], 'PC #11':principalComponents_r[10], 'PC #12':principalComponents_r[11], 'PC #13':principalComponents_r[12], 'PC #14':principalComponents_r[13], 'PC #15':principalComponents_r[14], 'PC #16':principalComponents_r[15], 'PC #17':principalComponents_r[16], 'PC #18':principalComponents_r[17]}
principalDf_r = pd.DataFrame(data = help_r)
variability_r = pca_r.explained_variance_ratio_
print('18 Principal Components')
print('Explained Variance of Each PC:', variability_r)
print('Sum of Variability:', sum(variability_r))


# 2) How many PCs did you use to capture at least 95% of the variability of the dataset?
# 
# After performing PCA on the yields data, it was apparent that 3 principal components was the minimum number of components to capture at least 95% of the variablility (0.9638).
# 
# After performing PCA on the returns data, it was apparent that 18 principal components was the minimum number of components to capture at least 95% of the variablility (0.9627).

# 3) (1) Generate a scree plot (proportion variance explained (PVE) & PCs)
# 

# In[27]:


#yields
pca_y_plt = PCA(n_components=10)
principalComponents_y_plt = pca_y_plt.fit_transform(x_y)

plt.plot(np.arange(1, 11, 1), pca_y_plt.explained_variance_ratio_)
plt.ylim(0, 0.8)
plt.xlim(1, 10.0)
plt.xticks(np.arange(1, 11, step=1))
plt.xlabel('Number of Components')
plt.ylabel('Proportion Variance Explained')
plt.title('Scree Plot for Yields Data (PVE & PC)')
plt.show()

#returns
pca_r_plt = PCA(n_components=20)
principalComponents_r_plt = pca_r_plt.fit_transform(x_r)

plt.plot(np.arange(1, 21, 1), pca_r_plt.explained_variance_ratio_)
plt.ylim(0, 0.5)
plt.xlim(1, 20.0)
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('Number of Components')
plt.ylabel('Proportion Variance Explained')
plt.title('Scree Plot for Returns Data (PVE & PC)')
plt.show()


# 3) (2) Generate a plot with the cumulative PVE & PCs

# In[28]:


#yields
variances_y = list()
sum_y = 0
for i in pca_y_plt.explained_variance_ratio_:
    sum_y = i + sum_y
    variances_y.append(sum_y)

plt.plot(np.arange(1, 11, 1), variances_y)
plt.ylim(0.5, 1.0)
plt.xlim(1, 10.0)
plt.xticks(np.arange(1, 11, step=1))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Scree Plot for Yields (Cum. PVE & PC)')
plt.show()

#returns
variances_r = list()
sum_r = 0
for i in pca_r_plt.explained_variance_ratio_:
    sum_r = i + sum_r
    variances_r.append(sum_r)

plt.plot(np.arange(1, 21, 1), variances_r)
plt.ylim(0.1, 1.0)
plt.xlim(1, 20.0)
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Scree Plot for Returns (Cum. PVE & PC)')
plt.show()


# 3) (3) Generate a scattered plot using the first two principal component loading vectors
# 
# principalDf is a dataframe that includes all of the principal component loading vectors. From there, I selected the first 2 principal components for each and displayed them.

# In[29]:


plt.scatter(principalDf_y['PC #1'], principalDf_y['PC #2'], c='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Yields: Loading Vectors PC 1 vs. PC 2')
plt.show()


# In[30]:


plt.scatter(principalDf_r['PC #1'], principalDf_r['PC #2'], c='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Returns: Loading Vectors PC 1 vs. PC 2')
plt.show()


# 3) (4) Generate a matrix or a dataframe with the scores of all the principal components
# 
# I created a dataframe with each principal component score and variable, then displayed them for each.

# In[31]:


column_names_y = yields.columns.values[1:]
help_y = {'Variable':column_names_y, 'PC #1':principalComponents_y[0], 'PC #2':principalComponents_y[1], 'PC #3':principalComponents_y[2]}
principalDf_y = pd.DataFrame(data = help_y)
print('Yields Data Principal Component Scores:')
print()
print(principalDf_y)


# In[32]:


column_names_r = returns.columns.values[1:]
help_r = {'Variable':column_names_r, 'PC #1':principalComponents_r[0], 'PC #2':principalComponents_r[1], 'PC #3':principalComponents_r[2], 'PC #4':principalComponents_r[3], 'PC #5':principalComponents_r[4], 'PC #6':principalComponents_r[5], 'PC #7':principalComponents_r[6], 'PC #8':principalComponents_r[7], 'PC #9':principalComponents_r[8], 'PC #10':principalComponents_r[9], 'PC #11':principalComponents_r[10], 'PC #12':principalComponents_r[11], 'PC #13':principalComponents_r[12], 'PC #14':principalComponents_r[13], 'PC #15':principalComponents_r[14], 'PC #16':principalComponents_r[15], 'PC #17':principalComponents_r[16], 'PC #18':principalComponents_r[17]}
principalDf_r = pd.DataFrame(data = help_r)
print('Returns Data Principal Component Scores:')
print()
print(principalDf_r)


# 4) (1) Aggregate the items (yields or stocks) in clusters using: K-means [Show the clusters for each case.]
# 
# First, I graphed SSE vs Number of clusters for each case. I identified where there was an 'elbow', where increasing clusters no longer had a big impact, and chose that for the model. I ran the k-means model for each dataset and displayed the centroids for each. However, for display purposes, I selected 2 variables and repeated the process to show what the clustering looks like. The centroids are the black dots in the plots. 
# 
# Yields: The elbow for both the whole data set and the subset were at 4 clusters, which is what I ran the model with.
# 
# Returns: At 4 clusters for the dataset and subset, you start seeing a decreased impact of increasing the number of clusters, so I selected 4 clusters for both models

# In[33]:


#Yields
sse_y = {}
for k in np.arange(1, 13, 1):
    kmeans = KMeans(n_clusters=k).fit(x_y)
    sse_y[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse_y.keys()), list(sse_y.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(np.arange(1, 13, step=1))
plt.show()
    
kmeans_y = KMeans(n_clusters=4)
kmeans_y.fit(x_y)
labels_y = kmeans_y.predict(x_y)
centroids_y = kmeans_y.cluster_centers_

print(centroids_y)

print()
print()
print('Clustering with 3yr and 5yr Yields')
sse_y = {}
for k in np.arange(1, 13, 1):
    kmeans = KMeans(n_clusters=k).fit(x_y[:,5:7])
    sse_y[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse_y.keys()), list(sse_y.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(np.arange(1, 13, step=1))
plt.show()

kmeans_y = KMeans(n_clusters=4)
kmeans_y.fit(x_y[:,5:7])
labels = kmeans_y.predict(x_y[:,5:7])
centroids_y = kmeans_y.cluster_centers_

print(centroids_y)

plt.figure(figsize=(10, 7))  
plt.scatter(x_y[:,5], x_y[:,6], c=kmeans_y.labels_, cmap='rainbow')  
plt.plot(centroids_y[0][0], centroids_y[0][1], 'or', color="black")
plt.plot(centroids_y[1][0], centroids_y[1][1], 'or', color="black")
plt.plot(centroids_y[2][0], centroids_y[2][1], 'or', color="black")
plt.plot(centroids_y[3][0], centroids_y[3][1], 'or', color="black")
plt.title('3yr vs. 5yr Yield Clustering')
plt.show()


# In[34]:


#Returns
sse_r = {}
for k in np.arange(1, 13, 1):
    kmeans = KMeans(n_clusters=k).fit(x_r)
    sse_r[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse_r.keys()), list(sse_r.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(np.arange(1, 13, step=1))
plt.show()

kmeans_r = KMeans(n_clusters=4)
kmeans_r.fit(x_r)
labels_r = kmeans_r.predict(x_r)
centroids_r = kmeans_r.cluster_centers_

print(centroids_r)

print()
print()
print('Clustering with GUC and Market Returns')
sse_y = {}
for k in np.arange(1, 13, 1):
    kmeans = KMeans(n_clusters=k).fit(x_r[:,19:21])
    sse_r[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse_r.keys()), list(sse_r.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(np.arange(1, 13, step=1))
plt.show()

kmeans_r = KMeans(n_clusters=4)
kmeans_r.fit(x_r[:,19:21])
labels = kmeans_r.predict(x_r[:,19:21])
centroids_r = kmeans_r.cluster_centers_

print(centroids_r)


plt.figure(figsize=(10, 7))  
plt.scatter(x_r[:,19], x_r[:,20], c=kmeans_r.labels_, cmap='rainbow')  
plt.plot(centroids_r[0][0], centroids_r[0][1], 'or', color="black")
plt.plot(centroids_r[1][0], centroids_r[1][1], 'or', color="black")
plt.plot(centroids_r[2][0], centroids_r[2][1], 'or', color="black")
plt.plot(centroids_r[3][0], centroids_r[3][1], 'or', color="black")
plt.title('GUC vs. Market Clustering')
plt.show()


# 4) (2) Aggregate the items (yields or stocks) in clusters using: Hierarchical Clustering  [Show the clusters for each case.]
# 
# For both cases, I determined the number of clusers that would be best usig the dendogram. From there, I ran a hierarchical clustering model. For display purposes, I selected 2 variables from each case to show what the clustering looks like.
# 
# 
# From the dendogram, you find the longest line without a horizontal line (or new cluster). From there, you draw a horizontal line across and the number of lines it intersects is the number of clusers to use in heirarchical clustering.
# 
# Yields: From the dendogram, it is apparent that 3 clusters would be best. For the subset of 3yr and 5yr, 3 clusters is best.
# 
# Returns: From the dendogram, it is apparent that 4 clusters would be best. For the subset of GUC and market, 3 clusters is best.

# In[35]:


#Yields

plt.figure(figsize=(10, 7))  
plt.title("Yields Dendograms")  
dend_y = shc.dendrogram(shc.linkage(x_y, method='ward'))  

cluster_y = AgglomerativeClustering(n_clusters=3, linkage='ward')  
cluster_y.fit_predict(x_y)



plt.figure(figsize=(10, 7))  
plt.title("Dendograms for 3yr vs 5yr Yields")
dend_y = shc.dendrogram(shc.linkage(x_y[:, 5:7], method='ward'))  

cluster_y = AgglomerativeClustering(n_clusters=3, linkage='ward')  
cluster_y.fit_predict(x_y[:, 5:7])

plt.figure(figsize=(10, 7))  
plt.scatter(x_y[:,5], x_y[:,6], c=cluster_y.labels_, cmap='rainbow') 
plt.title("Clustering for 3yr vs 5yr Yields")
plt.show()


# In[36]:


#Returns
plt.figure(figsize=(10, 7))  
plt.title("Returns Dendograms")  
dend_r = shc.dendrogram(shc.linkage(x_r, method='ward'))  

cluster_r = AgglomerativeClustering(n_clusters=4, linkage='ward')  
cluster_r.fit_predict(x_r)



plt.figure(figsize=(10, 7))  
plt.title("Dendograms for GUC vs Market Returns")
dend_r = shc.dendrogram(shc.linkage(x_r[:, 19:21], method='ward'))  

cluster_r = AgglomerativeClustering(n_clusters=3, linkage='ward')  
cluster_r.fit_predict(x_r[:, 19:21])

plt.figure(figsize=(10, 7))  
plt.scatter(x_r[:,19], x_r[:,20], c=cluster_r.labels_, cmap='rainbow') 
plt.title("Clustering for GUC vs Market Returns")
plt.show()


# 5) Briefly discuss the difference of results for both datasets.

# For principal component analysis, it took very few components for the yields data to reach the 95% variability mark. However, the returns data needed much more components, all the way up to 18. You can observe this in the scree plots, where the returns data has a much more gradual slope versus the steep slope of the yields data.
# 
# For k-means clustering, the SSE vs. Number of Clusters led me to the conclusion of 4 clusters for both datasets, although they may have seemed different in nature. 
# 
# Lastly, for the hierarchical clustering, the yields dataset had an optimal number of clusters at 3 according to the dendogram, while the returns data had an optimal number of 4.
# 
# Although the contents of the datasets were very different, you can observe that there were both similarities and differences between the two when it comes to PCA and clustering.
