# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:22:57 2018

@author: German G. Creamer
"""

#Install pandas-datareader. Open Anaconda Navigator. 
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import os

# Print clusters
def clusterPrint(k_clusters,data):
    cluster_listing = {}
    for cluster in range(k_clusters):
        cluster_listing['Cluster ' + str(cluster)] = [''] * len(data.index)
        where_in_cluster = np.where(clusters == cluster)[0]
        cluster_listing['Cluster ' + str(cluster)][0:len(where_in_cluster)] = data.index[where_in_cluster]
    # Print clusters
    print(pd.DataFrame(cluster_listing).loc[0:np.max(np.bincount(clusters)) - 1,:])


#Run the script twice: with stocks (STOCKS=1) and then with yields (STOCKS=0)
STOCKS=1

if(STOCKS):
    data1 = pd.read_csv("./midcapD.ts.csv") 
    data1 = data1.iloc[:, 1:22]    
    num_pc = 15
else:
    data1 = pd.read_csv("./yields2009.csv") 
    data1 = data1.iloc[:, 1:12]    
    num_pc = 2

data1.head()

X = np.asarray(data1)
[n,m] = X.shape
print('The number of timestamps is {}'.format(n))
print('The number of cases is {}'.format(m))

#1.
pca = PCA(n_components=num_pc) # number of principal components
pca.fit(X)

#2.
percentage =  pca.explained_variance_ratio_

percentage_cum = np.cumsum(percentage)

print('%0.2f of the variance is explained by the first %1d PCs' % ((percentage_cum[-1]*100),num_pc))
    
pca_components = pca.components_
x = np.arange(1,len(percentage)+1,1)

#3.1
plt.subplot(1, 2, 1)
plt.bar(x, percentage*100, align = "center")
plt.title('Contribution of principal components',fontsize = 16)
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.xticks(x,fontsize = 16) 
plt.yticks(fontsize = 16)
plt.xlim([0, num_pc+1])

#3.2
plt.subplot(1, 2, 2)
plt.plot(x, percentage_cum*100,'ro-')
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.title('Cumulative contribution of principal components',fontsize = 16)
plt.xticks(x,fontsize = 16) 
plt.yticks(fontsize = 16)
plt.xlim([1, num_pc])   
plt.ylim([30,100]);
        

#3.3.
#PC loadings (phi _ij):
factor_exposures = pd.DataFrame(index= list(range(1,num_pc+1)), 
                                columns=data1.columns,
                                data = pca.components_).T
factor_exposures
    
labels = factor_exposures.index
data0 = factor_exposures.values

plt.subplots_adjust(bottom = 0.1)

plt.title('Scatter Plot of Coefficients (loadings) of PC1 and PC2')
plt.xlabel('Loadings of PC1')
plt.ylabel('Loadings of PC2')
plt.scatter(
    data0[:, 0], data0[:, 1], marker='o', s=300, c='m',
    cmap=plt.get_cmap('Spectral'))

#3.4 PC scores (z_ij): inner product: <X,PC factor loadings)        
factor_data1 = X.dot(pca_components.T)
factor_data1 = pd.DataFrame(columns=list(range(1,num_pc+1)), 
                              index=data1.index,
                              data=factor_data1)
factor_data1.head()


#4. Clustering
data = data1.T
data.head()
    
#4.1. KMeans
k_clusters = 4
model = KMeans(k_clusters)
model.fit(data)   
print("Records in our dataset (rows): ", len(data.index))

clusters = model.predict(data)
#clusters
#pd.DataFrame(list(zip(data.index,model.predict(data))), columns=['Labels','Cluster_predicted']) 
clusterPrint(k_clusters,data)

#4.2. Hierarchical clustering
model = AgglomerativeClustering().fit(data)
clusters = model.labels_
k_clusters = clusters.max()+1
#pd.DataFrame(list(zip(data.index,clusters)), columns=['Labels','Cluster_predicted']) 
clusterPrint(k_clusters,data)
