# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:18:05 2017

@author: Sujay
"""

# needed imports
from matplotlib import pyplot as plt
import scipy.io as spio
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
import numpy as np
#import numpy as np
mat = spio.loadmat('dataset.mat', squeeze_me=True)
li = spio.loadmat('list.mat', squeeze_me=True)
X=mat['vect']
L=li['vect']
size=X.shape
print(size)
# generate the linkage matrix
Z = linkage(X,'complete')
nCls=5
c, coph_dists = cophenet(Z, pdist(X))
print('c= ',c)
T=fcluster(Z, nCls, criterion='maxclust')
label=[T,L]
label=np.transpose(np.reshape(label,(2,len(T))))
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
Cls=[]
for i in range(1,nCls+1):
    t=[]
    c=0
    for j in range(len(T)):
        if T[j]==i:
            c=c+1
            t.append(X[j,:])
    Cls.append(np.reshape(t,(c,size[1])))

centroid=[]
for i in range(nCls):
    temp=np.reshape(Cls[i],(len(Cls[i]),size[1]))
    t=[]
    for j in range(size[1]):
        t.append(np.median(np.sort(temp[:,j])))
    centroid.append(t)
centroid=np.reshape(centroid,(len(Cls),size[1]))        
spio.savemat('Centriod_list.mat',{'cen':centroid})
spio.savemat('Label_list.mat',{'list':label})      
            
        
    