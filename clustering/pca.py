#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:09:48 2019

@author: hirunima_j
"""

import scipy.io
from PIL import Image
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# load dataset into Pandas DataFrame
df_data = pd.read_csv('dataset_1.csv')
heads=list(df_data.head(0))
#names=['call_id','duration_queue','duration_tot','call_date','call_start','call_end','	call_duration','ucid','vdn','agent_id','cli','skill_id','acd','file_name','upload_date','time_key','date_id','bill_refill','	identification_number','msisdn_voice','province	network_stay','credit_category','credit_type','customer_priority_type','age','gender']
from sklearn.preprocessing import StandardScaler
df_data=df_data
features = ['duration_queue', 'duration_tot', 'network_stay', 'age']
# Separating out the features
x = df_data.loc[:, features].values
mask = np.any(np.isnan(x), axis=1)
x_data=x[~mask]
# Separating out the target
y = df_data.loc[:,['positive_negative']].values
# Standardizing the features
scal=StandardScaler()
x = scal.fit_transform(x_data)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2], cmap='Greens')
plt.show()

kmeans = KMeans(n_clusters=2) 
kmeans.fit(principalComponents)
prediction = kmeans.predict(principalComponents)
centroids = kmeans.cluster_centers_
colmap = {0: 'r', 1: 'g'}

plt.figure()
ax = plt.axes(projection='3d')
colors = map(lambda x: colmap[x], prediction)

ax.scatter3D(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2]
,color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    ax.scatter3D(*centroid, color=colmap[idx])
plt.show()

h=scal.inverse_transform(pca.inverse_transform(centroids))
#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
#    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
#    random_state=None, tol=0.0001, verbose=0)
