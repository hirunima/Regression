#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:54:34 2019

@author: hirunima_j
"""

import scipy.io
from PIL import Image
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
import glower
import kmedoids
from mpl_toolkits import mplot3d

# load dataset into Pandas DataFrame
df_data = pd.read_csv('dataset_1.csv')
heads=list(df_data.head(0))
#names=['call_id','duration_queue','duration_tot','call_date','call_start','call_end','	call_duration','ucid','vdn','agent_id','cli','skill_id','acd','file_name','upload_date','time_key','date_id','bill_refill','	identification_number','msisdn_voice','province	network_stay','credit_category','credit_type','customer_priority_type','age','gender']
le = LabelEncoder()
##df_data.loc[:, 'gender']=df_data.loc[:, 'gender'].values
df_data['gender'] =le.fit_transform(df_data['gender'].astype(str))
df_data['survey_name'] =le.fit_transform(df_data['survey_name'].astype(str))
features = ['duration_queue', 'duration_tot','survey_name', 'network_stay', 'age','gender']
# Separating out the features
x = df_data.loc[:, features].values
mask = np.any(np.isnan(x), axis=1)
x_data=x[~mask]
# Separating out the target
y = df_data.loc[:,['positive_negative']].values
e=list(np.asarray([y[~mask][:] == 0]*6)[:,:,0].transpose())
df0 = x_data[(np.asarray([y[~mask][:] == 0]*6)[:,:,0].transpose())].reshape(-1,len(features))
df1 = x_data[(np.asarray([y[~mask][:] == 1]*6)[:,:,0].transpose())].reshape(-1,len(features))
############################################
def clusterings(x_data):
    D = glower.gower_distances(x_data)
    M, C = kmedoids.kMedoids(D, 3)
    print('medoids:')
    for point_idx in M:
        print( x_data[point_idx])
    
    #print('clustering result:')
    #for label in C:
    #    for point_idx in C[label]:
    #        print('label {0}:　{1}'.format(label, x_data[point_idx]))
        
    clf = MDS(n_components=3, n_init=1, max_iter=100,dissimilarity='precomputed')
    X_mds = clf.fit_transform(D)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_mds[C[0],0],X_mds[C[0],1],X_mds[C[0],2], cmap='Greens')
    ax.scatter3D(X_mds[C[1],0],X_mds[C[1],1],X_mds[C[1],2], cmap='Greens')
    ax.scatter3D(X_mds[C[2],0],X_mds[C[2],1],X_mds[C[2],2], cmap='Greens')
#    ax.scatter3D(X_mds[C[3],0],X_mds[C[3],1],X_mds[C[3],2], cmap='Greens')
#    ax.scatter3D(X_mds[C[4],0],X_mds[C[4],1],X_mds[C[4],2], cmap='Greens')
    plt.show()
    #plot_embedding(X_mds, yconcat, titles=ttls)

clusterings(df0)
clusterings(df1)
##################analyze clusters
