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
features = ['duration_queue', 'duration_tot','survey_name', 'network_stay', 'age','gender','positive_negative']
# Separating out the features
x = df_data.loc[:, features].values
mask = np.any(np.isnan(x), axis=1)
x_data=x[~mask]
# Separating out the target
y = df_data.loc[:,['positive_negative']].values
############################################
D = glower.gower_distances(x_data)
M, C = kmedoids.kMedoids(D, 5)
print('medoids:')
for point_idx in M:
    print( x_data[point_idx] )

#print('')
#
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
ax.scatter3D(X_mds[C[3],0],X_mds[C[3],1],X_mds[C[3],2], cmap='Greens')
ax.scatter3D(X_mds[C[4],0],X_mds[C[4],1],X_mds[C[4],2], cmap='Greens')
plt.show()
#plot_embedding(X_mds, yconcat, titles=ttls)

##################analyze clusters
data_0=y[ C[0],:]
only_0=[i for i in data_0 if i==1]
pre_0=len(only_0)/len(data_0)*100

data_1=y[ C[1],:]
only_1=[i for i in data_1 if i==1]
pre_1=len(only_1)/len(data_1)*100

data_2=y[ C[2],:]
only_2=[i for i in data_2 if i==1]
pre_2=len(only_2)/len(data_2)*100

data_3=y[ C[3],:]
only_3=[i for i in data_3 if i==1]
pre_3=len(only_3)/len(data_3)*100

data_4=y[ C[4],:]
only_4=[i for i in data_4 if i==1]
pre_4=len(only_4)/len(data_4)*100

##print(data_1['positive_negative'])
bar_length_1=[len(only_0),len(only_1),len(only_2),len(only_3),len(only_4)]
bar_length_0=list(np.array([len(data_0),len(data_1),len(data_2),len(data_3),len(data_4)])-np.array(bar_length_1))
cluster_names=['cluster1','cluster2','cluster3','cluster4','cluster5']
target=['positive','negative']
colors=['blue','orange']
def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge",color=colors[i])   
    plt.xticks(_X, X)

subcategorybar(cluster_names, [bar_length_1,bar_length_0])
plt.legend(target,loc=2)
plt.show()