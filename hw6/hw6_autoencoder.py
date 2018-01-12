# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 00:48:38 2018

@author: USER
"""

from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np
import sys 
from sklearn.cluster import KMeans

train = np.load(sys.argv[1])
f = pd.read_csv(sys.argv[2])
image1 = f['image1_index'].values
image2 = f['image2_index'].values
train = train/255

x_emb = PCA(n_components=300,svd_solver='full',whiten=True,copy=True)
ff = x_emb.fit_transform(train)


kmeans = KMeans(n_clusters=2, random_state=0).fit(ff)
lab = list(kmeans.labels_)

ans=[]
for i,j in zip(image1,image2):
    temp = lab[i]
    temp_2 = lab[j]
    if temp == temp_2:
        ans.append(1)
    else :
        ans.append(0)
        
Id = [i for i in range(1980000)]
matrix={}
matrix['ID']=Id
matrix['Ans']=ans
df=pd.DataFrame(matrix)
df=df[['ID','Ans']]
df.to_csv(sys.argv[3],index=False)   