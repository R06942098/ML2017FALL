# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 23:30:24 2018

@author: USER
"""
import numpy as np
import os 
import sys
from numpy import dot,linalg
from skimage import io


def reconstruct():
    #train = preprocess()
    temp = os.path.join(sys.argv[1],sys.argv[2])
    reco = io.imread(temp).reshape(-1)[:,np.newaxis].astype('float32')
    b=os.listdir(sys.argv[1])
    a=[]
    for i in b:
        c = io.imread(sys.argv[1] + i).flatten()
        a.append(c)
    train = np.array(a).T
    mean = np.mean(train,axis=1)[:,np.newaxis]
    train_1 = train - mean
    reco -= mean
    #train,train_1,U,s,v = Svd()
    U,s,v = np.linalg.svd(train_1,full_matrices=False)
    eigen_face = U[:,0:4]
    temp = np.dot(eigen_face.T,reco)
    recon = np.dot(eigen_face,temp) + mean
    recon -= np.min(recon,0)
    recon /= np.max(recon,0)
    M = (recon*255).astype(np.uint8)
    reco = M.reshape(600,600,3)
    io.imsave('reconstruction.jpg',reco)
reconstruct()

