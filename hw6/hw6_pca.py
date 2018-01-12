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
    b=os.listdir(sys.argv[1])
    a=[]
    for i in b:
        c = io.imread(sys.argv[1] + i).flatten()
        a.append(c)
    train = np.array(a).T
    mean = np.mean(train,axis=1)[:,np.newaxis]
    train_1 = train - mean
    #train,train_1,U,s,v = Svd()
    U,s,v = np.linalg.svd(train_1,full_matrices=False)
    eigen_face = U[:,0:4]
    temp = np.dot(eigen_face.T,train_1)
    recon = np.dot(eigen_face,temp) + mean
    recon -= np.min(recon,0)
    recon /= np.max(recon,0)
    M = (recon*255).astype(np.uint8)
    reco_num  = int(sys.argv[2].split('.')[0])
    reco = M[:,reco_num].reshape(600,600,3)
    io.imsave('reconstruction.jpg',reco)
reconstruct()
