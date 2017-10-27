#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:52:30 2017

@author: cengbowei
"""

import pandas as pd
import numpy as np
import sys
valid_1 = pd.read_csv(sys.argv[1])
valid_2 = pd.read_csv(sys.argv[2])
fd = pd.read_csv(sys.argv[3])
target = pd.read_csv(sys.argv[4])
train_data = fd.values
train_target = target.values.reshape(-1)
winner = []
loser = []
for i in range(32561):
    if train_target[i] == 1:
        winner.append(i)
    else:
        loser.append(i)
for i in range(len(winner)):
    a = winner[i]
    winner_data1 = train_data[a,:][np.newaxis,:]
    if i == 0 :
        winner_data = winner_data1
        continue
    winner_data = np.concatenate((winner_data,winner_data1),axis=0)
for i in range(len(loser)) :
    a = loser[i]
    loser_data1 = train_data[a,:][np.newaxis,:]
    if i == 0 :
        loser_data = loser_data1
        continue
    loser_data = np.concatenate((loser_data,loser_data1),axis=0)
'''    --------------class of winner and loser----------------------'''
winner_mean = np.mean(winner_data,axis=0)
winner_cov = np.zeros([106,106])
for i in range(len(winner)):
    a = (winner_data[i,:] - winner_mean)[:,np.newaxis]
    b = a.dot(a.transpose())
    winner_cov += b
winner_cov = winner_cov / len(winner)

'''------------------mean and covariance matrix -------------------'''
loser_mean = (np.mean(loser_data,axis=0))
loser_cov = np.zeros([106,106])
for i in range(len(loser)):
    a = (loser_data[i,:] - loser_mean)[:,np.newaxis]
    b = a.dot(a.transpose())
    loser_cov += b
loser_cov = loser_cov / len(loser)
'''------------------mean and covarinace matrix (loser)------------'''
win_prob = len(winner)/32561
los_prob = len(loser)/32561
covariance = (win_prob * winner_cov) + (los_prob * loser_cov)
total_mean = winner_mean - loser_mean
inv_cov = np.linalg.inv(covariance)
gg = len(winner)/len(loser)
w_t = (total_mean.transpose()).dot(inv_cov)
c = winner_mean.transpose().dot(inv_cov).dot(winner_mean)
d = loser_mean.transpose().dot(inv_cov).dot(loser_mean)
def sigmoid(x):
    out = 1 / (1+np.exp(-x))
    return np.clip(out,0.000000001,0.9999999990)
def post(x):
    z = np.dot(w_t,x) - c/2 + d/2 + np.log(gg)
    y = sigmoid(z)
    return np.clip(y,0.000000001,0.9999999990)

fd = pd.read_csv(sys.argv[5])
test_data = fd.values
result_List = []
for i in range(16281):
    x = test_data[i,:]
    a = post(x)
    if a > 0.5 :
        result_List.append(1)
    else:
        result_List.append(0)
List_id = [i for i in range(1,16282)]
outmatrix = {}
outmatrix['id'] = List_id
outmatrix['label'] = result_List
final_result = pd.DataFrame(outmatrix)
List=[]
for i in result_List:
	if i == 1:
		List.append(i)
print(len(List))
final_result = pd.DataFrame(outmatrix)
final_result.to_csv(sys.argv[6],index=False)

        

