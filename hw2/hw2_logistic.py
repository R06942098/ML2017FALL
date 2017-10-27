#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:29:00 2017

@author: cengbowei
"""
import pandas as pd
import numpy as np
import sys
valid_1 = pd.read_csv(sys.argv[1])
valid_2 = pd.read_csv(sys.argv[2])
df = pd.read_csv(sys.argv[3])
target = pd.read_csv(sys.argv[4])
target = target.values.reshape(-1)
train_data1 = df.values
fd = pd.read_csv(sys.argv[5])
test_data1 = fd.values
normalization = np.concatenate((train_data1,test_data1),axis=0)
a = normalization - np.mean(normalization,axis = 0)
b = np.std(normalization,axis = 0)
data = a/b
train_data = data[0:32561,:]
test_data = data[32561:,:]
train_data = np.insert(train_data,0,np.ones(32561),axis=1)
test_data = np.insert(test_data,0,np.ones(16281),axis=1)
weight=np.zeros(107)
s_gra = np.zeros(107)
l_rate = 1
iteration = 1000
alpha = 0.0001
def sigmoid(x):
    a =  1 / (1 + np.exp(-x))
    return np.clip(a,0.0000000001,0.9999999999)
for i in range(iteration):
    y_sig = sigmoid(train_data.dot(weight))
    temp1 = weight[1:]
    temp = np.concatenate((np.zeros(1),temp1),axis=0)
    cost = y_sig - target
    gra = train_data.transpose().dot(cost)  + 2 * alpha * temp
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    weight = weight - gra/ada * l_rate
    print('iteration %d' %i)
    
List_a = []
s = train_data.dot(weight)
for i in s:
    if i > 0.5:
        List_a.append(1)
    else:
        List_a.append(0)
yt_hat = np.array(List_a)
accu = []
for i in range(32561):
    if yt_hat[i] == target[i]:
        accu.append(1)
score = len(accu)/32561
print(score)


o_sigmoid = test_data.dot(weight)
y_hat =  sigmoid(o_sigmoid)

List = []
for i in y_hat:
    if i > 0.5 :
        List.append(1)
    else:
        List.append(0)
List_id = [i for i in range(1,16282)]
outmatrix = {}
outmatrix['id'] = List_id
outmatrix['label'] = List
final_result = pd.DataFrame(outmatrix)
final_result.to_csv(sys.argv[6],index=False)
