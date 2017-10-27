#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:54:49 2017

@author: cengbowei
"""
import numpy as np
import pandas as pd
import sys
from xgboost import XGBClassifier
valid_1 = pd.read_csv(sys.argv[1])
valid_2 = pd.read_csv(sys.argv[2])
df = pd.read_csv(sys.argv[3])
target = pd.read_csv(sys.argv[4])
target = target.values.reshape(-1)
train_data1 = df.values
List_e = [0,2,3,4,10,24,25,26,27,29]
for i in List_e:
    train_data3 = train_data1[:,i][:,np.newaxis]
    if i == 0:
        train_data2 = train_data3
        continue
    train_data2 = np.concatenate((train_data2,train_data3),axis=1)
train_data4 = train_data2 * train_data2 
train_data = np.concatenate((train_data1,train_data2),axis=1)
fd = pd.read_csv(sys.argv[5])
test_data1 = fd.values
for i in List_e:
    test_data3 = test_data1[:,i][:,np.newaxis]
    if i == 0:
        test_data2 = test_data3
        continue
    test_data2 = np.concatenate((test_data2,test_data3),axis=1)
test_data4 = test_data2 * test_data2
test_data5 = test_data4 * test_data2
test_data = np.concatenate((test_data1,test_data2),axis=1)

s = XGBClassifier(max_depth=3,subsample=0.8,colsample_bytree=0.8,\
                  learning_rate=0.1,reg_lambda=10,n_estimators=1000,\
                  seed=27,nthread=4)
model = s.fit(train_data,target)
print(s)
y_hat = model.predict(test_data)
predictions = [round(value) for value in y_hat]
List_id = [i for i in range(1,16282)]
List = list(y_hat)
outmatrix = {}
outmatrix['id'] = List_id
outmatrix['label'] = List
final_result = pd.DataFrame(outmatrix)
cd = []

for i in y_hat:
    if i ==1:
        cd.append(i)
final_result.to_csv(sys.argv[6],index=False)


    
    
