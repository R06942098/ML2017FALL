# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:58:28 2017

@author: USER
"""

import pandas as pd 
import numpy as np
import sys
from keras.models import load_model
fd = pd.read_csv(sys.argv[1])
test =  fd.values
test = test[:,1]
j = 0
for i in range(len(test)):
    a = test[i].split()
    a = np.array(a,dtype='float64')
    temp = a[np.newaxis,:]
    if j == 0 :
        test_data = temp
        j+=1
        continue
    test_data = np.concatenate((test_data,temp),axis=0)
    print(j)
    j +=1
test_data = test_data.reshape(7178,48,48,1)
test_data /= 255
a = load_model('model.h5')
prediction = a.predict_classes(test_data)
List = [i for i in range(7178)]
c = list(prediction)
matrix = {}
matrix['id'] = List
matrix['label'] = c
final = pd.DataFrame(matrix)
final.to_csv(sys.argv[2],index=False)
