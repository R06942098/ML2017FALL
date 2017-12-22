# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:33:10 2017

@author: USER
"""

import pandas  as pd
import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense,Input,Flatten,dot,LSTM
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers.merge import dot , add,concatenate
from keras.models import load_model
import sys
e = pd.read_csv(sys.argv[1])
testID = e['TestDataID'].values
UserID = e['UserID'].values
MovieID = e['MovieID'].values
pp = load_model('best.h5')
temp = pp.predict([UserID,MovieID])
temp = temp.reshape(-1)
matrix = {}
matrix['Rating']=list(temp)
matrix['TestDataID']=list(testID)
e = pd.DataFrame(matrix)
e = e[['TestDataID','Rating']]
e.to_csv(sys.argv[2],index=False)