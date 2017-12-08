# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:27:20 2017

@author: USER
"""

import pandas as pd 
import numpy as np
from keras.models import load_model
import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

c = open(sys.argv[1],encoding = 'utf8')
test_List=[]
for i in range(0,200001):
    temp = c.readline()
    if i == 0:
        continue
    if  1<=i and i<=10:
        temp = temp[2:]
        test_List.append(temp)
        continue
    if i>10 and i<=100:
        temp = temp[3:]
        test_List.append(temp)
        continue
    if i>100 and i <=1000:
        temp = temp[4:]
        test_List.append(temp)
        continue
    if i > 1000 and i<=10000:
        temp=temp[5:]
        test_List.append(temp)
        continue
    if i > 10000 and i<=100000:
        temp = temp[6:]
        test_List.append(temp)
        continue
    if i > 100000 :
        temp = temp[7:]
        test_List.append(temp)
        continue
c.close()
h = open('tokenizer.pickle','rb')
tokenizer = pickle.load(h)
h.close()
model = load_model('model.h5')
sequences_test = tokenizer.texts_to_sequences(test_List)
test = pad_sequences(sequences_test, maxlen=28)     
b = model.predict_classes(test,batch_size=1000)
List_s = [i for i in range(200000)]
Matrix = {}
Matrix['id']=List_s
c = b.reshape(-1)
Matrix['label']=list(c)
final = pd.DataFrame(Matrix)
final.to_csv(sys.argv[2],index=False)