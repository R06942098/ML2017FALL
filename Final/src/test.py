# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:30:27 2018

@author: USER
"""

import numpy as np
import pandas as pd 
import pickle
import jieba
import sys
from keras.models import load_model
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
jieba.set_dictionary('dict.txt.big')

test_c = pd.read_csv(sys.argv[2])
pp=test_c.values
col = np.array(test_c.columns)
col = col[np.newaxis,:]
test = np.concatenate((col,pp),axis=0)
test_answer = list(test.reshape(-1))

val_3 = []
for i in test_answer:
    yy = ''
    for j in i:
        if j != ' ':
            yy = yy +j
    val_3.append(yy)
    
pre_test = [[] for i in range(len(test_answer))]

for i in range(len(val_3)):
    word = jieba.cut(val_3[i])
    word_1 = ' '.join(word)
    pre_test[i].append(word_1) 

pre_2= []
for i in range(len(pre_test)):
    temp =text_to_word_sequence(pre_test[i][0])
    pre_2.append(temp)
    
test = []
for i in pre_test:
    test.append(i[0])
    
tokenizer.fit_on_texts(test)
sequences_test = tokenizer.texts_to_sequences(test) 
test_answer  = pad_sequences(sequences_test,padding='post',maxlen=246)

def test_pad_0():    
    c = open(sys.argv[1],'rb')
    temp = pickle.load(c)
    List = []
    for i in temp:
        w,l = i.shape
        pad = 246-w
        zon = np.zeros((pad,39))
        temp_1 = np.concatenate((i,zon),axis=0)
        List.append(temp_1)
    return np.array(List)

test_data = test_pad_0()


def ensemble():
       w = load_model('donknow.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
       q = load_model('pre_0.0090.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})      
       s = load_model('pre_4900.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})      
       t = load_model('pre_541.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})      
       r = load_model('pre_0.13.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})      
       y = load_model('pre_527.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
       e = load_model('pre_bes.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
       u = load_model('563.h5',custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
       f=0
       ans=[]
       for i in range(0,8000,4):
           temp = test_data[f].reshape(1,246,39)
           temp_1 = []
           temp_2 = test_answer[i:i+4]
           for j in range(4):
               temp_3 = temp_2[j].reshape(1,246)
               d_1 = q.predict([temp,temp_3])[0][0]
               d_2 = s.predict([temp,temp_3])[0][0]
               d_3 = t.predict([temp,temp_3])[0][0]
               d_4 = r.predict([temp,temp_3])[0][0]
               d_5 = y.predict([temp,temp_3])[0][0]
               d_6 = w.predict([temp,temp_3])[0][0]
               d_7 = e.predict([temp,temp_3])[0][0]
               d_8 = u.predict([temp,temp_3])[0][0]
               d_9 = d_1 + d_2 + d_3 +d_4 +d_5 + d_6 + d_7 + d_8
               temp_1.append(d_9)
           answer = np.argmax(temp_1)
           ans.append(answer)
           f+=1
           if f%500 ==0:
               print(f)
       return ans

ans = ensemble()

matrix = {}
matrix['answer'] = ans
matrix['id'] = [i for i in range(1,2001)]
df  = pd.DataFrame(matrix)
df = df [['id','answer']]
df.to_csv(sys.argv[3],index=False)