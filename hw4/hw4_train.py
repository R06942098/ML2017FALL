# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:27:19 2017

@author: USER
"""

import pandas as pd 
import numpy as np
import random
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding ,Dense ,Flatten ,LSTM,Dropout,Activation,Input
from keras.layers import Conv1D, MaxPooling1D ,BatchNormalization
from keras.models import Sequential ,Model
from keras.activations import sigmoid
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import  l2,l1
from keras.layers import Bidirectional
from sklearn.svm import SVC
from gensim.models import Word2Vec
import sys
a = open(sys.argv[1],encoding='utf8')
List = []
List_label = []
for i in range(200000):
    temp = a.readline()
    temp_t = temp[10:]
    temp_l = temp[0]
    List.append(temp_t)
    List_label.append(temp_l)
    
labels = np.array(List_label,dtype='float64')
labels_1 = to_categorical(labels)
b = open(sys.argv[2],encoding='utf8')
nl_List = []
for i in range(1300000):
    temp = b.readline()
    nl_List.append(temp)
s_nl = np.load('remove.npy')  
nl_avList_1 = list(np.delete(nl_List,s_nl))

nl_avList=[]
for i in range(len(nl_List)):
    temp = len(nl_List[i])
    if temp < 193:
        nl_avList.append(nl_List[i])    
   
'''
test_List = []
c = open('testing_data.txt',encoding = 'utf8')
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
'''    
    
total=[]
total += List
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(List + nl_avList)
sequences_t = tokenizer.texts_to_sequences(List)
#sequences_test = tokenizer.texts_to_sequences(test_List)
sequences_nl_t = tokenizer.texts_to_sequences(nl_avList)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
train = pad_sequences(sequences_t, maxlen=28)  
#test = pad_sequences(sequences_test, maxlen=28)  
nl_t = pad_sequences(sequences_nl_t,maxlen=28)

emb = List + nl_avList
new_l = []
for i in range(len(emb)):
    temp = text_to_word_sequence(emb[i])
    new_l.append(temp)
model  = Word2Vec(new_l,size=192,min_count=3)

vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
embeddings_index = {}
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    coefs = vocab_list[i][1]
    embeddings_index[word]=coefs
    
embeddings_matrix = np.zeros((len(word_index) + 1, 192))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embeddings_matrix[i] = embedding_vector
    
embedding_layer = Embedding(len(word_index)+1, 192,
                            weights=[embeddings_matrix],input_length=28,
                            trainable=False)


t_data,v_data,t_label,v_label = train_test_split(train,labels_1,test_size=0.2,shuffle=True,random_state=42)     

model = Sequential()

model.add(embedding_layer)
#model.add(Bidirectional(LSTM(192,return_sequences=True,dropout=0.2)))
#model.add(Bidirectional(LSTM(512,return_sequences=True,dropout=0.3,recurrent_dropout=0.3)))
model.add(Bidirectional(LSTM(256,return_sequences=True,dropout=0.2)))
model.add(Bidirectional(LSTM(256,dropout=0.2)))
#model.add(Dense(units=192, activation='relu',kernel_initializer='he_normal'))
#model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(2,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(t_data ,t_label, epochs=15, validation_data=(v_data,v_label),shuffle=True ,batch_size=1000,\
                    callbacks=[ModelCheckpoint('best.h5',monitor='val_acc',save_best_only=True)])