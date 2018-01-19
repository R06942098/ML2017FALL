# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:47:20 2018

@author: USER
"""

import pandas as pd 
import numpy as np
import pickle
import  keras.backend as K
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding , Bidirectional
from keras.models import Sequential ,Model
from keras.layers import Bidirectional ,Dropout , Masking ,MaxPooling1D ,Conv1D
from keras.layers import Embedding ,Dense ,Flatten ,LSTM, Dropout,Activation,Input ,BatchNormalization
import random
import copy
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from keras.layers import Conv1D ,Lambda
from keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.layers.merge import Dot ,dot
import jieba 
from gensim.models import Word2Vec
import sys


jieba.set_dictionary('dict.txt.big')
temp = pd.read_csv(sys.argv[2])
temp_a = temp.values 
temp_l = []
temp_c = temp_l.append(temp.columns[0])
temp_c = np.array(temp_l)[np.newaxis,:]
train = np.concatenate((temp_c,temp_a),axis=0)
train_1 = train.reshape(-1)
val_1 = list(train_1)

test_c = pd.read_csv(sys.argv[4])
pp=test_c.values
col = np.array(test_c.columns)
col = col[np.newaxis,:]
test = np.concatenate((col,pp),axis=0)
test_answer = list(test.reshape(-1))

val_2=[]
for i in val_1:
    yy = ''
    for j in i:
        if j != ' ':
            yy = yy +j
    val_2.append(yy)
    
val_3 = []
for i in test_answer:
    yy = ''
    for j in i:
        if j != ' ':
            yy = yy +j
    val_3.append(yy)
    
pre_train = [[] for i in range(len(val_1))]
pre_test = [[] for i in range(len(test_answer))]

for i in range(len(val_1)):
    word = jieba.cut(val_2[i])
    word_1 = ' '.join(word)
    pre_train[i].append(word_1)
    
    
for i in range(len(val_3)):
    word = jieba.cut(val_3[i])
    word_1 = ' '.join(word)
    pre_test[i].append(word_1)    

pre_1 = []
for i in range(len(pre_train)):
    temp =text_to_word_sequence(pre_train[i][0])
    pre_1.append(temp)
pre_2= []
for i in range(len(pre_test)):
    temp =text_to_word_sequence(pre_test[i][0])
    pre_2.append(temp)

tra = []
for i in pre_train:
    tra.append(i[0])
test = []
for i in pre_test:
    test.append(i[0])
    
    
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(tra + test)
sequences_train = tokenizer.texts_to_sequences(tra)
sequences_test = tokenizer.texts_to_sequences(test) 
correct_train  = pad_sequences(sequences_train,padding='post',maxlen=246)
test_answer  = pad_sequences(sequences_test,padding='post',maxlen=246)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


temp = pre_1+pre_2
model  = Word2Vec(temp ,size=300,window=5,min_count=3,workers=5)
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
embeddings_index = {}

for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    coefs = vocab_list[i][1]
    embeddings_index[word]=coefs

embeddings_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embeddings_matrix[i] = embedding_vector 
        

        
def train_pad_0():
    c = open(sys.argv[1],'rb')
    temp = pickle.load(c)
    List = []
    for i in temp :
        w,l = i.shape
        pad = 246-w
        zon = np.zeros((pad,39))
        temp_1 = np.concatenate((i,zon),axis=0)
        List.append(temp_1)
    return np.array(List)
def test_pad_0():    
    c = open(sys.argv[3],'rb')
    temp = pickle.load(c)
    List = []
    for i in temp:
        w,l = i.shape
        pad = 246-w
        zon = np.zeros((pad,39))
        temp_1 = np.concatenate((i,zon),axis=0)
        List.append(temp_1)
    return np.array(List)


train = train_pad_0()
test_data = test_pad_0()
        
def es_error(correct_train):
    error = []
    for i in range(45036):
        temp = list(correct_train)
        del temp[i]
        f = random.sample(temp,1)
        error.append(f)
    return np.array(error).reshape(45036,246)
        

def load_train():
    train = np.load('train_data.npy')
    return train

def load_error():
    error_train = np.load('error_15.npy')
    return error_train

def concat_caption():
    error_train = load_error()
    qanswer = np.concatenate((correct_train,error_train),axis=0)
    return qanswer
def concat_train():
    train = load_train()
    train = np.concatenate((train,train),axis=0)
    return train

def handcraft_label():
    k=45036
    label  = []
    for i in range(k):
        label+=[1]
    for i in range(k):
        label+=[0]
    Ytrain = np.array(label)
    Ytrain = Ytrain.astype('float32')
    return Ytrain

def data_split(train,correct_train):
    error_train = es_error(correct_train)
    a , b = error_train.shape
    correct_train = correct_train.reshape(a,1,b)
    error_train = error_train.reshape(a,1,b)
    train_t,train_val,correct_t,correct_val,error_t,error_val = train_test_split(train,correct_train,error_train,test_size=0.1,shuffle=True)
    c = correct_t.shape[0]
    d = correct_val.shape[0]
    correct_t = correct_t.reshape(c,b)
    correct_val = correct_val.reshape(d,b)
    error_t = error_t.reshape(c,b)
    error_val = error_val.reshape(d,b)
    return train_t , train_val ,correct_t , correct_val , error_t , error_val

f = 40532
v = 4504
embedding_layer = Embedding(
    input_dim=len(word_index) + 1,
    output_dim=300,
    weights=[embeddings_matrix],
    input_length=246,
    trainable=True)

question = Input(shape=(246,39), dtype='float32')
biLSTM = Bidirectional(LSTM(128, return_sequences=True))(question)
biLSTM = Dropout(0.2)(biLSTM)

flat = Flatten()(biLSTM)
dense_question = Dense(20, activation='relu')(flat)



answer = Input(shape=(246,),dtype='int32')
answer_embedding = embedding_layer(answer)
biLSTM_a = Bidirectional(LSTM(128, return_sequences=True))(answer_embedding)
biLSTM_a = Dropout(0.2)(biLSTM_a)

attention = dot([biLSTM,biLSTM_a],axes=[1,1])
attention = Flatten()(attention)
attention = Dense(20,activation='relu')(attention)


cosine_sim = dot([dense_question, attention], normalize=True, axes=-1)
qa_model = Model([question, answer], [cosine_sim], name='qa_model')


correct_answer = Input(shape=(246,))
incorrect_answer = Input(shape=(246,))
correct_cos_sim = qa_model([question, correct_answer])
incorrect_cos_sim = qa_model([question, incorrect_answer])


def hinge_loss(cos_sims, margin=0.2):
    correct, incorrect = cos_sims
    return K.relu(margin - correct + incorrect)


contrastive_loss = Lambda(hinge_loss)([correct_cos_sim, incorrect_cos_sim])
contrastive_model = Model([question, correct_answer, incorrect_answer], [contrastive_loss], name='contrastive_model')
prediction_model = Model([question, answer], qa_model([question, answer]), name='prediction_model')


contrastive_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')
prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='adam')

y_train_dummy = np.zeros(shape=f)
y_val_dummy = np.zeros(shape=v)

for i in range(25):
    print(i)
    train_t , train_val ,correct_t , correct_val , error_t , error_val=data_split(train,correct_train)
    history = contrastive_model.fit(
            x=[train_t, correct_t, error_t],
            y=y_train_dummy,
            batch_size=256,
            epochs=2,
            validation_data=([train_val, correct_val, error_val], y_val_dummy))
    
f=0
ans=[]
for i in range(0,8000,4):
    temp = test_data[f].reshape(1,246,39)
    temp_1 = []
    temp_2 = test_answer[i:i+4]
    for j in range(4):
        temp_3 = temp_2[j].reshape(1,246)
        d = prediction_model.predict([temp,temp_3])[0][0]
        temp_1.append(d)
    answer = np.argmax(temp_1)
    ans.append(answer)
    f+=1
    if f%500 ==0:
        print(f)

matrix = {}
matrix['answer'] = ans
matrix['id'] = [i for i in range(1,2001)]
df  = pd.DataFrame(matrix)
df = df [['id','answer']]
df.to_csv(sys.argv[5],index=False)
