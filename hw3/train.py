# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:34:22 2017

@author: USER
"""
import pandas as pd 
import numpy as np
import sys 
from keras.models import Sequential
from keras.layers.core import Dense , Dropout , Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal ,TruncatedNormal ,Constant
from keras.optimizers import SGD ,Adam ,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1,l2
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint ,TensorBoard
from keras.layers.noise import GaussianNoise
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
from keras.utils import plot_model
from skimage import data,exposure,img_as_float
from skimage import exposure
import matplotlib.pyplot as plt

t = pd.read_csv(sys.argv[1])
t = t.values
t_data = t[:,1]
t_label = t[:,0]

def todim48(train):
    j=0
    for i in range(len(t_data)):
        a = t_data[i].split()
        a = np.array(a,dtype='float64')
        temp = a[np.newaxis,:]
        if j == 0 :
            test_data = temp
            j+=1
            continue
        test_data = np.concatenate((test_data,temp),axis=0)
        print(j)
        j +=1
def todim7(label):
    label = label.reshape(-1)
    j=0
    for i in label:
        a = np.zeros(7)
        a[i] = 1
        temp = a[np.newaxis,:]
        if  j==0 :
            label_data = temp
            j+=1
            continue
        label_data = np.concatenate((label_data,temp),axis=0)
        print(j)
        j+=1
    return label_data
train_data = todim48(t_data)
train_data = train-data.reshape(28709,48,48,1)
train_label=todim7(t_label)

t_data, v_data,t_label , v_label  = train_test_split(train_data,train_label, test_size=0.2, random_state=42,shuffle=True)
t_data /= 255
v_data /= 255


model = Sequential()

##layer1
model.add(Conv2D(filters=50,kernel_size=(3,3),padding='same',activation='relu',input_shape = (48,48,1),kernel_initializer='he_normal',name='conv_layer1'))
model.add(BatchNormalization(name='normal_1'))
model.add(Conv2D(filters=50,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal',name='conv_layer2'))
model.add(BatchNormalization(name='normal_2'))
model.add(MaxPooling2D((2,2),name='pooling_1'))
model.add(Dropout(0.3,name='dropout_layer_1'))


#cnn layer2 
model.add(Conv2D(filters=100,kernel_size=(3,3),padding='same',kernel_initializer='he_normal',activation='relu',name='conv_layer3'))
model.add(BatchNormalization(name='normal_3'))
model.add(Conv2D(filters=100,kernel_size=(3,3),padding='same',kernel_initializer='he_normal',activation='relu',name='conv_layer4'))
model.add(BatchNormalization(name='normal_4'))
model.add(MaxPooling2D((2,2),name='pooling_2'))
model.add(Dropout(0.3,name='dropout_layer_2'))

##layer 3
model.add(Conv2D(filters=150,kernel_size=(3,3),padding='same',kernel_initializer='he_normal',activation='relu',name='conv_layer5'))
model.add(BatchNormalization(name='normal_5'))
model.add(Conv2D(filters=150,kernel_size=(3,3),padding='same',kernel_initializer='he_normal',activation='relu',name='conv_layer6'))
model.add(BatchNormalization(name='normal_6'))
model.add(MaxPooling2D((2,2),name='pooling_3'))
model.add(Dropout(0.3,name='dropout_layer_3'))

model.add(Flatten(name='flatten'))

model.add(Dense(512,activation='relu',kernel_initializer='he_normal',name='fully_connected_layer1'))
model.add(BatchNormalization(name='normal_7'))
model.add(Dropout(0.4,name='dropout_layer_4'))

model.add(Dense(512,activation='relu',kernel_initializer='he_normal',name='fully_connected_layer2'))
model.add(BatchNormalization(name='normal_8'))
model.add(Dropout(0.5,name='dropout_layer_5'))         

model.add(Dense(units=7,activation='softmax',kernel_initializer='he_normal',name='perdiction'))
model.compile(loss = 'categorical_crossentropy',optimizer ='adam' ,
              metrics=['accuracy'])

model.summary()


datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2,zoom_range=0.2,shear_range=0.2)
datagen.fit(t_data)

s = datagen.flow(t_data,t_label,batch_size=256)
history = model.fit_generator(s,steps_per_epoch=1000, epochs=35,validation_data=(v_data,v_label),\
                     callbacks=[ModelCheckpoint('test.h5',monitor='val_acc',save_best_only=True)])
