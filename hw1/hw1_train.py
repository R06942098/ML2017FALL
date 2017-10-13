import pandas as pd
import numpy as np
import math
df = pd.read_csv('train.csv',encoding='big5')
df=df.replace('NR',0)
df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']] = df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].astype('float')
pm25_1 = df[df['測項'] == 'PM2.5'].drop(['測項','測站'],axis=1)
target_1 = df[df['測項'] == 'PM2.5'].drop(['測項','測站','日期'],axis=1)
target  = target_1.astype('float')
y_1 = target.values.reshape(-1)
for i in range(0,5760,480):
    target0 = y_1[9+i:480+i][np.newaxis,:]
    if i == 0:
        y = target0
        continue
    y = np.concatenate((y,target0),axis=0)
y = y.reshape(-1)
df= df.drop(['日期','測項','測站'],axis=1)
df1 = df.values
for i in range(0,4303,18) :
        matrix = df1[i:i+18,:]
        if i == 0:
                matrix1 = matrix 
                continue
        matrix1 = np.concatenate((matrix1,matrix),axis=1)

for i in range(0,5760,480):
    for j in range(471):
        train_data0 = matrix1[0:18,i+j:i+j+9].reshape(-1)[np.newaxis,:]
        if i == 0 and j==0:
            train_data1 = train_data0
            continue
        train_data1 = np.concatenate((train_data1,train_data0),axis=0)
train_data2 = np.square(train_data1)
train_data1 = np.concatenate((train_data1,train_data2),axis=1)
train_cross1 = train_data1[0:2875,:]
train_cross2 = train_data1[:,63:]
train_cross = np.insert(train_cross1,0,np.ones(2875),axis=1)
y_corss = y[0:2875]
train_data = np.insert(train_data1,0,np.ones(5652),axis=1)
lamba = 0.0001
weight=np.zeros(325)
repeat = 300000
l_rate=10
s_gra = np.zeros(325)
for i in range(repeat):
    hypo = np.dot(train_cross,weight)
    loss = hypo - y_corss
    temp1 = weight[1:]
    temp = np.concatenate((np.zeros(1),temp1),axis=0)
    gra = np.dot(train_cross.transpose(),loss) + 2*lamba*temp
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    weight = weight - l_rate * gra/ada


for i in range(repeat):
    hypo = np.dot(train_data,weight)
    loss = hypo - y
    gra = np.dot(train_data.transpose(),loss)
    temp1 = weight[1:]
    temp = np.concatenate((np.zeros(1),temp1),axis=0)+2*lamba*temp
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    weight = weight - l_rate * gra/ada
