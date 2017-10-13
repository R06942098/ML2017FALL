import pandas as pd
import numpy as np
import math
import sys
weight = np.load('Model.npy')
fd = pd.read_csv(sys.argv[1],encoding = 'big5')
fd=fd.replace('NR',0)
a =fd.values
b=np.array(['id_0','AMB_TEMP','15','14','14','13','13','13','13','13','12'])
c =np.insert(a,0,b,axis=0)
df2=pd.DataFrame(c,columns=['0','1','2','3','A','B','C','D','E','F','G'])
df2[['2','3','A','B','C','D','E','F','G']]=df2[['2','3','A','B','C','D','E','F','G']].astype('float')
df2 = df2.drop('0',axis=1)
df2 = df2.drop('1',axis=1)
df2 =df2.values
for i in range(0,4313,18) :
        matrix = df2[i:i+18,0:9]
        if i == 0:
                matrix1 = matrix 
                continue
        matrix1 = np.concatenate((matrix1,matrix),axis=1)


for i in range(0,2152,9):
    test_data0 = matrix1[0:18,i:i+9].reshape(-1)[np.newaxis,:]
    if i == 0:
        test_data1 = test_data0
        continue
    test_data1 = np.concatenate((test_data1,test_data0),axis=0)
test_data2 = np.square(test_data1)
test_data1 = np.concatenate((test_data1,test_data2),axis=1)
test_data = np.insert(test_data1,0,np.ones(240),axis=1)
y_hat = test_data.dot(weight)
List=[]
for i in range(240):
	List.append('id_'+str(i))
Value=[]
for i in range(240):
    Value.append(y_hat[i])
outmatrix = {}
outmatrix['id'] = List
outmatrix['value'] = Value
df = pd.DataFrame(outmatrix)
df.to_csv(sys.argv[2],index=False)
