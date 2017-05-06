import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

arch = open('Dorothea/dorothea_train.data')
train_data = np.zeros([800,100000],dtype='int')

for i in range(800):
    linea = arch.readline()
    dat_lin = linea.split()
    dat = np.array(dat_lin,dtype='int')
    n = 0
    for j in range(100000):
        if (j == dat[n]):
            train_data[i,j] = 1
            n += 1
            if (n>=len(dat)):
                n = 0

arch = open('Dorothea/dorothea_train.labels')
train_labels = np.zeros([800,1],dtype='int')

for i in range(800):
    linea = arch.readline()
    dat_lin = linea.split()
    dat = np.array(dat_lin,dtype='int')
    train_labels[i] = dat[0]

arch = open('Dorothea/dorothea_valid.data')
valid_data = np.zeros([350,100000],dtype='int')

for i in range(350):
    linea = arch.readline()
    dat_lin = linea.split()
    dat = np.array(dat_lin,dtype='int')
    n = 0
    for j in range(100000):
        if (j == dat[n]):
            valid_data[i,j] = 1
            n += 1
            if (n>=len(dat)):
                n = 0

arch = open('Dorothea/dorothea_valid.labels')
valid_labels = np.zeros([350,1],dtype='int')

for i in range(350):
    linea = arch.readline()
    dat_lin = linea.split()
    dat = np.array(dat_lin,dtype='int')
    valid_labels[i] = dat[0]


N=10
est=50
predict=np.zeros(350)
ii=np.zeros(100000)
for n in range(N):
    rf = RandomForestClassifier(n_estimators=est)
    rf.fit(train_data,train_labels.T[0])

    predict += rf.predict(valid_data)
    ERR = 1 - np.sum(predict == valid_labels.T[0])/350.0

    sort = np.argsort(rf.feature_importances_)
    for i in range(len(sort)):
        ii[sort[i]]+=i                   

for p in range(len(predict)):
    if(predict[p]>=5):
        predict[p]=1
    else:
        predict[p]=0
                   
ERR = 1 - np.sum(predict == valid_labels.T[0])/350.0
print(ERR)
