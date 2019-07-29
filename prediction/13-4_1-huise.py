#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GM11 import GM11 #引入自己编写的灰色预测函数

from keras.models import Sequential
from keras.layers.core import Dense, Activation



data = pd.read_csv('data1.csv') #读取数据
data.index = range(1994, 2014)

data.loc[2014] = None
data.loc[2015] = None
l = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
for i in l:
  f = GM11(data[i][0:-2].values)[0]
  data[i][2014] = f(len(data)-1) #2014年预测结果
  data[i][2015] = f(len(data)) #2015年预测结果
  data[i] = data[i].round(2) #保留两位小数

data.index.name = 'Id'
data[l+['y']].to_excel('data1_GM11.xls') #结果输出



feature = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7'] #特征所在列
data = pd.read_excel('data1_GM11.xls',index_col = 'Id')
data_train = data.loc[range(1994,2014)].copy() #取2014年前的数据建模

data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化

x_train = data_train[feature].values #特征数据
y_train = data_train['y'].values

model = Sequential() #建立模型
model.add(Dense(12, input_dim= 6))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(1, input_dim= 12 ))
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型
##model.fit(x_train, y_train, nb_epoch = 1000, batch_size = 16) #训练模型，学习一万次
##model.save_weights('1-net.model') #保存模型参数

model.load_weights('1-net.model')

x = ((data[feature] - data_mean[feature])/data_std[feature]).values
data[u'y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
data.to_excel('revenue.xls')

p = data[['y','y_pred']].plot(subplots = True, style=['b-o','r-*'])
plt.show()
