#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GM11 import GM11 #引入自己编写的灰色预测函数

##from keras.models import Sequential
##from keras.layers.core import Dense, Activation



data = pd.read_excel('exam.xlsx',index_col = 'Id') #读取数据
#data.index = range(1994, 2014)

data.loc[30] = None
data.loc[31] = None


l=[ 1 ]

for i in l:
  f = GM11(data[i][0:-2].values)[0]
  data[i][30] = f(len(data)-1) #2014年预测结果
  data[i][31] = f(len(data)) #2015年预测结果
  data[i] = data[i].round(2) #保留两位小数


data[l].to_excel('jan_data.xls') #结果输出




