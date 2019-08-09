#-*- coding: utf-8 -*-
#使用K-Means算法提取用户用电特征曲线

from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np
import cx_Oracle
import os
import csv
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR
from sqlalchemy import inspect
import traceback


os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def engine():
    engine = create_engine('oracle://dec:dec@192.168.100.104:1521/pvdb',encoding='utf-8', echo=True)
    return engine


#定义通用方法函数，插入数据库表。
def insert_db(data, table_name):
    
    data.to_sql(name=table_name, con=engine, if_exists='append', index = False)


##yc_power_sample 用采功率数据的获取          
def get_data(data1, data2, home_no):

    data1 = '\''+ data1 + '\''
    data2 = '\''+ data2 + '\''
    home_no = '\''+ home_no + '\''

    
    #连接数据库后获取数据
    conn = cx_Oracle.connect('dec','dec','192.168.100.104/pvdb')
    cr = conn.cursor()
    
    sql = 'select * from YC_POWER_SAMPLE where DATA_DATE between ' + data1 + ' and ' + data2 + ' and CONS_NO = ' + home_no
    yc_power_data = pd.read_sql(sql, conn)
    
    yc_power_data = yc_power_data[ (yc_power_data['P1'] > 0) & (yc_power_data['P80'] > 0)]
    
    data_no = pd.DataFrame(( yc_power_data.T[0:7]).T)
    data_no.index.name = 'ID'
    data_no.to_csv('data_no.csv')
    
    #对数据进行预处理，选取与用电数据有关的列，对其中的Nan数据用0值填充，去除一行中全为0的数据
    yc_power_data = ( yc_power_data.T[7:103] ).T
    yc_power_data = yc_power_data.fillna(0)
    yc_power_data.index.name = 'ID'
    yc_power_data.to_csv('yc_power_data.csv')

    sql = 'select * from mode_ele_stat_day '
    mode_data = pd.read_sql(sql, conn)
    mode_data.to_csv('modedata.csv',index = None)
                              
    

def K_means():
    
    iteration = 500 #聚类最大循环次数
    data = pd.read_csv('yc_power_data.csv',index_col = 'ID') #读取数据

    data_zs = data #打印结果用这个变量

    data= data.values#将dataframe转换成array类型
    data1 = data #保存归一化后的数据


    data_min = data.min(axis = 1)#找出每一行的最小值
    data_max = data.max(axis = 1)#找出每一行的最大值

    m,n = data.shape#获取行和列  行代表数据对象个数 列代表数据维度

    #数据标准化
    for i in range(0,m):
        for j in range(0,n):
            data1[i,j] = ( data[i,j] - data_min[i]) / ( data_max[i] -  data_min[i] )


    a = []
    for i in range(2,8):##根据chi指标确定最优聚类数 chi值越大，聚类效果越好
        kmeans_model = KMeans(n_clusters=i).fit(data1)
        labels = kmeans_model.labels_
        a.append(metrics.calinski_harabasz_score(data1, labels) )
    
    k = a.index(max(a))#找出CHI指标最大值对应的索引

    k = 2 + k ##根据CHI指标确定最优K值

    model = KMeans(n_clusters = k, max_iter = iteration) #分为k类
    model.fit(data1) #开始聚类

    data_no1 = pd.read_csv('data_no.csv', index_col = 'ID')
    
    
    mode_data = pd.read_csv('modedata.csv')
    
    r1 = []
    for i in range(1,k+1):
        r1.append(i)
    r1 = pd.Series(r1)
    
    r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
    r3 = (data_no1[0:k]).T[0:1].T
    r = pd.concat([r1,r3,r2], axis = 1)
    r.columns =  mode_data.columns[0:98]
    r.to_csv('result.csv',index = None )


    
   
    

# main 函数入口
if __name__ == '__main__':
    get_data('2017/11/20', '2017/11/26', '6623660203')
    K_means()
    data1 = pd.read_csv("result.csv",index_col = None,encoding = 'gbk')
    data1 = data1[2:3]
    engine = engine()
    insert_db(data1, 'mode_ele_stat_day')
    print('completed!')



