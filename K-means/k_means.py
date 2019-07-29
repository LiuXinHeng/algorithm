#-*- coding: utf-8 -*-
#使用K-Means算法提取用户用电特征曲线

from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlwt

iteration = 500 #聚类最大循环次数

##显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


##保存文件函数
def save(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
            f.save(path)
            

data = pd.read_excel('jan_data.xlsx',index_col = 'Id') #读取数据


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





##简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data_zs.columns) + [u'类别数目'] #重命名表头
print(r)

##详细输出原始数据及其类别
r = pd.concat([data_zs, pd.Series(model.labels_, index = data_zs.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(data_zs.columns) + [u'聚类类别'] #重命名表头
r.to_excel('out1.xlsx') #保存结果





##画出聚类结果
x = np.linspace(1,24,n)
y = model.cluster_centers_

b = ['第一类用电特征曲线','第二类用电特征曲线','第三类用电特征曲线','第四类用电特征曲线','第五类用电特征曲线','第六类用电特征曲线','第七类用电特征曲线','第八类用电特征曲线']
for i in range(0,k):
    plt.plot(x,y[i],label=b[i]);

plt.xlabel("时刻")
plt.title("用电特征曲线")
plt.legend()
plt.show()






