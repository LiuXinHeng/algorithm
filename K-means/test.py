import pandas as pd

data = pd.read_csv('yc_power_data.csv',index_col = 'ID') #读取数据
print(data.columns)
