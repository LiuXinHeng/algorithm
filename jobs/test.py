# -*- coding: utf-8 -*-


import sys
sys.path.append('../libs/')

import common
import pandas as pd
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
engine = common.engine()

##data = pd.read_csv('1.csv', encoding = 'gbk')
##
##
##data.to_sql(name='YC_POWER_SAMPLE', con=engine, if_exists='append')
