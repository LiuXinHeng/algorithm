#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# apk add py-mysqldb or


import datetime
import time
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR
from sqlalchemy import inspect
import pandas as pd
import traceback




    


def engine():
    engine = create_engine('oracle://dec:dec@192.168.100.104:1521/pvdb',encoding='utf-8', echo=True)
    return engine


#定义通用方法函数，插入数据库表，并创建数据库主键，保证重跑数据的时候索引唯一。
def insert_db(data, table_name):
    # 定义engine
    #engine = engine()
    # 使用 http://docs.sqlalchemy.org/en/latest/core/reflection.html5
    # 使用检查检查数据库表是否有主键。
    insp = inspect(engine)
    data.to_sql(name=table_name, con=engine, schema=common.MYSQL_DB, if_exists='append')
   





# main函数入口
if __name__ == '__main__':
    print('complete!')













