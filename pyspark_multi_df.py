# -*- coding: utf-8 -*-

import os, sys, timeit
import base64
import numpy as np
import pandas as pd
import pyspark
import logging

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.Logger('Main')

try:
    sc = pyspark.SparkContext(appName="Test")
except Exception as err:
    logger.warning(err)
    
sql = pyspark.SQLContext(sc)
    
rdd = sc.parallelize([
    ('A', 1),
    ('B', 5),
    ('A', 6),
    ('B', 9),
    ('A', 2)
])

def func(x):
    key = x[0]
    value = x[1]
    print(value)
    arr1 = [
            ["Movies" , np.array([1.0, 2.5], dtype=np.float32)],
            ["Sports" , np.array([1.0, 2.5], dtype=np.float32)],
            ["Coding" , np.array([1.0, 2.5], dtype=np.float32)],
            ["Fishing", np.array([1.0, 2.5], dtype=np.float32)],
            ["Dancing", np.array([1.0, 2.5], dtype=np.float32)],
            ["cooking", np.array([1.0, 2.5], dtype=np.float32)]
    ]
    arr2 = [
            ["AAA", 6590],
            ["BBB", 8054]
    ]
    
    df1 = pd.DataFrame(arr1, columns = ["name", "num"])
    df2 = pd.DataFrame(arr2, columns = ["name", "num"])
    
    #df1['num'] = df1['num'].map(lambda x: bytearray(x))
    df1['num'] = df1['num'].map(lambda x: base64.b64encode(x).decode('ascii'))
    
    d1 = df1.to_dict(orient='records')
    d2 = df2.to_dict(orient='records')
    
    return [('df1', d1), ('df2', d2)]

############
# Method 1 #
############
timer_start = timeit.default_timer()

rdd_1 = rdd.groupByKey()\
        .flatMap(lambda x: func(x))\
        .reduceByKey(lambda x,y: x+y)\
        .persist()

df1 = rdd_1.filter(lambda kv: kv[0] == 'df1').flatMap(lambda x: x[1]).toDF()
df2 = rdd_1.filter(lambda kv: kv[0] == 'df2').flatMap(lambda x: x[1]).toDF()

df1.show()
df2.show()

timer_end = timeit.default_timer()
print(timer_end - timer_start)

############
# Method 2 #
############
timer_start = timeit.default_timer()

rdd_2 = rdd.groupByKey()\
        .flatMap(lambda x: func(x))\
        .persist()

df1 = rdd_2.filter(lambda kv: kv[0] == 'df1').flatMap(lambda x: x[1]).toDF()
df2 = rdd_2.filter(lambda kv: kv[0] == 'df2').flatMap(lambda x: x[1]).toDF()

df1.show()
df2.show()

timer_end = timeit.default_timer()
print(timer_end - timer_start)

pdf = df1.toPandas()
#a = pdf['num'].map(lambda x: np.frombuffer(x, np.float32))
