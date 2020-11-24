title: Pandas学习笔记
date: 2017-08-28
tags: [Python,Pandas]
---
Pandas是一个开源的BSD许可证库，为Python编程语言提供了高性能，易于使用的数据结构和数据分析工具。

<!--more-->
## DataFrame
```
import pandas as pd
import numpy as np

np.random.seed(1234)
df_a = pd.DataFrame({"A":np.random.randn(4),
                     "Z":np.random.randint(1,9,(4,)),
                     "B":pd.Timestamp("20170309"),
                     "C":pd.Series(1,index=list(range(4)),dtype="float32"),
                     "D":pd.Series([1,2,np.nan,4]),
                     "E":np.array([3]*4,dtype="int32"),
                     "F":pd.Categorical(["test","train","test","train"]),
                     "G":"foo"},columns=list('AZBCDEFG'))
```

`df_a.dtypes`:

    A           float64
    Z             int64
    B    datetime64[ns]
    C           float32
    D           float64
    E             int32
    F          category
    G            object

`df_a.index`:

    Int64Index([0, 1, 2, 3], dtype='int64')

`df_a.columns`:

    Index(['A', 'Z', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='object')

`df_a.values`:

    array([[0.47143516373249306, 8, Timestamp('2017-03-09 00:00:00'), 1.0, 1.0, 3, 'test', 'foo'],
           [-1.1909756947064645, 2, Timestamp('2017-03-09 00:00:00'), 1.0, 2.0, 3, 'train', 'foo'],
           [1.4327069684260973, 8, Timestamp('2017-03-09 00:00:00'), 1.0, nan, 3, 'test', 'foo'],
           [-0.3126518960917129, 2, Timestamp('2017-03-09 00:00:00'), 1.0, 4.0, 3, 'train', 'foo']], dtype=object)

## Selection

| id  | Operation                      | Syntax        | Result    |
| --- | :----------------------------- | :------------ | :-------- |
| 0   | Select column                  | df[col]       | Series    |
| 1   | Select row by label            | df.loc[label] | Series    |
| 2   | Select row by integer location | df.iloc[loc]  | Series    |
| 3   | Slice rows                     | df[5:10]      | DataFrame |
| 4   | Select rows by boolean vector  | df[bool_vec]  | DataFrame |

```
df_a['B']
df_a[['A','C']]

df_a.loc[:,'C']
df_a.loc[3,:]

df_a.iloc[:,2]
df_a.iloc[3,:]
df_a.iloc[3,:2]
df_a.iloc[3,[0,2]]

df_a[(df_a['Z']>1) & (df_a['Z']<9)]
df_a[(df_a['Z']<4) | (df_a['Z']>5)]
```

## Update

```
df_a['D'] = [4,3,2,1]
df_a['D'] = pd.Series([4,3,2,1])

#注意：F列是Category类型，不接受train|test之外的值
df_a.loc[2,:] = [91.0,91,pd.Timestamp('2017-02-09 00:00:00'),91.0,91.0,91,'test','91']
```

## Delete

```
#axis=行:0|列:1，默认是行
df_a.drop('C',axis=1)

#axis=行:0|列:1，默认是行
df_a.drop(0,axis=0)
```

## null/na/unique

```
a = pd.DataFrame({'a':[1,1,3,1],'b':[101,102,103,101]},index=list('abcd'))
a.dropna(how='any')  # 去掉包含缺失值的行
a.fillna(value=5)  # 对缺失值进行填充
a.isnull()  # 对数据进行布尔填充
a['a'].notnull()
a.drop_duplicates(['a'])
```

## index/columns

```
a = pd.DataFrame({'a':[1,1,3,1],'b':[101,101,103,101]},index=list('abcd'))
a.rename(columns={'a':'col_a'}, inplace=True)
a.reset_index(drop=True,inplace=True)
```

## groupby

```
df_tmp = pd.DataFrame({'a':[1,2,1,2],'b':[1,2,3,4],'c':['a','a','b','b']})
for sub_i in df_tmp.groupby(['a','c']):
    sub_i[0]
    sub_i[1]

df_tmp.groupby('c').agg([('Count','count')])
```

## append

```
>>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
>>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
>>> df1.append(df2)
   A  B
0  1  2
1  3  4
0  5  6
1  7  8
>>> df1.append(df2, ignore_index=True)
   A  B
0  1  2
1  3  4
2  5  6
3  7  8
>>> pd.concat([df1, df2], ignore_index=True)
```

## merge
参考[Merge, join, and concatenate](http://pandas.pydata.org/pandas-docs/stable/merging.html):
```
>>> df1 = pd.DataFrame({'key': list('aabb'), 'data1': range(4)}, index=[0,1,2,3])
>>> df2 = pd.DataFrame({'key': list('bbcc'), 'data2': range(4)}, index=[0,1,2,3])
>>> pd.merge(df1, df2)
   data1 key  data2
0      2   b      0
1      2   b      1
2      3   b      0
3      3   b      1
>>> pd.merge(df1, df2, on=['key'])
   data1 key  data2
0      2   b      0
1      2   b      1
2      3   b      0
3      3   b      1
>>> pd.merge(df1, df2, left_on=['key'], right_on=['key'])
   data1 key  data2
0      2   b      0
1      2   b      1
2      3   b      0
3      3   b      1
>>> pd.merge(df1, df2, how='left')
   data1 key  data2
0      0   a    NaN
1      1   a    NaN
2      2   b    0.0
3      2   b    1.0
4      3   b    0.0
5      3   b    1.0
>>> pd.merge(df1, df2, left_index=True, right_index=True)
   data1 key_x  data2 key_y
0      0     a      0     b
1      1     a      1     b
2      2     b      2     c
3      3     b      3     c
```

## nlargest

```
df = pd.DataFrame({'a':[1,10,8,11,8],'b':list('abdce'),'c':[1.0,2.0,np.nan,3.0,4.0]})
df.nlargest(7,['a','c'])

tmp = df.nlargest(10,['a','c']).index.unique()
df.loc[tmp]
```

## io
[io-read-csv-table](http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table):
```
df.to_csv('_tmp/data_foo.csv', index=False, encoding="utf-8")
pd.read_csv('_tmp/data_foo.csv', encoding="utf-8")
```

[io-excel](http://pandas.pydata.org/pandas-docs/stable/io.html#io-excel):
```
df.to_excel('_tmp/data_foo.xlsx', sheet_name='Sheet1')
pd.read_excel('_tmp/data_foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

[io-sql](http://pandas.pydata.org/pandas-docs/stable/io.html#io-sql):
```
import pymysql
import pandas as pd

db_host = 'host'
db_port = 3306
db_user = 'user name'
db_pass = 'pass word'
db_dbnm = 'database'

from sqlalchemy import create_engine
link = '%s:%s@%s:%d/%s?charset=utf8' % (db_user, db_pass, db_host, db_port, db_dbnm)
conn = create_engine('mysql+pymysql://' + link, encoding="utf8")

# write
df = pd.DataFrame({"id":[1,2,3],"name":['大','数','据']})
df.to_sql('tmp_flystar_test', conn, if_exists='replace', index=False, chunksize=1000)

# read
sql = "select * from tmp_flystar_test"
df = pd.read_sql(sql, conn)
```

## plot
```
%pylab inline
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000',periods=1000))
ts = ts.cumsum()
ts.plot(kind='line');
```

```
df = pd.DataFrame(np.random.randn(1000,4), index=pd.date_range('1/1/2000',periods=1000), columns=list('ABCD'))
df = df.cumsum()
df.plot(kind='line');
```

## 参考资料:
- [10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
- [Comparison with R](http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html)
- [Intro to Data Structures](http://pandas.pydata.org/pandas-docs/stable/dsintro.html)