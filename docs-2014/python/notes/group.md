title: Python分组计算
date: 2017-08-01
tags: [Python]
---
利用pandas库进行数据分组分析,最常用的方法有:groupby,pivot_table,crosstab,以下分别介绍.

<!--more-->
## 测试数据
```
import pandas as pd

df = {'A': ['foo','foo','foo','foo','foo','bar','bar','bar','bar'],
      'B': ['one','one','one','two','two','one','one','two','two'],
      'C': [1,    2,    2,    3,    3,    4,    5,    6,    7]}
df = pd.DataFrame(df)
```

## 分组: groupby
Pandas中最为常用和有效的分组函数.

### 按列分组
`groupby`函数生成的`g1`是中间分组变量,为`GroupBy`类型:
```python
g1 = df.groupby(['A'])
```

使用推导式`[x for x in g1]`显示分组内容:

    [('bar',      A    B  C
    5  bar  one  4
    6  bar  one  5
    7  bar  two  6
    8  bar  two  7),
     ('foo',      A    B  C
    0  foo  one  1
    1  foo  one  2
    2  foo  one  2
    3  foo  two  3
    4  foo  two  3)]

### 分组统计
在`g1`上应用`size`,`sum`,`count`等函数,能实现分组统计:
```python
g1.size()
## A
## bar    4
## foo    5
## dtype: int64
g1.sum()
##       C
## A      
## bar  22
## foo  11
g1.count()
##      B  C
## A        
## bar  4  4
## foo  5  5
```

### 应用: agg
```python
g1['C'].agg(['mean', 'sum'])
##      mean  sum
## A             
## bar   5.5   22
## foo   2.2   11
```

## 透视表: pivot_table
类似于Excel数据透视表:
```python
pd.pivot_table(df, values='C', index='A', columns='B',
               aggfunc=lambda x: len(x))
## B    one  two
## A            
## bar    2    2
## foo    3    2
```

## 交叉表: crosstab
按照指定的行和列统计分组频数,虽然`groupby`也可以实现,但是这个更方便:
```python
tmp = pd.crosstab(df['A'], df['B'], margins=True)
print(tmp)
```

输出为:

    B    one  two  All
    A                 
    bar    2    2    4
    foo    3    2    5
    All    5    4    9
