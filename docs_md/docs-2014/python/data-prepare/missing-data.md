title: 数据清洗之缺失值
date: 2017-07-19
tags: [数据清洗,缺失值]
---
本文首先介绍数据清洗任务两大基本对象,错误数据和缺失值.然后,对缺失值处理方法进行简单归纳.最后,使用Python进行缺失值处理练习.

<!--more-->
## 数据清洗

### 错误数据

- 脏数据或错误数据，比如`Age=-2`
- 数据不正确，`0`是真的`0`，还是代表缺失
- 数据不一致，比如一个单位是美元，一个是人民币
- 数据重复

### 缺失值处理

#### 缺失值`0%~20%`

- 连续变量使用均值或中位数填补
- 分类变量不需要填补，另算一类即可，或用众数填补

#### 缺失值`20%~80%`

- 填补方法同上
- 每个有缺失值的变量生成一个指示哑变量，参与建模

#### 缺失值`80%~`

- 每个有缺失值的变量生成一个指示哑变量，参与建模
- 原始变量不使用

## 缺失值处理实践(Python)
标记缺失值,删除缺失值,估算缺失值,支持缺失值的算法.

### 标记缺失值
``` python
def output(txt, val):
    print(txt)
    print(val)

import pandas as pd
import numpy as np
np.random.seed(1234)
dataset = pd.DataFrame({
    "f1":np.random.rand(100),
    "f2":np.random.randint(2, size=100),
    "f3":np.random.randint(2, size=100),
    "class":np.random.randint(2, size=100)
})

dataset.describe()  #统计摘要
dataset.head(6)  #前6行的数据

dataset[['class','f2','f3']]==0).sum()  #各列中零值的数量
dataset[['class','f2','f3']] = dataset[['class','f2','f3']].replace(0, np.NaN)  #将零值标记为NaN
dataset.isnull().sum()  #各列中缺失值的数量
```

在Python中，特别是Pandas，NumPy和Scikit-Learn，我们将缺失值标记为NaN。sum，count等操作将忽略NaN。

### 删除缺失值
``` python
def output(txt, val):
    print(txt)
    print(val)

import pandas as pd
import numpy as np
np.random.seed(1234)
dataset = pd.DataFrame({
    "f1":np.random.rand(100),
    "f2":np.random.randint(2, size=100),
    "f3":np.random.randint(2, size=100),
    "class":np.random.randint(2, size=100)
})

dataset[['class','f2','f3']] = dataset[['class','f2','f3']].replace(0, np.NaN)  #将零值标记为NaN

dataset.shape  #维度信息
dataset.dropna(inplace=True)
dataset.shape  #维度信息
```

### 估算缺失值
``` python
def output(txt, val):
    print(txt)
    print(val)

import pandas as pd
import numpy as np
np.random.seed(1234)
dataset = pd.DataFrame({
    "f1":np.random.rand(100),
    "f2":np.random.randint(2, size=100),
    "f3":np.random.randint(2, size=100),
    "class":np.random.randint(2, size=100)
})

dataset[['class','f2','f3']] = dataset[['class','f2','f3']].replace(0, np.NaN)  #将缺失值标记为NaN

dataset.fillna(dataset.mean(), inplace=True)  #fillna --用平均值替换缺失值
```

### 支持缺失值的算法
当缺少数据时，并不是所有的算法都会失效。有一些可以灵活对待缺失值的算法，例如KNN，当值缺失时，它可以将其不计入距离测量。另一些算法，例如分类和回归树，可以在构建预测模型时将缺失值看作唯一且不同的值。遗憾的是，决策树和KNN对于缺失值并不友好。不管怎样，如果你考虑使用其他算法（如XGBoost）或开发自己的执行，依然是一个选择。

## 参考资料：
- [如何使用Python处理Missing Data](http://www.dataguru.cn/article-11047-1.html)
