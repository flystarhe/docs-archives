title: sklearn cross validation
date: 2017-08-29
tags: [Python]
---
学习预测函数的参数并在相同的数据上进行测试是一个方法上的错误：只会重复其刚刚看到的样本的标签的模型将具有完美的分数，但无法预测任何有用的，看不见的数据。这种情况称为过度配合。为了避免这种情况，通常的做法是将部分可用数据作为测试集。

<!--more-->
## 测试数据
```
import pandas as pd
import numpy as np

np.random.seed(1234)
df_tmp = pd.DataFrame(np.random.randint(0,9,(9,4)), columns=list('ABCD'))
```

## Train/Test
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_tmp[['A','B','C']], df_tmp[['D']], test_size=0.2, random_state=0)
```

## ShuffleSplit
```
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train, test in ss.split(df_tmp):
    print("%s %s" % (train, test))
```

    [4 8 6 3 0 5] [7 2 1]
    [6 8 0 4 2 5] [3 1 7]
    [4 1 0 8 7 6] [5 2 3]

## KFold
```
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train,test in kf.split(df_tmp):
    print("%s %s" % (train, test))
```

    [2 3 4 5 6 7 8] [0 1]
    [0 1 4 5 6 7 8] [2 3]
    [0 1 2 3 6 7 8] [4 5]
    [0 1 2 3 4 5 8] [6 7]
    [0 1 2 3 4 5 6 7] [8]

## LeaveOneOut
```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train, test in loo.split(df_tmp):
    print("%s %s" % (train, test))
```

    [1 2 3 4 5 6 7 8] [0]
    [0 2 3 4 5 6 7 8] [1]
    [0 1 3 4 5 6 7 8] [2]
    [0 1 2 4 5 6 7 8] [3]
    [0 1 2 3 5 6 7 8] [4]
    [0 1 2 3 4 6 7 8] [5]
    [0 1 2 3 4 5 7 8] [6]
    [0 1 2 3 4 5 6 8] [7]
    [0 1 2 3 4 5 6 7] [8]

## LeavePOut
```
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
for train, test in lpo.split(df_tmp):
    print("%s %s" % (train, test))
```

    [2 3 4 5 6 7 8] [0 1]
    [1 3 4 5 6 7 8] [0 2]
    [1 2 4 5 6 7 8] [0 3]
    [1 2 3 5 6 7 8] [0 4]
    [1 2 3 4 6 7 8] [0 5]
    [1 2 3 4 5 7 8] [0 6]
    [1 2 3 4 5 6 8] [0 7]
    [1 2 3 4 5 6 7] [0 8]
    [0 3 4 5 6 7 8] [1 2]
    [0 2 4 5 6 7 8] [1 3]
    [0 2 3 5 6 7 8] [1 4]
    [0 2 3 4 6 7 8] [1 5]
    ..

## 参考资料:
- [General examples](http://scikit-learn.org/stable/auto_examples/index.html)
- [Cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html)