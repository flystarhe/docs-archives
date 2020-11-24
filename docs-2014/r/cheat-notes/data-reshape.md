title: R入门小抄 | reshape2
date: 2015-05-20
tags: [R]
---
reshape2.

<!--more-->
```
install.packages("reshape2")
library(reshape2)
data = airquality[1:6,]
```

      Ozone Solar.R Wind Temp Month Day
    1    41     190  7.4   67     5   1
    2    36     118  8.0   72     5   2
    3    12     149 12.6   74     5   3
    4    18     313 11.5   62     5   4
    5    NA      NA 14.3   56     5   5
    6    28      NA 14.9   66     5   6

```
data1 = melt(data, id.vars=c('Month','Day'),
        measure.vars=c('Ozone','Solar.R','Wind'),
        variable.name="variable",
        value.name="value")
```

       Month Day variable value
    1      5   1    Ozone  41.0
    2      5   2    Ozone  36.0
    3      5   3    Ozone  12.0
    4      5   4    Ozone  18.0
    5      5   5    Ozone    NA
    6      5   6    Ozone  28.0
    7      5   1  Solar.R 190.0
    8      5   2  Solar.R 118.0
    9      5   3  Solar.R 149.0
    10     5   4  Solar.R 313.0
    11     5   5  Solar.R    NA
    12     5   6  Solar.R    NA
    13     5   1     Wind   7.4
    14     5   2     Wind   8.0
    15     5   3     Wind  12.6
    16     5   4     Wind  11.5
    17     5   5     Wind  14.3
    18     5   6     Wind  14.9

```
data2 = dcast(data1, Month + Day ~ variable, value.var='value')
```

      Month Day Ozone Solar.R Wind
    1     5   1    41     190  7.4
    2     5   2    36     118    8
    3     5   3    12     149 12.6
    4     5   4    18     313 11.5
    5     5   5  <NA>    <NA> 14.3
    6     5   6    28    <NA> 14.9

## 虚拟变量
在R语言中对包括分类变量(factor)的数据建模时，一般会将其自动处理为虚拟变量或哑变量(dummy variable)。

但有一些特殊的函数，如neuralnet包中的neuralnet函数就不会预处理。如果直接将原始数据扔进去，会出现需要“数值/复数矩阵/矢量参数”错误。这个时候，除了将这些变量删除，我们只能手动将factor转换为取值(0,1)的虚拟变量。

所用的函数一般有`model.matrix()`或`nnet`中的`class.ind()`：
```r
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data","./german.data")
data <- read.table("./german.data")
str(data)
```

使用`model.matrix()`：
```r
dummyV1 <- model.matrix(~V1, data)
head(dummyV1)
```

输出为：
```r
  (Intercept) V1A12 V1A13 V1A14
1           1     0     0     0
2           1     1     0     0
3           1     0     0     1
4           1     0     0     0
5           1     0     0     0
6           1     0     0     1
```

使用`nnet`中的`class.ind()`：
```r
data = data.frame(x1=c('f','f','b','b','c','c'),x2=c(1,2,3,4,5,6))

install.packages('nnet')
library(nnet)

tmp = class.ind(data$x1)
dimnames(tmp)[[1]] = 1:dim(tmp)[1]
dimnames(tmp)[[2]] = paste('var_',dimnames(tmp)[[2]],sep='')

cbind(data,tmp)
```

输出为：
```r
  x1 x2 var_b var_c var_f
1  f  1     0     0     1
2  f  2     0     0     1
3  b  3     1     0     0
4  b  4     1     0     0
5  c  5     0     1     0
6  c  6     0     1     0
```
