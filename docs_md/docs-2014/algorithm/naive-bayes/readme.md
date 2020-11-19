title: 朴素贝叶斯(R)
date: 2017-05-22
tags: [朴素贝叶斯,R]
---
朴素贝叶斯分类的优势在于不怕噪声和无关变量，其Naive之处在于它假设各特征属性是无关的。而贝叶斯网络（Bayesian Network）则放宽了变量无关的假设，将贝叶斯原理和图论相结合，建立起一种基于概率推理的数学模型,对于解决复杂的不确定性和关联性问题有很强的优势。

<!--more-->
## 朴素贝叶斯

朴素贝叶斯算法仍然是流行的十大挖掘算法之一，该算法是有监督的学习算法，解决的是分类问题，如客户是否流失、是否值得投资、信用等级评定等多分类问题。该算法的优点在于简单易懂、学习效率高、在某些领域的分类问题中能够与决策树、神经网络相媲美。但由于该算法以自变量之间的独立（条件特征独立）性和连续变量的正态性假设为前提，就会导致算法精度在某种程度上受影响。接下来我们就详细介绍该算法的知识点及实际应用。

## R语言函数简介

R语言中的`klaR`包就提供了朴素贝叶斯算法实现的函数`NaiveBayes`，我们来看一下该函数的用法及参数含义：

```
NaiveBayes(formula, data, ..., subset, na.action=na.pass)
NaiveBayes(x, grouping, prior, usekernel=FALSE, fL=0, ...)
```

- `formula`指定参与模型计算的变量，以公式形式给出，类似于`y=x1+x2+x3`；
- `data`用于指定需要分析的数据对象；
- `na.action`指定缺失值的处理方法，默认情况下不将缺失值纳入模型计算，也不会发生报错信息，当设为`na.omit`时则会删除含有缺失值的样本；
- `x`指定需要处理的数据，可以是数据框形式，也可以是矩阵形式；
- `grouping`为每个观测样本指定所属类别；
- `prior`可为各个类别指定先验概率，默认情况下用各个类别的样本比例作为先验概率；
- `usekernel`指定密度估计的方法（在无法判断数据的分布时，采用密度密度估计方法），默认情况下使用正态分布密度估计，设为TRUE时，则使用核密度估计方法；
- `fL`指定是否进行拉普拉斯修正，默认情况下不对数据进行修正，当数据量较小时，可以设置该参数为1，即进行拉普拉斯修正。

## R语言实战

本次实战内容的数据来自于UCI机器学习网站。Download: [Dota Folder](http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/), [Data Set Description](http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names)

```
myurl <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
mydat <- read.csv(myurl,header=FALSE,sep=",",fileEncoding="utf-8",stringsAsFactors=TRUE)
names(mydat) <- c('classes','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat')
str(mydat)
```

抽样，并将总体分为训练集和测试集：

```
set.seed(12)
index <- sample(1:nrow(mydat), size=0.8*nrow(mydat))
train <- mydat[index,]
test <- mydat[-index,]
```

原始数据中毒蘑菇与非毒蘑菇之间的比较比较接近，通过抽选训练集和测试集，发现比重与总体比例大致一样，故可认为抽样的结果能够反映总体状况，可进一步进行建模和测试：

```
prop.table(table(mydat$classes))
prop.table(table(train$classes))
prop.table(table(test$classes))
```

由于影响蘑菇是否有毒的变量有21个，可以先试着做一下特征选择，这里我们就采用随机森林方法（借助caret包实现特征选择的工作）进行重要变量的选择。使用随机森林函数和10重交叉验证抽样方法，并抽取5组样本：

```
require('caret')

rfeControls_rf <- rfeControl(
    functions=rfFuncs,
    method='cv',
    repeats=5)

fs_nb <- rfe(
    x=train[,-1],
    y=train[,1],
    sizes=seq(4,21,2),
    rfeControl=rfeControls_rf)

fs_nb
plot(fs_nb, type=c('g','o'))
fs_nb$optVariables
```

结果显示，只需要选择6个变量即可。

接下来，我们就针对这6个变量，使用朴素贝叶斯算法进行建模和预测：

```
require('klaR')
vars <- c('classes',fs_nb$optVariables)
fit <- NaiveBayes(classes ~ ., data=train[,vars])
pred <- predict(fit, newdata=test[,vars][,-1])

freq <- table(pred$class, test[,1])
freq
accuracy <- sum(diag(freq))/sum(freq)
accuracy

require('pROC')
modelroc <- roc(as.integer(test[,1]), as.integer(factor(pred$class)))

plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1,0.2), grid.col=c('green','red'), max.auc.polygon=TRUE, auc.polygon.col='steelblue')
```

通过朴素贝叶斯模型，在测试集中，模型的准确率约为97%，而且AUC的值也非常高，一般超过0.8就说明模型比较理想了。