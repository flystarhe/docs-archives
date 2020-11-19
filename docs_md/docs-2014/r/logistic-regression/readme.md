title: Logistic Regression
date: 2017-05-22
tags: [R,LR]
---
本文将以著名的婚外情数据介绍使用R完成逻辑回归.

<!--more-->
婚外情数据即著名的`Fair’s Affairs`，取自于1969年《今日心理》（Psychology Today）所做 的一个非常有代表性的调查，而Greene（2003）和Fair（1978）都对它进行过分析。该数据从601个参与者身上收集了9个变量，包括一年来婚外私通的频率以及参与者性别、年龄、婚龄、是否有小孩、宗教信仰程度（5分制，1分表示反对，5分表示非常信仰）、学历、职业（逆向编号的戈登7种分类），还有对婚姻的自我评分（5分制，1表示非常不幸福，5表示非常幸福）。我们先看一些描述性的统计信息：
```
install.packages('AER')
data(Affairs,package='AER')
summary(Affairs)
##     affairs          gender         age         yearsmarried    children  religiousness     education       occupation        rating     
##  Min.   : 0.000   female:315   Min.   :17.50   Min.   : 0.125   no :171   Min.   :1.000   Min.   : 9.00   Min.   :1.000   Min.   :1.000  
##  1st Qu.: 0.000   male  :286   1st Qu.:27.00   1st Qu.: 4.000   yes:430   1st Qu.:2.000   1st Qu.:14.00   1st Qu.:3.000   1st Qu.:3.000  
##  Median : 0.000                Median :32.00   Median : 7.000             Median :3.000   Median :16.00   Median :5.000   Median :4.000  
##  Mean   : 1.456                Mean   :32.49   Mean   : 8.178             Mean   :3.116   Mean   :16.17   Mean   :4.195   Mean   :3.932  
##  3rd Qu.: 0.000                3rd Qu.:37.00   3rd Qu.:15.000             3rd Qu.:4.000   3rd Qu.:18.00   3rd Qu.:6.000   3rd Qu.:5.000  
##  Max.   :12.000                Max.   :57.00   Max.   :15.000             Max.   :5.000   Max.   :20.00   Max.   :7.000   Max.   :5.000  
```

从这些统计信息可以看到，52%的调查对象是女性，72%的人有孩子，样本年龄的中位数为32岁。对于响应变量，72%的调查对象表示过去一年中没有婚外情（451/601），而婚外偷腥的多次数为12（占了6%）。
affairs表示过去一年中婚外情的次数，可将affairs转化为二值型因子ynaffair：
```r
Affairs$ynaffair[Affairs$affairs >0]=1
Affairs$ynaffair[Affairs$affairs==0]=0
Affairs$ynaffair=factor(Affairs$ynaffair,levels=c(0,1),labels=c('No','Yes'))
table(Affairs$ynaffair)
```

输出为：
```r
 No Yes 
451 150 
```

该二值型因子现可作为Logistic回归的结果变量：
```r
fit.full=glm(formula=ynaffair ~ gender + age + yearsmarried + children + religiousness + education + occupation + rating,
    data=Affairs,family=binomial(link='logit'))
summary(fit.full)
```

输出为：
```r
Call:
glm(formula = ynaffair ~ gender + age + yearsmarried + children + 
    religiousness + education + occupation + rating, family = binomial(link = "logit"), 
    data = Affairs)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.5713  -0.7499  -0.5690  -0.2539   2.5191  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)    1.37726    0.88776   1.551 0.120807    
gendermale     0.28029    0.23909   1.172 0.241083    
age           -0.04426    0.01825  -2.425 0.015301 *  
yearsmarried   0.09477    0.03221   2.942 0.003262 ** 
childrenyes    0.39767    0.29151   1.364 0.172508    
religiousness -0.32472    0.08975  -3.618 0.000297 ***
education      0.02105    0.05051   0.417 0.676851    
occupation     0.03092    0.07178   0.431 0.666630    
rating        -0.46845    0.09091  -5.153 2.56e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 675.38  on 600  degrees of freedom
Residual deviance: 609.51  on 592  degrees of freedom
AIC: 627.51

Number of Fisher Scoring iterations: 4
```

从回归系数的p值（后一栏）可以看到，性别、是否有孩子、学历和职业对方程的贡献都不显著（你无法拒绝参数为0的假设）。去除这些变量重新拟合模型，检验新模型是否拟合得好：
```r
fit.reduced=glm(formula=ynaffair ~ age + yearsmarried + religiousness + rating,
    data=Affairs,family=binomial(link='logit'))
summary(fit.reduced)
```

输出为：
```r
Call:
glm(formula = ynaffair ~ age + yearsmarried + religiousness + 
    rating, family = binomial(link = "logit"), data = Affairs)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.6278  -0.7550  -0.5701  -0.2624   2.3998  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)    1.93083    0.61032   3.164 0.001558 ** 
age           -0.03527    0.01736  -2.032 0.042127 *  
yearsmarried   0.10062    0.02921   3.445 0.000571 ***
religiousness -0.32902    0.08945  -3.678 0.000235 ***
rating        -0.46136    0.08884  -5.193 2.06e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 675.38  on 600  degrees of freedom
Residual deviance: 615.36  on 596  degrees of freedom
AIC: 625.36

Number of Fisher Scoring iterations: 4
Correlation of Coefficients:
              (Intercept) age   yearsmarried religiousness
age           -0.65                                       
yearsmarried   0.31       -0.75                           
religiousness -0.36        0.01 -0.18                     
rating        -0.63        0.06  0.08         0.00        
```

新模型的每个回归系数都非常显著（p<0.05）。由于两模型嵌套（fit.reduced是fit.full的一个子集），你可以使用anova()函数对它们进行比较，对于广义线性回归，可用卡方检验。
```r
anova(fit.reduced,fit.full,test='Chisq')
```

输出为：
```r
Analysis of Deviance Table

Model 1: ynaffair ~ age + yearsmarried + religiousness + rating
Model 2: ynaffair ~ gender + age + yearsmarried + children + religiousness + 
    education + occupation + rating
  Resid. Df Resid. Dev Df Deviance Pr(>Chi)
1       596     615.36                     
2       592     609.51  4   5.8474   0.2108
```

结果的卡方值不显著（p=0.21），表明四个预测变量的新模型与九个完整预测变量的模型拟合程度一样好。这使得你更加坚信添加性别、孩子、学历和职业变量不会显著提高方程的预测精度，因此可以依据更简单的模型进行解释。先看看回归系数：
```r
coef(fit.reduced)
```

输出为：
```r
  (Intercept)           age  yearsmarried religiousness        rating 
   1.93083017   -0.03527112    0.10062274   -0.32902386   -0.46136144 
```

在Logistic回归中，响应变量是Y=1的对数优势比（log）。回归系数含义是当其他预测变量不变时，1单位预测变量的变化可引起的响应变量对数优势比的变化。
由于对数优势比解释性差，你可对结果进行指数化：
```r
exp(coef(fit.reduced))
```

输出为：
```r
  (Intercept)           age  yearsmarried religiousness        rating 
    6.8952321     0.9653437     1.1058594     0.7196258     0.6304248 
```

可以看到婚龄增加一年，婚外情的优势比将乘以1.106（保持年龄、宗教信仰和婚姻评定不变）；相反，年龄增加一岁，婚外情的的优势比则乘以0.965。因此，随着婚龄的增加和年龄、宗教信仰与婚姻评分的降低，婚外情优势比将上升。因为预测变量不能等于0，截距项在此处没有什么特定含义。

如果有需要，你还可使用`confint()`函数获取系数的置信区间。例如，`exp(confint(fit.reduced))`可在优势比尺度上得到系数95%的置信区间。

对于我们大多数人来说，以概率的方式思考比使用优势比更直观。使用`predict()`函数，你可观察某个预测变量在各个水平时对结果概率的影响。首先创建一个包含你感兴趣预测变量值的虚拟数据集，然后对该数据集使用`predict()`函数，以预测这些值的结果概率。

现在我们使用该方法评价婚姻评分对婚外情概率的影响。首先，创建一个虚拟数据集，设定年龄、婚龄和宗教信仰为它们的均值，婚姻评分的范围为`1:5`。
```r
testdata=data.frame(rating=1:5,age=mean(Affairs$age),yearsmarried=mean(Affairs$yearsmarried),religiousness=mean(Affairs$religiousness))
testdata$prod=predict(fit.reduced,newdata=testdata,type='response')
testdata
```

输出为：
```r
  rating      age yearsmarried religiousness      prod
1      1 32.48752     8.177696      3.116473 0.5302296
2      2 32.48752     8.177696      3.116473 0.4157377
3      3 32.48752     8.177696      3.116473 0.3096712
4      4 32.48752     8.177696      3.116473 0.2204547
5      5 32.48752     8.177696      3.116473 0.1513079
```

从这些结果可以看到，当婚姻评分从1（很不幸福）变为5（非常幸福）时，婚外情概率从0.53降低到了0.15（假定年龄、婚龄和宗教信仰不变）。同样可以看看年龄等的影响。

## RWeka.LR
```r
## Logistic regression:
## Using standard data set 'infert'.
STATUS <- factor(infert$case, labels = c("control", "case"))
Logistic(STATUS ~ spontaneous + induced, data = infert)
## Compare to R:
glm(STATUS ~ spontaneous + induced, data = infert, family = binomial())
```
