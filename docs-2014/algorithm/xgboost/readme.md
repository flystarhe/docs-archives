title: XGBoost,速度快效果好的boosting
date: 2017-05-01
tags: [GBDT,XGBoost]
---
在数据分析的过程中，我们经常需要对数据建模并做预测。在众多的选择中，randomForest,gbm,glmnet是三个尤其流行的R包，它们在Kaggle的各大数据挖掘竞赛中的出现频率独占鳌头，被坊间人称为R数据挖掘包中的三驾马车。gbm包比同样是使用树模型的randomForest包占用的内存更少，同时训练速度较快，尤其受到大家的喜爱。在python的机器学习库sklearn里也有GradientBoostingClassifier的存在。

<!--more-->
## GBDT
GBDT(Gradient Boosting Decision Tree)又叫MART(Multiple Additive Regression Tree)，是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力（generalization)较强的算法。

GBDT的核心在于：每一棵树学的是之前所有树结论和的残差，这个残差就是一个加预测值后能得真实值的累加量。比如A的真实年龄是18岁，但第一棵树的预测年龄是12岁，差了6岁，即残差为6岁。那么在第二棵树里我们把A的年龄设为6岁去学习，如果第二棵树真的能把A分到6岁的叶子节点，那累加两棵树的结论就是A的真实年龄；如果第二棵树的结论是5岁，则A仍然存在1岁的残差，第三棵树里A的年龄就变成1岁，继续学习。[推荐阅读](http://blog.csdn.net/w28971023/article/details/8240756)。

### DT
决策树分为两大类：回归树和分类树。前者用于预测实数值，如明天的温度、用户的年龄、网页的相关程度；后者用于分类标签值，如晴天/阴天/雾/雨、用户性别、网页是否是垃圾页面。这里要强调的是，前者的结果加减是有意义的，如`10岁+5岁-3岁`等于`12岁`，后者则无意义，如`男+男+女`到底是男是女？

GBDT的核心在于累加所有树的结果作为最终结果，就像前面对年龄的累加，而分类树的结果显然是没办法累加的，所以GBDT中的树都是回归树，不是分类树，这点对理解GBDT相当重要，尽管GBDT调整后也可用于分类但不代表GBDT的树是分类树。

### GB
GBDT的核心就在于，每一棵树学的是之前所有树结论和的残差，这个残差就是一个加预测值后能得真实值的累加量。比如A的真实年龄是18岁，但第一棵树的预测年龄是12岁，差了6岁，即残差为6岁。那么在第二棵树里我们把A的年龄设为6岁去学习，如果第二棵树真的能把A分到6岁的叶子节点，那累加两棵树的结论就是A的真实年龄；如果第二棵树的结论是5岁，则A仍然存在1岁的残差，第三棵树里A的年龄就变成1岁，继续学。这就是Gradient Boosting在GBDT中的意义。

### Boosting
这是boosting，不是Adaboost。提到决策树大家会想起C4.5，提到boost多数人也会想到Adaboost。

Adaboost是另一种boost方法，它按分类对错，分配不同的weight，计算cost function时使用这些weight，从而让“错分的样本权重越来越大，使它们更被重视”。

Bootstrap也有类似思想，它在每一步迭代时不改变模型本身，也不计算残差，而是从N个instance训练集中按一定概率重新抽取N个instance出来（单个instance可以被重复sample），对着这N个新的instance再训练一轮。由于数据集变了迭代模型训练结果也不一样，而一个instance被前面分错的越厉害，它的概率就被设的越高，这样就能同样达到逐步关注被分错的instance，逐步完善的效果。

Adaboost的方法被实践证明是一种很好的防止过拟合的方法，但至于为什么则至今没从理论上被证明。GBDT也可以在使用残差的同时引入Bootstrap re-sampling，GBDT多数实现版本中也增加的这个选项，但是否一定使用则有不同看法。re-sampling一个缺点是它的随机性，即同样的数据集合训练两遍结果是不一样的，也就是模型不可稳定复现，这对评估是很大挑战，比如很难说一个模型变好是因为你选用了更好的feature，还是由于这次sample的随机因素。

### Shrinkage
Shrinkage（缩减）的思想认为，每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。即它不完全信任每一个棵残差树，它认为每棵树只学到了真理的一小部分，累加的时候只累加一小部分，通过多学几棵树弥补不足。

### GBDT的适用范围
该版本GBDT几乎可用于所有回归问题（线性/非线性），相对LR仅能用于线性回归，GBDT的适用面非常广。亦可用于二分类问题（设定阈值，大于阈值为正例，反之为负例）。

### GBDT in Python
```python
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

clf.score(X_test, y_test)
```

参考：[gradient-tree-boosting](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)。

## XGBoost
XGBoost也是以（CART）为基学习器的GB算法，但是扩展和改进了GDBT：

1. 传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。
2. 传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。
3. xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性。
4. Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。
5. 列抽样（column subsampling）。xgboost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算，这也是xgboost异于传统gbdt的一个特性。
6. 对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。
7. xgboost工具支持并行。注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）。
8. 可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点。

### XGBoost in Python
安装`xgboost`：
```bash
pip install xgboost
```

训练&预测：
```python
import xgboost as xgb
# read in data
# data : string/numpy array/scipy.sparse/pd.DataFrame
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtests = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
params = {'max_depth':2, 'min_child_weight':1, 'scale_pos_weight':1, 'eta':0.3, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'auc'}
num_round = 200
bst = xgb.train(params, dtrain, num_round)
# make prediction
preds = bst.predict(dtests)
```

加载数据到对象DMatrix中：
```python
#加载libsvm格式的数据或二进制的缓存文件
dtrain = xgb.DMatrix('train.svm.txt')
#加载numpy的数组到DMatrix对象
import numpy as np
data = np.random.rand(5, 10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix(data, label=label)
#将scipy.sparse格式的数据转化为DMatrix格式
csr = scipy.sparse.csr_matrix( (dat, (row,col)) )
dtrain = xgb.DMatrix(csr)
#将DMatrix格式的数据保存成XGBoost的二进制格式
dtrain = xgb.DMatrix('train.svm.txt')
dtrain.save_binary("train.buffer")
#DMatrix中的缺失值
dtrain = xgb.DMatrix(data, label=label, missing=-999.0)
#给样本设置权重
w = np.random.rand(5, 1)
dtrain = xgb.DMatrix(data, label=label, missing=-999.0, weight=w)
```

参考：[xgboost/get_started](https://xgboost.readthedocs.io/en/latest/get_started/)，[Notes on Parameter Tuning](http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html)，[XGBoost Parameters](http://xgboost.readthedocs.io/en/latest/parameter.html)。

### 控制过拟合
当你观察到训练精度高,但是测试精度低时,你可能遇到了过拟合的问题.通常有两种方法可以控制xgboost中的过拟合:

- 第一个方法是直接控制模型的复杂度:这包括`max_depth`,`min_child_weight`和`gamma`;
- 第二种方法是增加随机性,使训练对噪声强健:这包括`subsample`,`colsample_bytree`,你也可以减小步长`eta`,但是当你这么做的时候需要记得增加`num_round`.

### 处理不平衡的数据集
对于广告点击日志等常见情况,数据集是极不平衡的.这可能会影响xgboost模型的训练,有两种方法可以改善它.

如果你只关心预测的排名顺序(AUC):

- 通过`scale_pos_weight`来平衡`positive`和`negative`权重;
- 使用AUC进行评估

如果你关心预测正确的概率:

- 在这种情况下,您无法重新平衡数据集
- 在这种情况下,将参数`max_delta_step`设置为有限数字(比如说1)将有助于收敛

### 模板
线性回归:
```python
train_xs, train_ys, test_xs, test_ys = (ndarrays, ..)
dtrain = xgb.DMatrix(train_xs, train_ys[:, 0])
dtest = xgb.DMatrix(test_xs, test_ys[:, 0])

params = {
            "eta":0.1,
            "gamma":0,
            "max_depth":5,
            "min_child_weight":10,
            "subsample":0.5,
            "colsample_bytree":1,
            "objective":"reg:linear",
            "eval_metric": "mae",  # or `rmse`
            "booster":"gbtree",
            "silent":1
        }
num_boost_round = 500
watchlist = [(dtest, "eval"), (dtrain, "train")]

# training
bst = xgb.train(params, dtrain, num_boost_round, watchlist)
bst.save_model("model.bin")

# evaluation
preds = bst.predict(dtest)
labels = dtest.get_label()

# loading
bst = xgb.Booster()
bst.load_model("model.bin")
```

## 参考资料：
- [xgboost/get_started](http://xgboost.apachecn.org/cn/latest/get_started/index.html)
- [xgboost/python_intro](http://xgboost.apachecn.org/cn/latest/python/python_intro.html)
- [xgboost/param_tuning](http://xgboost.apachecn.org/en/latest/how_to/param_tuning.html)
- [xgboost/parameter](http://xgboost.apachecn.org/cn/latest/parameter.html)
- [xgboost/速度快效果好的boosting模型](https://cos.name/2015/03/xgboost/)
- [demo/kaggle-higgs/higgs-train.R](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-train.R)
- 中文文档地址: http://xgboost.apachecn.org/cn/latest/
- 英文文档地址: http://xgboost.apachecn.org/en/latest/
- GitHub地址: https://github.com/dmlc/xgboost