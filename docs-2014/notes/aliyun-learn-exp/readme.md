title: 阿里云机器学习体验笔记
date: 2017-05-22
tags: [机器学习]
---
阿里云机器学习PAI，提供最丰富的算法。PAI包含特征工程、数据预处理、统计分析、机器学习、深度学习框架、预测与评估这一整套的机器学习算法组件，共100余种。[doc](https://help.aliyun.com/document_detail/42709.html)

<!--more-->
## 准备数据
```r
subdata <- iris[iris$Species != 'virginica', ]
subdata$Species <- factor(subdata$Species)
names(subdata) <- tolower(gsub("[.]", "_", names(subdata)))
write.csv(subdata, file="demo_iris_subdata.csv", fileEncoding="utf-8", row.names=FALSE)
```

```sql
CREATE TABLE flystar_demo_iris_subdata (
    sepal_length DOUBLE,
    sepal_width DOUBLE,
    petal_length DOUBLE,
    petal_width DOUBLE,
    species STRING
)
LIFECYCLE 100000;
```

进入[大数据开发套件](https://ide.shuju.aliyun.com)工作区，新建表`flystar_demo_iris_subdata`，然后导入`demo_iris_subdata.csv`中的数据。（数据必须先导入到项目）

## 新建实验
首先进入[机器学习.实验](https://pai.base.shuju.aliyun.com/experiment.htm)，点击`我的实验`面板最下端的`新建实验`，新建空白实验：

- 名称: demo_classification_binary
- 项目: cloud_service
- 描述: flystar demo classification
- 未知: 我的实验/demo/classification

## 读入数据
导航到`组件.源/目标`，拖拽`读数据表`到画布，在右侧`表选择`栏选择表`flystar_demo_iris_subdata`。切换到`字段信息`栏，可以查看输入表的字段名、数据类型和前100行数据的数值分布。

## 数据预处理
考虑使用`GBDT二分类`模型，Aliyun文档说了GBDT特征只支持`double`和`bigint`，GBDT标签只支持`bigint`。所以，要转换`species STRING`为`species int`，顺便在做个归一化吧！

### string to int
导航到`组件.工具`，拖拽`SQL脚本`到画布，连接`读数据表`与`SQL脚本`组件，在右侧`参数设置`栏的`SQL脚本`中输入SQL语句：
```sql
select sepal_length,sepal_width,petal_length,petal_width,
(case species when '"setosa"' then 0 else 1 end) as species
from ${t1};
```

注意，这里的值`"setosa"`带有双引号，因为Aliyun导入csv数据时，不识别文本引用符号。

### min-max 标准化
导航到`组件.数据预处理`，拖拽`归一化`到画布，连接`SQL脚本`与`归一化`(输入数据表)组件，在右侧`字段设置`栏选择`sepal_length`，`sepal_width`，`petal_length`，`petal_width`。

### 数据合并
将两张表的数据按列合并，需要表的行数保持一致，否则报错。

导航到`组件.数据预处理.数据合并`，拖拽`合并列`到画布，连接`归一化`(输出结果表)到`合并列.左表`，连接`SQL脚本`到`合并列.右表`，左表选择：`sepal_length`，`sepal_width`，`petal_length`，`petal_width`，右表选择：`species`。

## 拆分
导航到`组件.数据预处理`，拖拽`拆分`到画布，连接`合并列`与`拆分`组件，在右侧`参数设置`栏输入`切分比例:0.8`，`随机数种子:1234`。

## 数据可视化
向画布拖入`统计分析.全表统计`和`统计分析.数据视图`组件，连线并点击画布底部的运行按钮，待实验运行结束。右击`全表统计`，点击`查看数据`，可看到数据的全表统计信息。右击`数据视图`，点击`查看分析报告`，可看到数据的直方图。

## 算法建模
向画布拖入`机器学习.二分类.GBDT二分类`组件，连接`拆分.输出表1`与`GBDT二分类`组件，在右侧`字段设置`栏，选择特征列：`sepal_length`，`sepal_width`，`petal_length`，`petal_width`，选择标签列：`species`。

切换到`参数设置`栏，设置`树的数目: 500`，`学习速率: 0.05`，`树最大深度: 5`，一般来说设置调整这几个参数就够了。但是我们的测试数据集很小，所以修改设置`叶节点最小样本数: 10`。

再向画布拖入一个`机器学习.二分类.GBDT二分类`组件，其他保持不变，只是修改`学习速率: 0.1`。

## 模型评估
向画布拖入两个`机器学习.预测`组件和两个`机器学习.评估.二分类评估`组件，分别连接对应的组件流和数据流。对于`预测`组件设置`特征列`和`原样输出列`(类标签)，对于`评估`组件设置`原始标签列列名: species`和`计算KS/PR等指标时按等频率分成多少个桶: 10`。

点击`运行`，得到不同参数下训练的GBDT模型的综合评估。里面包括综合指数、详细信息和KS/PR/LIFT/ROC曲线。

右键`预测`点击`查看数据`，全部预测正确，右键`二分类评估`点击`查看评估报告`，看到`AUC: 1`和`F-Score: 1`就更放心了，这当然要归功于我们的特征足够好。

## 参考资料：
- [机器学习 > 快速开始](https://help.aliyun.com/document_detail/30350.html)