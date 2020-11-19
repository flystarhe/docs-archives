# Optimization technique for keras
本文介绍了几个深度学习模型的简单优化技巧,包括迁移学习,dropout,学习率调整等,并展示了如何用Keras实现.

[引用:深度学习模型的简单优化技巧](https://www.jiqizhixin.com/articles/061002)

## 为什么要改进模型?
像卷积神经网络(CNN)这样的深度学习模型具有大量的参数.实际上,我们可以调用这些超参数,因为它们原本在模型中并没有被优化.你可以网格搜索这些超参数的最优值,但需要大量硬件计算和时间.那么,一个真正的数据科学家能满足于猜测这些基本参数吗?

改进模型的最佳方法之一是基于在你的领域进行过深入研究的专家的设计和体系结构,他们通常拥有强大的硬件可供使用.而且,他们经常慷慨地开源建模架构和原理.

## 深度学习技术
以下是一些通过预训练模型来改善拟合时间和准确性的方法:

- 研究理想的预训练体系架构:了解迁移学习的好处,或了解一些功能强大的CNN体系架构.考虑那些看起来不太适合但具有潜在共享特性的领域
- 使用较小的学习率:由于预训练的权重通常优于随机初始化的权重,因此修改要更为精细!你在此处的选择取决于学习环境和预训练的表现,但请检查各个时期的误差,以了解距离收敛还要多久
- 使用dropout:与回归模型的Ridge和LASSO正则化一样,没有适用于所有模型的优化alpha或dropout.这是一个超参数,取决于具体问题,必须进行测试
- 限制权重大小:可以限制某些层的权重的最大范数(绝对值),以泛化我们的模型
- 不要动前几层:神经网络的前几个隐藏层通常用于捕获通用和可解释的特征
- 修改输出层:使用适合你的领域的新激活函数和输出大小替换模型默认值,不要把自己局限于最明显的解决方案.尽管MNIST看起来似乎需要10个输出类,也许`12-16`个类可能会更好地解决并提高模型性能

## Keras中的技术
在Keras中修改MNIST的dropout和限制权重大小的方法如下:
```python
# dropout in input and hidden layers
# weight constraint imposed on hidden layers
# ensures the max norm of the weights does not exceed 5
model = Sequential()
model.add(Dropout(0.2, input_shape=(784,))) # dropout on the inputs
model.add(Dense(128, input_dim=784, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
```

## Dropout最佳实践
- 使用`20%-50%`的dropout,建议输入`20%`.太低,影响可以忽略;太高,可能欠拟合
- 在输入层和隐藏层上使用dropout.这已被证明可以提高深度学习的性能
- 使用伴有衰减的较大的学习速率,以及较大的动量
- 限制权重!较大的学习速率会导致梯度爆炸,通过对网络权值施加约束可以改善结果
- 使用更大的网络.在较大的网络上使用dropout可能会获得更好的性能,从而使模型有更多的机会学习独立的表征

下面是Keras中的最终层修改示例,其中包含14个MNIST类:
```python
from keras.layers.core import Activation, Dense
model.layers.pop() # defaults to last
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(14, activation='softmax'))
```

以及如何冻结前五层权重的示例:
```python
for layer in model.layers[:5]:
    layer.trainable = False
```

## 预训练网络库
- Kaggle列表:https://www.kaggle.com/gaborfodor/keras-pretrained-models
- Keras应用:https://keras.io/applications/
- OpenCV示例:https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

## 可视化你的模型
```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

两个可选参数:

- show_shapes(默认False)控制输出形状是否显示在图中
- show_layer_names(默认True)控制层命名是否显示在图中

也可以在notebook中显示:
```python
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
