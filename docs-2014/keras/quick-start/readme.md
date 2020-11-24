title: Keras | Quick Start
date: 2017-07-10
tags: [深度学习,Keras]
---
Keras的核心数据结构是[模型](#)，模型是一种组织网络层的方式。Keras中主要的模型是[Sequential](#)模型，一系列网络层按顺序构成的栈。使用[函数式](#)可以构建更复杂的模型。

<!--more-->
## Keras后端
如果你至少运行过一次Keras，你将在下面的目录下找到Keras的配置文件：
```
{HOME}/.keras/keras.json
```

如果该目录下没有该文件，你可以手动创建一个：
```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

[Kera后端函数对照表](http://keras-cn.readthedocs.io/en/latest/backend/#kera)

## Sequential模型
```
from keras.models import Sequential
model = Sequential()
```

网络层通过`.add()`堆叠起来，就构成了一个模型：
```
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```

完成模型的搭建后，需要使用`.compile()`方法来编译模型：
```
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

编译模型时必须指明[损失函数](#)和[优化器](#)，如果你需要的话，也可以自己定制损失函数。Keras的一个核心理念就是简明易用同时，保证用户对Keras的绝对控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码：
```
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

完成模型编译后，我们在训练数据上按`batch`进行一定次数的迭代来训练网络：
```
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

当然，我们也可以手动将一个个`batch`的数据送入网络中训练：
```
model.train_on_batch(x_batch, y_batch)
```

随后，我们可以对我们的模型进行评估：
```
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

使用我们的模型对新的数据进行预测：
```
classes = model.predict(x_test, batch_size=128)
```

为了更深入的了解Keras，我们建议你查看一下下面的两个教程：

- [快速开始Sequntial模型](http://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model)
- [快速开始函数式模型](http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API)

## 基本概念
张量,函数式模型,batch,epochs.

### 张量
张量，或tensor，张量可以看作是向量、矩阵的自然推广，我们用张量来表示广泛的数据类型。

规模最小的张量是0阶张量，即标量，也就是一个数。

当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量。

如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵。

把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体。

把立方体摞起来，好吧这次我们真的没有给它起别名了，就叫4阶张量了，不要去试图想像4阶张量是什么样子，它就是个数学上的概念。

张量的阶数有时候也称为[维度](#)，或者[轴](#)，轴这个词翻译自英文axis。譬如一个矩阵`[[1,2],[3,4]]`，是一个2阶张量，有两个维度或轴，沿着第0个轴（为了与python的计数方式一致，本文档维度和轴从0算起）你看到的是`[1,2]`，`[3,4]`两个向量，沿着第1个轴你看到的是`[1,3]`，`[2,4]`两个向量。

在如何表示一组彩色图片的问题上，Theano和TensorFlow发生了分歧，`th`模式，也即`Theano`模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式`（100,3,16,32）`，Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。这种theano风格的数据组织方法，称为[channels_first](#)，即通道维靠前。而TensorFlow，的表达形式是`（100,16,32,3）`，即把通道维放在了最后，这种数据组织方式称为[channels_last](#)。

Keras默认的数据组织形式在`{HOME}/.keras/keras.json`中规定，可查看该文件的`image_data_format`一项查看，也可在代码中通过`keras.backend.image_data_format()`函数返回，请在网络的训练和测试中保持维度顺序一致。

### 函数式模型
模型其实有两种，一种叫`Sequential`，称为[序贯模型](#)，也就是单输入单输出，一条路通到底，层与层之间只有相邻关系，跨层连接统统没有。这种模型编译速度快，操作上也比较简单。

由于`functional model API`在使用时利用的是[函数式编程](#)的风格，我们这里将其译为[函数式模型](#)。总而言之，只要这个东西接收一个或一些张量作为输入，然后输出的也是一个或一些张量，那不管它是什么鬼，统统都称作“模型”。

### batch
深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为`Batch gradient descent`，[批梯度下降](#)。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为[随机梯度下降](#)，`stochastic gradient descent`。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，`mini-batch gradient decent`，[小批的梯度下降](#)，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

基本上现在的梯度下降都是基于`mini-batch`的，所以Keras的模块中经常会出现`batch_size`，就是指这个。顺便说一句，Keras中用的优化器SGD是`stochastic gradient descent`的缩写，但不代表是一个样本就更新一回，还是基于`mini-batch`的。

### epochs
简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。
