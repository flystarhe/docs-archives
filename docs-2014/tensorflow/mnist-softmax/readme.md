title: TensorFlow Start MNIST Softmax
date: 2017-05-31
tags: [TensorFlow]
---
[MNIST For ML Beginners](https://www.tensorflow.org/get_started/mnist/beginners)

<!--more-->
MNIST是一个简单的计算机视觉数据集。它由以下手写数字的图像组成：

![](readme01.png)

它还包括每个图像的标签，告诉我们是哪个数字。例如，上述图像的标签是5,0,4和1。

在本教程中，我们将训练一个模型来查看图像并预测它们的数字。我们的目标不是训练一个真正精致的模型，而是使用TensorFlow。我们将从一个非常简单的模型开始，称为Softmax回归。接下来，将逐行解释[mnist_softmax.py](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py)中的代码。

## MNIST数据集
MNIST数据托管在[MNIST Page](http://yann.lecun.com/exdb/mnist/)。这两行代码，将自动下载和读取数据：


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data1/hejian_lab/_temp/mnist/", one_hot=True)
```

    Extracting /data1/hejian_lab/_temp/mnist/train-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/train-labels-idx1-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-labels-idx1-ubyte.gz


这些文件本身并没有使用标准的图片格式储存，需要使用`read_data_sets()`函数读取，`mnist`分为三部分：

- `mnist.train`: 55000组图片和标签，用于训练。
- `mnist.validation`: 5000组图片和标签，用于迭代验证训练的准确性。
- `mnist.test`: 10000组图片和标签，用于最终测试训练的准确性。

数据集的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）。

每个MNIST数据点都有两部分：手写数字的图像和相应的标签。例如，训练图像是`mnist.train.images`，训练标签是`mnist.train.labels`。每个图像是`28x28`像素，我们可以把它解释为一大批数字：

![](readme02.png)

我们把这个数组变成一个`28x28=784`的向量。平铺数据会丢弃有关图像2D结构的信息，好的计算机视觉方法会利用这个结构。`mnist.train.images`形如：

![](readme03.png)

MNIST中的每个图像都具有相应的标签，0到9之间的数字表示图像中绘制的数字，`mnist.train.labels`形如：

![](readme04.png)

## Softmax Regression
我们知道MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%，判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。

这是一个使用`softmax regression`模型的经典情况。softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率。softmax回归有两个步骤：首先我们将我们的输入的证据加在某些类中，然后将该证据转换成概率。

为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。下面的图片显示了一个模型学习到的图片上每个像素对于特定数字类的权值。红色代表负数权值，蓝色代表正数权值。

![](readme05.png)

我们也需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片$x$它代表的是数字$i$的证据可以表示为：

\begin{align}
evidence_i = \sum_{j} W_{i,j}x_{j} + b_{i}
\end{align}

其中$W_i$代表权重，$b_i$代表数字$i$类的偏置量，$j$代表给定图片$x$的像素索引用于像素求和。然后用Softmax函数可以把这些证据转换成概率$y$：

\begin{align}
y = softmax(evidence)
\end{align}

这里的softmax可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。因此，给定一张图片，它对于每一个数字的吻合度可以被softmax函数转换成为一个概率值。softmax函数可以定义为：

\begin{align}
softmax(x) = normalize(\exp (x)) \\
softmax(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
\end{align}

softmax回归如下图所示：

![](readme06.png)

\begin{align}
y = softmax(Wx + b)
\end{align}

## 实现回归模型
为了用python实现高效的数值计算，我们通常会使用函数库，比如NumPy，会把类似矩阵乘法这样的复杂运算使用其他外部语言实现。不幸的是，从外部计算切换回Python的每一个操作，仍然是一个很大的开销。如果你用GPU来进行外部计算，这样的开销会更大。用分布式的计算方式，也会花费更多的资源用来传输数据。

TensorFlow也把复杂的计算放在python之外完成，但是为了避免前面说的那些开销，它做了进一步完善。Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。）

使用TensorFlow之前，首先导入它：


```python
import tensorflow as tf
print(tf.__version__)
```

    1.1.0


我们通过操作符号变量来描述这些可交互的操作单元，我们创建一个：


```python
x = tf.placeholder(tf.float32, [None, 784])
```

$x$不是一个具体的值，而是一个占位符`placeholder`，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是`[None, 784]`。（这里的None表示此张量的第一个维度可以是任何长度的。）

我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：`Variable`。一个`Variable`代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用`Variable`表示。


```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

我们赋予`tf.Variable`不同的初值来创建不同的`Variable`。在这里，我们都用全为零的张量来初始化`W`和`b`。因为我们要学习`W`和`b`的值，它们的初值可以随意设置。注意，`W`的维度是`[784, 10]`，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。`b`的形状是`[10]`，所以我们可以直接把它加到输出上面。

现在，我们可以实现我们的模型啦。只需要一行代码！


```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

首先，我们用`tf.matmul(x, W)`表示`x`乘以`W`，对应之前等式里面的，这里`x`是一个2维张量拥有多个输入，然后再加上`b`。

至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型。TensorFlow不仅仅可以使softmax回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型。一旦被定义好之后，我们的模型就可以在不同的设备上运行：计算机的CPU，GPU，甚至是手机！

## 训练模型
为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是，这两种方式是相同的。

一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：

\begin{align}
H_{y'}(y) = - \sum_{i} y'_{i} \log(y_{i})
\end{align}

$y$是我们预测的概率分布，$y'$是实际的分布（我们输入的`one-hot vector`）。比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。更详细的关于交叉熵的解释超出本教程的范畴，但是你很有必要好好[理解它](http://colah.github.io/posts/2015-09-Visual-Information/)。为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

然后可以用$- \sum y' \log(y)$计算交叉熵：


```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

现在我们知道我们需要我们的模型做什么啦，用TensorFlow来训练它是非常容易的。因为TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。


```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.5的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。

TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。

我们现在可以在`Session`中启动该模型：


```python
sess = tf.Session()
```

在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：


```python
sess.run(tf.global_variables_initializer())
```

然后开始训练模型，这里我们让模型循环训练1000次：


```python
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行`train_step`。

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training），在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

## 评估模型
首先让我们找出那些预测正确的标签。`tf.argmax`是一个非常有用的函数，它能给出某个`tensor`对象在某一维上的其数据最大值所在的索引值。由于标签向量是由`0,1`组成，因此最大值`1`所在的索引位置就是类别标签，比如`tf.argmax(y,1)`返回的是模型对于任一输入`x`预测到的标签值，而`tf.argmax(y_,1)`代表正确的标签，我们可以用`tf.equal`来检测我们的预测是否真实标签匹配：


```python
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
```

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，`[True,False,True,True]`会变成`[1,0,1,1]`，取平均值后得到`0.75`：


```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后，我们计算所学习到的模型在测试数据集上面的正确率：


```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    0.9174


这个最终结果值应该大约是`91%`。嗯，并不太好。事实上，这个结果是很差的。这是因为我们仅仅使用了一个非常简单的模型。不过，做一些小小的改进，我们就可以得到`97％`的正确率。最好的模型甚至可以获得超过`99.7％`的准确率！想了解更多信息，可以看看这个关于各种模型的[性能对比列表](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)。

## 完整代码(1)


```python
# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data1/hejian_lab/_temp/mnist/", one_hot=True)

import tensorflow as tf
sess = tf.Session()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# True distribution
y_ = tf.placeholder(tf.float32, [None, 10])

# Loss Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Init
sess.run(tf.global_variables_initializer())

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    Extracting /data1/hejian_lab/_temp/mnist/train-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/train-labels-idx1-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-labels-idx1-ubyte.gz
    0.9177


## 完整代码(2)


```python
# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data1/hejian_lab/_temp/mnist/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Predicted
y = tf.matmul(x,W) + b

# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Init
tf.global_variables_initializer().run()

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    Extracting /data1/hejian_lab/_temp/mnist/train-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/train-labels-idx1-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-labels-idx1-ubyte.gz
    0.9162

