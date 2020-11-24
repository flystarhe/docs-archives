title: TensorFlow Start MNIST Deep Network
date: 2017-05-31
tags: [TensorFlow]
---
[Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros)

本教程的第一部分解释了[mnist_softmax.py](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py)代码中发生了什么，这是Tensorflow模型的基本实现。第二部分显示了一些提高精度的方法。从[mnist_deep.py](https://www.github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py)下载深层网络实施代码。

<!--more-->
## Softmax
在创建我们的模型之前，我们将首先加载MNIST数据集，并启动TensorFlow会话：


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data1/hejian_lab/_temp/mnist/', one_hot=True)
```

    Extracting /data1/hejian_lab/_temp/mnist/train-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/train-labels-idx1-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-images-idx3-ubyte.gz
    Extracting /data1/hejian_lab/_temp/mnist/t10k-labels-idx1-ubyte.gz


这里mnist是一个轻量级的类，它将训练，验证和测试集存储为NumPy数组。它还提供了一个迭代数据服务的功能，我们将在下面使用。

TensorFlow依靠高效的C ++后端来进行计算。与此后端的连接称为会话。TensorFlow程序的常见用法是首先创建一个图形，然后在会话中启动它。这里我们使用方便的`InteractiveSession`类：


```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

建立一个Softmax回归模型：


```python
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

    0.9162


## Network

### Weight Initialization
要创建这个模型，我们将需要创建很多权重和偏差。这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题：


```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

### Convolution and Pooling
TensorFlow为卷积和池化操作提供了很大的灵活性。我们如何处理边界？我们的步幅是多少？在这个例子中，我们总是选择`vanilla`版本。我们的卷积使用1步长的模板，保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板。为了代码更简洁，我们把这部分抽象成一个函数。


```python
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

### First Convolutional Layer
现在我们可以开始实现第一层了。它由一个卷积接一个`max pooling`完成。卷积在每个`5x5`的patch中算出32个特征。卷积的权重张量形状是`[5,5,1,32]`，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。而对于每一个输出通道都有一个对应的偏置量。


```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

为了用这一层，我们把$x$变成一个4d张量，其第2、3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图，所以为1)。


```python
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
```

我们将`x_image`与权重张量进行卷积，添加偏差，然后应用`ReLU`激活函数，最后`max pool`。`max_pool_2x2`方法将图像大小减小到`14x14`：


```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### Second Convolutional Layer
为了建立一个深度网络，我们堆叠这种类型的几层。第二层中，每个`5x5`的patch会得到64个特征：


```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

### Densely Connected Layer
现在图像尺寸已经缩小到`7x7`，我们加入一个有1024个神经元的全连接层，以便对整个图像进行处理。我们把池化层输出的张量reshape成一批向量，乘以权重矩阵，添加偏倚并应用`ReLU`激活函数。


```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

为了减少过拟合，我们在输出层之前加入`dropout`。我们用一个`placeholder`来代表一个神经元的输出在`dropout`中保持不变的概率。这样我们可以在训练过程中启用`dropout`，在测试过程中关闭`dropout`。TensorFlow的`tf.nn.dropout`操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的`scale`。所以用`dropout`的时候可以不用考虑`scale`。


```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### Readout Layer
最后，我们添加一个softmax层，就像前面的`Softmax regression`一样。


```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

## Train and Evaluate
为了进行训练和评估，我们使用与之前简单的单层Softmax神经网络模型几乎相同的一套代码，只是我们将用更加复杂的ADAM优化器替换梯度下降，在`feed_dict`中加入额外的参数`keep_prob`来控制`dropout`比例。

我们也将使用`tf.Session`而不是`tf.InteractiveSession`。这更好地分离了创建图形（模型分离）的过程和评估图形（模型拟合）的过程。它通常使代码清洁。`tf.Session`是在一个块内创建的，一旦该块被退出就自动销毁。

随意运行这段代码。请注意，它会执行2000次训练迭代，每200次迭代输出一次日志，可能需要一段时间（2分钟），具体取决于您的处理器。


```python
y_ = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import datetime
now = datetime.datetime.now()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 200 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print('time: %s' % (datetime.datetime.now() - now))
```

    step 0, training accuracy 0.12
    step 200, training accuracy 0.9
    step 400, training accuracy 0.92
    step 600, training accuracy 0.96
    step 800, training accuracy 0.94
    step 1000, training accuracy 1
    step 1200, training accuracy 0.98
    step 1400, training accuracy 0.96
    step 1600, training accuracy 0.96
    step 1800, training accuracy 0.96
    test accuracy 0.9758
    time: 0:02:12.860346


到此，我们已经学会了如何使用TensorFlow快速，轻松地构建，训练和评估一个相当复杂的深度学习模型。
