title: TensorFlow Quick Start
date: 2017-07-10
tags: [TensorFlow]
---
[Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)

<!--more-->
## 基本概念
Tensor,Variable,placeholder,Session

### Tensor
张量是TensorFlow中的重要数据单元。张量由一组任意维度的数组的原始值组成。张量的`rank`是其维数。以下是张量的一些例子：


```python
3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]; # a rank 3 tensor with shape [2, 1, 3]
```


```python
import tensorflow as tf

tensor = tf.zeros(shape=[1,2])
print(tensor)

sess = tf.Session()
print(sess.run(tensor))
```

    Tensor("zeros:0", shape=(1, 2), dtype=float32)
    [[ 0.  0.]]


### Variable
是变量的意思，用来表示图中的各计算参数。与Tensor不同，Variable必须初始化以后才有具体的值。


```python
variable = tf.Variable(tensor)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(variable))
```

    [[ 0.  0.]]


### placeholder
又叫占位符，同样是一个抽象的概念。用于表示输入输出数据的格式。告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，不过正式运行的时候会补上的！因为没有具体数值，所以只要指定尺寸即可。


```python
x = tf.placeholder(tf.float32,[1, 5],name='input')
y = tf.placeholder(tf.float32,[None, 5],name='input')
```

上面有两种形式，第一种x，表示输入是一个`[1,5]`的横向量。第二种y，表示输入是一个`[None,5]`的矩阵。那么什么情况下会这么用呢`None`就是需要输入一批`[1,5]`的数据的时候。比如我有一批共10个数据，那我可以表示成`[10,5]`的矩阵。如果是一批5个，那就是`[5,5]`的矩阵。tensorflow会自动进行批处理。

### session
也就是会话。我的理解是，session是抽象模型的实现者。为什么之前的代码多处要用到session？因为模型是抽象的嘛，只有实现了模型以后，才能够得到具体的值。同样，具体的参数训练，预测，甚至变量的实际值查询，都要用到session。

## The Computational Graph

可能会认为`TensorFlow Core`程序由两个独立部分组成：

- 构建计算图。
- 运行计算图。

计算图每个节点采用零个或多个张量作为输入，并产生张量作为输出。

一种类型的节点是一个[常数](#)，它不需要任何输入，它输出一个内部存储的值。创建两个浮点传感器node1和node2，如下所示：


```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

    Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)


请注意，打印节点不会按预期输出值3.0和4.0。它们是在评估时分别产生3.0和4.0的节点。要实际评估节点，我们必须在会话中运行计算图。会话封装了TensorFlow运行时的控制和状态。

以下代码创建一个`Session`对象，然后调用其运行方法运行足够的计算图来评估node1和node2。通过在会话中运行计算图如下：


```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

    [3.0, 4.0]


我们可以通过将Tensor节点与操作（操作也是节点）组合来构建更复杂的计算。如下所示：


```python
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
```

    node3:  Tensor("Add:0", shape=(), dtype=float32)
    sess.run(node3):  7.0


TensorFlow提供了一个名为TensorBoard的实用程序，可以显示计算图的图片。

这个计算图不是特别有趣，因为它总是产生一个恒定的结果。可以将图形参数化为接受外部输入，称为[占位符](#)。占位符是以后提供值的承诺：


```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

前面的三行有点像一个函数，其中我们定义了两个输入参数（a和b），然后对它们进行一个操作。我们可以通过`feed_dict`参数来为这些占位符提供具体值：


```python
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))
```

    7.5
    [ 3.  7.]


我们可以通过添加另一个操作来使计算图更加复杂。例如：


```python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))
```

    22.5


在机器学习中，我们通常会想要一个可以接受任意输入的模型。为了使模型可训练，我们需要能够修改图形以获得具有相同输入的新输出。[Variables](#)允许我们向图中添加可训练的参数。它们的构造类型和初始值：


```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b
```

[调用`tf.constant`时，常量被初始化，](#)它们的值永远不会改变。相比之下，调用`tf.Variable`时，变量不会被初始化，要初始化TensorFlow程序中的所有变量，必须显式调用特殊操作，如下所示：


```python
init = tf.global_variables_initializer()
sess.run(init)
```

[在我们调用`sess.run`之前，这些变量是未初始化的。](#)由于`x`是占位符，我们可以同时评估`x`的几个值的`linear_model`，如下所示：


```python
print(sess.run(linear_model, feed_dict={x:[1,2,3,4]}))
```

    [ 0.          0.30000001  0.60000002  0.90000004]


我们创建了一个模型，但是我们不知道它有多好。为了评估模型，我们需要一个`y`占位符来提供所需的值，我们需要编写一个[损失函数](#)。

损失函数测量当前模型与提供的数据之间的距离。我们将使用线性回归的标准损失模型，其将当前模型和提供的数据之差的平方和：


```python
y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

    23.66


我们可以手动[重新分配变量的值](#)。假设，`W = -1`和`b = 1`是我们的模型的最优参数。我们可以相应地改变：


```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

sess.run([fixW, fixb])
print(sess.run(loss, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

    0.0


我们猜测的`W`和`b`是“完美”值，但机器学习的全部要点是自动找到正确的模型参数。我们将在下一节中展示如何完成此项工作。

### tf.train
TensorFlow提供了[优化器](#)，缓慢地更改每个变量，以便最大程度地减少损失函数。最简单的优化器是梯度下降。它根据相对于该变量的损失导数的大小修改每个变量。通常，手动计算符号导数是乏味且容易出错的。因此，TensorFlow可以使用该函数自动生成仅给出模型描述的导数`tf.gradients`：


```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
```

    [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]


## Complete program
完成的可训练线性回归模型如下所示：


```python
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, feed_dict={x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], feed_dict={x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

    W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11


## 新手入门

- 使用[张量](#)表示数据
- 使用[计算图](#)来表示计算任务
- 在被称之为[会话](#)的上线文中执行图
- 使用[feed](#)和[fetch](#)可以为任意操作赋值或从中获取数据

### 交互式使用
文档中的示例使用一个会话`Session`来启动图，并调用`Session.run()`方法执行操作。

为了便于使用诸如IPython之类的Python交互环境，可以使用`InteractiveSession`代替`Session`类，使用`Tensor.eval()`和`Operation.run()`方法代替`Session.run()`，这样可以避免使用一个变量来持有会话：


```python
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

x.initializer.run()

sub = tf.add(x, a)
print(sub.eval())
```

    [ 4.  5.]
