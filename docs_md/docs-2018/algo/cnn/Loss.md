# Loss
目前已经有很多损失函数，对于损失函数的选择依赖于具体任务。然而，所有损失函数具有一个共同特性，它必须能以精确的数学表达式表示损失函数。

- 交叉熵误差：通常用于分类任务。
- Dice损失(IoU)：通常用于分割任务。
- KL散度：用于衡量两种分布之间的差异。
- L0损失函数定义为：\\( (| f_{\theta}(\hat{x}) - \hat{y} | + \epsilon)^{\gamma} \\)，其中\\( \epsilon = 10^{-8} \\)，其中\\( \gamma \\)在训练期间从2线性退火至0。
- L1损失函数定义为：\\( (| f_{\theta}(\hat{x}) - \hat{y} |)^1 \\)，在观察的中位数处具有最佳值。绝对误差，多用于回归任务。
- L2损失函数定义为：\\( (| f_{\theta}(\hat{x}) - \hat{y} |)^2 \\)，在观察的平均值处具有最佳值。与L1类似，但对于异常值更加敏感。

## F
Sigmoid:
$$
\begin{aligned}
f(x) = \frac{1}{1 + \exp(-x)}
\end{aligned}
$$

Softmax:
$$
\begin{aligned}
f(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
\end{aligned}
$$

Tanh:
$$
\begin{aligned}
f(x) = \frac{ e^{x} - e^{-x} }{ e^{x} + e^{-x} }
\end{aligned}
$$

## HDR
⾼动态范围图像质量度量是相对MSE（标准MSE损失将由⽬标中长尾效应异常值⽀配），其中平⽅差除以像素的近似亮度的平⽅，损失函数定义为：
$$
\begin{aligned}
(f_{\theta}(\hat{x}) - \hat{y})^2 / (\hat{y} + \epsilon)^2
\end{aligned}
$$

该度量存在的⾮线性问题，我们建议替换为：
$$
\begin{aligned}
(f_{\theta}(\hat{x}) - \hat{y})^2 / (f_{\theta}(\hat{x}) + 0.01)^2
\end{aligned}
$$

## MSE
MSE是常见的Loss函数，就是均方平方差(Mean Squared Error)，定义如下：
$$
\begin{aligned}
l(x,y) = L = \{l_1, \,...\, , l_N\}^T, \, l_n = (x_n - y_n)^2
\end{aligned}
$$

平方差可以表达预测值与真实值的差异，但在分类问题中效果并不如交叉熵好，原因可以参考[这篇博文](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)。

## 交叉熵(Cross-Entropy)
交叉熵损失(cross-entropy Loss)又称为对数似然损失(Log-likelihood Loss)、对数损失，二分类时还可称之为逻辑斯谛回归损失(Logistic Loss)。交叉熵损失函数表达式为\\( L = - \sum_{i=1}^{N} y_i log(x_i) \\)。pytroch这里不是严格意义上的交叉熵损失函数，而是先将input经过softmax激活函数，将向量“归一化”成概率形式，然后再与target计算严格意义上交叉熵损失。在多分类任务中，经常采用softmax激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要softmax激活函数将一个向量进行“归一化”成概率分布的形式，再采用交叉熵损失函数计算loss。再回顾PyTorch的`CrossEntropyLoss()`，官方文档中提到时将`nn.LogSoftmax()`和`nn.NLLLoss()`进行了结合，`nn.LogSoftmax()`相当于激活函数，`nn.NLLLoss()`是损失函数。

交叉熵可以作为Loss函数的原因：首先是交叉熵得到的值一定是正数，其次是预测结果越准确值越小，反之Loss函数为无限大，非常符合我们对Loss函数的定义。

## pytorch

### nll_loss
负对数似然损失，对C分类问题是有用的。通过在网络的最后一层添加`LogSoftmax`层，可以轻松获得神经网络中的对数概率。如果您不希望添加额外的层，则可以改用`CrossEntropyLoss`。损失预期的目标应该是范围内的类别索引`[0, C-1]`，其中C为类别数量。如果指定了`ignore_index`，则此损失也将接受该类索引（此索引可能不一定在类范围内）。

### binary_cross_entropy/BCELoss/BCEWithLogitsLoss
度量目标和输出之间的二分类交叉熵。

$$
\begin{aligned}
l(x,y) = L = \{l_1, \,...\, , l_N\}^T, \, l_n = - [y_n log(x_n) + (1 - y_n) log(1 - x_n)]
\end{aligned}
$$

- where N is the batch size.
- targets y should be numbers between 0 and 1.

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
input = autograd.Variable(torch.randn(3, 2), requires_grad=True)
target = autograd.Variable(torch.FloatTensor(3, 2).random_(2))
loss = F.binary_cross_entropy(F.sigmoid(input), target)
loss.backward()
```

### cross_entropy/CrossEntropyLoss
将`LogSoftMax`和`NLLLoss`合并到一个类中，对C分类问题是有用的。

$$
\begin{aligned}
loss(x, class) = - log \Bigg ( \frac{\exp(x[class])}{\sum_j \exp(x[j])} \Bigg ) \\
loss(x, class) = - x[class] + log \Bigg ( \sum_j \exp(x[j]) \Bigg )
\end{aligned}
$$

- input has to be a 2D Tensor of size (minibatch, C).
- target is a 1D tensor of size minibatch, a class index (0 to C-1) for each value.

## TensorFlow
`sigmoid_cross_entropy_with_logits`的输入是logits和targets，而targets的shape和logits相同，就是正确的label值。注释中还提到分类之间是独立的，不要求是互斥，这种问题我们称为多目标。例如判断图片中是否包含10种动物，label值可以包含多个1或0个1。还有一种问题是多分类问题，例如我们对年龄特征分为5段，只允许5个值有且只有1个值为1，这种问题可以直接用这个函数吗？答案是不可以。

为什么`softmax_cross_entropy_with_logits`只适合单目标的二分类或者互斥的多分类问题(只有一个分类目标)？

这就是TensorFlow目前提供的有关Cross Entropy的函数实现，用户需要理解多目标和多分类的场景，根据业务需求(分类目标是否独立和互斥)来选择基于`sigmoid`或者`softmax`的实现。

## 参考资料：
- [Loss functions](https://pytorch.org/docs/stable/nn.functional.html#loss-functions)