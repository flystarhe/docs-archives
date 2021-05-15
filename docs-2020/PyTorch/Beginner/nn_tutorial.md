# NN Tutorial
PyTorch提供了设计精美的模块和类`torch.nn`，`torch.optim`，`Dataset`和`DataLoader`来帮助您创建和训练神经网络。为了充分利用它们的功能并针对您的问题对其进行自定义，您需要真正了解它们在做什么。为了建立这种理解，我们将首先在MNIST数据集上训练基本神经网络，而无需使用这些模型的任何功能；我们最初只会使用最基本的PyTorch张量功能。然后，我们将逐步改进，正好显示每一块做什么，以及如何使代码更简洁灵活。

## MNIST
```
import gzip
import pickle
import requests
from pathlib import Path

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```

每个图像为`28x28`，并存储为长度为`784(28x28)`的扁平行。我们需要先将其重塑为2d。
```
from matplotlib import pyplot

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
## (50000, 784)
```

PyTorch使用`torch.tensor`，而不是numpy数组，因此我们需要转换数据。
```
import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

print(x_train.shape)
print(y_train.min(), y_train.max())
## torch.Size([50000, 784])
## tensor(0) tensor(9)
```

## Neural Net From Scratch
首先，我们仅使用PyTorch张量操作创建模型。我们假设您已经熟悉神经网络的[基础知识](https://course.fast.ai/)。PyTorch提供了创建随机或零填充张量的方法，我们将使用它们来为简单的线性模型创建权重和偏差。这些只是常规张量，还有一个非常特殊的附加值：我们告诉PyTorch它们需要梯度。对于权重，我们在初始化后设置了`require_grad`，因为我们不希望初始化包含在梯度中。（请注意，PyTorch中的尾随`_`表示该操作是就地执行的。）
```
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()

bias = torch.zeros(10, requires_grad=True)
```

由于PyTorch具有自动计算梯度的功能，我们可以将任何标准的Python函数（或可调用对象）用作模型！因此，让我们编写一个普通矩阵乘法和广播加法来创建一个简单的线性模型。我们还需要一个激活函数，因此我们将编写`log_softmax`并使用它。尽管PyTorch提供了许多预写的损失函数，激活函数等，但是您可以使用纯Python轻松编写自己的函数。PyTorch甚至会自动为您的函数创建快速GPU或矢量化的CPU代码。
```
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
```

在上面，`@`代表点积运算。我们将对一批数据（64张图像）调用函数。由于我们从随机权重开始，因此在这一阶段，我们的预测不会比随机预测更好。
```
bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
print(preds[0], preds.shape)
## tensor([-2.5563, -2.6373, -2.4165, -2.1595, -2.6809, -1.8110, -2.6341, -2.8322,
##         -2.1175, -1.8124], grad_fn=<SelectBackward>) torch.Size([64, 10])
```

`preds`张量不仅包含张量值，还包含梯度函数。稍后我们将使用它进行反向传播。让我们实现对数似然法以用作损失函数：
```
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
```

让我们用随机模型来检查损失，以便我们以后看向后传播后是否可以改善。
```
yb = y_train[0:bs]
print(loss_func(preds, yb))
## tensor(2.4395, grad_fn=<NegBackward>)
```

我们还实现一个函数来计算模型的准确性。对于每个预测，如果具有最大值的索引与目标值匹配，则该预测是正确的。
```
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))
## tensor(0.1562)
```

现在，我们可以运行一个训练循环。对于每次迭代，我们将：

- 选择一个小批量的数据
- 使用模型进行预测
- 计算损失
- 更新模型的梯度

我们使用这些梯度来更新权重和偏差。我们在`torch.no_grad()`上下文管理器中执行此操作，因为我们不希望在下一步的梯度计算中记录这些操作。您可以在[此处](https://pytorch.org/docs/stable/notes/autograd.html)详细了解PyTorch的Autograd如何记录操作。

然后，将渐变设置为零，以便为下一个循环做好准备。默认，我们的渐变将记录所有已发生操作的运行记录（即`loss.backward()`将渐变添加到已存储的内容中，而不是替换它们）。您可以使用标准的python调试器逐步浏览PyTorch代码，从而可以在每一步检查各种变量值。取消注释下面的`set_trace()`即可尝试。
```
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
```

就是这样：我们完全从头开始创建并训练了一个最小的神经网络（在这种情况下，是逻辑回归，因为我们没有隐藏的层）！让我们检查损失和准确性，并将其与我们之前获得的进行比较。我们希望损失会减少，准确性会增加，而且确实如此。
```
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
## tensor(0.0838, grad_fn=<NegBackward>) tensor(1.)
```

## Using `torch.nn.functional`
现在，我们将重构代码，使其与以前相同，只是我们将开始利用PyTorch的nn类使其更简洁，更灵活。

第一步也是最简单的步骤，就是用`torch.nn.functional`替换我们的手写激活和丢失函数，从而缩短代码长度。该模块包含`torch.nn`库中的所有函数（而该库的其他部分包含类）。除了广泛的损失和激活函数外，您还可以在这里找到一些方便的函数来创建神经网络，例如池化函数。（还有一些用于进行卷积，线性图层等的函数，但是正如我们将看到的那样，通常可以使用库的其他部分来更好地处理这些函数。）

如果您使用的是负对数似然损失和对数`softmax`激活，那么Pytorch会提供一个将两者结合的函数`F.cross_entropy`。因此，我们甚至可以从模型中删除激活函数。
```
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
```

注意，我们不再在模型函数中调用`log_softmax`。让我们确认我们的损失和准确性与以前相同：
```
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
## tensor(0.0838, grad_fn=<NllLossBackward>) tensor(1.)
```

## Refactor Using `nn.Module`
接下来，我们将使用`nn.Module`和`nn.Parameter`来实现更清晰，更简洁的训练循环。我们将`nn.Module`子类化（它本身是一个类并且能够跟踪状态）。在这种情况下，我们要创建一个类，该类包含权重，偏执和向前方法。`nn.Module`具有许多我们将要使用的属性和方法（例如`.parameters()`和`.zero_grad()`）。
```
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
```

由于我们现在使用的是对象而不是仅使用函数，因此我们首先必须实例化模型：
```
model = Mnist_Logistic()
```

现在我们可以像以前一样计算损失。请注意，`nn.Module`对象的使用就像它们是函数一样，但是在后台Pytorch将自动调用我们的`forward`方法。
```
print(loss_func(model(xb), yb))
## tensor(2.2969, grad_fn=<NllLossBackward>)
```

以前，在我们的训练循环中，我们必须按名称更新每个参数的值，并手动将每个参数的`grads`分别归零，如下所示：
```
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()
```

现在我们可以利用`model.parameters()`和`model.zero_grad()`（它们都由PyTorch为`nn.Module`定义）来使这些步骤更简洁，并且更不会出现忘记某些参数的错误，尤其是当我们有一个更复杂的模型时：
```
with torch.no_grad():
    for p in model.parameters():
        p -= p.grad * lr
    model.zero_grad()
```

我们将小小的训练循环包装在`fit`函数中，以便稍后再运行。
```
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
```

让我们仔细检查一下我们的损失是否减少了：
```
print(loss_func(model(xb), yb))
## tensor(0.0829, grad_fn=<NllLossBackward>)
```

## Refactor Using `nn.Linear`
我们继续重构我们的代码。Pytorch具有许多类型的预定义层，可以大大简化我们的代码，并且通常也可以使其速度更快。
```
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
```

我们以与以前相同的方式实例化模型并计算损失：
```
model = Mnist_Logistic()
print(loss_func(model(xb), yb))
## tensor(2.3089, grad_fn=<NllLossBackward>)
```

我们仍然可以使用与以前相同的拟合方法。
```
fit()

print(loss_func(model(xb), yb))
## tensor(0.0818, grad_fn=<NllLossBackward>)
```

## Refactor Using `optim`
Pytorch还提供了一个包含各种优化算法的软件包`torch.optim`。我们可以使用优化器中的`step`方法采取向前的步骤，而不是手动更新每个参数。这将使我们替换之前的手动编码优化步骤：
```
with torch.no_grad():
    for p in model.parameters():
        p -= p.grad * lr
    model.zero_grad()
```

而是只使用：
```
opt.step()
opt.zero_grad()
```

`optim.zero_grad()`将梯度重置为0，我们需要在计算下一个小批量的梯度之前调用它。我们将定义一个小函数来创建模型和优化器，以便将来再次使用。
```
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
## tensor(2.3519, grad_fn=<NllLossBackward>)
## tensor(0.0824, grad_fn=<NllLossBackward>)
```

## Refactor Using `Dataset`
PyTorch有一个抽象的Dataset类。数据集可以是具有`__len__`函数（由Python的标准`len`函数调用）和`__getitem__`函数作为对其进行索引的一种方法。PyTorch的`TensorDataset`是包装张量的数据集。通过定义索引的长度和方式，这也为我们提供了沿张量的第一维进行迭代，索引和切片的方法。这将使我们在训练的同一行中更容易访问自变量和因变量。`x_train`和`y_train`都可以组合在一个`TensorDataset`中，这将更易于迭代和切片。
```
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
## tensor(0.0816, grad_fn=<NllLossBackward>)
```

## Refactor Using `DataLoader`
Pytorch的`DataLoader`负责管理批次。您可以从任何数据集创建一个`DataLoader`。`DataLoader`使迭代批次变得更加容易。
```
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
## tensor(0.0819, grad_fn=<NllLossBackward>)
```

## Add Validation
我们只是试图建立一个合理的训练循环以用于我们的训练数据。实际上，您始终还应该具有一个验证集，以识别您是否过度拟合。我们将验证集的批次大小设为训练集的两倍。这是因为验证集不需要反向传播，因此占用的内存更少（不需要存储渐变）。我们利用这一优势来使用更大的批量，并更快地计算损失。
```
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
## 0 tensor(0.3051)
## 1 tensor(0.2858)
```

## Create `fit()` And `get_data()`
由于我们经历了两次相似的过程来计算训练集和验证集的损失，让我们将其设为自己的函数`loss_batch`。
```
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
## 0 0.37742782249450685
## 1 0.3059203647851944
```

## Switch To CNN
现在，我们将构建具有三个卷积层的神经网络。由于前面的任何函数均不假设任何有关模型形式的内容，因此我们将能够使用它们来训练CNN，而无需进行任何修改。
```
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
## 0 0.3528137544393539
## 1 0.28793846560120584
```

## Closing
```
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

loss_func = F.cross_entropy

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

bs = 64
lr = 0.1
epochs = 2
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
## 0 0.3731710307359695
## 1 0.26514258354902265

def accuracy(x, y):
    with torch.no_grad():
        out = model(x)
    preds = torch.argmax(out, dim=1)
    return (preds == y).float().mean()

print(accuracy(x_train, y_train))
print(accuracy(x_valid, y_valid))
## tensor(0.9172)
## tensor(0.9228)
```

## 参考资料：
- [WHAT IS TORCH.NN REALLY?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)