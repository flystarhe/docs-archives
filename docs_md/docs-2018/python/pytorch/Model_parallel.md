# Model parallel
模型并行在分布式训练技术中被广泛使用。先前的文章已经解释了如何使用`DataParallel`在多个GPU上训练神经网络。此功能将相同的模型复制到所有GPU，其中每个GPU消耗输入数据的不同分区。尽管它可以极大地加快训练过程，但不适用于某些模型太大而单个GPU无法容纳的用例。这篇文章展示了如何通过使用并行模型来解决该问题，与`DataParallel`相反，将模型拆分到不同的GPU上，而不是在每个GPU上复制整个模型（具体来说，模型`m`包含10层：使用时`DataParallel`，则每个GPU都具有这10层中每个层的副本，而在两个GPU上并行模型时，每个GPU可以托管5层）。

模型并行化的高级思想是将模型的不同子网放置到不同的设备上，并相应地实现`forward`方法以在设备之间移动中间输出。由于模型的一部分只能在任何单个设备上运行，因此一组设备可以共同为更大的模型服务。

## Basic Usage
让我们从包含两个线性层的玩具模型开始。要在两个GPU上运行此模型，只需将每个线性层放在不同的GPU上，然后移动输入和中间输出以匹配层设备。
```python
import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
```

调用损失函数时，只需确保标签与输出在同一设备上。
```python
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()
```

## Apply Model Parallel to Existing Modules
只需更改几行，就可以在多个GPU上运行现有的单GPU模块。以下代码显示了如何分解`torchvision.models.reset50()`为两个GPU。想法是从现有ResNet模块继承，并在构建过程中将层拆分为两个GPU。然后，`forward`通过相应地移动中间输出来覆盖用于缝合两个子网的方法。
```python
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
```

对于模型太大而无法放入单个GPU的情况，上述实现解决了该问题。但是，您可能已经注意到，如果模型合适，它将比在单个GPU上运行它要慢。这是因为在任何时间点，两个GPU中只有一个在工作，而另一个在那儿什么也没做。性能进一步恶化作为中间输出需要从`cuda:0`复制到`cuda:1`。

让我们进行实验以更定量地了解执行时间。在此实验中，我们通过运行随机输入和标签来训练`ModelParallelResNet50`和现有对象`torchvision.models.reset50()`。
```python
import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
```

我们使用该方法运行10次​​：
```python
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
mp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
```

结果表明，模型并行实现的执行时间`4.02/3.75-1=7%`比现有的单GPU实现更长。因此，我们可以得出结论，在GPU之间来回复制张量大约有7％的开销。有待改进的地方，因为我们知道两个GPU之一在整个执行过程中处于空闲状态。一种选择是将每个批次进一步划分为拆分流水线，以便当一个拆分到达第二子网时，可以将下一个拆分馈入第一子网。这样，两个连续的拆分可以在两个GPU上同时运行。

## Speed Up by Pipelining Inputs
我们将每个批次120图像进一步划分为20图像分割。当PyTorch异步启动CUDA操作时，该实现无需生成多个线程即可实现并发。
```python
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')

        ret = []
        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
```

请注意，设备到设备的张量复制操作在源设备和目标设备上的当前流上同步。如果创建多个流，则必须确保复制操作正确同步。在完成复制操作之前写入源张量或读取/写入目标张量可能导致不确定的行为。上面的实现仅在源设备和目标设备上都使用默认流，因此没有必要强制执行其他同步。

实验结果表明，对并行ResNet50进行建模的流水线输入可大致加快训练过程`3.75/2.51-1=49%`。距离理想的100％加速还有很长的路要走。由于我们`split_sizes`在管道并行实现中引入了新参数，因此尚不清楚新参数如何影响整体培训时间。直观地讲，使用较小的结果`split_size`会导致许多微小的CUDA内核启动，而使用较大的`split_size`结果会导致在第一个和最后一个分割期间出现较长的空闲时间。都不是最佳选择。`split_size`此特定实验可能有最佳配置。让我们尝试通过使用几个不同的`split_size`值进行实验来找到它。

## 参考资料：
- [MODEL PARALLEL BEST PRACTICES](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)