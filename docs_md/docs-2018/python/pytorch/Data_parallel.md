# Data parallel
数据并行化是指将样本的微型批次分成多个较小的微型批次，并并行运行每个较小的微型批次的计算。数据并行使用`nn.DataParallel`来实现。`DataParallel`自动分割您的数据，并将作业发送到多个GPU上的多个模型。每个模型完成工作后，`DataParallel`会收集并合并结果，然后再将结果返回给您。

在Pytorch中使用GPU非常容易，可以将模型放在GPU上，也可以将张量复制到GPU：
```python
device = torch.device("cuda:0")

model.to(device)
mytensor = my_tensor.to(device)
```

>请注意，`my_tensor.to(device)`仅会返回在GPU上的新副本，而不是重写`my_tensor`。

在多个GPU上执行前向、后向传播是很自然的。但是，pytorch默认仅使用一个GPU。需要使用`DataParallel`使模型并行，从而在多个GPU上运行操作：
```python
model = nn.DataParallel(model)
```

## Imports and parameters
导入PyTorch模块并定义参数：
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Dummy DataSet
制作一个虚拟（随机）数据集：
```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)
```

## Simple Model
对于演示，我们的模型仅获取输入，执行线性运算并给出输出。但是，`DataParallel`可以在任何模型（CNN，RNN，Capsule Net等）上使用。我们已经在模型中放置了一条打印语句，以监视输入和输出张量的大小。
```python
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())
        return output
```

## Create Model and DataParallel
首先，我们需要制作一个模型实例，并检查是否有多个GPU。如果我们有多个GPU，可以使用`nn.DataParallel`封装模型。然后我们可以将模型放在GPU上`model.to(device)`。
```python
model = Model(input_size, output_size)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
```

## Run the Model
```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())
```

结果：
```python
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
...

Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
...
```

## 数据并行包装
```python
import torch
import torch.nn as nn


class DataParallelModel(nn.Module):

    def __init__(self):
        super(DataParallelModel, self).__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
```

## 数据并行原语
通常，可以独立使用`nn.parallel`。我们已经实现了类似于MPI的简单原语：

- `replicate`：在多个设备上复制模块
- `scatter`：在第一维上分配输入
- `gather`：收集和连接第一维中的输入
- `parallel_apply`：将一组已分配的输入应用于一组已分配的模型

```python
def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)
```

## 参考资料：
- [DATA PARALLELISM](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [MULTI-GPU EXAMPLES](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)