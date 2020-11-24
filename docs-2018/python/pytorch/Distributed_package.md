# Distributed package
PyTorch（即`torch.distributed`）中包含的分布式软件包，使研究人员和从业人员可以轻松地跨进程和机器集群并行化其计算。为此，它利用传递消息的语义来允许每个进程将数据传递给其他任何进程。与`torch.multiprocessing`包相反，进程可以使用不同的通信后端，而不仅限于在同一台计算机上执行。为了开始，我们需要能够同时运行多个进程的能力。对于本教程而言，我们将使用一台计算机并使用以下模板来分叉多个进程。
```python
"""run.py:"""
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

上面的脚本产生了两个进程，每个进程将设置分布式环境，初始化进程组`dist.init_process_group`，最后执行给定的`run`函数。

>为了在每个节点上生成多个进程，可以使用`torch.distributed.launch`或`torch.multiprocessing.spawn`。

## Point-to-Point Communication
数据从一个进程到另一个进程的传输称为点对点通信。这些可通过`send`和`recv`函数，或`isend`以及`irecv`来实现。
```python
"""Blocking point-to-point communication."""
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

在上面的示例中，两个进程都从零张量开始，然后`进程0`递增张量并将其发送到`进程1`。请注意，`进程1`需要分配内存以存储它将接收的数据。

另请注意，`send/recv`是阻塞的，直到通信完成。另一方面，`isend/irecv`是非阻塞的，脚本继续执行，方法返回一个`Work`对象，可以选择对象的`wait()`。
```python
"""Non-blocking point-to-point communication."""
def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

## Collective Communication
与点对点通信相反，集合允许跨组中所有进程的通信模式。小组是我们所有进程的子集。要创建组，我们可以将等级列表传递给`dist.new_group(group)`。默认情况下，集合在所有进程（也称为world）上执行。例如，为了获得所有过程中所有张量的总和，我们可以使用`dist.all_reduce(tensor, op, group)`。
```python
"""All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```

由于我们需要组中所有张量的总和，因此我们将`dist.reduce_op.SUM`用作约简运算符。一般而言，任何可交换的数学运算都可以用作运算符。PyTorch开箱即用，带有4个这样的运算符，它们都在元素级运行：
```python
dist.reduce_op.SUM
dist.reduce_op.PRODUCT
dist.reduce_op.MAX
dist.reduce_op.MIN
```

除之外，PyTorch中目前共有7个集合通信：

- `dist.broadcast(tensor, src, group)`：从`src`复制`tensor`到所有其他进程
- `dist.reduce(tensor, dst, op, group)`：应用`op`于所有`tensor`并将结果存储在`dst`中
- `dist.all_reduce(tensor, op, group)`：与`reduce`相同，但是结果存储在所有进程中
- `dist.scatter(tensor, src, scatter_list, group)`：复制`i_th`张量`scatter_list[i]`到`i_th`进程
- `dist.gather(tensor, dst, gather_list, group)`：从所有进程复制`tensor`到`dst`中
- `dist.all_gather(tensor_list, tensor, group)`：从所有进程复制`tensor`到`tensor_list`，在所有进程上
- `dist.barrier(group)`：阻塞组内所有进程，直到每一个进程都进入了该函数

## Distributed Training
现在我们了解了分布式模块的工作原理，让我们编写一些有用的东西。我们的目标是复制[DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)的功能。当然，这将是一个教学示例，在现实世界中，您应该使用上面链接的经过官方测试，优化的最佳版本。

很简单，我们想要实现随机梯度下降的分布式版本。我们的脚本将允许所有进程在其数据批次上计算其模型的梯度，然后平均其梯度。为了在更改进程数时确保相似的收敛结果，我们首先必须对数据集进行分区。（您也可以使用[tnt.dataset.SplitDataset](https://github.com/pytorch/tnt/blob/master/torchnet/dataset/splitdataset.py#L4)，而不是下面的代码段）
```python
"""Dataset partitioning helper"""
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
```

使用上面的代码片段，我们现在可以使用以下几行简单地对任何数据集进行分区：
```python
"""Partitioning MNIST"""
def partition_dataset():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=trans)
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz
```

假设我们有2个副本，则每个进程将具有`train_set`的`60000/2=30000`个样本。我们还将批量大小除以副本数，以保持整体批量大小为128。

现在，我们可以编写我们通常的向前-向后-优化训练代码，并添加一个函数调用来平均模型的梯度。
```python
"""Distributed Synchronous SGD Example"""
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
```

仍然需要执行该`average_gradients(model)`功能，该功能只需要一个模型并在整个世界上平均其梯度即可。
```python
"""Gradient averaging."""
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
```

等等！我们成功实现了分布式同步SGD，并且可以在大型计算机集群上训练任何模型。（技术上是正确的，但要实现同步SGD的生产级实现还需要更多技巧）

## Communication Backends
`torch.distributed`最优雅的方面之一是它具有抽象能力，并可以在不同的后端之上构建。如前所述，目前在PyTorch中实现了三个后端：`Gloo`，`NCCL`和`MPI`。它们各自具有不同的规格和权衡，具体取决于所需的用例。可在[此处](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed)找到支持功能的比较表。经验法则：

- 使用NCCL后端进行分布式GPU训练
- 使用Gloo后端进行分布式CPU训练

### Gloo
到目前为止，我们已经广泛使用了Gloo后端。它作为开发平台非常方便，因为它已包含在预编译的PyTorch二进制文件中，并且可在Linux和macOS上运行。它支持CPU上的所有点对点和集合操作，以及GPU上的所有集合操作。CUDA张量的集体运算的实现未像NCCL后端提供的那样优化。

您肯定已经注意到，如果您`model`使用GPU，我们的分布式SGD示例将无法正常工作。为了使用多个GPU，让我们还进行以下修改：

1. `device = torch.device("cuda:{}".format(rank))`
2. `model = Net()`替换为`model = Net().to(device)`
3. `data, target = data.to(device), target.to(device)`

经过上述修改，我们的模型现在可以在两个GPU上训练，您可以使用`watch nvidia-smi`监视它们的使用。

### NCCL
[NCCL](https://github.com/nvidia/nccl)后端提供了一个优化的实现对CUDA张量共同操作的。如果仅将CUDA张量用于集体操作，请考虑使用此后端以获得最佳性能。NCCL后端包含在具有CUDA支持的预构建二进制文件中。

## Example

### Single node, multiple GPUs
[ImageNet implementation](https://github.com/pytorch/examples/tree/master/imagenet):
```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes
[ImageNet implementation](https://github.com/pytorch/examples/tree/master/imagenet):
```bash
#Node 0
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
#Node 1
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' \
--multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

### PyTorch distributed launch
用分布式训练，我们需要对脚本进行升级，使其能够独立的在节点中运行。我们想要完全实现分布式，并且在每个结点的每个GPU上独立运行进程。初始化分布式后端，封装模型以及准备数据，这些数据用于在独立的数据子集中训练进程。

`Single-Process Multi-GPU`，将在每个主机/节点上生成一个进程，并且每个进程将在运行该节点的节点的所有GPU上运行。除非模型非常大，必须模型并行，否则不推荐。
```python
torch.distributed.init_process_group(backend='nccl')
model = DistributedDataParallel(model) # device_ids will include all GPU devices by default
```

`Multi-Process Single-GPU`，强烈建议将`DistributedDataParallel`与多个进程配合使用，每个进程都在单个GPU上运行。这是目前使用PyTorch进行数据并行训练的最快方法，适用于单节点（multi-GPU）和多节点数据并行训练。
```python
# on each host with N GPUs, spawn up N processes
torch.cuda.set_device(i) # where i is from 0 to N-1
torch.distributed.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[i], output_device=i)
DistributedSampler(...)
```

使用`torch.distributed.launch`运行脚本。最主要的是第一台机器，所有的机器都要求能对它进行访问。因此，它需要拥有一个可以访问的IP地址以及一个开放的端口。例如：
```bash
#Node 0
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 \
--node_rank=0 --master_addr='192.168.1.1' --master_port=1234 \
train.py [--arg1 --arg2 ...]
#Node 1
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 \
--node_rank=1 --master_addr='192.168.1.1' --master_port=1234 \
train.py [--arg1 --arg2 ...]
```

这种方式启动任务，系统除了提供命令行参数`--local_rank=LOCAL_PROCESS_RANK`，还会在`os.environ`中添加环境变量`LOCAL_RANK,MASTER_ADDR,MASTER_PORT,RANK,WORLD_SIZE`。如果你觉得在计算机集群上运行一组几乎相同的命令有些枯燥，可点击此处了解[GNU并行](https://www.gnu.org/software/parallel/)。

## 参考资料：
- [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [DISTRIBUTED COMMUNICATION PACKAGE - TORCH.DISTRIBUTED](https://pytorch.org/docs/stable/distributed.html)