# Distributed data parallel
`DistributedDataParallel`（DDP）在模块级别实现数据并行性。它使用`torch.distributed`包中的通信集合体来同步梯度，参数和缓冲区。并行性在流程内和跨流程均可用。在一个流程中，DDP将输入模块复制到`device_ids`中指定的设备中，将输入沿批次维度分散，然后将输出收集到`output_device`中，类似于`DataParallel`。在整个过程中，DDP在正向传递中插入必要的参数同步，在反向传递中插入梯度同步。用户可以将进程映射到可用资源，只要进程不共享GPU设备即可。推荐的（通常是最快的）方法是为每个模块副本创建一个进程，即，在进程内不进行任何模块复制。本教程中的代码在8GPU服务器上运行，但可以轻松地推广到其他环境。

- `rank`取值范围为`[0, world_size)`
- `world_size`指定进程数量，建议设置与GPU数量相同
- 进程内部存在并行，建议依据`torch.cuda.device_count() // world_size`分配GPU

>为了在每个节点上生成多个进程，可以使用`torch.distributed.launch`或`torch.multiprocessing.spawn`。

## DataParallel vs DistributedDataParallel
为什么尽管增加了复杂性，但您还是考虑使用`DistributedDataParallel`覆盖`DataParallel`：

- 回想一下模型并行的教程，如果模型太大而无法容纳在单个GPU上，则必须使用并行模型将其拆分到多个GPU中。`DistributedDataParallel`可与模型并行工作，`DataParallel`目前没有。
- `DataParallel`是单进程，多线程的，并且只能在单台机器上运行；而单机`DistributedDataParallel`是多进程的，并且可以在单机和多机训练中使用。因此，即使对于单机训练，您的数据也足够小以适合单台机器，但`DistributedDataParallel`仍预计会比`DataParallel`快。`DistributedDataParallel`还可以预先复制模型，而不是在每次迭代时复制模型，并且可以避免全局解释器锁定。
- 如果您的两个数据都太大而无法容纳在一台计算机上，而您的模型又太大了以至于无法容纳在单个GPU上，则可以将并行模型（在多个GPU上拆分单个模型）与`DistributedDataParallel`结合。在这种情况下，每个`DistributedDataParallel`流程可以并行使用模型，而所有流程共同使用并行数据。

## Basic Use Case
要创建DDP模块，请首先正确设置进程组。在[使用PyTorch编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html)中可以找到更多详细信息。
```python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()
```

现在，让我们创建一个玩具模块，将其与DDP封装在一起，并提供一些虚拟输入数据。请注意，如果训练是从随机参数开始的，则可能要确保所有DDP进程都使用相同的初始值。否则，全局梯度同步将没有意义。
```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 2)
```

>直接在ipynb文件中执行可能会不成功，推荐的方式是把代码写入`lab.py`，然后ipynb中执行`!python lab.py`。

## Skewed Processing Speeds
在DDP中，构造函数，转发方法和输出的差异是分布式同步点。期望不同的过程以相同的顺序到达同步点，并在大致相同的时间进入每个同步点。否则，快速的流程可能会提早到达，并在等待流浪者时超时。因此，用户负责平衡流程之间的工作负载分配。有时，由于（例如）网络延迟，资源争用，不可预测的工作负载峰值，不可避免地会出现处理速度偏差。为了避免在这些情况下超时，请确保`timeout`在调用`init_process_group`时传递足够大的值。

## Save and Load Checkpoints
在训练期间使用`torch.save`保存检查点和`torch.load`从检查点恢复是很常见的。有关更多详细信息，请参见[保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)。使用DDP时，一种优化方法是仅在一个进程中保存模型，然后将其加载到所有进程中，从而减少写开销。这是正确的，因为所有过程都从相同的参数开始，并且梯度在向后传递中同步，因此优化程序应将参数设置为相同的值。如果使用此优化，请确保在保存完成之前不要启动所有进程。此外，在加载模块时，您需要提供适当的`map_location`参数以防止进程进入其他人的设备。如果`map_location`丢失，`torch.load`首先将模块加载到CPU，然后将每个参数复制到保存位置，这将导致同一台计算机上的所有进程使用同一组设备。
```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```

## Combine DDP with Model Parallelism
DDP还可以与多GPU模型一起使用，但是不支持进程内的复制。您需要为每个模块副本创建一个进程，与每个进程多个副本相比，通常可以提高性能。当训练具有大量数据的大型模型时，DDP包装多GPU模型特别有用。使用此功能时，需要小心地实现多GPU模型，以避免使用硬编码的设备，因为会将不同的模型副本放置到不同的设备上。
```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

当DDP与一个多GPU模型结合时，`device_ids`和`output_device`不能设置。输入和输出数据将通过应用程序或模型`forward()`方法放置在适当的设备中。
```python
def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    run_demo(demo_basic, 2)
    run_demo(demo_checkpoint, 2)

    if torch.cuda.device_count() >= 8:
        run_demo(demo_model_parallel, 4)
```

## 参考资料：
- [GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)