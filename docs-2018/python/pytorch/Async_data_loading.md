# Async data loading
在训练神经网络的时候，都是在CPU(准备数据)和GPU(模型学习)间交替进行。这种症状的体现是`nvidia-smi`查看GPU使用率时，GPU-Util时常为0。如何解决这种问题呢？在NVIDIA提出的分布式框架[Apex](https://github.com/NVIDIA/apex)里面，找到了一个简单的解决方案。
```python
d0 = TestDataset()
%timeit [d for d in d0]
# 2.04 s ± 9.75 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
d1 = AsyncDataset()
%timeit [d for d in d1]
# 153 µs ± 23.6 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Dataset
```python
import torch
import numpy as np


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

## TestDataset
```python
class TestDataset(Dataset):

    def __init__(self, length=10, mean=[0., 0., 0.], std=[1.0, 1.0, 1.0]):
        self.length = length
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda().view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).cuda().view(3, 1, 1)

    def __getitem__(self, index):
        if index < self.length:
            input = torch.full((3, 2048, 2048), index, dtype=torch.float32, device="cuda")
            target = torch.tensor(index, dtype=torch.float32, device="cuda")
            input.sub_(self.mean).div_(self.std)
            return input, target
        else:
            raise IndexError

    def __len__(self):
        return self.length
```

## AsyncDataset
```python
class AsyncDataset(Dataset):

    def __init__(self, length=10, shuffle=False, mean=[0., 0., 0.], std=[1.0, 1.0, 1.0]):
        self.length = length
        self.shuffle = shuffle
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda().view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).cuda().view(3, 1, 1)

        self.index_queue = list()
        self.index_iter = iter(self.index_queue)

        self.stream = torch.cuda.Stream()
        self.preload()

    def __getitem__(self, index):
        if index < self.length:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if input is not None:
                input.record_stream(torch.cuda.current_stream())
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
            self.preload()
            return input, target
        else:
            raise IndexError

    def preload(self):
        try:
            index = next(self.index_iter)
        except StopIteration:
            if self.shuffle:
                self.index_queue = np.random.permutation(self.length)
            else:
                self.index_queue = np.arange(self.length)
            self.index_iter = iter(self.index_queue)

            index = next(self.index_iter)

        with torch.cuda.stream(self.stream):
            self.next_input = torch.full((3, 2048, 2048), index, dtype=torch.float32, device="cuda")
            self.next_target = torch.tensor(index, dtype=torch.float32, device="cuda")
            self.next_input.sub_(self.mean).div_(self.std)

    def __len__(self):
        return self.length
```

## AsyncDataLoader
```python
import torch
import numpy as np
from torch.utils.data import DataLoader


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TempDataset(Dataset):

    def __init__(self, length=10):
        self.length = length

    def __getitem__(self, index):
        if index < self.length:
            return np.full((3, 4, 4), index, dtype=np.float32), index
        else:
            raise IndexError

    def __len__(self):
        return self.length


class AsyncDataLoader(object):

    def __init__(self, loader, mean=[0., 0., 0.], std=[1.0, 1.0, 1.0]):
        self.loader = iter(loader)
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).cuda().view(1, 3, 1, 1)

        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


if __name__ == "__main__":
    dataset = TempDataset()
    data_loader = DataLoader(dataset, batch_size=2, num_workers=2)
    async_loader = AsyncDataLoader(data_loader, [0., 0., 0.], [2.0, 2.0, 2.0])

    iteration = 0
    input, target = async_loader.next()
    while input is not None:
        iteration += 1
        input, target = async_loader.next()
```

## 参考资料：
- [apex/examples/imagenet](https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L258)
- [Optimizing PyTorch training code](https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/)