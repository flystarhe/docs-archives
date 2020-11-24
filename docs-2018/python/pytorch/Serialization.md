# Serialization
本文档为有关保存和加载PyTorch模型的各种用例提供​​了解决方案。[doc](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## What is a `state_dict`?
在PyTorch中，模型的可学习参数(即权重和偏差)，包含在`torch.nn.Module`的参数中，通过访问`model.parameters()`。`state_dict`是一个简单的Python字典对象，每个层映射到其参数张量。请注意，只有具有可学习参数的层(卷积层/线性层等)和已注册的缓冲区(`batchnorm`的`running_mean`)才在模型的`state_dict`中具有条目。优化器对象`torch.optim`也具有`state_dict`，其中包含有关优化器状态以及所用超参数的信息。让我们从简单模型看一下`state_dict`。
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print("{:<30}".format(param_tensor), model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print("{:<30}".format(var_name), optimizer.state_dict()[var_name])
```

输出：
```
Model's state_dict:
conv1.weight                   torch.Size([16, 3, 7, 7])
bn1.weight                     torch.Size([16])
bn1.bias                       torch.Size([16])
bn1.running_mean               torch.Size([16])
bn1.running_var                torch.Size([16])
bn1.num_batches_tracked        torch.Size([])
fc.weight                      torch.Size([10, 16])
fc.bias                        torch.Size([10])
Optimizer's state_dict:
state                          {}
param_groups                   [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [140479599097464, 140479599099048, 140479599098976, 140479599098184, 140479599098472]}]
```

## Saving & Loading Model for Inference
Save:
```python
torch.save(model.state_dict(), PATH)
```

Load:
```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

请记住，`model.eval()`在运行推理之前，必须先设置`dropout`和`batch normalization`图层到评估模式。如果不这样做，将导致不一致的推理结果。

## Saving & Loading a General Checkpoint for Inference and/or Resuming Training
Save:
```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
```

Load:
```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

保存用于检查或继续训练的常规检查点时，您必须保存的不仅仅是模型的`state_dict`。保存优化器的`state_dict`也是很重要的，因为它包含随着模型训练而更新的缓冲区和参数。要加载项目，请首先初始化模型和优化器，然后使用本地加载字典`torch.load()`。

## Saving & Loading Model Across Devices
Save on GPU, Load on CPU:
```python
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

Save on GPU, Load on GPU:
```python
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

Save on CPU, Load on GPU:
```python
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

Saving `torch.nn.DataParallel` Models:
```python
torch.save(model.module.state_dict(), PATH)

# Load to whatever device you want
```

## Load Checkpoint
```python
import os
import torch
import torch.distributed as dist
from torch.hub import load_state_dict_from_url


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): A filepath or URL or torchvision://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('torchvision://'):
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)

    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_url_dist(url):
    """ In distributed setting, this function only download checkpoint at local rank 0 """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = load_state_dict_from_url(url)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_state_dict_from_url(url)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    shape_mismatch_pairs = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[name].size():
            shape_mismatch_pairs.append([name, own_state[name].size(), param.size()])
            continue
        own_state[name].copy_(param)

    all_missing_keys = set(own_state.keys()) - set(state_dict.keys())
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
    if shape_mismatch_pairs:
        mismatch_info = 'these keys have mismatched shape:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(mismatch_info + table.table)

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
```

[mmcv/runner/checkpoint.py](https://github.com/open-mmlab/mmcv)