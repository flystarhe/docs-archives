# Pruning Tutorial
最新的深度学习技术依赖于难以部署的过度参数化模型。相反，已知生物神经网络使用有效的稀疏连通性。为了减少内存，电池和硬件消耗，找出减少模型中参数数量的最佳技术很重要，而又不牺牲准确性，在设备上部署轻量级模型，并通过私有设备上计算来确保隐私。在研究方面，修剪用于研究参数过多和参数不足网络之间学习动态的差异，研究幸运的稀疏子网络和初始化作为破坏性神经体系结构搜索技术的作用等。在本教程中，您将学习如何使用`torch.nn.utils.prune`稀疏化神经网络，以及如何扩展它以实现自己的自定义修剪技术。
```
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
```

## Create A Model
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)
```

## Inspect A Module
检查一下LeNet模型中的（未修剪的）`conv1`层。到目前为止，它将包含两个参数`weight`和`bias`，并且没有缓冲区。
```
module = model.conv1
print(list(module.named_parameters()))
print(list(module.named_buffers()))
```

## Pruning A Module
要修剪模块（在本示例中，为LeNet体系结构的`conv1`层），请首先在`torch.nn.utils.prune`中可用的修剪技术中选择一种修剪技术（或通过对`BasePruningMethod`进行子类化来实现自己的修剪技术）。然后，指定模块和该模块中要修剪的参数的名称。最后，使用所选修剪技术所需的适当关键字参数，指定修剪参数。

在此示例中，我们将在`conv1`层中的参数`weight`中随机修剪30％的连接。模块作为第一个参数传递给函数；名称使用其字符串标识符在该模块内标识参数；和数量表示连接到修剪的百分比（如果它是介于0和1之间的浮点数），或者表示到修剪的绝对连接数（如果它是一个非负整数）。
```
prune.random_unstructured(module, name="weight", amount=0.3)
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.weight)
```

修剪是通过从参数中删除权重并将其替换为名为`weight_orig`的新参数（即在初始参数名称后附加`_orig`）来进行的。`weight_orig`存储未修剪的张量版本。`bias`没有被修剪，因此它将保持不变。通过上面选择的修剪技术生成的修剪掩码被保存为名为`weight_mask`的模块缓冲区（即，在初始参数名称后附加` _mask`）。为了使前向通行不加修改，权重属性必须存在。在`torch.nn.utils.prune`中实现的修剪技术计算权重的修剪版本（通过将掩码与原始参数组合）并将其存储在属性权重中。注意，这不再是模块的参数，现在只是一个属性。

最后，使用PyTorch的`forward_pre_hooks`在每次正向传递之前应用修剪。具体来说，如我们在此处所做的那样，在修剪模块时，它将为与之相关的每个参数的修剪获取一个`forward_pre_hook`。在这种情况下，由于到目前为止我们只修剪了名为`weight`的原始参数，因此将只存在一个钩子。
```
print(module._forward_pre_hooks)
## OrderedDict([(0, <torch.nn.utils.prune.RandomUnstructured object at 0x7ff91860cb70>)])
```

为了完整起见，我们现在也可以修剪`bias`，以查看模块的参数，缓冲区，挂钩和属性如何变化。仅出于尝试另一种修剪技术的目的，在这里，我们按照`l1_unstructured`修剪函数中的实现，按`L1`范数修剪偏差中的3个最小条目。
```
prune.l1_unstructured(module, name="bias", amount=3)
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.bias)
print(module._forward_pre_hooks)
```

现在，我们希望命名的参数同时包含`weight_orig`（来自之前）和`bias_orig`。缓冲区将包括`weight_mask`和`bias_mask`。这两个张量的修剪版本将作为模块属性存在，并且该模块现在将具有两个`forward_pre_hooks`。

## Iterative Pruning
可以多次修剪模块中的同一参数，而各种修剪调用的效果等于串联应用的各种蒙版的组合。举例来说，假设我们现在要进一步修剪`module.weight`，这次使用沿张量的第0轴的结构化修剪（第0轴对应于卷积层的输出通道，并且对于conv1具有6维），根据渠道的L2规范。这可以使用`ln_structured(n=2, dim=0)`来实现。
```
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

# As we can verify, this will zero out all the connections corresponding to
# 50% (3 out of 6) of the channels, while preserving the action of the
# previous mask.
print(module.weight)
```

## Serializing A Pruned Model
所有相关的张量，包括掩码缓冲区和用于计算修剪的张量的原始参数，都存储在模型的`state_dict`中，因此可以根据需要轻松地序列化和保存。
```
print(model.state_dict().keys())
```

## Remove Pruning Re-Parametrization
要使修剪永久化，请根据`weight_orig`和`weight_mask`删除重新参数化，然后删除`forward_pre_hook`，我们可以使用`torch.nn.utils.prune`中的`remove`函数。请注意，这不会撤消修剪，好像从未发生过。通过将参数权重重新分配给模型参数（修剪后的版本），它只是使其永久化。

删除重新参数化之前：
```
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.weight)
```

删除重新参数化后：
```
prune.remove(module, 'weight')

print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.weight)
```

## Pruning Multiple Parameters In A Model
通过指定所需的修剪技术和参数，我们可以轻松地修剪网络中的多个张量，也许根据它们的类型，如在本示例中将看到的那样。
```
new_model = LeNet()
for name, module in new_model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist
## dict_keys(['conv1.weight_mask', 'conv2.weight_mask', 'fc1.weight_mask', 'fc2.weight_mask', 'fc3.weight_mask'])
```

## Global Pruning
到目前为止，我们仅研究了通常称为“局部”修剪的方法，即在模型中一对一修剪张量的做法，通过将每个条目的统计信息（权重，激活，梯度等）专门与该张量中的其他条目进行比较。但是，一种常见且可能更强大的技术是通过删除（例如）删除整个模型中最低的20％的连接，而不是删除每一层中最低的20％的连接来一次修剪模型。这很可能导致每个层的修剪百分比不同。让我们看看如何使用`torch.nn.utils.prune`中的`global_unstructured`来做到这一点。
```
model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

现在，我们可以检查在每个修剪参数中引起的稀疏性，该稀疏性将不等于每层中的20％。但是，全球稀疏度将（大约）为20％。

## Extending `torch.nn.utils.prune` With Custom Pruning Functions
要实现自己的修剪函数，可以通过将`nn.utils.prune`模块`BasePruningMethod`作为基类来扩展子类，这与所有其他修剪方法相同。基类为您实现以下方法：`__call__`，`apply_mask`，`apply`，`prune`和`remove`。除了一些特殊情况外，您不必为新的修剪技术重新实现这些方法。但是，您将必须实现`__init__`（构造函数）和`compute_mask`（有关如何根据修剪技术的逻辑为给定张量计算掩码的说明）。此外，您将必须指定该技术实现的修剪类型（受支持的选项是`global`，`structured`和`unstructured`）。需要确定在迭代应用修剪的情况下如何组合蒙版。换句话说，当修剪预修剪的参数时，当前的修剪技术应作用于参数的未修剪部分。指定`PRUNING_TYPE`将启用`PruningContainer`（用于处理修剪掩码的迭代应用程序）正确识别要修剪的参数的范围。

例如，假设您要实施一种修剪技术，以修剪张量中的所有其他条目（或者-如果先前已修剪过张量，则在张量的其余未修剪部分中）。这将是`PRUNING_TYPE='unstructured'`，因为它作用于层中的单个连接，而不作用于整个单元/通道（“结构化”）或不同参数（“全局”）。
```
class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask
```

现在，要将其应用于`nn.Module`中的参数，还应该提供一个简单的函数来实例化该方法并将其应用。
```
def foobar_unstructured(module, name):
    """
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    FooBarPruningMethod.apply(module, name)
    return module

model = LeNet()
foobar_unstructured(model.fc3, name='bias')

print(model.fc3.bias_mask)
## tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1.])
```

## 参考资料：
- [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)