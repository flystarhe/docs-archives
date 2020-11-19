title: PyTorch | Custom C extensions
date: 2018-02-05
tags: [PyTorch,C]
---
自定义的C扩展pytorch.[原文](http://pytorch.org/tutorials/advanced/c_extension.html)

<!--more-->
## 准备你的C代码
首先,你必须编写你的C函数.在下面,您可以找到一个模块的向前和向后函数的示例实现,该模块添加了两个输入.

在你的`.c`文件中,你可以使用`#include <TH/TH.h>`包含TH,而THC使用`#include <THC/THC.h>`.
```c
/* src/my_lib.c */
#include <TH/TH.h>

int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output)
{
    if (!THFloatTensor_isSameSizeAs(input1, input2))
        return 0;
    THFloatTensor_resizeAs(output, input1);
    THFloatTensor_cadd(output, input1, 1.0, input2);
    return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return 1;
}
```

代码没有限制,除了你将不得不准备一个头,这将列出所有要从Python调用的函数.
```c
/* src/my_lib.h */
int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);
int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
```

现在,您将需要一个超级短文件,它将构建您的自定义扩展:
```python
# build.py
from torch.utils.ffi import create_extension
ffi = create_extension(
    name='_ext.my_lib',
    headers='src/my_lib.h',
    sources=['src/my_lib.c'],
    with_cuda=False
)
ffi.build()
```

## 将其包含在您的Python代码中
运行后,pytorch会创建一个`_ext`目录并放入`my_lib`.软件包名称可以包含最终模块名称前面的任意数量的软件包(包括无).如果构建成功,您可以像常规python文件一样导入您的扩展.
```python
# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib


class MyAddFunction(Function):
    def forward(self, input1, input2):
        output = torch.FloatTensor()
        my_lib.my_lib_add_forward(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = torch.FloatTensor()
        my_lib.my_lib_add_backward(grad_output, grad_input)
        return grad_input
```

```python
# modules/add.py
from torch.nn import Module
from functions.add import MyAddFunction

class MyAddModule(Module):
    def forward(self, input1, input2):
        return MyAddFunction()(input1, input2)
```

```python
# main.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.add import MyAddModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

model = MyNetwork()
input1, input2 = Variable(torch.randn(5, 5)), Variable(torch.randn(5, 5))
print(model(input1, input2))
print(input1 + input2)
```
