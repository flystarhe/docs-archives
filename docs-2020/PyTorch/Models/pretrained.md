# Pretrained

## 加载部分参数
模型可能是一些经典模型改掉一部分，比如一般算法中提取特征的网络常见的会直接使用`vgg/resnet`的`features extraction`部分，在训练的时候可以直接加载已经在imagenet上训练好的预训练参数，这种方式实现如下：
```
net = Net()
model_dict = net.state_dict()

import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)

pretrained_dict = vgg16.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
```

也可通过设置`strict`参数为`False`来忽略那些没有匹配到的`keys`：
```
net = Net()

import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)

net.load_state_dict(vgg16.state_dict(), strict=False)
```

## 微调经典网络
因为pytorch中的`torchvision`给出了很多经典常用模型，并附加了预训练模型。利用好这些训练好的基础网络可以加快不少自己的训练速度。首先比如加载vgg16（带有预训练参数的形式）：
```
import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)

import torch.nn as nn
vgg16.features[0] = nn.Conv2d(4, 64, 3, 1, 1)
```

## 修改经典网络
先简单介绍一下需要需改的部分，在vgg16的基础模型下，每一个卷积都要加一个`dropout`层，并将`ReLU`激活函数换成`PReLU`，最后两层的`Pooling`层`stride`改成`1`。
```
def feature_layer():
    layers = []
    pool1 = ['4', '9', '16']
    pool2 = ['23', '30']
    vgg16 = models.vgg16(pretrained=True).features
    for name, layer in vgg16._modules.items():
        if isinstance(layer, nn.Conv2d):
            layers += [layer, nn.Dropout2d(0.5), nn.PReLU()]
        elif name in pool1:
            layers += [layer]
        elif name == pool2[0]:
            layers += [nn.MaxPool2d(2, 1, 1)]
        elif name == pool2[1]:
            layers += [nn.MaxPool2d(2, 1, 0)]
        else:
            continue
    features = nn.Sequential(*layers)
    #feat3 = features[0:24]
    return features
```

大概的思路就是，创建一个新的网络`layers`列表，遍历`vgg16`里每一层，如果遇到卷积层就把该层保持原样加进去，随后增加一个`dropout`层，再加一个`PReLU`层。然后如果遇到最后两层`pool`，就修改响应参数加进去，其他的`pool`正常加载。最后将这个`layers`列表转成网络的`nn.Sequential`的形式，最后返回`features`。然后再你的新的网络层就可以用以下方式来加载：
```
class SNet(nn.Module):

    def __init__(self):
        super(SNet, self).__init__()
        self.features = feature_layer()

    def forward(self, x):
        x = self.features(x)
        return x
```

## 去除某些模块
下面是在不修改原模型代码的情况下，通过`resnet18.named_children()`和`resnet18.children()`的方法去除子模块`fc`和`avgpool`：
```
import torch
import torchvision.models as models
from collections import OrderedDict

resnet18 = models.resnet18(pretrained=False)
print("resnet18", resnet18)

# use named_children()
resnet18_v1 = OrderedDict(resnet18.named_children())
resnet18_v1.pop("avgpool")
resnet18_v1.pop("fc")
resnet18_v1 = torch.nn.Sequential(resnet18_v1)
print("resnet18_v1", resnet18_v1)

# use children()
resnet18_v2 = torch.nn.Sequential(*list(resnet18.children())[:-2])
print("resnet18_v2", resnet18_v2)
```
