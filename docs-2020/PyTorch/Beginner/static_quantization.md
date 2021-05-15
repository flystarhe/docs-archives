# Static Quantization
本教程介绍了如何进行训练后的静态量化，并说明了两种更高级的技术-每通道量化和量化感知训练-可以进一步提高模型的准确性。请注意，目前仅支持CPU量化，因此在本教程中我们将不使用GPU/CUDA。在本教程结束时，您将看到PyTorch中的量化如何导致模型大小显着减小同时提高速度。此外，您将看到如何轻松应用[此处](https://arxiv.org/abs/1806.08342)显示的一些高级量化技术 ，以使您的量化模型受到的准确性打击比其他方式要少得多。我们将从进行必要的导入开始：
```
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

# Specify random seed for repeatable results
torch.manual_seed(191009)
```

## Model Architecture
我们首先定义`MobileNetV2`模型体系结构，并进行了一些值得注意的修改以实现量化：

- 替换`torch.add`为`nn.quantized.FloatFunctional`
- 在网络的开头和结尾处插入`QuantStub`和`DeQuantStub`
- 用`ReLU`替换`ReLU6`

注意：此代码来自[此处](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py)。
```
from torch.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)

        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)
```

## Helper Functions
接下来，我们定义一些帮助程序功能以帮助模型评估。这些大多来自[这里](https://github.com/pytorch/examples/blob/master/imagenet/main.py)。
```
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5
    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
```

## Define Dataset And Data Loaders
作为最后的主要设置步骤，我们为培训和测试集定义了数据加载器。我们为本教程创建的特定数据集仅包含来自ImageNet数据的1000张图像，每个类别都有一张（该数据集的大小刚好超过250 MB，可以相对轻松地下载）。
```
import requests

url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip`
filename = 'data/imagenet_1k.zip'

r = requests.get(url)
with open(filename, 'wb') as f:
    f.write(r.content)
```

另一方面，要使用整个ImageNet数据集运行本教程中的代码，可以使用`torchvision`[下载数据](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)。下载完数据后，我们在下面显示了一些函数，这些函数定义了用于读取该数据的数据加载器。这些功能主要来自[这里](https://github.com/pytorch/vision/blob/master/references/detection/train.py)。
```
def prepare_data_loaders(data_path):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test
```

接下来，我们将加载经过预先​​训练的MobileNetV2模型。我们在此处提供URL，以便从[torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py#L9)中下载数据。
```
# wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -P data
# wget https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip -P data
# cp data/mobilenet_v2-b0353104.pth data/mobilenet_pretrained_float.pth
# unzip -qo data/imagenet_1k.zip -d data

data_path = 'data/imagenet_1k'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')
```

接下来，我们将“融合模块”；通过节省内存访问量，这可以使模型更快，同时还可以提高数值精度。尽管这可以用于任何模型，但在量化模型中尤为常见。
```
print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)
```

最后，要获得“基准”精度，让我们看看带有融合模块的未量化模型的精度：
```
num_eval_batches = 10

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
## Size of baseline model
## Size (MB): 13.981375
## ..........Evaluation accuracy on 300 images, 78.00
```

## Post-Training Static Quantization
训练后的静态量化不仅涉及像动态量化中那样将权重从float转换为int，而且还涉及执行额外的步骤，即首先通过网络馈送一批数据并计算不同激活的结果分布（具体来说，这是通过在记录数据的不同点插入观察者模块来完成的）。然后将这些分布用于确定在推理时如何具体量化不同的激活（一种简单的技术将简单地将整个激活范围划分为256个级别，但我们也支持更复杂的方法）。重要的是，此附加步骤使我们能够在操作之间传递量化值，而不是在每次操作之间将这些值转换为浮点数，然后再转换为整数，从而显着提高了速度。
```
num_calibration_batches = 10

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
## Size of model after quantization
## Size (MB): 3.58906
## ..........Evaluation accuracy on 300 images, 62.67
```

对于这个量化模型，我们发现在这300张相同的图像上，准确率仅低至62％。但是，我们确实将模型的大小减小到了3.6 MB以下，几乎减少了4倍。

此外，我们可以简单地通过使用不同的量化配置来显着提高准确性。我们使用推荐的配置对x86架构进行量化，重复相同的练习。此配置执行以下操作：

- 量化每个通道的权重
- 使用直方图观察器，该直方图观察器收集激活的直方图，然后以最佳方式选择量化参数

```
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
## ..........Evaluation accuracy on 300 images, 76.33
```

仅更改这种量化配置方法，就可以将准确性提高到76％以上！尽管如此，这仍比上述实现的基准78％差1-2％。

## Quantization-Aware Training
量化意识训练（QAT）是通常导致最高准确性的量化方法。使用QAT，在训练的前进和后退过程中，所有权重和激活都被“伪量化”：也就是说，浮点值会四舍五入以模拟int8值，但所有计算仍将使用浮点数进行。因此，在训练过程中进行所有权重调整，同时“意识到”模型将最终被量化的事实。因此，在量化之后，此方法通常比动态量化或训练后静态量化具有更高的准确性。实际执行QAT的总体工作流程与之前非常相似：

- 我们可以使用与以前相同的模型：量化意识训练不需要额外的准备
- 我们需要使用`qconfig`来指定要在权重和激活之后插入哪种伪量化，而不是指定观察者

我们首先定义一个训练函数：
```
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)
            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            return

    print('Full imagenet train set: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(top1=top1, top5=top5))
    return
```

我们像以前一样融合模块：
```
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```

最后，`prepare_qat`执行“伪量化”，为量化感知训练准备模型：
```
torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n', qat_model.features[1].conv)
```

高精度训练量化模型需要在推断时对数字进行精确建模。因此，对于量化感知训练，我们通过以下方式修改训练循环：

- 在训练快要结束时切换批处理规范以使用运行均值和方差，以更好地匹配推理数字
- 我们还冻结了量化器参数（比例和零点），并对权重进行了微调

```
num_train_batches = 20

# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
## Training: * Acc@1 55.500 Acc@5 78.167
## ..........Epoch 6 :Evaluation accuracy on 300 images, 77.33
## ....................Loss tensor(1.8969, grad_fn=<DivBackward0>)
## Training: * Acc@1 56.333 Acc@5 80.333
## ..........Epoch 7 :Evaluation accuracy on 300 images, 76.33
```

有关量化意识训练的更多信息：

- QAT是后期训练量化技术的超集，可以进行更多调试。例如，我们可以分析模型的准确性是否受到权重或激活量化的限制
- 我们还可以在浮点中模拟量化模型的准确性，因为我们使用伪量化来对实际量化算法的数值建模
- 我们也可以轻松地模拟训练后量化

## Speedup From Quantization
最后，让我们确认一下我们上面提到的内容：量化模型实际上执行推理的速度更快吗？让我们测试一下：
```
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)
## Elapsed time:  45 ms
## Elapsed time:  25 ms
```

## 参考资料：
- [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)