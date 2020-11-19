# Getting started

## Installation
```bash
nvcc --version
# For CUDA 9.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
# For CUDA 10.0
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```
[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)

## Defining the pipeline
为分类器定义一个非常简单的管道。准备目录结构，其中包含猫和狗的图片：
```bash
images/
├── cat
│   ├── cat0.jpg
│   ├── cat1.jpg
│   └── cat2.jpg
└── dog
    ├── dog0.jpg
    ├── dog1.jpg
    └── dog2.jpg
```

管道将从该目录中读取图像，对其进行解码并返回`(image, label)`对：
```python
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

image_dir = "images"
batch_size = 8

class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)
```

`SimplePipeline`类是`dali.pipeline.Pipeline`的子类，该类提供创建和启动管道的大多数方法。我们仅需要实现的两个方法是构造函数和`define_graph`函数：

- `FileReader` - 遍历目录并返回`(encoded image, label)`对
- `ImageDecoder` - 接受编码的图像输入并输出解码的RGB图像

## Building the pipeline
为了使用我们的`SimplePipeline`，我们需要构建它。这可以通过调用`build`函数来实现。
```python
pipe = SimplePipeline(batch_size, 1, 0)
pipe.build()
```

## Running the pipeline
构建管道之后，我们可以运行它来获取一批结果。
```python
pipe_out = pipe.run()
print(pipe_out)
```

是2个元素的列表（按预期，我们在`SimplePipeline`的`define_graph`函数中指定了2个输出）。这两个元素都是`TensorListCPU`对象，每个对象都包含CPU上的张量列表。

为了显示结果（仅出于调试目的，在实际培训中我们不会执行该步骤，因为这会使我们的图像批处理从GPU到CPU往返），我们可以从DALI的Tensor发送数据到NumPy数组。为了检查是否可以直接将其发送给NumPy，我们可以调用TensorList的`is_dense_tensor`函数。

```python
images, labels = pipe_out
print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))
#Images is_dense_tensor: False
#Labels is_dense_tensor: True
```

事实证明，包含标签的TensorList可以由张量表示，而包含图像的TensorList不能。让我们看看返回标签的形状和内容是什么。
```python
labels_array = labels.as_array()

print(labels_array.shape())
labels_array
```

为了查看图像，我们将需要遍历TensorList中包含的所有张量，并使用其`at`方法进行访问。
```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(images.at(0))
```

## Adding augmentations

### Random shuffle
```python
class ShuffledSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ShuffledSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)
```

- `random_shuffle` - 可以在阅读器中对图像进行混洗，使用从磁盘读取的图像缓冲区执行的
- `initial_fill` - 设置缓冲区的容量，在此示例中将其设置为数据集大小

### Augmentations
DALI不仅可以从磁盘读取图像并将其批处理成张量，还可以对这些图像执行各种增强操作，以改善深度学习训练的效果。该管道在输出图像之前先旋转它们。
```python
class RotatedSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RotatedSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.rotate = ops.Rotate(angle=10.0)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        rotated_images = self.rotate(images)
        return (rotated_images, labels)
```

将每个图像旋转10度并不是很有趣。为了进行有意义的增强，我们希望图像旋转给定范围内的随机角度。
```python
class RandomRotatedSimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RandomRotatedSimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.rotate = ops.Rotate()
        self.rng = ops.Uniform(range=(-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        angle = self.rng()
        rotated_images = self.rotate(images, angle=angle)
        return (rotated_images, labels)
```

## GPU acceleration
DALI可以访问GPU加速，从而可以提高输入和增强流水线的速度，并使其能够扩展到多GPU系统。

### Copying tensors to GPU
让我们修改前面的`RandomRotatedSimplePipeline`示例，以将GPU用作旋转运算符。
```python
class RandomRotatedGPUPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RandomRotatedGPUPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.rotate = ops.Rotate(device="gpu")
        self.rng = ops.Uniform(range=(-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        angle = self.rng()
        rotated_images = self.rotate(images.gpu(), angle=angle)
        return (rotated_images, labels)
```

我们在Rotate操作中添加了`device="gpu"`参数。`images.gpu()`将`images`复制到GPU。

>DALI管道不支持将数据从GPU移至CPU。在所有执行路径中，CPU操作无法跟随GPU操作。

### Hybrid decoding
有时，尤其是对于更高分辨率的图像，解码以JPEG格式存储的图像可能会成为瓶颈。为了解决这个问题，开发了`nvJPEG`库。它在CPU和GPU之间划分了解码过程，从而大大减少了解码时间。在ImageDecoder中指定`mixed`设备参数可启用`nvJPEG`支持。
```python
class HybridPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(HybridPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=21)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        # images are on the GPU
        return (images, labels)
```

`device=mixed`同时使用CPU和GPU的混合计算方法。这意味着它接受CPU输入，但返回GPU输出。这就是从管道返回的图像对象为`TensorListGPU`类型的原因。