# SimpleITK image basics
SimpleITK图像基础。本文档将简要介绍`Image`类。按照惯例，我们先导入相关的模块。
```python
import numpy as np
from PIL import Image
from IPython.display import display

import SimpleITK as sitk
```

## 创建图像
有多种方法可以创建图像。所有图像的初始值都很好地定义为零。
```python
image = sitk.Image(256, 128, 64, sitk.sitkInt16)
image_2D = sitk.Image(64, 64, sitk.sitkFloat32)
image_2D = sitk.Image([32, 32], sitk.sitkUInt32)
image_RGB = sitk.Image([128, 128], sitk.sitkVectorUInt8, 3)
```

完整定义图像需要以下组件：

1. 像素类型`[固定在创建时，无默认值]`：无符号32位整数，`sitkVectorUInt8`等。
2. 尺寸`[固定在创建时，无默认值]`：每个维度的像素/体素数。此数量隐式定义图像尺寸。
3. 原点`[默认为零]`：物理单位`(mm)`，具有索引`(0,0,0)`的像素/体素的坐标。
4. 间距`[默认为1]`：以物理单位给出的每个维度中相邻像素/体素之间的距离。
5. 方向矩阵`[默认为恒等]`：像素/体素轴方向与物理方向之间的映射。

## 访问属性
如果您熟悉ITK，那么这些方法将遵循您的期望：
```python
print(image.GetSize())
print(image.GetOrigin())
print(image.GetSpacing())
print(image.GetDirection())
print(image.GetNumberOfComponentsPerPixel())
## (256, 128, 64)
## (0.0, 0.0, 0.0)
## (1.0, 1.0, 1.0)
## (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
## 1
```

图像尺寸的大小具有显式访问者：
```python
print(image.GetWidth(), image.GetHeight(), image.GetDepth())
## 256 128 64
```

由于SimpleITK图像的尺寸和像素类型是在运行时确定的，因此需要访问：
```python
print(image.GetDimension())
print(image.GetPixelIDValue())
print(image.GetPixelIDTypeAsString())
## 3
## 2
## 16-bit signed integer
```

2D图像的深度是多少？
```python
print(image_2D.GetSize())
print(image_2D.GetDepth())
## (32, 32)
## 0
```

矢量图像的尺寸和大小是多少？
```python
print(image_RGB.GetDimension())
print(image_RGB.GetSize())
print(image_RGB.GetNumberOfComponentsPerPixel())
## 2
## (128, 128)
## 3
```

对于某些文件类型（如DICOM），有关图像的其他信息包含在元数据字典中：
```python
for key in image.GetMetaDataKeys():
    print("\"{0}\":\"{1}\"".format(key, image.GetMetaData(key)))
```

## 访问像素
有成员函数`GetPixel`，`SetPixel`它提供类似ITK的像素访问接口。
```python
print(image.GetPixel(0, 0, 0))
image.SetPixel(0, 0, 0, 1)
print(image.GetPixel(0, 0, 0))
## 0
## 1
```

或：
```python
print(image[0,0,0])
image[0,0,0] = 10
print(image[0,0,0])
## 1
## 10
```

## Numpy和SimpleITK之间的转换
从SimpleITK图像中获取Numpy ndarray。这是图像缓冲区的深层副本，完全安全且没有潜在的副作用。
```python
nda = sitk.GetArrayFromImage(image_RGB)
img = sitk.GetImageFromArray(nda)
img.GetSize()
## (3, 128, 128)
nda = sitk.GetArrayFromImage(image_RGB)
img = sitk.GetImageFromArray(nda, isVector=True)
img.GetSize()
## (128, 128)
```

>注意：SimpleITK像素访问`image[x,y,z]`，Numpy像素访问`array[z,y,x]`。

## 索引和尺寸的顺序在转换过程中需要注意
ITK的`Image`类没有括号运算符。它有一个`GetPixel`，它将一个`Index`对象作为参数，其排序为`(x,y,z)`。这是SimpleITK的`Image`类用于`GetPixel`方法和切片运算符的约定。在numpy中，数组以相反的顺序索引`(z,y,x)`。另请注意，对通道的访问是不同的。在SimpleITK中，您不直接访问通道，而是返回表示特定像素的所有通道的像素值，然后您可以访问该像素的通道。在numpy数组中，您将直接访问该通道。
```python
multi_channel_3Dimage = sitk.Image([2,4,8], sitk.sitkVectorFloat32, 5)
x = multi_channel_3Dimage.GetWidth() - 1
y = multi_channel_3Dimage.GetHeight() - 1
z = multi_channel_3Dimage.GetDepth() - 1
multi_channel_3Dimage[x,y,z] = np.random.random(multi_channel_3Dimage.GetNumberOfComponentsPerPixel())

nda = sitk.GetArrayFromImage(multi_channel_3Dimage)

print("Image size: " + str(multi_channel_3Dimage.GetSize()))
print("Numpy array size: " + str(nda.shape))

# Notice the index order and channel access are different:
print("First channel value in image: " + str(multi_channel_3Dimage[x,y,z][0]))
print("First channel value in numpy array: " + str(nda[z,y,x,0]))
```

## 可视化
通过转换为numpy数组可用于可视化以集成到科学python环境中。
```python
nda = np.random.randint(-1000, 3000, (64,128,256))  # Numpy (z,y,x)

img = sitk.GetImageFromArray(nda)  # ITK (x,y,z)

z = 0
slice = sitk.GetArrayViewFromImage(img)[z,:,:]  # Numpy (z,y,x)

slice = slice - slice.min()
if slice.max() > 0:
    slice = slice / slice.max() * 255
display(Image.fromarray(slice.astype("uint8")))
```

>注意：PIL格式为`(width, height, channel)`，Numpy格式为`(height, width, channel)`。

## 参考资料：
- [SimpleITK Notebooks](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/)