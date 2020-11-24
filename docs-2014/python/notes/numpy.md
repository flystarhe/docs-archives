title: NumPy学习笔记
date: 2017-08-09
tags: [Python]
---
NumPy的主要对象是同种元素的多维数组.这是一个所有的元素都是一种类型,通过一个正整数元组索引的元素表格.在NumPy中维度(dimensions)叫做轴(axes),轴的个数叫做秩(rank).

<!--more-->
## 数组属性
NumPy的数组类被称作`ndarray`,重要`ndarray`对象属性有:

- `ndarray.ndim`:数组轴的个数,在python的世界中,轴的个数被称作秩
- `ndarray.shape`:数组的维度.这是一个指示数组在每个维度上大小的整数元组
- `ndarray.size`:数组元素的总个数,等于shape属性中元组元素的乘积
- `ndarray.dtype`:一个用来描述数组中元素类型的对象,NumPy提供它自己的数据类型
- `ndarray.itemsize`:数组中每个元素的字节大小.例如,元素类型为float64时值为8`(=64/8)`
- `ndarray.data`:包含实际数组元素的缓冲区,通常我们不需要使用这个属性

## 创建数组
NumPy创建数据有两个方法,`array`和`asarray`:
```
arr1 = np.array([1, 2], dtype=np.int16)
arr2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
```

`asarray`通过转换`array_like`对象创建数组,通常不会拷贝数组,除非`dtype`不匹配,或数组不存在:
```
a = [1, 2, 3]
np.asarray(a) is a
## False
a = np.array([1, 2, 3], dtype=np.float32)
np.asarray(a) is a
## True
np.asarray(a, dtype=np.float32) is a
## True
np.asarray(a, dtype=np.float64) is a
## False
```

函数`zeros`创建一个全是0的数组,函数`ones`创建一个全1的数组,函数`empty`创建一个内容随机并且依赖与内存状态的数组:
```
np.zeros((2, 4))
np.ones((2, 4), dtype=np.int16)
np.empty((2, 4))
```

NumPy提供一个类似`range`的函数`arange`,返回数组而不是列表:
```
np.arange(10, 30, 5)
## array([10, 15, 20, 25])
np.arange(0, 2, 0.3)
## array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```

## 数组变形
`reshape`函数改变参数形状并返回它,而`resize`函数改变数组自身:
```
a = np.arange(15)
b = a.reshape(3, 5)
a.resize((3, 5))
```

## 数组组合
数据拼接可用方法`append`和`concatenate`,不过`concatenate`更高效:
```
import time
import numpy as np

n = 10000
a = np.array([range(100), range(100)])
b = np.array([range(100), range(100)])

ts = time.time()
for _ in range(n):
    np.append(a, b, axis=0)
print('time loss: %s' % (time.time() - ts))
## time loss: 0.025145530700683594

ts = time.time()
for _ in range(n):
    np.concatenate((a, b), axis=0)
print('time loss: %s' % (time.time() - ts))
## time loss: 0.013432502746582031
```

对那些维度比二维更高的数组,hstack沿着第二个轴组合,vstack沿着第一个轴组合,concatenate允许可选参数给出组合时沿着的轴:
```
a = np.arange(6).reshape((2, 3))
b = np.arange(6).reshape((2, 3))
np.vstack((a, b))
np.hstack((a, b))
```

## 基本运算
NumPy中的乘法运算符`*`指示按元素计算,矩阵乘法可以使用`dot`函数:
```
a = np.array([[1, 1], [0, 1]])
b = np.array([[2, 0], [3, 4]])
np.dot(a, b)
```

有些操作符像`+=`和`*=`被用来更改已存在数组而不创建一个新的数组:
```
a = np.ones((2, 3), dtype=np.int16)
a += 3
```

`sum`,`min`,`max`等运算默认作为列表理解,无关数组的形状.然而,指定`axis`参数可以把运算应用到指定的轴:
```
a = np.arange(12).reshape(3, 4)
a.min()
a.min(axis=0)
```

## 随机数器
返回随机的整数,位于半开区间`[low, high)`:
```
np.random.randint(9, size=(2,5))
np.random.randint(0, 9, size=(2,5))
```

返回一个样本,服从标准正态分布:
```
np.random.randn(2,5)
np.random.normal(0, 1, size=(2,5))
```

返回一个样本,服从`[0, 1)`的均匀分布:
```
np.random.random_sample((5,))  # [0.0, 1.0)
2 * np.random.random_sample((3, 2)) + 3  # [3.0, 5.0)
```

生成一个随机样本,从一个给定的一维数组:
```
np.random.choice(5, 3)
np.random.choice([1, 2, 3, 4, 5], 3)
np.random.choice([1, 2, 3, 4, 5], 3, replace=True)
```

洗牌,打乱顺序,改变自身内容:
```
arr = np.arange(10)
np.random.shuffle(arr)
```

返回一个随机排列:
```
np.random.permutation(10)
```

## IO
`load()`和`save()`用NumPy专用的二进制格式保存数据:
```
arr = np.arange(10)
np.save("_tmp/np_arr.npy", arr)
np.load("_tmp/np_arr.npy")
```

将多个数组保存到一个文件中,可以使用`savez()`:
```
np.savez("_tmp/np_arr.npz", a=arr, b=arr)
np.load("_tmp/np_arr.npz").files
```

`savez()`输出的是一个扩展名为npz的压缩文件,其中每个文件都是一个`save()`保存的npy文件,文件名和数组名相同.`load()`自动识别npz文件,并且返回一个类似于字典的对象,可以通过数组名作为键获取数组的内容.

`savetxt()`和`loadtxt()`可以读写保存1维和2维数组的文本文件.例如可以用它们读写CSV格式的文本文件:
```
a = np.arange(0,12,0.5).reshape(4,-1)
np.savetxt("a.txt", a, fmt="%d", delimiter=",")
np.loadtxt("a.txt", delimiter=",")
```

## 方法总览
下面列出了一些有用的函数和方法:

### Array Creation
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

### Conversions
ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

### Manipulations
array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

### Questions
all, any, nonzero, where

### Ordering
argmax, argmin, argsort, max, min, ptp, searchsorted, sort

### Operations
choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum

### Basic Statistics
cov, mean, std, var

### Basic Linear Algebra
cross, dot, outer, linalg.svd, vdot

## 参考资料:
- [Quickstart tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)