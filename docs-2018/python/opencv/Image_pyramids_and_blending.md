# Image pyramids and blending

## 图像金字塔
通常，我们使用恒定大小的图像。但在某些情况下，我们需要处理同一图像的不同分辨率的图像。例如，在搜索图像中的某些内容时，如脸部，我们不确定对象在图像中的大小。在这种情况下，我们需要创建一组具有不同分辨率的图像，并在所有图像中搜索对象。这些具有不同分辨率的图像被称为图像金字塔（因为它们被保存在堆叠中，底部最大图像，顶部最小图像看起来像金字塔）。图像金字塔有两种：

1. 高斯金字塔和
2. 拉普拉斯金字塔

通过去除较低级别（较高分辨率）图像中的连续行和列来形成高斯金字塔中的较高级别（低分辨率）。然后，较高级别的每个像素由来自基础级别中的5个像素的贡献形成，具有高斯权重。通过这样做，`MxN`图像变成`M/2xN/2`图像。因此面积减少到原始面积的四分之一。当我们在金字塔中上升时（即分辨率降低），相同的模式继续。同样，在扩展时，每个级别的区域变为4次。我们可以使用`cv2.pyrDown()`和`cv2.pyrUp()`函数找到高斯金字塔。

```python
from PIL import Image
from IPython.display import display

import cv2 as cv

bgr = cv.imread("origin.png")
rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

display(Image.fromarray(rgb, "RGB"))
```

以下是高斯金字塔中的4个级别：

![](Image_pyramids_and_blending.md.01.jpg)

拉普拉斯金字塔由高斯金字塔形成。没有专属函数。拉普拉斯金字塔图像仅与边缘图像相似。它的大部分元素都是零。它们用于图像压缩。拉普拉斯金字塔中的一个层次由高斯金字塔中的该层次与高斯金字塔中的上层的扩展版本之间的差异形成。拉普拉斯级别的三个级别如下所示（调整对比度以增强内容）：

![](Image_pyramids_and_blending.md.02.jpg)

## 使用金字塔的图像混合
金字塔的一个应用是图像混合。例如，在图像拼接中，您需要将两个图像堆叠在一起，但由于图像之间的不连续性，它可能看起来不太好。在这种情况下，使用金字塔进行图像混合可以实现无缝混合，而不会在图像中留下太多数据。其中一个典型的例子是混合了两种水果，橙子和苹果。现在查看结果以了解我在说什么：

![](Image_pyramids_and_blending.md.03.jpg)

简单地完成如下：

1. 加载苹果和橙色的两个图像
2. 找到苹果和橙色的高斯金字塔（在这个特定的例子中，级别数是6）
3. 从高斯金字塔，找到他们的拉普拉斯金字塔
4. 现在加入左半部分的苹果和右半部分的拉普拉斯金字塔
5. 最后，从这个联合图像金字塔，重建原始图像。

以下是完整的代码：

```python
import cv2 as cv
import numpy as np

A = cv.imread('apple.jpg')
B = cv.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows , cols, dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols//2], B[:,cols//2:]))

# display
rgb = cv.cvtColor(real, cv.COLOR_BGR2RGB)
display(Image.fromarray(rgb, "RGB"))
rgb = cv.cvtColor(ls_, cv.COLOR_BGR2RGB)
display(Image.fromarray(rgb, "RGB"))
```

## 参考资料：
- [Image Pyramids](https://docs.opencv.org/master/dc/dff/tutorial_py_pyramids.html)
- [Panoramic Image Mosaic](http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/imagemosaic.html)