# Histogram Backprojection
在本章中，我们将学习直方图反投影。

它用于图像分割或在图像中查找感兴趣的对象。用简单的话说，它创建的图像大小与输入图像相同（但只有一个通道），其中每个像素对应于该像素属于我们物体的概率。用更简单的话来说，与其余部分相比，输出图像将使我们感兴趣的对象具有更多的白色。好吧，这是一个直观的解释。（我无法使其更简单）。直方图反投影与camshift算法等配合使用。

## Numpy中的算法
首先，我们需要计算我们要查找的对象（使其为`M`）和要搜索的图像（使其为`I`）的颜色直方图。
```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# roi is the object or region of object we need to find
roi = cv.imread('rose_red.png')
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# target is the image we search in
target = cv.imread('rose.png')
hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
# Find the histograms using calcHist. Can be done with np.histogram2d also
M = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
```

求出比率`R=M/I`。然后反向投影R，即使用R作为调色板，并以每个像素作为其对应的目标概率创建一个新图像。h是色调，s是饱和度。
```python
R = M / I
h, s, v = cv.split(hsvt)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B, 1)
B = B.reshape(hsvt.shape[:2])
```

现在对圆盘应用卷积。
```python
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cv.filter2D(B, -1, disc, B)
B = np.uint8(B)
cv.normalize(B, B, 0, 255, cv.NORM_MINMAX)
```

现在最大强度的位置给了我们物体的位置。如果我们期望图像中有一个区域，则对合适的值进行阈值处理会得到不错的结果。
```python
ret, thresh = cv.threshold(B, 50, 255, 0)
```

## OpenCV中的反投影
OpenCV提供了一个内置函数`cv.calcBackProject()`。它的参数与`cv.calcHist()`函数几乎相同。它的参数之一是直方图，它是对象的直方图，我们必须找到它。同样，对象直方图应在传递到`backproject`函数之前进行标准化。它返回概率图像。然后，我们将图像与磁盘内核卷积并应用阈值。下面是我的代码和输出：
```python
import numpy as np
import cv2 as cv

roi = cv.imread('rose_red.png')
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

target = cv.imread('rose.png')
hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# normalize histogram and apply backprojection
cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cv.filter2D(dst, -1, disc, dst)
# threshold and binary AND
ret, thresh = cv.threshold(dst, 50, 255, 0)
thresh = cv.merge((thresh, thresh, thresh))
res = cv.bitwise_and(target, thresh)
res = np.vstack((target, thresh, res))
cv.imwrite('res.jpg', res)
```

## 参考资料：
- [Histogram - 4 : Histogram Backprojection](https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html)