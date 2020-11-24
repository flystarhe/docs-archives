# Image arithmetic operations
学习对图像的几种算术运算，如加法，减法，按位运算等。

## 图像相加
您可以通过OpenCV函数，`cv.add()`或简单地通过numpy操作添加两个图像，`res = img1 + img2`。两个图像应该具有相同的深度和类型，或者第二个图像可以是标量值。

```python
import cv2 as cv
import numpy as np

x = np.uint8([250])
y = np.uint8([10])
print( cv.add(x,y) ) # 250+10 = 260 => 255
## [[255]]
print( x+y )         # 250+10 = 260 % 256 = 4
## [4]
```

>OpenCV添加和Numpy添加之间存在差异。OpenCV添加是饱和操作，而Numpy添加是模运算。

## 图像混合
这也是图像添加，但是对图像赋予不同的权重，从而给出混合感或透明感。我拍了两张图片将它们混合在一起。第一图像的权重为`\alpha=0.7`，第二图像的权重为`\beta=0.3`，最后`\gamma=0`。`cv.addWeighted()`在图像上应用以下等式。

$$
\begin{aligned}
dst = \alpha \cdot img1 + \beta \cdot img2 + \gamma
\end{aligned}
$$

```python
img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')

dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

from PIL import Image
from IPython.display import display

rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
display(Image.fromarray(rgb, "RGB"))
```

## 图像位操作
这包括按位`AND`，`OR`，`NOT`和`XOR`运算。它们在提取图像的任何部分时非常有用，定义和使用非矩形ROI等。下面我们将看到如何更改图像的特定区域的示例。

我想将OpenCV徽标放在图像上方。如果我添加两个图像，它将改变颜色。如果我混合它，我会得到一个透明的效果。但我希望它不透明。如果它是一个矩形区域，我可以像上一章那样使用ROI。但OpenCV徽标不是矩形。所以你可以通过按位操作来完成，如下所示：

```python
# Load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
```

## 参考资料：
- [图像的算术运算](https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html)