# Extracting ROI

ROI是英文Region Of Interest的缩写，很多时候我们对图像的分析就是对图像特定ROI的分析与理解，对细胞与医疗图像来说，ROI提取正确才可以进行后续的分析、测量、计算密度等，而且这些ROI往往不是矩形区域，一般都是不规则的多边形区域。

## Mask
在提取ROI之前，首先了解一下图像处理中的Mask（遮罩），OpenCV中的Mask定义：八位单通道的Mat对象，每个像素点值为零或者非零区域。当Mask对象添加到图像上时，只有非零的区域是可见的。OpenCV中完成上述操作只需要简单调用API函数`bitwise_and`。

## 生成Mask

### 方法1
通过手工描点，然后多边形区域填充。代码：
```python
import cv2 as cv
import numpy as np
from IPython.display import display, Image

src = cv.imread("test.jpg")
h, w, c = src.shape

display(Image("test.jpg"))

mask = np.zeros((h, w), dtype=np.uint8)
x_data = np.array([79, 131, 169, 183, 178, 74, 40, 33, 44])
y_data = np.array([26, 22, 37, 74, 107, 140, 125, 92, 55])
pts = np.stack((x_data, y_data), axis=1)
cv.fillPoly(mask, [pts], (255), 8, 0)

res = cv.bitwise_and(src, src, mask=mask)

out = np.concatenate((src, res), axis=1)
cv.imwrite("temp.jpg", out)
display(Image("temp.jpg"))
```

![](Extracting_ROI.md.01.jpg)

### 方法2
生成Mask可以根据轮廓、二值化连通组件分析、inRange等处理方法得到。这里基于inRange方式得到Mask区域。
```python
import cv2 as cv
import numpy as np
from IPython.display import display, Image

src = cv.imread("test.jpg")
h, w, c = src.shape

display(Image("test.jpg"))

hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (120, 70, 70), (180, 255, 255))

res = cv.bitwise_and(src, src, mask=mask)

out = np.concatenate((src, res), axis=1)
cv.imwrite("temp.jpg", out)
display(Image("temp.jpg"))
```

![](Extracting_ROI.md.02.jpg)