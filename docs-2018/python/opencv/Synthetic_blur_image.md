# Synthetic blurred image
在图像去模糊和图像质量评估等任务，需要对清晰图片进行处理获得模糊的图片从而进行训练。如由于对焦，运动等造成的模糊图像。

## 运动模糊
一般来说，运动模糊的图像都是朝同一方向运动的，那么就可以利用`cv.filter2D`函数。
```python
from PIL import Image
from IPython.display import display

import cv2 as cv
import numpy as np

def motion_blur(image, degree=10, angle=20):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

img = cv.imread("origin.png")
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst = motion_blur(rgb, 10, 90)

display(Image.fromarray(np.hstack((rgb, dst)), "RGB"))
```

![](Synthetic_blur_image.md.01.png)

## 对焦模糊
opencv提供了`GaussianBlur`函数：
```python
degree = 15

img = cv.imread("origin.png")
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst = cv.GaussianBlur(rgb, ksize=(degree, degree), sigmaX=0, sigmaY=0)

display(Image.fromarray(np.hstack((rgb, dst)), "RGB"))
```

![](Synthetic_blur_image.md.02.png)

## 重影
如果你知道偏移方向`(t_x,t_y)`，你可以创建转换矩阵如下：

$$
\begin{aligned}
M = 
\left[
\begin{matrix}
1 & 0 & t_x \\
0 & 1 & t_y
\end{matrix}
\right]
\end{aligned}
$$

平移：
```python
img = cv.imread("origin.png")
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

rows, cols = rgb.shape[:2]

M = np.float32([[1, 0, 10], [0, 1, 5]])
dst1 = cv.warpAffine(rgb, M, (cols, rows))

M = np.float32([[1, 0, -10], [0, 1, -5]])
dst2 = cv.warpAffine(rgb, M, (cols, rows))

display(Image.fromarray(np.hstack((rgb, dst1, dst2)), "RGB"))
```

>`cv.warpAffine()`函数的第三个参数是输出图像的大小，它应该是`(width,height)`的形式。记住width是列数，height是行数。

![](Synthetic_blur_image.md.03.png)

混合：
```python
dst = cv.addWeighted(dst1, 0.6, dst2, 0.4, 0)

display(Image.fromarray(np.hstack((rgb, dst)), "RGB"))
```

![](Synthetic_blur_image.md.04.png)