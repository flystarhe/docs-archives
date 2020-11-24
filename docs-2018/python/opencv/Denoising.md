# Denoising
您将了解有关去除图像中噪声的非局部均值去噪算法。

我们已经看到了许多图像平滑技术，例如高斯模糊，中值模糊等，它们在某种程度上可以消除少量噪声。在那些技术中，我们在像素周围采取了一个小的邻域，并进行了一些操作（例如高斯加权平均值，值的中位数等）来替换中心元素。简而言之，在像素处去除噪声是其周围的局部现象。

我们获取一个像素，在其周围获取一个小窗口，在图像中搜索相似的窗口，对所有窗口求平均，然后用得到的结果替换该像素。此方法是“非本地均值消噪”。与我们之前看到的模糊技术相比，它花费了更多时间，但效果非常好。对于彩色图像，图像将转换为CIELAB色彩空间，然后分别对L和AB分量进行降噪。

## OpenCV的图像去噪
- `cv.fastNlMeansDenoising()`使用单个灰度图像。
- `cv.fastNlMeansDenoisingColored()`使用彩色图像。

常见的参数有：
- h：决定滤波器强度的参数。较高的h值可以更好地消除噪点，但同时也可以消除图像细节。（10可以）
- hForColorComponents：与h相同，但仅用于彩色图像。（通常与h相同）
- templateWindowSize：应为奇数。（推荐7）
- searchWindowSize：应该为奇数。（建议21）

### fastNlMeansDenoisingColored
如上所述，它用于消除彩色图像中的噪声。（噪声可能是高斯的）。请参见下面的示例：
```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('die.png')
dst = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
```

## 参考资料：
- [Image Denoising](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html)
- [Non-Local Means Denoising](http://www.ipol.im/pub/art/2011/bcm_nlm/)