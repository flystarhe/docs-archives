# Template matching
- 使用模板匹配在图像中查找对象
- 学习`cv2.matchTemplate(), cv2.minMaxLoc()`

## 理论
模板匹配是一种用于在较大图像中搜索和查找模板图像位置的方法。它只是将模板图像滑动到输入图像上（像2D卷积一样），然后在模板图像下比较模板和输入图像的补丁。`cv2.matchTemplate()`返回一个灰度图像，其中每个像素表示该像素的邻域与模板匹配多少。

如果输入图像大小`(WxH)`，模板图片的大小`(wxh)`，输出图像的大小为`(W-w+1, H-h+1)`。得到结果后，就可以使用`cv2.minMaxLoc()`查找最大值/最小值在哪里。将其作为矩形的左上角，并以`(w,h)`作为矩形的宽度和高度。该矩形是模板区域。

## 模板匹配
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', 0)
template = cv2.imread('template.jpg', 0)
w, h = template.shape[::-1]
img2 = img.copy()

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
```

## 多个目标
在上一部分中，我们在图像中搜索了梅西的脸，该脸在图像中仅出现一次。假设您要搜索具有多次出现的对象，`cv2.minMaxLoc()`不会为您提供所有位置。在这种情况下，我们将使用阈值。
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('mario.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.png', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
cv2.imwrite('res.png', img_rgb)
```

## 参考资料：
- [Template Matching](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html)