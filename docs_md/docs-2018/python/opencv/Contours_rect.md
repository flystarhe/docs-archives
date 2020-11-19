# Contours rect

## 数据准备
```python
import cv2 as cv
import numpy as np

img = np.zeros((100, 100), dtype='uint8')
img[30:70, 30:40] = 255
img[30:40, 30:70] = 255

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

from IPython.display import display, Image
cv.imwrite('0.jpg', img)
display(Image('0.jpg'))
```

## 正外接矩形
```python
def f1(img):
    ret, thresh = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return cv.boundingRect(contours[0])

def f2(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return cv.boundingRect(contours[0])

def f3(img):
    inds = np.argwhere(img > 0)
    x0, y0 = inds.min(axis=0)
    x1, y1 = inds.max(axis=0)
    return x0, y0, x1, y1

def f4(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    inds = np.squeeze(contours[0])
    x0, y0 = inds.min(axis=0)
    x1, y1 = inds.max(axis=0)
    return x0, y0, x1, y1

def f5(img):
    inds = np.argwhere(img > 0)
    inds = np.expand_dims(inds, 1)
    return cv.boundingRect(inds)
```

`f1(img), f2(img), f3(img), f4(img), f5(img)`结果：
```python
(30, 30, 40, 40)
(30, 30, 40, 40)
(30, 30, 69, 69)
(30, 30, 69, 69)
(30, 30, 40, 40)
```

`%timeit`速度测试：
```bash
f1 9.71 µs ± 78 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
f2 8.04 µs ± 75.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
f3 38.7 µs ± 188 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
f4 16.4 µs ± 101 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
f5 34.8 µs ± 258 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
