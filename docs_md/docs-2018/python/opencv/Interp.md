# Interp

- （上采样）要放大图像，通常使用`cv.INTER_LINEAR`和`cv.INTER_CUBIC`。
- （下采样）要缩小图像，OpenCV的文档建议使用`cv.INTER_AREA`。

```
import cv2 as cv

image = cv.imread("1.png")
cv.imshow("image", image)

h, w = image.shape[:2]
dsize = (int(w/2), int(h))
small_image = cv.resize(image, dsize, interpolation=cv.INTER_AREA)
cv.imshow("small_image", small_image)
```
