# Histogram

## OpenCV中的直方图计算
```python
cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
```

- `images`它是`uint8`或`float32`类型的源图像。它应该放在方括号中，即`[img]`。
- `channels`也在方括号中给出。它是我们为其计算直方图的通道的索引。例如，如果输入为灰度图像，则其值为`[0]`。对于彩色图像，您可以传递`[0]`，`[1]`或`[2]`分别计算蓝色，绿色或红色通道的直方图。
- `mask`为了找到完整图像的直方图，将其指定为`None`。但是，如果要查找图像特定区域的直方图，则必须为此创建一个遮罩图像并将其作为遮罩。
- `histSize`这表示我们的BIN计数。需要放在方括号中。对于全尺寸，我们使用`[256]`。
- `ranges`这是我们的RANGE。通常为`[0,256]`。

因此，让我们从示例图像开始。只需以灰度模式加载图像并找到其完整的直方图即可。
```python
img = cv.imread('home.jpg', 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])
```

## Numpy中的直方图计算
Numpy还为您提供了一个函数，您可以尝试以下行：
```python
hist, bins = np.histogram(img.ravel(), 256, [0, 256])
```

Numpy还有另一个函数`np.bincount()`，它比`np.histogram()`快10倍左右。
```python
hist = np.bincount(img.ravel(), minlength=256)
```

## 参考资料：
- [Histograms - 1 : Find, Plot, Analyze !!!](https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html)