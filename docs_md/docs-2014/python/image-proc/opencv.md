title: Image Processing in OpenCV
date: 2017-08-12
tags: [Python,OpenCV]
---
OpenCV是一个跨平台计算机视觉库,它轻量级而且高效,同时提供了Python,Ruby,MATLAB等语言的接口,实现了图像处理和计算机视觉方面的很多通用算法.

<!--more-->
## OpenCV in Windows
我们这里考虑从预构建的二进制文件安装OpenCV,当然你也可以从源代码构建OpenCV.Numpy是必须的,因为所有的opencv数组结构都被转化为Numpy的数组.Matplotlib是可选的,但建议安装,因为在很多教程中使用了.如下:

```
$ python -m pip install numpy
$ python -m pip install matplotlib
```

下载[opencv-3.3.0-vc14.exe](http://opencv.org/releases.html),然后安装,我的路径为`D:/program`.最后,将`D:/program/opencv/build/python/2.7/x64`下的`cv2.pyd`复制到`Anaconda/Lib/site-packages`目录下.写个小程序:

```
import cv2
print(cv2.__version__)
image = cv2.imread('test.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)
```

注意,这个方法目前仅仅适用于py27,要想在py3里工作,还得自己折腾.

## OpenCV in Ubuntu
如果你是Anaconda用户,命令`conda install opencv`就能帮你完成.否则你可能需要更多的操作.这种ez的方式很多时候时是管用,对于OpenCV深度用户,还是建议编译安装.下载[opencv-3.2.0.zip](http://opencv.org/releases.html),参考[Installation in Linux 3.2.0](http://docs.opencv.org/3.2.0/d7/d9f/tutorial_linux_install.html)完成安装.安装依赖包:
```
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

获取最新的OpenCV稳定版本,编译安装:
```
unzip opencv-3.2.0.zip && cd opencv-3.2.0
mkdir build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENCV_EXTRA_MODULES_PATH=/home/hejian/opencv_contrib-3.2.0/modules \
-DPYTHON3_EXECUTABLE=/home/hejian/anaconda3/envs/py35-tf12-gpu/bin/python \
-DPYTHON_INCLUDE_DIR=/home/hejian/anaconda3/envs/py35-tf12-gpu/include/python3.5m \
-DPYTHON_LIBRARY=/home/hejian/anaconda3/envs/py35-tf12-gpu/lib/libpython3.5m.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/home/hejian/anaconda3/envs/py35-tf12-gpu/lib/python3.5/site-packages/numpy/core/include ..
sudo make -j8 # runs 8 jobs in parallel
sudo make install
cd /home/hejian/anaconda3/envs/py35-tf12-gpu/lib/python3.5/site-packages
cp /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
```

## 基本操作
图片读,写和显示操作.直接使用cv2的`imread`,`imwrite`和`imshow`函数:
```
import numpy as np
import cv2

img = cv2.imread('test.jpg')
cv2.imshow('test-image-show', img)

k = cv2.waitKey(0)
# wait for ESC key to exit
# wait for 's' key to save and exit
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('test-output.jpg', img)
    cv2.destroyAllWindows()
```

`imread`函数还可以定义加载的`mode`,默认是以`RGB`模式处理图片.可选参数`cv2.CV_LOAD_IMAGE_COLOR`彩色,`cv2.CV_LOAD_IMAGE_GRAYSCALE`灰度.

## 图片属性
```
import cv2
img = cv2.imread('test-output.jpg')
img.shape  # (640, 640, 3)
img.size  # 1228800
img.dtype  # uint8
```

## 输出文本
在处理图片的时候,我们经常把一些信息直接以文字的形式输出在图片上,比如:
```
cv2.putText(img, 'Hello World', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
```

## 缩放图片
实现缩放图片并保存,这在OpenCV做图像处理的时候都是很常用的操作:
```
import cv2
import numpy as np
img = cv2.imread('test.jpg')

height, width = img.shape[:2]
res = cv2.resize(img, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
```

## 图像平移
```
import cv2
import numpy as np

img = cv2.imread('test.jpg')
rows, cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])

dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 图像旋转
```
img = cv2.imread('test.jpg')
rows, cols = img.shape

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
```

## 仿射变换
```
import cv2
import numpy as np

img = cv2.imread('test.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('image', dst)
cv2.waitKey(0)
```

## 颜色变换
`cv2.cvtColor(input_image, flag)`函数实现图片颜色空间的转换,`flag`参数决定变换类型.如`BGR->Gray`设置为`cv2.COLOR_BGR2GRAY`.下面的代码实现识别视频中蓝色部分:
```
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # 读取视频的每一帧
    _, frame = cap.read()

    # 将图片从 BGR 空间转换到 HSV 空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义在 HSV 空间中蓝色的范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # 根据以上定义的蓝色的阈值得到蓝色的部分
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame, frame, mask= mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5)
    if k == 27:
        break

cv2.destroyAllWindows()
```

## 通道拆分/合并
```
import cv2
img = cv2.imread('test.jpg')
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
```

## 参考资料:
- [OpenCV-Python Tutorials](https://docs.opencv.org/3.3.1/d6/d00/tutorial_py_root.html)
- [Python-OpenCV 图像与视频处理](https://segmentfault.com/a/1190000003742481)
- [几何变换(仿射与投影)的应用](http://blog.csdn.net/a352611/article/details/51418178)
- [几何变换(仿射与投影)的原理](http://blog.csdn.net/a352611/article/details/51417779)