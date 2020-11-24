# OpenCV
```python
import cv2 as cv
print(cv.__version__)
```

## CentOS
快速安装：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

执行`import cv2 as cv`可能会报如下错误：
```
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```

这时你需要安装缺少的共享库：
```
yum whatprovides libSM.so.6
# libSM-1.2.2-2.el7.i686 : X.Org X11 SM runtime library
yum -y install libSM-1.2.2-2.el7.i686 --setopt=protected_multilib=false
```

[Building OpenCV from Source Using CMake](https://github.com/opencv/opencv/tree/master/doc/py_tutorials/py_setup):
```
yum -y install epel-release && yum -y update
yum install cmake3 python-devel numpy gcc gcc-c++
yum install gtk2-devel libdc1394-devel ffmpeg-devel gstreamer-plugins-base-devel

which python
# /root/anaconda3/bin/python

git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build

cmake3 -DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DPYTHON3_EXECUTABLE=/root/anaconda3/bin/python \
-DPYTHON_LIBRARY=/root/anaconda3/lib/libpython3.7m.so \
-DPYTHON_INCLUDE_DIR=/root/anaconda3/include/python3.7m \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/root/anaconda3/lib/python3.7/site-packages/numpy/core/include ..

make -j8
make install
```

拷贝`lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so`到`site-packages`目录。

## Ubuntu
快速安装：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

[Building OpenCV from Source Using CMake](https://docs.opencv.org/4.0.0/d7/d9f/tutorial_linux_install.html)：
```
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

which python
# /home/hej/anaconda3/bin/python

git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build

sudo cmake -DCMAKE_BUILD_TYPE=Release \
-DOPENCV_ENABLE_NONFREE=ON \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENCV_EXTRA_MODULES_PATH=/data/tmps/opencv_contrib/modules \
-DPYTHON3_EXECUTABLE=/home/hej/anaconda3/bin/python \
-DPYTHON_INCLUDE_DIR=/home/hej/anaconda3/include/python3.6m \
-DPYTHON_LIBRARY=/home/hej/anaconda3/lib/libpython3.6m.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/home/hej/anaconda3/lib/python3.6/site-packages/numpy/core/include ..

sudo make -j8
sudo make install
```

拷贝`lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so`到`site-packages`目录。

## Use conda
[conda-forge/packages/opencv](https://anaconda.org/conda-forge/opencv)：
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c conda-forge opencv
conda install -c conda-forge/label/gcc7 opencv
conda install -c conda-forge/label/broken opencv
conda install -c conda-forge/label/cf201901 opencv
```
