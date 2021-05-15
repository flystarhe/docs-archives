# Install OpenCV-Python 4.3.0
```
import cv2 as cv
print(cv.__version__)
```

* [github.com/opencv](https://github.com/opencv/opencv-python)
* [Installation in Linux](https://docs.opencv.org/4.3.0/d7/d9f/tutorial_linux_install.html)
* [Install OpenCV-Python in Fedora](https://docs.opencv.org/4.3.0/dd/dd5/tutorial_py_setup_in_fedora.html)
* [Install OpenCV-Python in Ubuntu](https://docs.opencv.org/4.3.0/d2/de6/tutorial_py_setup_in_ubuntu.html)
* [Install OpenCV-Python in Windows](https://docs.opencv.org/4.3.0/d5/de5/tutorial_py_setup_in_windows.html)

## Notes
Q: Import fails `ImportError: libSM.so.6: cannot open shared object file`

这时你需要安装缺少的共享库：
```
yum whatprovides libSM.so.6
# libSM-1.2.2-2.el7.i686 : X.Org X11 SM runtime library
yum -y install libSM-1.2.2-2.el7.i686 --setopt=protected_multilib=false
```

## From pip
用于Python的非官方的预构建的仅CPU的OpenCV软件包。[github.com/opencv](https://github.com/opencv/opencv-python)
```
pip install --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python-headless
```

>`opencv-python`主要模块，`opencv-contrib-python`完整软件包(主要模块和contrib/extra模块)，`*-headless`适合服务器环境(例如docker，云环境等)，没有GUI库依赖项。如果您希望从源代码编译绑定以启用其他模块(例如CUDA)，请查看手动构建部分。

## Ubuntu from source
我们将安装一些依赖项。有些是必需的，有些是可选的。如果不想，可以跳过可选的依赖项。
```
sudo apt-get update
sudo apt-get install build-essential

which python
# /home/hej/anaconda3/bin/python

sudo apt-get install cmake
sudo apt-get install gcc g++
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
sudo apt-get install libgtk2.0-dev
# Optional Dependencies
sudo apt-get install libpng-dev
sudo apt-get install libjpeg-dev
sudo apt-get install libopenexr-dev
sudo apt-get install libtiff-dev
sudo apt-get install libwebp-dev
```

您可以使用最新的[稳定版本](https://opencv.org/releases/)，也可以从[Git存储库](https://github.com/opencv/opencv)中获取最新的快照。
```
cd /data/tmps
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd /data/tmps/opencv && mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
-DOPENCV_ENABLE_NONFREE=ON \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENCV_EXTRA_MODULES_PATH=/data/tmps/opencv_contrib/modules \
-DPYTHON3_EXECUTABLE=/home/hej/anaconda3/bin/python \
-DPYTHON_INCLUDE_DIR=/home/hej/anaconda3/include/python3.7m \
-DPYTHON_LIBRARY=/home/hej/anaconda3/lib/libpython3.7m.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/home/hej/anaconda3/lib/python3.7/site-packages/numpy/core/include/ ..

make -j7 # runs 7 jobs in parallel
sudo make install
```

拷贝`lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so`到`site-packages`目录。执行`numpy.__path__`可查询numpy的安装位置。

## CentOS from source
我们将安装一些依赖项。有些是必需的，有些是可选的。如果不想，可以跳过可选的依赖项。
```
yum -y install epel-release
yum -y update

# https://rpmfusion.org/Configuration
yum localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm

which python
# /home/hej/anaconda3/bin/python

yum -y install cmake
yum -y install gcc gcc-c++
yum -y install gtk2-devel
yum -y install libdc1394-devel
yum -y install ffmpeg-devel
yum -y install gstreamer-plugins-base-devel
# Optional Dependencies
yum -y install libpng-devel
yum -y install libjpeg-turbo-devel
yum -y install jasper-devel
yum -y install openexr-devel
yum -y install libtiff-devel
yum -y install libwebp-devel
```

您可以使用最新的[稳定版本](https://opencv.org/releases/)，也可以从[Git存储库](https://github.com/opencv/opencv)中获取最新的快照。
```
cd /data/tmps
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd /data/tmps/opencv && mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
-DWITH_TBB=OFF \
-DWITH_EIGEN=OFF \
-DBUILD_DOCS=OFF \
-DOPENCV_ENABLE_NONFREE=ON \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENCV_EXTRA_MODULES_PATH=/data/tmps/opencv_contrib/modules \
-DPYTHON3_EXECUTABLE=/home/hej/anaconda3/bin/python \
-DPYTHON_INCLUDE_DIR=/home/hej/anaconda3/include/python3.7m \
-DPYTHON_LIBRARY=/home/hej/anaconda3/lib/libpython3.7m.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/home/hej/anaconda3/lib/python3.7/site-packages/numpy/core/include/ ..

make -j7 # runs 7 jobs in parallel
sudo make install
```

拷贝`lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so`到`site-packages`目录。执行`numpy.__path__`可查询numpy的安装位置。

## Conda
要使用Conda安装此软件包，请执行以下操作之一：[url](https://anaconda.org/conda-forge/opencv)
```
conda install -c conda-forge opencv
conda install -c conda-forge/label/gcc7 opencv
conda install -c conda-forge/label/broken opencv
conda install -c conda-forge/label/cf201901 opencv
conda install -c conda-forge/label/cf202003 opencv
```
