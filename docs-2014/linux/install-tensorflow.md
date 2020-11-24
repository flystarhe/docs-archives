title: Install TensorFlow
date: 2017-07-06
tags: [TensorFlow,Ubuntu]
---
Windows下安装TensorFlow之`CPU & GPU`版本,Ubuntu下安装TensorFlow之`GPU`版本,实现`(tf1.1,tf1.2)x(norm,nomkl)`交叉组合4个环境.最后进行了TensorFlow安装验证,Jupyter-notebook服务搭建,安装并验证Keras.

<!--more-->
## Install TensorFlow with CPU
新建环境:
```
conda info -e
conda create -n tensorflow python=3.5
conda install python=3.5
```

激活与关闭环境:(Win)
```
activate tensorflow
deactivate tensorflow
```

激活与关闭环境:(Linux)
```
source activate tensorflow
source deactivate tensorflow
```

删除环境:
```
conda remove -n tensorflow --all
```

### TensorFlow with CPU(tensorflow-cpu env)
创建一个名为`tensorflow-cpu`的`conda`环境，并激活:
```
conda create -n tensorflow-cpu python=3.5
#Windows: activate tensorflow-cpu
#Linux: source activate tensorflow-cpu
pip install --upgrade pip
pip install numpy
pip install pandas
pip install jupyter
```

安装CPU版本的TensorFlow:
```
pip install --upgrade tensorflow
```

或:
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl
```

## Install TensorFlow with GPU(Windows)
您需要以下内容:

- 支持CUDA的GPU
- 受支持的Windows版本
- [cuda_8.0.61_win10.exe](http://developer.nvidia.com/cuda-downloads)
- [cudnn-8.0-windows10-x64-v5.1.zip](https://developer.nvidia.com/rdp/cudnn-download)

`cuda_8.0.61_win10.exe`默认安装就可以了，正常情况会自动添加变量`CUDA_PATH`和`CUDA_PATH_V8_0`到`系统变量`，值为`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`。`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin`和`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp`也被添加到`Path`变量中。

`cudnn-8.0-windows10-x64-v5.1.zip`安装很简单，分别拷贝`cuda\bin` `cuda\include` `cuda\lib`三个目录中的内容到`CUDA\v8.0`对应的目录。

创建一个名为`tensorflow-gpu`的`conda`环境，并激活:
```
conda create -n tensorflow-gpu python=3.5
#Windows: activate tensorflow-gpu
#Linux: source activate tensorflow-gpu
pip install --upgrade pip
pip install numpy
pip install pandas
pip install jupyter
```

安装GPU版本的TensorFlow:
```
pip install --upgrade tensorflow-gpu
```

或:
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-win_amd64.whl
```

## Install TensorFlow with GPU(Ubuntu)

### CUDA
下载[cuda_8.0.61_375.26_linux.run](https://developer.nvidia.com/cuda-downloads)，安装:

1. 执行`sudo sh cuda_8.0.61_375.26_linux.run`。
2. 按照命令行提示操作。NVIDIA驱动我们已经安装了，所以安装CUDA过程中，第二项“是否安装显卡驱动”选择“no”，其他全部按照默认设定。

`sudo vim /etc/profile`添加:
```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```

执行`source /etc/profile`使生效，`sudo vim /etc/ld.so.conf`添加:
```
/usr/local/cuda-8.0/lib64
```

执行`sudo ldconfig`,提示错误:
```
/sbin/ldconfig.real: /usr/lib/nvidia-375/libEGL.so.1 不是符号连接
/sbin/ldconfig.real: /usr/lib32/nvidia-375/libEGL.so.1 不是符号连接
```

系统找的是符号连接，而不是文件。对这两个文件更名,重新建立符号连接:
```
sudo mv /usr/lib/nvidia-375/libEGL.so.1 /usr/lib/nvidia-375/libEGL.so.1.org
sudo mv /usr/lib32/nvidia-375/libEGL.so.1 /usr/lib32/nvidia-375/libEGL.so.1.org
sudo ln -s /usr/lib/nvidia-375/libEGL.so.375.66 /usr/lib/nvidia-375/libEGL.so.1
sudo ln -s /usr/lib32/nvidia-375/libEGL.so.375.66 /usr/lib32/nvidia-375/libEGL.so.1
```

执行`sudo ldconfig`,查看版本信息`nvcc --version`:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```

后面发现上面的繁琐操作或者可以由以下命令代替(未验证):
```
sudo apt install nvidia-cuda-toolkit
```

下载[cudnn-8.0-linux-x64-v5.1.tgz](https://developer.nvidia.com/cudnn)，安装:
```
sudo tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include/
```

执行`sudo ldconfig`,提示错误:
```
/sbin/ldconfig.real: /usr/local/cuda-8.0/lib64/libcudnn.so.5 不是符号连接
```

系统找的是符号连接，而不是文件。对这两个文件更名,重新建立符号连接:
```
cd /usr/local/cuda-8.0/lib64
sudo rm -rf libcudnn.so.5 libcudnn.so
sudo ln -s libcudnn.so.5.1.10 libcudnn.so.5
sudo ln -s libcudnn.so.5 libcudnn.so
```

执行`sudo ldconfig`.

### BLAS
可选[ATLAS](http://math-atlas.sourceforge.net/)或[Intel MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/):

- 安装ATLAS:`sudo apt-get install libatlas-base-dev`
- 安装OpenBLAS:`sudo apt-get install libopenblas-dev`

MKL的安装稍显麻烦,如果使用Anaconda管理Python环境,则跳过此步,因为Anaconda在安装Numpy时,会自己准备好MKL.Intel MKL,下载[l_mkl_2017.0.098.tgz](https://software.intel.com/en-us/mkl),可查阅[安装指南](https://software.intel.com/en-us/articles/intel-mkl-103-install-guide):
```
tar -zxf l_mkl_2017.0.098.tgz
cd l_mkl_2017.0.098/
sudo ./install.sh
```

记得下载的时候你填写的邮箱吗?你有一封新邮件,邮件中Download链接地址里的`SN=33RM-RM349RZ7`就是安装过程中需要的序列号.`sudo vim /etc/profile`添加:
```
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
```

### libcupti-dev
```
sudo apt-get install libcupti-dev
```

### OpenCV
如果你是Anaconda用户,命令`conda install opencv`就能帮你完成.否则你可能需要更多的操作.这种ez的方式很多时候时是管用,对于OpenCV深度用户,还是建议编译安装.下载[opencv-3.3.0](http://opencv.org/releases.html)，参考[Installation in Linux 3.3.0](http://docs.opencv.org/3.3.0/d7/d9f/tutorial_linux_install.html)完成安装。安装依赖包:
```
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libdc1394-22-dev
```

获取最新的OpenCV稳定版本，编译安装:
```
git clone https://github.com/opencv/opencv_contrib.git

wget https://github.com/opencv/opencv/archive/3.3.0.zip
mv 3.3.0.zip opencv-3.3.0.zip && unzip opencv-3.3.0.zip && cd opencv-3.3.0
mkdir build && cd build

sudo cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENCV_EXTRA_MODULES_PATH=/data1/bins/opencv_contrib/modules \
-DPYTHON3_EXECUTABLE=/root/anaconda3/bin/python \
-DPYTHON_INCLUDE_DIR=/root/anaconda3/include/python3.5m \
-DPYTHON_LIBRARY=/root/anaconda3/lib/libpython3.5m.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=/root/anaconda3/lib/python3.5/site-packages/numpy/core/include ..

sudo make -j8
sudo make install

cd /root/anaconda3/lib/python3.5/site-packages
cp /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
```

若提示`numpy/npy_common.h:17:5: warning: "NPY_INTERNAL_BUILD" is not defined`.则修改`site-packages/numpy/core/include/numpy/npy_common.h`:
```
#if NPY_INTERNAL_BUILD
#ifndef NPY_INLINE_MATH
#define NPY_INLINE_MATH 1
#endif
#endif
```

修改为:
```
#ifndef NPY_INTERNAL_BUILD
#define NPY_INTERNAL_BUILD
#ifndef NPY_INLINE_MATH
#define NPY_INLINE_MATH 1
#endif
#endif
```

进入Python测试`import cv2`,可能会遇到如下问题:
```
ImportError: /root/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
```

如果使用的是Anaconda管理Python环境,执行`conda install libgcc`即可解决.

### Anaconda
```
bash Anaconda3-4.4.0-Linux-x86_64.sh
```

添加国内镜像:
```
conda config --add channels https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

执行完上述命令后,会生成`~/.condarc`文件,记录着我们对conda的配置,直接手动创建,编辑该文件是相同的效果.`cat .condarc`:
```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

### tensorflow-gpu
python3.5, tensorflow1.2:
```
conda info -e
conda create -n py35-tf12-gpu python=3.5
source activate py35-tf12-gpu
conda install numpy scipy nose scikit-learn pandas jupyter pillow
conda list
pip install --upgrade pip
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl
```

python3.5, tensorflow1.2, nomkl:
```
conda info -e
conda create -n py35-tf12-gpu-nomkl python=3.5
source activate py35-tf12-gpu-nomkl
conda install nomkl numpy scipy nose scikit-learn pandas jupyter pillow
conda list
pip install --upgrade pip
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl
```

[link](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package),测试安装:
```
python -c "import numpy; numpy.test()"
python -c "import scipy; scipy.test()"
```

## Validate TensorFlow
```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
## Hello, TensorFlow!
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
## 42
quit()
```

如果系统输出`Hello, TensorFlow!`则可以开始编写TensorFlow程序。

## jupyter-notebook
`cd /root && sudo vim apps/python-jupyter-notebook.py`:
```
if __name__ == '__main__':
    import sys
    import notebook.notebookapp
    sys.exit(notebook.notebookapp.main())
```

设置开机启动,在`/etc/rc.local`添加:
```
export JAVA_HOME=/home/hejian/apps/jdk1.8.0_161
cd /root/apps/
/root/anaconda3/bin/python python-jupyter-notebook.py --allow-root --NotebookApp.iopub_data_rate_limit=0 --ip=127.0.0.1 --port=9091 --no-browser --notebook-dir /data1/hej &
```

注意,记得开放端口,或关闭防火墙`service ufw stop && sudo ufw disable`.然后在浏览器中输入服务地址。

### 免密码登录
```bash
$ jupyter-notebook --generate-config --allow-root
$ sudo vim /root/.jupyter/jupyter_notebook_config.py
#-----------------------------------------------------
## new line
c.NotebookApp.token = ''
#-----------------------------------------------------
```

## keras
```
sudo apt-get install libhdf5-serial-dev
pip install numpy scipy
pip install pyyaml h5py
pip install keras
sudo apt-get install graphviz
pip install graphviz
pip install pydot_ng
```

测试安装:
```
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

model = Sequential()
model.add(Dense(32, input_shape=(50,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file='model.png')
```

NumPy(~22s),SciPy(~208s).你可能会遇到`Intel MKL ERROR`,你可以执行`conda install nomkl`或安装`Intel MKL`.安装了还有问题,请尝试`export LD_PRELOAD=/root/anaconda3/lib/libmkl_avx2.so:$LD_PRELOAD`或`export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_avx2.so:$LD_PRELOAD`.[参考](http://www.tuicool.com/articles/eE3yeuj)
