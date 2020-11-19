title: Install CUDA and cudnn
date: 2018-03-16
tags: [Linux,CUDA]
---
CUDA,cudnn安装笔记.

<!--more-->
## CUDA8(Ubuntu)
下载[cuda_8.0.61_375.26_linux.run](https://developer.nvidia.com/cuda-downloads),安装:

1. 执行`sudo sh cuda_8.0.61_375.26_linux.run`
2. 按照提示操作.NVIDIA驱动我们已经安装,“是否安装显卡驱动”选择“no”,其他按照默认设定

`sudo vim /etc/profile`添加:
```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```

执行`source /etc/profile`使生效,查看版本信息`nvcc --version`:
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

下载[cudnn-8.0-linux-x64-v5.1.tgz](https://developer.nvidia.com/cudnn),安装:
```
tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include/
```

`sudo vim /etc/ld.so.conf`添加:
```
/usr/local/cuda-8.0/lib64
```

执行`sudo ldconfig`,提示错误:
```
/sbin/ldconfig.real: /usr/lib/nvidia-375/libEGL.so.1 不是符号连接
/sbin/ldconfig.real: /usr/lib32/nvidia-375/libEGL.so.1 不是符号连接
```

系统找的是符号连接,而不是文件.对这文件更名,重建符号连接:
```
sudo mv /usr/lib/nvidia-375/libEGL.so.1 /usr/lib/nvidia-375/libEGL.so.1.old
sudo mv /usr/lib32/nvidia-375/libEGL.so.1 /usr/lib32/nvidia-375/libEGL.so.1.old
sudo ln -s /usr/lib/nvidia-375/libEGL.so.375.66 /usr/lib/nvidia-375/libEGL.so.1
sudo ln -s /usr/lib32/nvidia-375/libEGL.so.375.66 /usr/lib32/nvidia-375/libEGL.so.1
```

执行`sudo ldconfig`,提示错误:
```
/sbin/ldconfig.real: /usr/local/cuda-8.0/lib64/libcudnn.so.5 不是符号连接
```

系统找的是符号连接,而不是文件.对这文件更名,重建符号连接:
```
cd /usr/local/cuda-8.0/lib64
sudo rm -rf libcudnn.so.5 libcudnn.so
sudo ln -s libcudnn.so.5.1.10 libcudnn.so.5
sudo ln -s libcudnn.so.5 libcudnn.so
```

执行`sudo ldconfig`.

## CUDA9(Ubuntu)
其实我已经安装了CUDA8,但又需要CUDA9,还想默认CUDA8.比如我的TensorFlow需要CUDA9,而PyTorch使用CUDA8.

首先到[web](https://developer.nvidia.com/cuda-downloads)下载匹配的版本,如`cuda_9.0.176_384.81_linux.run`,下载清单末端有`Installation Guide for Linux`有必要看看:
```bash
#验证是否有支持的版本
$ uname -m && cat /etc/*release
#验证是否安装了GCC
$ gcc --version
```

首先`Disabling Nouveau`:
```bash
$ sudo vim /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
$ sudo update-initramfs -u
```

运行安装程序并按照屏幕提示操作:
```bash
$ sudo sh cuda_9.0.176_384.81_linux.run
```

如果你已经手动安装了合适的驱动,`Install NVIDIA Accelerated Graphics Driver`选择`no`.因为我们希望默认使用`CUDA8`,所以`symbolic link at /usr/local/cuda`选择`no`.执行卸载脚本即可卸载CUDA,默认路径为`/usr/local/cuda-9.0/bin`.

最后,配置环境`sudo vim /etc/profile`:
```bash
$ export PATH=/usr/local/cuda-9.0/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
```

执行`source /etc/profile`使生效,安装补丁:
```bash
$ sudo sh cuda_9.0.176.1_linux.run
$ sudo sh cuda_9.0.176.2_linux.run
```

验证安装&查看版本信息:
```bash
$ nvcc --version
```

下载[cudnn-9.0-linux-x64-v7.1.tgz](https://developer.nvidia.com/cudnn),安装:
```bash
$ tar -xzf cudnn-9.0-linux-x64-v7.1.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
```

`sudo vim /etc/ld.so.conf`,添加:
```bash
/usr/local/cuda-9.0/lib64
```

执行`sudo ldconfig`,若提示`/usr/local/cuda-9.0/lib64/libcudnn.so.7 不是符号连接`:
```bash
$ cd /usr/local/cuda-9.0/lib64
$ sudo mv libcudnn.so.7 libcudnn.so.7.old
$ sudo ln -s libcudnn.so.7.1.1 libcudnn.so.7
```

执行`sudo ldconfig`.测试`TensorFlow 1.6`:
```python
import tensorflow
import keras
tensorflow.__version__, keras.__version__
## ('1.6.0', '2.1.5')
```
