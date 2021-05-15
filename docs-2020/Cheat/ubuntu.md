# Ubuntu
设置OS启动默认进入图形界面还是文本模式。
```
systemctl get-default
systemctl set-default multi-user.target
systemctl set-default graphical.target
```

## GCC
```
sudo apt update
sudo apt install build-essential
```

该命令将安装一堆新包，包括gcc，g++和make，Ubuntu 18.04存储库中可用的默认GCC版本是7.4.0。下载[GCC Releases](https://gcc.gnu.org/releases.html)并安装：
```
tar -xzf gcc-7.5.0.tar.gz
cd gcc-7.5.0/
./contrib/download_prerequisites
./configure --prefix=/usr/local/gcc-7.5.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib
make -j8
make install
echo 'export PATH=/usr/local/gcc-7.5.0/bin:$PATH' >> /etc/profile
source /etc/profile
gcc --version
```

## CUDA
CUDA打包了显卡驱动，如果你确定要安装CUDA，且不在意驱动是否是最新版，可跳过此步骤。对于使用NVIDIA的游戏玩家，推荐下载最新[官方驱动程序](https://www.nvidia.com/Download/index.aspx)。安装前的准备：
```
lspci | grep -i VGA
lspci | grep -i nvidia
# 04:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
# 0b:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
uname -m && cat /etc/*release
# x86_64
# DISTRIB_ID=Ubuntu
# DISTRIB_RELEASE=18.04
gcc --version
```

要安装显示驱动程序，必须首先禁用Nouveau驱动程序。Linux的每个发行版都有禁用Nouveau的不同方法。如果以下命令可打印任何内容，则将加载Nouveau驱动程序：
```
lsmod | grep nouveau
```

创建文件`/etc/modprobe.d/blacklist-nouveau.conf`具有以下内容：
```
blacklist nouveau
options nouveau modeset=0
```

执行`sudo update-initramfs -u`重新生成内核initramfs。重新启动进入文本模式（运行级别3）。通常可以通过在系统的内核引导参数的末尾添加数字`3`来实现。由于尚未安装NVIDIA驱动程序，因此文本终端可能无法正确显示。将`nomodeset`临时添加到系统的内核引导参数可能会解决此问题。需要重新启动才能完全卸载Nouveau驱动程序并阻止图形界面加载。加载Nouveau驱动程序或激活图形界面时，无法安装CUDA驱动程序。（文本模式`CTRL+ALT+Fn(n=1..6)`，图形界面`CTRL+ALT+F7`）

显卡驱动的安装十分简单，执行`sudo sh NVIDIA-Linux-x86_64-450.57.run`即可，然后用`nvidia-smi`命令查看显卡和驱动情况，列出GPU的信息列表则表示驱动安装成功。你可能切到图形界面，在登录界面耗着，死活进不了图形界面。可能原因是你卸载了所有图形驱动，仅安装了独显驱动，但是你的主板的显卡初始选项确是集显(IGFX)。你可以尝试重启，修改BIOS初始显卡选项为独显(PCIE)。[CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)：
```
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```

如果你已经手动安装了合适的驱动，`Install NVIDIA Accelerated Graphics Driver`选择`no`。执行卸载脚本即可卸载CUDA，默认路径为`/usr/local/cuda-10.2/bin`。配置环境：
```
# sudo vim /etc/profile
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
# sudo vim /etc/ld.so.conf
/usr/local/cuda-10.2/lib64
# source /etc/profile
nvcc --version
```

2020年8月26日发布了补丁1，下载并执行`sudo sh cuda_10.2.1_linux.run`完成安装。

安装[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)：
```
tar -xzf cudnn-10.2-linux-x64-v8.0.3.33.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-10.2/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64
sudo chmod a+r /usr/local/cuda-10.2/include/cudnn*.h /usr/local/cuda-10.2/lib64/libcudnn*
```

删除CUDA工具包和NVIDIA驱动程序：
```
sudo apt-get --purge remove '*cublas*' 'cuda*' 'nsight*'
sudo apt-get --purge remove '*nvidia*'
```
