# CUDA on WSL
Windows Subsystem for Linux (WSL)是Windows 10的一项功能，使用户可以直接在Windows上运行本机Linux命令行工具。WSL是一个容器化的环境，用户可以在其中从Windows 10 Shell的命令行运行Linux本机应用程序，而无需双启动环境的复杂性。在内部，WSL与Microsoft Windows操作系统紧密集成，从而使其可以与传统的Windows桌面和现代商店应用程序一起运行Linux应用程序。[wsl-user-guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

![](cuda_on_wsl.md.01.png)

## Getting Started
借助WSL 2和GPU半虚拟化技术，Microsoft使开发人员能够在Windows上运行GPU加速的应用程序。以下文档描述了在WSL 2环境中开始运行CUDA应用程序或容器的工作流程。在WSL上运行CUDA的入门要求您按顺序完成以下步骤：

1. 从Microsoft Windows Insider程序安装最新版本
2. 为WSL 2安装NVIDIA预览驱动程序
3. 安装WSL 2

注册[Microsoft Windows Insider Program](https://insider.windows.com/en-us/getting-started/#register)。从[Fast ring](https://insider.windows.com/en-us/getting-started/#install)安装最新版本。

>确保安装了Build 20145或更高版本。您可以通过运行`winver`检查内部版本号。

从[CUDA on WSL](https://developer.nvidia.com/cuda/wsl)页面CUDA的下载部分下载NVIDIA驱动程序。根据您系统中的NVIDIA GPU的类型（GeForce/Quadro）选择合适的驱动程序。使用可执行文件安装驱动程序。这是您需要安装的唯一驱动程序。

>不要在WSL中安装任何Linux显示驱动程序。Windows显示驱动程序将同时安装用于本机Windows和WSL支持的常规驱动程序组件。

按照[此处](https://docs.microsoft.com/windows/wsl/install-win10)提供的Microsoft文档中的说明安装WSL 2。通过点击`Windows Update > Check for updates`确保您具有最新的内核。如果安装了带有内核`4.19.121+`的正确更新，则您应该能够在Windows Update历史记录中看到它。或者，您可以通过在PowerShell中运行`wsl cat /proc/version`来检查版本号。如果没有看到此更新，请确保在Windows Update Advanced选项中，启用推荐的Microsoft更新并再次检查。启动Linux发行版，并使用以下命令确保它在WSL 2模式下运行：
```
wsl.exe --list -v command
```

## Setting up CUDA Toolkit
设置CUDA网络存储库。此处显示的说明适用于Ubuntu 18.04。有关其他发行版的更多信息，请参见[《CUDA Linux安装指南》](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。
```
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/" > /etc/apt/sources.list.d/cuda.list'
apt-get update
apt-get install -y cuda-toolkit-11-0
```

>不要选择`cuda`，`cuda-11-0`，`cuda-drivers`，因为这些软件包将导致尝试在WSL 2下安装Linux NVIDIA驱动程序。

## Setting up to Run Containers
设置NVIDIA Container Toolkit的工作流程，以准备运行GPU加速的容器。[安装Docker](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-docker)，使用Docker安装脚本来为您选择的WSL 2 Linux发行版安装Docker。请注意，NVIDIA Container Toolkit尚不支持[Docker Desktop WSL 2](https://docs.docker.com/docker-for-windows/wsl/)后端。对于此发行版，请安装用于Linux发行版的标准Docker-CE。
```
curl https://get.docker.com | sh
```

[安装NVIDIA Container Toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-nvidia-docker)，现在安装NVIDIA容器工具包。WSL 2支持始于`nvidia-docker2` v2.3和基础运行时库（`libnvidia-container >= 1.2.0-rc.1`）。为简便起见，此处提供的安装说明适用于Ubuntu 18.04 LTS。
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container-experimental.list | sudo tee /etc/apt/sources.list.d/libnvidia-container-experimental.list
```

更新软件包清单后，安装NVIDIA运行时软件包（及其依赖项）。
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

打开一个单独的WSL 2窗口，然后使用以下命令再次启动Docker守护程序以完成安装。
```
sudo service docker stop
sudo service docker start
```

## Running CUDA Containers
我们将逐步介绍在WSL 2环境中运行GPU容器的一些示例。在此示例中，我们运行一个`N-body`模拟CUDA示例。该示例已被容器化，可以从NGC获得。
```
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

在此示例中，让我们运行Jupyter notebook。
```
docker run --gpus all -it -p 8000:8000 tensorflow/tensorflow:latest-gpu-py3-jupyter
```

在此示例中，使用GPU进行ResNet-50训练。
```
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.03-tf2-py3
```
