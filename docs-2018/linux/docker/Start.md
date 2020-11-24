# Start
Docker是世界领先的软件容器平台.开发人员使用Docker来消除与同事的代码协作时的"我机器上的工作"问题.运营商使用Docker在独立的容器中并行运行和管理应用程序,以获得更好的计算密度.企业使用Docker构建灵活的软件传送流程,可以更快,更安全地发布新功能.

## Docker for Ubuntu
Docker需要64的操作系统.此外你的kernel内核至少要在`3.10`版本之上:
```bash
$ uname -r
## 4.10.0-32-generic
```

较老版本的Docker被称为docker或docker-engine.如果这些已安装,请卸载它们:
```bash
$ sudo apt-get remove docker docker-engine docker.io
```

您可以根据需要以不同的方式安装Docker:

- 大多数用户[设置Docker存储库](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository)并从中安装,安装方便,容易升级.这是推荐的方法
- 部分用户下载DEB软件包[手动安装](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-from-a-package),并手动管理升级.这在没有访问互联网的系统上安装Docker的情况下是有用的
- 在测试和开发环境中,用户选择使用[自动化脚本](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-convenience-script)来安装Docker

在新主机上首次安装Docker之前,需要设置Docker存储库:
```bash
$ sudo apt-get update
$ sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
```

添加Docker的官方GPG密钥:
```bash
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

使用以下命令设置稳定版本库:
```bash
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

`lsb_release -cs`子命令返回您的Ubuntu发行版的名称,例如`16.04`为`xenial`.

安装最新版本的Docker:
```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce
```

在生产系统上,您应该安装特定版本的Docker.列出可用的版本:
```bash
$ apt-cache madison docker-ce
```

安装特定版本:
```bash
$ sudo apt-get install docker-ce=<VERSION>
```

通过运行`hello-world`映像验证Docker是否正确安装:
```bash
$ docker info
$ sudo docker run hello-world
```

修改镜像/容器文件存储位置.`vim /etc/docker/daemon.json`:
```
{
    "graph": "/data2/docker-images"
}
```

卸载Docker:
```bash
$ sudo apt-get purge docker-ce
$ sudo rm -rf /var/lib/docker
```

## nvidia-docker
参考[Installation-(version-2.0)](https://github.com/NVIDIA/nvidia-docker/wiki).Removing nvidia-docker 1.0:
```
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
apt-get purge -y nvidia-docker
```

Installing version 2.0:
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd
```

Basic usage:
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

## Download Images
我们使用Docker的第一步,应该是获取一个官方的镜像,例如mysql,wordpress,基于这些基础镜像我们可以开发自己个性化的应用.我们可以使用Docker命令行工具来下载官方镜像,但是因为网络原因,我们下载一个300M的镜像需要很长的时间,甚至下载失败.因为这个原因,阿里云容器Hub服务提供了官方的镜像站点加速官方镜像的下载速度.[link](https://yq.aliyun.com/articles/29941)

当你下载安装的Docker不低于`1.10`时,建议直接通过`daemon config`进行配置,使用配置文件`/etc/docker/daemon.json`,没有时新建该文件:
```
{
    "registry-mirrors": ["https://r9x4ucwq.mirror.aliyuncs.com"]
}
```

获取你的专属加速器地址:进入[Docker Hub 镜像站点](https://cr.console.aliyun.com/#/accelerator),然后你可以看到`您的专属加速器地址`,应该会要求你登录.

重新启动Docker守护程序:
```bash
$ sudo service docker restart
```

## Install TensorFlow
TensorFlow社区在Docker Hub提供了[image](https://hub.docker.com/r/tensorflow/tensorflow/),使用非常简单.

CPU版本:
```bash
$ sudo docker pull tensorflow/tensorflow
$ sudo docker run -it -p 9999:8888 tensorflow/tensorflow
```

GPU版本:
```bash
$ sudo nvidia-docker pull tensorflow/tensorflow:latest-gpu-py3
$ sudo nvidia-docker run -it -p 9999:8888 tensorflow/tensorflow:latest-gpu-py3
```

然后,浏览器访问[http://localhost:9999/](http://localhost:9999/).详情请参阅[Using TensorFlow via Docker](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker).

## Build an image from a Dockerfile
`PATH`指定为`.`:
```bash
$ git clone https://github.com/CSAILVision/LabelMeAnnotationTool.git
$ cd LabelMeAnnotationTool/
$ sudo docker build -t hejian/labelme -f Dockerfile .
$ sudo docker run --name labelme -d -p 9000:80 hejian/labelme
```

`PATH`指定为`URL`:
```bash
$ sudo docker build -t hejian/labelme -f Dockerfile github.com/CSAILVision/LabelMeAnnotationTool
$ sudo docker run --name labelme -d -p 9000:80 hejian/labelme
```

## docker diff
检查对容器文件系统上文件或目录的更改.用法:
```bash
docker diff CONTAINER
```

## docker inspect
返回有关Docker对象的低级信息.用法:
```bash
docker inspect [OPTIONS] NAME|ID [NAME|ID...]
```

比如执行以下命令,查看容器`diff`:
```
docker inspect labelme | grep UpperDir
## "UpperDir": "/var/lib/docker/overlay2/3e3dbc06e763198924f08484b05f22cab5eb4fb3eec2703e859042e5c0f557b7/diff"
```

当容器崩溃/异常/损坏,就可以从里面抽取数据,降低损失:
```
cd /var/lib/docker/overlay2/3e3dbc06e763198924f08484b05f22cab5eb4fb3eec2703e859042e5c0f557b7/diff
```

## Notes
系统启动时启动服务:
```bash
$ sudo systemctl enable docker
```

重新启动Docker守护程序:
```bash
$ sudo service docker restart
```

查你的Docker镜像:
```bash
$ sudo docker images
```

退出/关闭容器:
```bash
## ctrl+d,退出容器且关闭
## ctrl+d+q,退出容器不关闭
$ sudo docker stop [OPTIONS] CONTAINER [CONTAINER...]
```

已停止的容器可以重新启动,并保持原来的所有更改不变:
```bash
$ sudo docker start [OPTIONS] CONTAINER [CONTAINER...]
```

连接到正在运行的容器:
```bash
$ sudo docker attach <containerId>
```

在正在运行的容器中运行命令/打开终端:
```bash
$ docker exec [OPTIONS] CONTAINER COMMAND [ARG...]
$ sudo docker exec labelme pwd
$ sudo docker exec -it labelme bash
```

删除容器:
```bash
$ sudo docker rm [OPTIONS] CONTAINER [CONTAINER...]
```

列出容器:
```bash
$ sudo docker ps  # 显示正在运行的容器
$ sudo docker ps -a  # 显示所有容器,包括未运行的
```

添加用户:
```bash
$ sudo useradd new-user -s /bin/bash
$ sudo passwd new-user
$ sudo usermod -aG docker new-user
```

## 参考资料:
- [官方参考文档](https://docs.docker.com/reference/)
- [Docker Get Started Tutorial](https://docs.docker.com/get-started/)
- [Get Docker CE for CentOS](https://docs.docker.com/install/linux/docker-ce/centos/)
- [Get Docker CE for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Docker Engine Utility for NVIDIA GPUs](https://github.com/NVIDIA/nvidia-docker)
- [Docker Hub](https://hub.docker.com/)