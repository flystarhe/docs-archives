# Docker
Docker是一个用于开发，交付和运行应用程序的开放平台。Docker使您能够将应用程序与基础架构分开，从而可以快速交付软件。借助Docker，您可以以与管理应用程序相同的方式来管理基础架构。通过利用Docker的方法来快速交付，测试和部署代码，您可以大大减少编写代码和在生产环境中运行代码之间的延迟。Docker Engine具有三种类型的更新通道：稳定，测试和每晚更新。通过与主分支不同的发布分支进行的。使用`<year>.<month>`格式创建分支，例如`18.09`。

* [Install Docker Engine on CentOS](https://docs.docker.com/engine/install/centos/)
* [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

大多数Linux发行版都用`systemctl`启动服务。如果没有`systemctl`，请使用`service`命令。
```
sudo systemctl enable docker
sudo systemctl start docker
sudo service docker start
```

以非`root`用户管理Docker：`sudo usermod -aG docker $USER`将您的用户添加到`docker`组，`newgrp docker`注销并重新登录，`docker run hello-world`验证不带`sudo`是否可以运行`docker`。[Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

## mirror
您可以通过修改daemon配置文件`/etc/docker/daemon.json`来使用加速器：
```
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": ["https://r9x4ucwq.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

获取你的专属加速器地址：进入[Docker Hub 镜像站点](https://cr.console.aliyun.com)，然后你可以看到`您的专属加速器地址`，应该会要求你登录。

## commands
```
docker version
docker images -a

docker build -t flystarhe/python:3.8 .
docker build -t flystarhe/python:3.8 - < Dockerfile
docker run --rm --gpus device=0,1 nvidia/cuda:10.1-base nvidia-smi
docker run --gpus all -d -p 9000:9000 --ipc=host --name test -v "$(pwd)":/workspace flystarhe/python:3.8

docker ps -a
docker rm test
docker stop test
docker start test
docker exec -it test bash
docker inspect -f "{{json .Config}}" NAME|ID
docker inspect -f "{{json .Mounts}}" NAME|ID
docker inspect -f "{{ .Config.Env }}" c3f279d17e0a
docker commit -c "ENV DEBUG=true" c3f279d17e0a none/testimage:version3
docker inspect -f "{{ .Config.Env }}" f5283438590d

docker save -o flystarhe-python-3.8.tar flystarhe/python:3.8
docker load -i flystarhe-python-3.8.tar
```

## entrypoint.sh
```
#!/bin/bash
set -e

MODE=${1:-dev}
if [ "${MODE}" = 'dev' ]; then
    /usr/sbin/sshd -p 9001
else
    nohup python /workspace/app_tornado.py 9001 ${@:2} > /workspace/nohup.out 2>&1 &
fi

jupyter lab --ip='*' --port=9000 --notebook-dir='/workspace' --NotebookApp.token='hi' --no-browser --allow-root
```

## Dockerfile - pytorch

* Python: 3.8
* PyTorch: 1.7.0
* SSH INFO: `ssh root@ip -p 9001`
* Jupyter: `http://ip:9000/?token=hi`

```
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install jupyterlab tornado && \
    conda clean -ya

WORKDIR /usr/src
# conda-develop /usr/src/SimpleCV
RUN git clone -b main --depth 1 https://github.com/flystarhe/SimpleCV.git SimpleCV && cd SimpleCV && \
    git checkout -B main

RUN mkdir -p /run/sshd && mkdir -p ~/.ssh && \
    echo "# ssh keys" > ~/.ssh/authorized_keys
WORKDIR /workspace
COPY entrypoint.sh /usr/src/
ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
```

* `docker pull flystarhe/python:3.8-torch1.7.0`
* `t=test && docker run --gpus all -d -p 9000:9000 -p 9001:9001 --ipc=host --name ${t} -v "$(pwd)"/${t}:/workspace flystarhe/python:3.8-torch1.7.0`

## Dockerfile - multi-stage
```
# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy && \
    /opt/conda/bin/conda clean -ya

FROM conda as conda-installs
RUN /opt/conda/bin/pip install jupyterlab && \
    /opt/conda/bin/conda clean -ya

FROM ${BASE_IMAGE} as official
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-server \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"
RUN mkdir -p /run/sshd && mkdir -p ~/.ssh && \
    echo "# ssh keys" > ~/.ssh/authorized_keys && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile
WORKDIR /workspace
ENTRYPOINT /usr/sbin/sshd -p 9000 -D
```

进行多阶段构建：
```
export DOCKER_BUILDKIT=1
# context
docker build -t flystarhe/python:3.8 --target official .
# no context
docker build -t flystarhe/python:3.8 --target official - < Dockerfile
# docker run
t=test && docker run -d -p 9000:9000 --ipc=host --name ${t} -v "$(pwd)"/${t}:/workspace flystarhe/python:3.8
```

## 参考资料：
* [Docker Hub](https://hub.docker.com/search?q=&type=image)
* [Docker base command](https://docs.docker.com/engine/reference/commandline/docker/)
* [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
* [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)