# Anaconda
在安装[Anaconda个人版](https://www.anaconda.com/products/individual)之前，请查看系统要求。如果您不希望Anaconda包含数百个软件包，则可以安装Miniconda，这是Anaconda的微型版本，仅包含conda，其依赖项和Python。

* [Miniconda Installers - Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/)
* [Anaconda Installers - Anaconda3-2020.07-Linux-x86_64.sh](https://repo.anaconda.com/archive/)

```
# Ubuntu
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
# CentOS
yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver

curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
/bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

/opt/conda/bin/conda install -y python=3.8 conda-build pyyaml numpy
/opt/conda/bin/conda install -y -c conda-forge jupyterlab
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate base" >> ~/.bashrc

conda info -e
conda install python=3.8
conda create -n py3.8 python=3.8
conda activate py3.8
conda deactivate py3.8
conda remove -n py3.8 --all
```

* `-b`批处理模式，假设您同意许可协议，不编辑`.bashrc`或`.bash_profile`文件。
* `-p`安装前缀/路径。如果设置了`-f`，即使前缀已经存在，也强制安装。
* [https://hub.docker.com/r/continuumio/miniconda3/dockerfile](https://hub.docker.com/r/continuumio/miniconda3/dockerfile)

## JupyterLab
```
jupyter --paths
jupyter lab --generate-config
```

设置开机启动：
```
# vim /etc/rc.local
export PATH=/opt/conda/bin:$PATH
nohup jupyter lab --ip='*' --port=9000 --notebook-dir='/workspace' --NotebookApp.token='hi' --no-browser --allow-root > ~/log.jupyter 2>&1 &
```

[magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)魔法命令都以`%`或者`%%`开头，以`%`开头的成为行命令，`%%`开头的称为单元命令。执行`%magic`可以查看关于各个命令的说明，而在命令之后添加`?`可以查看该命令的详细说明。

Nbextensions
```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install
# Autopep8, ExecuteTime
```

nbconvert
```jupyter
%time !jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook.ipynb
```

## Docker
```
mkdir -p docker
tee docker/Dockerfile <<EOF
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-server \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    /opt/conda/bin/conda install -y python=3.8 conda-build pyyaml numpy && \
    /opt/conda/bin/pip install jupyterlab && \
    /opt/conda/bin/conda clean -ya && \
    mkdir -p /run/sshd && mkdir -p ~/.ssh && \
    echo "# ssh keys" > ~/.ssh/authorized_keys && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile

WORKDIR /workspace
ENV PATH="/opt/conda/bin:$PATH"
ENTRYPOINT /usr/sbin/sshd -p 9001 && \
    jupyter lab --ip='*' --port=9000 --notebook-dir='/workspace' --NotebookApp.token='hi' --no-browser --allow-root
# Build and Run
# docker build -t flystarhe/python:3.8 .
# docker build -t flystarhe/python:3.8 - < Dockerfile
# t=test && docker run -d -p 9000:9000 -p 9001:9001 --ipc=host --name ${t} -v "$(pwd)"/${t}:/workspace flystarhe/python:3.8
EOF
```

## Notes
```
# Jupyter
!cd {PROJ_HOME}
LLIB = "/path/to/module"
!PYTHONPATH="$(pwd)":{LLIB} python *.py
!CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)":{LLIB} python *.py
!PYTHONPATH="$(pwd)":{LLIB} nohup python *.py args > log.python 2>&1 &
# Shell
cd ${PROJ_HOME}
LLIB="/path/to/module"
PYTHONPATH="$(pwd)":${LLIB} python *.py
```

使用镜像加速：
```
!pip install -i $URL $PACKAGE_NAME
!conda install -y -c $URL $PACKAGE_NAME
```

将本地文件夹包含在Anaconda环境的PYTHONPATH中：
```
conda-develop /local/package/path
```

`requirements.txt`
```
pip freeze > requirements.txt
pip install -r requirements.txt

conda list -e > requirements.txt
conda install -y --file requirements.txt
conda create -y -n <env> --file requirements.txt
```

Anaconda仓库的镜像[TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。如果`~/.condarc`文件不存在，可先执行`conda config --set show_channel_urls yes`生成文件：
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

运行`conda clean -i`清除索引缓存，保证用的是镜像站提供的索引。运行`conda create -n myenv numpy`测试一下吧。

## 参考资料：
* [Installation](https://docs.anaconda.com/anaconda/install/)
* [Getting started with Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/)
* [Managing packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)
* [Command reference](https://docs.conda.io/projects/conda/en/latest/commands.html)
* [Frequently asked questions](https://docs.anaconda.com/anaconda/user-guide/faq/)
* [Jupyter: magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)