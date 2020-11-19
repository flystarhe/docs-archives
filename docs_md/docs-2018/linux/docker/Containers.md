# Containers

## Dockerfile
[Dockerfile:centos](https://hub.docker.com/r/library/centos/):
```
FROM centos:6.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
```

[Dockerfile:ubuntu](https://hub.docker.com/r/library/ubuntu/):
```
FROM ubuntu:16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
```

[Dockerfile:python](https://hub.docker.com/_/python/):
```
FROM python:3.6-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./your-daemon-or-script.py" ]
```

[images:anaconda3](https://hub.docker.com/r/continuumio/anaconda3/):
```
docker pull continuumio/anaconda3:5.1.0
docker run --rm -it continuumio/anaconda3 /bin/bash
docker run --rm -it -p 8888:8888 continuumio/anaconda3 /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && mkdir /opt/notebooks && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"
```

## Example: flask
[Dockerfile](https://hub.docker.com/r/_/python/):
```
# Dockerfile
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

再创建两个文件,`requirements.txt`和`app.py`,将它们放在`Dockerfile`所在文件夹中.

`requirements.txt`:
```
Flask
```

`app.py`:
```python
from flask import Flask
import os
import socket

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
```

构建应用程序:
```bash
$ docker build -t demo/flask .
$ docker image ls
$ docker run --name demo00 -d -p 4000:80 demo/flask
$ docker container ls
$ docker container stop demo00
```

## Example: Share your image
为了演示我们刚才创建的可移植性,我们上传我们构建的映像并在其他地方运行它.

登录你的`Docker ID`,如果您没有,请到[web](https://cloud.docker.com/)注册:
```bash
$ docker login
```

将本地映像与注册表中的存储库相关联的符号是`username/repository:tag`.该标签是可选的,但建议使用.语法为`docker tag image username/repository:tag`.例如:
```bash
$ docker tag demo/flask flystarhe/demo:flask
$ docker image ls
```

将您的标记图像上传到存储库:
```bash
$ docker push flystarhe/demo:flask
```

从远程存储库中提取并运行图像:
```bash
$ docker run --name demo01 -p 4000:80 flystarhe/demo:flask
```

## 与主机交换数据
首先,我们要知道容器ID的查询方法:
```bash
$ sudo docker ps
$ sudo docker ps -a
```

从主机到Docker容器:
```bash
$ sudo docker cp /host/path <containerId>:/path/within/container
```

从Docker容器到主机:
```bash
$ sudo docker cp <containerId>:/path/within/container /host/path
```

使用`docker run -v`,将本机的`/data`目录分享到Docker的`/mnt`目录:
```bash
$ sudo docker run -v /data:/mnt <image>
```

`-v`参数,冒号前为宿主机目录,冒号后为镜像内挂载的路径.

## 参考资料:
- [Docker 命令大全](http://www.runoob.com/docker/docker-command-manual.html)
- [Docker Hub](https://hub.docker.com/)