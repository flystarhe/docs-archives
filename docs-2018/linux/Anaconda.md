# Anaconda
Anaconda是集成了很多Python优秀Packages的项目,免去开发者手工配置各种依赖包的麻烦.

## Install
```bash
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
echo $'export PATH="/root/anaconda3/bin:$PATH"' >> /etc/profile
source /etc/profile
conda info
conda update conda
```

你可能需要`py35`而不是默认的`py36`:
```bash
conda install python=3.5
```

或者是卸载:
```bash
rm -rf ~/anaconda3
```

## Package
```bash
conda list | grep -nE "^pillow"
conda search package_name
conda install package_name
conda update package_name
conda remove package_name
```

## Environments
```bash
conda info -e
conda create -n lab_py2 python=2
Linux: source activate lab_py2
Linux: source deactivate lab_py2
conda remove -n lab_py2 --all
```

## Jupyter
禁用防火墙:
```bash
service iptables stop
chkconfig iptables off
```

默认配置文件:
```bash
jupyter --paths
## config:
##    /root/.jupyter
##    /root/anaconda3/etc/jupyter
##    /usr/local/etc/jupyter
##    /etc/jupyter
jupyter notebook --generate-config
vim /root/.jupyter/jupyter_notebook_config.py
##    c.NotebookApp.token = 'hi'
##    c.NotebookApp.open_browser = False
##    c.NotebookApp.iopub_data_rate_limit = 0
```

设置开机启动,`vim /etc/rc.local`:
```bash
## https://www.oracle.com/technetwork/java/javase/downloads/index.html
## tar -zxf jdk-8u211-linux-x64.tar.gz
## export JAVA_HOME=/root/jdk1.8.0_211
export PATH="/root/anaconda3/bin:$PATH"
export JAVA_HOME=/root/jdk1.8.0_161
nohup jupyter notebook --allow-root --ip='*' --port=9000 --notebook-dir='/data' --NotebookApp.token='hi' > /root/nohup.out 2>&1 &
```

为了导出PDF,需要安装[TeX](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex)生态系统:
```bash
sudo apt-get install texlive-xetex
```

除了Jupyter的导出,还可以命令行执行`jupyter nbconvert file_name.ipynb --to pdf`.命令行执行笔记本:
```bash
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook0.ipynb > log.00 2>&1 &
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook*.ipynb > log.00 2>&1 &
nohup jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook1.ipynb notebook2.ipynb > log.00 2>&1 &
```

IPython提供了许多魔法命令，使得在IPython环境中的操作更加得心应手。魔法命令都以`%`或者`%%`开头，以`%`开头的成为行命令，`%%`开头的称为单元命令。行命令只对命令所在的行有效，而单元命令则必须出现在单元的第一行，对整个单元的代码进行处理。执行`%magic`可以查看关于各个命令的说明，而在命令之后添加`?`可以查看该命令的详细说明。

## python
```bash
cd $PROJ_HOME
PYTHONPATH=`pwd`/module_* python *.py
PYTHONPATH=`pwd`/module_* nohup python *.py > log.txt 2>&1 &
```

## requirements.txt
优雅的Python项目,会包含一个`requirements.txt`文件,用于记录所有依赖包及其精确的版本号,以便新环境部署.

生成:
```bash
pip freeze > requirements.txt
```

使用:
```bash
pip install -r requirements.txt
conda install -y --file requirements.txt
conda create -y --name <env> --file requirements.txt
```

## mirrors
[TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)还提供了Anaconda仓库的镜像:
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

## conda-forge
[conda-forge](https://conda-forge.org/#about)软件包安装和[packages](https://anaconda.org/conda-forge/repo)：
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install <package-name>
```

## 参考资料：
- [Anaconda installer](https://www.anaconda.com/distribution)
- [Anaconda installer archive](https://repo.anaconda.com/archive/)
- [Getting started with Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/)
- [Frequently asked questions](https://docs.anaconda.com/anaconda/user-guide/faq/)
- [Jupyter magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)