title: IPython/Jupyter与kernels
date: 2016-06-27
tags: [IPython,Jupyter]
---
IPython是一个python的交互式shell，比默认的python shell好用得多。还提供了Python之外的许多编程语言支持，[IPython/Jupyter kernels list](https://github.com/ipython/ipython/wiki/IPython-kernels-for-other-languages)：R、Scala、Spark等等，不过需要用户手动安装。

<!--more-->
## Install Jupyter Notebook
虽然Jupyter可以运行很多编程语言，但Python是安装Jupyter的先决条件。推荐新用户使用[Anaconda](https://www.continuum.io/downloads)来安装Python和Jupyter。[文档](http://jupyter.readthedocs.io/en/latest/install.html)

    $ yum -y install wget epel-release # 安装epel扩展源
    $ wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
    $ bash Anaconda3-4.1.1-Linux-x86_64.sh
    $ conda info
    $ conda update conda //更新conda

Anaconda还提供了一个强大的包管理工具，可以通过conda进行各种包的操作。比如：

    $ conda update python //更新python
    $ conda update pip //更新包
    $ conda list //查看安装的包和框架
    $ conda search jieba //搜索包
    $ conda install swapi //安装包
    $ conda install setuptools==19.2 //安装包：特定版本
    $ conda remove setuptools //卸载包
    $ rm -rf ~/anaconda3 //卸载conda

更多`conda`的使用参考[文档](http://conda.pydata.org/docs/test-drive.html)。下面说说使用`pip`安装Jupyter：

    $ yum -y install python3-devel # if python2 use python-devel
    $ pip3 install jupyter # if python2 use pip

运行Jupyter笔记本：(默认`ip='localhost'`，`port=8888`)

    $ jupyter --version
    $ jupyter notebook --no-browser --ip=192.168.190.136 --port=8888
    $ service iptables stop //停止firewall
    $ chkconfig iptables off //禁止firewall开机启动

浏览器打开`http://192.168.190.136:8888/tree`就可以体验了。远程访问需要开放端口访问权限，我习惯直接关闭防火墙。

## Install Scala Kernel
这里以2.10.x为例，更多内容参考[链接](https://github.com/alexarchambault/jupyter-scala)。

    $ curl -L -o jupyter-scala-2.10 https://git.io/vrHh7
    $ chmod +x jupyter-scala-2.10
    $ ./jupyter-scala-2.10
    $ rm -f jupyter-scala-2.10
    $ jupyter kernelspec list

现在应该看到kernels里面除了python，还有scala了。运行Jupyter笔记本：

    $ jupyter notebook --no-browser --ip=192.168.190.136 --port=8888

浏览器打开`http://192.168.190.136:8888/tree`就可以体验了。远程访问需要开放端口访问权限，我习惯直接关闭防火墙。

## Install R Kernel
安装ZMQ，更多内容参考[链接](http://irkernel.github.io/installation/#linux-panel)：

    $ yum update
    $ yum info R
    $ yum list R
    $ yum list installed | grep R
    $ yum -y erase R //如果需要卸载
    $ yum -y install epel-release //感觉很省事，但用最新版本有时并不轻松
    $ yum -y install R
    $ yum -y install czmq-devel
    $ yum -y install gcc gcc-c++ gcc-gfortran readline-devel libXt-devel tcl tcl-devel tclx tk tk-devel curl-devel openssl-devel

启动R，执行命令：(注：推荐R 3.2，R 3.3可能会失败)

    $ R
    > install.packages(c('rzmq','repr','IRkernel','IRdisplay'), repos = c('http://irkernel.github.io/', getOption('repos')), type = 'source') //推荐
    # install.packages(c('pbdZMQ', 'repr', 'devtools')) //选择http线路
    # devtools::install_github(c('IRkernel/IRdisplay', 'IRkernel/IRkernel'))
    > IRkernel::installspec() //IRkernel::installspec(name='ir33', displayname='R 3.3')
    > q()
    $ jupyter kernelspec list

现在应该看到kernels里面除了python、scala，还有R了。运行Jupyter笔记本：

    $ jupyter notebook --no-browser --ip=192.168.190.136 --port=8888

浏览器打开`http://192.168.190.136:8888/tree`就可以体验了。远程访问需要开放端口访问权限，我习惯直接关闭防火墙。

## 魔法命令
IPython提供了许多魔法命令，使得在IPython环境中的操作更加得心应手。魔法命令都以`%`或者`%%`开头，以`%`开头的成为行命令，`%%`开头的称为单元命令。行命令只对命令所在的行有效，而单元命令则必须出现在单元的第一行，对整个单元的代码进行处理。

执行`%magic`可以查看关于各个命令的说明，而在命令之后添加`?`可以查看该命令的详细说明。

## Matplotlib
`matplotlib`是最著名的Python图表绘制扩展库，它支持输出多种格式的图形图像，并且可以使用多种GUI界面库交互式地显示图表。使用`%matplotlib`命令可以将`matplotlib`的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定`matplotlib`图表的显示方式。

在下面的例子中，`inline`表示将图表嵌入到Notebook中。因此最后一行`pl.plot()`所创建的图表将直接显示在该单元之下，由于我们不需要查看最后一行返回的对象，因此以分号结束该行。

```
%matplotlib inline
import pylab as pl
pl.seed(1)
data = pl.randn(100)
pl.plot(data);
```

内嵌图表的输出格式缺省为`PNG`，可以通过`%config`命令修改这个配置。`%config`命令可以配置IPython中的各个可配置对象，其中`InlineBackend`对象为`matplotlib`输出内嵌图表时所使用的对象，我们配置它的`figure_format="svg"`，这样将内嵌图表的输出格式修改为`SVG`。

```
%config InlineBackend.figure_format="svg"
%matplotlib inline
pl.plot(data);
```

内嵌图表很适合制作图文并茂的Notebook，然而它们是静态的无法进行交互。这时可以将图表输出模式修改为使用GUI界面库，下面的qt4表示使用QT4界面库显示图表。请读者根据自己系统的配置，选择合适的界面库：‘gtk’, ‘osx’, ‘qt’, ‘qt4’, ‘tk’, ‘wx’。

## 性能分析
性能分析对编写处理大量数据的程序非常重要，特别是Python这样的动态语言，一条语句可能会执行很多内容，有的是动态的，有的调用二进制扩展库，不进行性能分析，就无法对程序进行优化。IPython提供了许多进行性能分析的魔法命令。

`%timeit`调用`timeit`模块对单行语句重复执行多次，计算出其执行时间。下面的代码测试修改列表单个元素所需的时间。

```
a = [1,2,3]
%timeit a[1] = 10
```

`%%timeit`则用于测试整个单元中代码的执行时间。下面的代码测试空列表中循环添加10个元素所许的时间：

```
%%timeit
a = []
for i in range(10):
    a.append(i)
```

`timeit`命令会重复执行代码多次，而`time`则只执行一次代码，输出代码的执行情况，和`timeit`命令一样，它可以作为行命令和单元命令。下面的代码统计往空列表中添加10万个元素所需的时间。

```
%%time
a = []
for i in range(100000):
    a.append(i)
```

`time`和`timeit`命令都将信息使用`print`输出，如果希望用程序分析这些信息，可以使用`%%capture`命令，将单元格的输出保存为一个对象。下面的程序对不同长度的数组调用`sort()`函数进行排序，并使用`%timeit`命令统计排序所需的时间。为了加快程序的计算速度，这里通过`-n20`指定代码的运行次数为20次。由于使用了`%%capture`命令，程序执行之后没有输出，所有输出都被保存进了`result`对象。

```
%%capture result
import numpy as np
for n in [1000, 5000, 10000, 50000, 100000, 500000]:
    arr = np.random.rand(n)
    print("n={0}".format(n))
    %timeit -n20 np.sort(arr)
```

`result.stdout`属性中保存通过标准输出管道中的输出信息.下面的代码使用`re`模块从上面的字符串中获取数组长度和排序执行时间的信息，并将其绘制成图表。图表的横坐标为对数坐标轴，表示数组的长度；纵坐标为平均每个元素所需的排序时间。可以看出每个元素所需的平均排序时间与数组的长度的对数成正比，因此可以计算出排序函数`sort()`的时间复杂度为：

```
def tosec(t):
    units = {"ns":1e-9, "us":1e-6, "ms":1e-3, "s":1}
    value, unit = t.strip().split()
    return float(value) * units[unit]

import re
info = re.findall(r"n=(.*?)\s+(.*?(?:s|ms|us|ns))", result.stdout)
info = [(int(t0), tosec(t1)) for t0, t1 in info]
x, y = np.r_[info].T
pl.semilogx(x, y/x, "-o");
```

`%%prun`命令调用`profile`模块，对单元中的代码进行性能剖析。下面的性能剖析显示`fib()`运行了`21891`次，而`fib_fast()`则只运行了`20`次。

## 使用技巧
在Jupyter中使用系统命令非常容易：

```
!pwd
```

有时可能临时生成文件，总有那么些脚本是需要动态生成的。如下：

```
%%writefile _tmp/r_hello_world.R
args <- commandArgs(TRUE)

print(args)
```

然后执行它们,并删除:
```
!Rscript _tmp/r_hello_world.R 1 TRUE abc
!rm -rf tmp_r_hello_world.R
```

使用`Image`和`IPython.display`显示图片：

```
from PIL import Image
from IPython.display import display
display(Image.open("_images/yk-logo-1220.png"))
```

`Jupyter`不仅仅能运行`python`，还可以执行`shell`。如下：

```
%%bash
for i in {1..3}
do
    echo "i is $i"
done
```

## R
执行`conda install rpy2`安装，安装后`R_Home`为`/root/anaconda3/lib/R`，`R_Library`为`/root/anaconda3/lib/R/library`。可从这里启动环境来实现对`R`的维护和管理，比如`install.packages`，虽然在Python脚本也能做到，但是操作起来会麻烦的多。[参考](http://rpy2.readthedocs.io/en/version_2.8.x/introduction.html)

```
import rpy2.robjects as robjects
print(robjects.r("x = 1:5")) #eval
print(robjects.r["x"]) #get
```

```
import rpy2.robjects as robjects
from rpy2.robjects import globalenv

globalenv["var_int_vec"] = robjects.IntVector([1,2,3,4,5])
globalenv["var_str_vec"] = robjects.StrVector(["a","b","c"])
globalenv["var_bool_vec"] = robjects.BoolVector([True,False,True])
globalenv["var_factor_vec"] = robjects.FactorVector(["a","b","a","a","b"])
globalenv["var_list_vec"] = robjects.ListVector({"a": 1,"b": "text","c": True})

import pandas as pd
from rpy2.robjects import pandas2ri
pandas2ri.activate()
df = pd.DataFrame({'txt':['a','b','c'], 'int':[1,2,3]})
globalenv["var_data_frame"] = robjects.DataFrame(df)

print(robjects.r("ls()"))
print(globalenv["var_data_frame"])
print(robjects.r["var_data_frame"])

import rpy2.robjects as robjects
from rpy2.robjects import Environment

eval_env = Environment()
eval_env["var_txt"] = robjects.py3str(u"何剑你好")

print(robjects.r.ls(eval_env))
print(eval_env["var_txt"])

import rpy2.robjects as robjects
sqr = robjects.r('function(x) x^2')
print(sqr(2))
```

## 参考资料:
- [IPython交互环境](http://hyry.dip.jp/tech/book/page/scipynew/ipython-200-notebook-magic.html)
- [IPython Documentation](http://ipython.readthedocs.io/en/stable/index.html)