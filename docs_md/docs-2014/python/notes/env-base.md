title: 打造Python科学计算环境
date: 2015-06-02
tags: [Python]
---
Python的设计哲学是“优雅”、“明确”、“简单”。随着numpy、scipy、matplotlab等扩展包逐渐成熟大有一副要干掉R的气势，当然R的价值其实也不在语言。
>系统环境：windows 7 ultimate 64bit + Sublime Text。

<!--more-->
## 安装Python
Python的安装非常简单，下载[python-3.4.3.msi](https://www.python.org/downloads/windows/)，然后双击执行。我习惯是安装在`D:\Python34`。记得将`D:\Python34`与`D:\Python34\Scripts`添加到PATH环境变量。

## 安装扩展包
我们的目标是搭建科学计算环境，故仅演示ipython、numpy、scipy、matplotlib等扩展包的安装。我相信搞定了这些钉子户，其他的都不是问题。

**2.1 如何安装扩展包？**
Python在windows下扩展包常见的安装方式有5种：

1. **setup.py：**下载其zip压缩包并解压，命令行切换到解压目录，执行这个模块提供的setup.py文件`python setup.py install`。
2. **easy_install：**执行命令`easy_install packagename`，很简单是吧！不过需先安装setuptools扩展，下载[ez_setup.py](https://pypi.python.org/pypi/setuptools)，然后`python ez_setup.py`安装。
3. **pip：**执行命令`pip install packagename`，这也是推荐的。好消息是Python34已经给你准备好了，不过建议升级一下`easy_install --upgrade pip`。当然记住一些命令参数是有用的，`pip uninstall packagename`卸载包，`pip list`列出安装的包。
4. **wheel：**wheel本质上是一个zip包格式，它使用`.whl`扩展名。首先下载[packagename-..-none-any.whl](http://www.lfd.uci.edu/~gohlke/pythonlibs/)，然后`pip install packagename-..-none-any.whl`安装。
5. **exe：**绝对是windows用户最受欢迎的，没有之一。我安装scipy时经历各种失败后，就去下载了[scipy-0.15.1-win32-superpack-python3.4.exe](http://sourceforge.net/projects/scipy/files/scipy/0.15.1/)。

**2.2 安装ipython**
有了前面 **pip**的准备工作这一步就变得非常简单了。具体安装命令是这样的`pip install --upgrade "ipython[all]"`，其中参数这里就不赘述了。

**2.3 安装numpy**
根据前面的经验我们执行`pip install numpy`就行了，如果你直接成功了，那么我恭喜你。如果遇到`error Microsoft Visual C++ 10 ..`什么的...何必那么费劲，干嘛不去下载[numpy-1.9.2-win32-superpack-python3.4.exe](http://sourceforge.net/projects/numpy/files/)呢？难道作为windows用户的你不爱exe了！也许你会遇到`python version *.* required, which was not found in the registry`的问题，麻烦你把64bit-python换为32bit-python，谢谢！

**2.4 安装scipy**
这个就有点调皮了，遗憾的是`pip install scipy`这条路不通。不用紧张，我们还有 **wheel**和 **exe**这两招没使出来呢。方案1，下载[scipy-0.15.1-cp34-none-win32.whl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)，然后`pip install scipy-0.15.1-cp34-none-win32.whl`安装；方案2，下载[scipy-0.15.1-win32-superpack-python3.4.exe](http://sourceforge.net/projects/scipy/files/scipy/)，然后双击。

**2.5 安装matplotlib**
这家伙看起来很懂事的样子，真的`pip install matplotlib`就成功了。不过接下来的事就好玩了，环境组建好了是不是该见证奇迹呢？下面是我的代码：

    # coding=utf-8
    import numpy as np
    import scipy as sp
    import pylab as pl
    x = np.linspace(0, 4*np.pi, 100)
    pl.plot(x, np.sin(x), label='y=sin(x)')
    pl.title(u'哈哈哈哈哈')
    pl.legend()
    pl.show()

事实证明我高兴的早了点，我那张扬的“哈哈哈哈哈”被该死的方框给取代了。中文显示问题，然而都不是事。所以我又添加了两行代码：

    # coding=utf-8
    import numpy as np
    import scipy as sp
    import pylab as pl
    x = np.linspace(0, 4*np.pi, 100)
    pl.plot(x, np.sin(x), label='y=sin(x)')
    pl.title(u'哈哈哈哈哈')
    pl.legend()
    pl.mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    pl.mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块
    pl.show()

到这里，我们的环境正式组建完毕。嗯，是时候伸个懒腰了！

## 参考资料：
- [pip user guide](https://pip.pypa.io/en/latest/user_guide.html)
- [Python包管理工具解惑](http://www.tuicool.com/articles/FNJZNr)
- [Windows下pip安装包报错](http://www.cnblogs.com/ldm1989/p/4210743.html)
- [Matplotlib输出中文显示问题](http://my.oschina.net/u/1180306/blog/279818)