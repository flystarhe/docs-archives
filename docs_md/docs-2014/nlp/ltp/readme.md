title: 哈工大LTP初接触
date: 2016-02-23
tags: [NLP,LTP,NER]
---
命名实体识别主流算法有：CRF、字典法和混合方法。评价较好的系统有：哈工大的LTP·中文、斯坦福的NER·英文和hankcs的HanLP·中文。其中哈工大语言云以REST方式提供服务，当然也可以从[源代码](https://github.com/HIT-SCIR/ltp/blob/master/doc/install.rst)编译安装。HanLP未公布PR指标，哈工大公布其NER的PR指标分别为0.939552和0.936372。

<!--more-->
## 编译安装ltp
LTP使用编译工具CMake构建项目。在安装LTP之前，你需要首先安装CMake[官网](http://www.cmake.org)。如果你是Windows用户，请下载CMake的二进制安装包；如果你是Linux的用户，可以通过编译源码的方式安装CMake，当然，你也可以使用Linux的软件源来安装。

    $ yum -y install gcc-c++ ncurses
    $ wget https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz
    $ tar -zxf cmake-3.4.3.tar.gz
    $ cd cmake-3.4.3
    $ ./bootstrap
    $ gmake
    $ make install
    $ cmake --version

注：在编译前，请修改`src/server/ltp_server.cpp`122-152行，在“ltp_data/*”前加上您的LTP项目的路径(/root/ltp-3.3.0/)。若需要外部词典你还可以指定`std::string segmentor_lexicon = "";127行`和`std::string postagger_lexcion = "";137行`的真实路径。

    $ wget https://github.com/HIT-SCIR/ltp/archive/v3.3.0.tar.gz
    $ mv v3.3.0.tar.gz ltp-v3.3.0.tar.gz
    $ tar -zxf ltp-v3.3.0.tar.gz
    $ cd ltp-3.3.0
    $ vim src/server/ltp_server.cpp
    $ ./configure
    $ make

## 模型文件ltp_data
LTP的模型文件[ltp-data-v3.3.0.zip](http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569)，解压并拷贝`ltp_data`目录到LTP根目录。

    $ unzip ltp-data-v3.3.0.zip
    $ mv 3.3.0 ltp-data-v3.3.0
    $ cd ltp-data-v3.3.0
    $ cp -r ltp_data /root/ltp-3.3.0/

## 测试验证
转到LTP编译安装目录启动ltp_server就可以通过网页简单测试了。

    $ cd /root/ltp-3.3.0
    $ ./bin/ltp_server --port 9090

使用curl测试ltp_server：

    $ curl --data "s=我爱北京天安门&t=ws&x=n" http://127.0.0.1:9090/ltp
    $ curl --data "s=我爱北京天安门&t=ner&x=n" http://127.0.0.1:9090/ltp
    $ curl --data "s=我爱北京天安门&t=all&x=n" http://127.0.0.1:9090/ltp

使用python测试ltp_server：

    # coding=utf-8
    import urllib, urllib2
    uri_base = "http://192.168.190.136:9090/ltp"
    data = {'s':'我爱北京天安门', 'x':'n', 't':'all'}
    request = urllib2.Request(uri_base)
    params = urllib.urlencode(data)
    response = urllib2.urlopen(request, params)
    content = response.read().strip()
    print(content)

## in python
首先你需要有python环境，安装过程如下：

    $ yum -y install wget epel-release # 安装epel扩展源
    $ wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
    $ wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
    $ bash Anaconda2-4.1.1-Linux-x86_64.sh
    $ conda info
    $ conda update conda //更新conda

安装pyltp有两种方法。使用pip：

    $ pip install pyltp #请尽量使用conda install

或从源代码安装：

    $ git clone https://github.com/HIT-SCIR/pyltp
    $ git submodule init
    $ git submodule update
    $ python setup.py install

然后下载模型文件[百度云](http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569)，当前模型版本3.3.1。

使用pyltp进行分句示例如下：
```python
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
print '\n'.join(sents)
```

使用pyltp进行分词示例如下：
```python
from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load("/lab/ltp/ltp_data/cws.model")  # 加载模型
words = segmentor.segment("元芳你怎么看")  # 分词
print "|".join(words)
segmentor.release()  # 释放模型
```

使用pyltp进行词性标注如下：
```python
from pyltp import Postagger
postagger = Postagger()  # 初始化实例
postagger.load('/lab/ltp/ltp_data/pos.model')  # 加载模型
postags = postagger.postag(words)  # 词性标注
print '\t'.join(postags)
postagger.release()  # 释放模型
```

使用pyltp进行命名实体识别示例如下:
```python
from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load('/lab/ltp/ltp_data/ner.model')  # 加载模型
netags = recognizer.recognize(words, postags)  # 命名实体识别
print '\t'.join(netags)
recognizer.release()  # 释放模型
```

除了分词之外，pyltp还提供词性标注、命名实体识别、依存句法分析、语义角色标注等功能。详细使用方法请参考在线文档。

## 参考资料:
- [编译安装LTP](https://github.com/HIT-SCIR/ltp/blob/master/doc/install.rst)
- [LTP在线文档](http://ltp.readthedocs.org/zh_CN/latest/)
- [LTP训练套件](http://ltp.readthedocs.io/zh_CN/latest/train.html)
- [LTP实现原理](http://ltp.readthedocs.io/zh_CN/latest/theory.html)
- [HIT-SCIR/pyltp](https://github.com/HIT-SCIR/pyltp)