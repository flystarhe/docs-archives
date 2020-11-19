title: Java to Python
date: 2018-03-06
tags: [Python]
---
JPype是一个能够让Python代码方便地调用Java代码的工具,从而克服了Python在某些领域中的不足.

<!--more-->
## 安装Java
```bash
$ cd /home/hejian/apps
## http://www.oracle.com/technetwork/java/javase/downloads/index.html
## 下载二进制文件: jdk-8u161-linux-x64.tar.gz
$ tar zxf jdk-8u161-linux-x64.tar.gz
$ rm -rf jdk-8u161-linux-x64.tar.gz
$ echo $'export JAVA_HOME=/home/hejian/apps/jdk1.8.0_161' >> /etc/profile
$ echo $'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
$ echo $'export CLASSPATH=.:$JAVA_HOME/lib:$JAVA_HOME/jre/lib' >> /etc/profile
$ source /etc/profile
$ java -version
```

## 安装Jpype1
```bash
$ conda install -c conda-forge jpype1
```

Debian/Ubuntu:
```bash
$ sudo apt-get install g++ python3-dev
```

Red Hat/Fedora:
```bash
$ su -c 'yum install gcc-c++ python3-devel'
```

## 使用Jpype1
```python
import jpype
jvmPath = jpype.getDefaultJVMPath()
print(jvmPath)
```

>如果报错,可能是`JAVA_HOME`没有生效.如果你在使用`Jupyter`,可以在启动`Jupyter`时,先执行`export JAVA_HOME=/home/hejian/apps/jdk1.8.0_161`.总之,必须设置环境变量`JAVA_HOME`到JDK的根目录.

优雅的方式,代码前端添加:
```python
import os
os.environ['JAVA_HOME'] = 'home/hejian/apps/jdk1.8.0_161'
```

执行Java语句:
```python
import jpype

jpype.startJVM(jpype.getDefaultJVMPath(), '-ea')
jpype.java.lang.System.out.println('hello world')
jpype.shutdownJVM()
```

## 调用Jar
自定义第三方Jar包,打包为`speech_to_text.jar`:
```java
package com.hej.speech_to_text;

import java.io.InputStream;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

public class Demo {
    public String sayHello(String user) {
        return "hello, " + user;
    }

    public String load() {
        String result = "load():\n";
        try {
            InputStream is = Demo.class.getClassLoader().getResourceAsStream("config.properties");
            Properties prop = new Properties();
            prop.load(is);
            Set<Map.Entry<Object, Object>> set = prop.entrySet();
            Iterator<Map.Entry<Object, Object>> it = set.iterator();
            String key = null;
            String value = null;
            while (it.hasNext()) {
                Map.Entry<Object, Object> entry = (Map.Entry) it.next();
                key = String.valueOf(entry.getKey());
                value = String.valueOf(entry.getValue());
                result += "=>" + key + "=" + value + "\n";
            }
        } catch (Exception e) {
            System.out.println("The config file load exception");
        }
        return result;
    }

    public static void main(String[] args) {
        Demo test = new Demo();
        System.out.println(test.sayHello("test"));
        System.out.println(test.load());
    }
}
```

有时候需要在Jar包内部包含`properties`文件,在Maven工程里,将`properties`文件放在`src/main/resources`目录,就会自动打包到`classes`目录下.然后在Jar包的代码中读取如下:
```java
InputStream is = Demo.class.getClassLoader().getResourceAsStream("config.properties");
}
```

`config.properties`:
```
# APP ID
app_id=xxx
# secret key
secret_key=xxxxxx
# we support both http and https prototype
lfasr_host=http://lfasr.xfyun.cn/lfasr/api
```

Python调用Jar包程序:
```python
import jpype
import os.path as osp

jvmPath = jpype.getDefaultJVMPath()
jarPath = osp.join(osp.abspath(osp.dirname(__file__)), 'speech_to_text.jar')

jpype.startJVM(jvmPath, '-ea', '-Djava.class.path={}'.format(jarPath))
JDClass = jpype.JClass('com.hej.speech_to_text.Demo')
jd = JDClass()
print(jd.sayHello('wow'))
print(jd.load())
jpype.shutdownJVM()
```

## 调用HanLP
HanLP是由一系列模型与算法组成的Java工具包,目标是普及自然语言处理在生产环境中的应用.HanLP由3部分组成:`hanlp.jar`,`data.zip`,`hanlp.properties`.请前往[github](https://github.com/hankcs/HanLP)下载最新版.

`hanlp.properties`配置文件的作用是告诉HanLP数据包的位置,只需修改第一行:
```
root=usr/home/HanLP/
```

为`data`的父目录即可,比如data目录是`/Users/hankcs/Documents/data`,那么`root=/Users/hankcs/Documents/`.如果选用mini数据包的话,则需要修改配置文件:
```
CoreDictionaryPath=data/dictionary/CoreNatureDictionary.mini.txt
BiGramDictionaryPath=data/dictionary/CoreNatureDictionary.ngram.mini.txt
```

>为了方便用户,特提供内置了数据包的Portable版`hanlp-portable.jar`,零配置即可使用基本功能.请前往[mvnrepository](http://mvnrepository.com/artifact/com.hankcs/hanlp)下载最新版.

比如,创建目录`/data2/tmps/pyhej-nlp/java/jars/`,把`hanlp.jar`和`hanlp.properties`放进去:
```
jars
├── hanlp-1.5.4.jar
├── hanlp-portable-1.5.4.jar
├── hanlp.properties
```

### Python调用
```python
import jpype

# 连接符(Linux冒号:,Windows分号;)
dPath = '/data2/tmps/pyhej-nlp/java/jars/hanlp-1.5.4.jar:/data2/tmps/pyhej-nlp/java/jars'
jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s'%dPath, '-Xms1g', '-Xmx1g')
HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
# 中文分词
print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
jpype.shutdownJVM()
```

也可是Portable版:
```python
import jpype

# 连接符(Linux冒号:,Windows分号;)
dPath = '/data2/tmps/pyhej-nlp/java/jars/hanlp-portable-1.5.4.jar'
jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s'%dPath, '-Xms1g', '-Xmx1g')
HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
# 中文分词
print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
jpype.shutdownJVM()
```
