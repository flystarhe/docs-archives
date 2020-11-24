title: Linux | Science Lab on CentOS
date: 2015-08-10
tags: [Linux,CentOS]
---
CentOS是Linux发行版之一，它是来自于Red Hat Enterprise Linux依照开放源代码规定释出的源代码所编译而成。由于出自同样的源代码，因此有些要求高度稳定性的服务器以CentOS替代商业版的Red Hat Enterprise Linux使用。

<!--more-->
## 准备
做些必要的准备工作能省去不少麻烦。如：GCC系列之C编译器、C++编译器、Fortran编译器，辅助工具之make、wget、vim、ssh，以及其它依赖关系。
```
$ hostname
master
$ rpm -Uvh http://mirrors.ustc.edu.cn/epel/epel-release-latest-7.noarch.rpm
$ yum -y update
$ yum -y install gcc gcc-c++ gcc-gfortran make vim ssh epel-release lrzsz
$ yum -y install libtool lapack lapack-devel blas blas-devel readline-devel libXt-devel zlib-devel libxml2-devel 
$ mkdir -p /flab/abfs #lab目录
$ cat /etc/hosts
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
192.168.0.108  master    master.com
192.168.0.109  slave1    slave1.com
192.168.0.110  slave2    slave2.com
```

## 安装vsftp
全称是Very Secure FTP，一个安全、高速、稳定的FTP服务器。添加FTP账号www，指定/home/www为主目录，且默认不能登陆系统。
```
$ systemctl stop firewalld.service #停止firewall // service iptables stop
$ systemctl disable firewalld.service #禁止firewall开机启动 // chkconfig iptables off
$ vim /etc/selinux/config #关闭selinux
SELINUX=disabled
#SELINUXTYPE=targeted
$ setenforce 0 #使配置生效
$ yum -y install vsftpd
$ useradd -s /sbin/nologin -d /home/www www #添加FTP账号
$ passwd www #修改密码
$ chown -R www /home/www #修改权限
$ vim /etc/vsftpd/vsftpd.conf #修改配置
anonymous_enable=NO
$ systemctl restart vsftpd.service #重启vsftpd配置生效
$ systemctl enable vsftpd.service #设置vsftpd开机启动
```

## 安装jdk & scala & spark
执行`java -version`检查是否安装openJDK，或执行`rpm -qa | grep java`查看是否有openJDK相关的项，如果存在请先行卸载`rpm -e --nodeps xxx`。
```
$ ll
-rw-r--r--. 1 www www 153530841 Aug 31 21:24 jdk-7u80-linux-x64.tar.gz
-rw-r--r--. 1 www www  29924685 Aug 31 21:24 scala-2.10.5.tgz
-rw-r--r--. 1 www www 250331360 Aug 31 21:24 spark-1.4.1-bin-hadoop2.6.tgz
$ tar -zxvf jdk-7u80-linux-x64.tar.gz -C /flab
$ tar -zxvf scala-2.10.5.tgz -C /flab
$ tar -zxvf spark-1.4.1-bin-hadoop2.6.tgz -C /flab
$ echo $'export JAVA_HOME=/flab/jdk1.7.0_80' >> /etc/profile
$ echo $'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
$ echo $'export CLASSPATH=.:$JAVA_HOME/lib:$JAVA_HOME/jre/lib' >> /etc/profile
$ echo $'export SCALA_HOME=/flab/scala-2.10.5' >> /etc/profile
$ echo $'export PATH=$PATH:$SCALA_HOME/bin' >> /etc/profile
$ echo $'export SPARK_HOME=/flab/spark-1.4.1-bin-hadoop2.6' >> /etc/profile
$ echo $'export PATH=$PATH:$SPARK_HOME/bin' >> /etc/profile
$ source /etc/profile
$ java -version
$ scala -version
$ spark-shell
$ scala> 9*9
res0: Int = 81
$ scala> :quit
```

## 安装python3
设置Python3为默认版本可能会导致yum不可用，执意如此可尝试将`/usr/bin/yum`首行`#!/usr/bin/python`改为`#!/usr/bin/python2`。
```
$ which python
$ tar -zxvf Python-3.4.3.tgz
$ cd Python-3.4.3
$ ./configure --prefix=/flab/python-3.4.3
$ make && make install
$ mv /usr/bin/python /usr/bin/python-old
$ ln -s /flab/python-3.4.3/bin/python3 /usr/bin/python3
$ echo $'export PYTHON_HOME=/flab/python-3.4.3' >> /etc/profile
$ echo $'export PATH=$PATH:$PYTHON_HOME/bin' >> /etc/profile
$ source /etc/profile
$ python -V #默认版本
$ python3 -V
```

## 安装python扩展
这里只关注numpy(数组操作库)、scipy(科学计算库)和matplotlib(2D绘图库)。(依赖`lapack lapack-devel blas blas-devel`)
```
# if python2
$ yum -y install python-pip
$ pip install --upgrade pip
$ pip install numpy scipy python-matplotlib //或：yum -y install numpy scipy python-matplotlib
# if python3
$ cd /flab/python-3.4.3/bin
$ ./pip3 install numpy
```

## 安装R
R是用于统计分析、绘图的语言和操作环境。R是属于GNU系统的一个自由、免费、源代码开放的软件，它是一个用于统计计算和统计制图的优秀工具。(依赖`readline-devel libXt-devel`)
```
$ yum -y install gcc gcc-c++ gcc-gfortran readline-devel libXt-devel tcl tcl-devel tclx tk tk-devel curl-devel openssl-devel
$ ll
-rw-r--r--. 1 www  www   28942883 Aug 31 21:24 R-3.0.3.tar.gz
$ tar -zxvf R-3.0.3.tar.gz
$ cd R-3.0.3
$ ./configure --enable-BLAS-shlib --enable-R-shlib --prefix=/flab/R-3.0.3
$ make && make install
$ echo $'export R_HOME=/flab/R-3.0.3' >> /etc/profile
$ echo $'export PATH=$PATH:$R_HOME/bin' >> /etc/profile
$ source /etc/profile
$ R
```

推荐采用如下比较省事的方式安装：
```
$ yum -y install epel-release
$ yum -y install R #epel源安装
```

## 安装sublime text
她是开发代码编辑的神器，具有代码高亮、语法提示、自动完成且反应快速的编辑器软件，不仅具有华丽的界面，还支持插件扩展机制，用她来写代码，绝对是一种享受。

### yum
Install the GPG key:
```
sudo rpm -v --import https://download.sublimetext.com/sublimehq-rpm-pub.gpg
```

Select the channel to use:
```
sudo yum-config-manager --add-repo https://download.sublimetext.com/rpm/stable/x86_64/sublime-text.repo
```

Update yum and install Sublime Text:
```
sudo yum install sublime-text
```

[link](http://www.sublimetext.com/docs/3/linux_repositories.html)

## 安装slave1 & 2
从`hosts`内容来看，除了master，还有两个slave，再来一遍吗？不，为什么不试试`scp`！(前提是master能ping通slave)
```
# [root@master ~]
$ scp -qr /flab root@slave1:/flab
$ scp -qr /etc/profile root@slave1:/etc/profile
$ scp -qr /etc/hosts root@slave1:/etc/hosts
$ scp -qr /flab root@slave2:/flab
$ scp -qr /etc/profile root@slave2:/etc/profile
$ scp -qr /etc/hosts root@slave2:/etc/hosts
# [root@slave* ~]
$ yum -y update
$ yum -y install gcc gcc-c++ gcc-gfortran make vim ssh epel-release lrzsz
```

到这里，怪物们就算凑齐了。接下来会探索 **为我所用**！