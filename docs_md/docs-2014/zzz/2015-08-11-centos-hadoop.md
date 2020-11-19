title: Hadoop on CentOS
date: 2015-08-11
tags: [CentOS,Hadoop]
---
Hadoop是Apache软件基金会旗下的一个开源分布式计算平台。以Hadoop分布式文件系统（HDFS，Hadoop Distributed Filesystem）和MapReduce（Google MapReduce的开源实现）为核心的Hadoop为用户提供了系统底层细节透明的分布式基础架构。
>系统环境：CentOS 7 + JDK 1.7.0 + Hadoop 2.6.0。

<!--more-->
## 准备
该步骤要完成`/etc/hosts`文件配置和ssh无密码登录，假设已有3台软件环境雷同且能互连的机器`master:192.168.0.108;slave1:192.168.0.109;slave2:192.168.0.110`，否则请参考[Spark on CentOS](#)先行准备。

    $ hostname
    master
    $ pwd
    /root
    $ cat /etc/hosts
    127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
    ::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
    192.168.0.108  master    master.com
    192.168.0.109  slave1    slave1.com
    192.168.0.110  slave2    slave2.com
    $ ssh-keygen -t rsa
    $ cat .ssh/id_rsa.pub >> .ssh/authorized_keys
    $ ssh master #无密码登录测试
    $ exit #退出登录
    $ ssh root@slave1 mkdir -p /root/.ssh
    $ scp -qr .ssh/authorized_keys root@slave1:/root/.ssh/
    $ ssh root@slave2 mkdir -p /root/.ssh
    $ scp -qr .ssh/authorized_keys root@slave2:/root/.ssh/
    $ ssh slave1 #无密码登录测试
    $ exit #退出登录
    $ ssh slave2 #无密码登录测试
    $ exit #退出登录

## 安装Hadoop
Hadoop的安装方式很简单，下载安装包并解压到目标目录即可。

    $ hostname
    master
    $ pwd
    /home/www
    $ ll
    -rw-r--r--. 1 www www 195257604 Aug 31 21:24 hadoop-2.6.0.tar.gz
    $ tar -zxvf hadoop-2.6.0.tar.gz -C /flab
    $ echo $'export HADOOP_HOME=/flab/hadoop-2.6.0' >> /etc/profile
    $ echo $'export PATH=$PATH:$HADOOP_HOME/bin' >> /etc/profile
    $ source /etc/profile

## 配置Hadoop
要修改的配置文件都在`$HADOOP_HOME/etc/hadoop`目录，涉及`hadoop-env.sh, yarn-env.sh, slaves, core-site.xml, hdfs-site.xml, mapred-site.xml, yarn-site.xml`。

    $ hostname
    master
    $ pwd
    /flab/hadoop-2.6.0/etc/hadoop
    $ mkdir -p /flab/hadoop-2.6.0/hadoop-tmp /flab/hadoop-2.6.0/hadoop-hdfs-name /flab/hadoop-2.6.0/hadoop-hdfs-data
    $ vim hadoop-env.sh
    export JAVA_HOME=/flab/jdk1.7.0_80
    $ vim yarn-env.sh
    export JAVA_HOME=/flab/jdk1.7.0_80
    $ vim slaves
    slave1
    slave2
    $ vim core-site.xml
    <configuration>
        <property>
            <name>hadoop.tmp.dir</name>
            <value>file:///flab/hadoop-2.6.0/hadoop-tmp</value>
        </property>
        <property>
            <name>fs.default.name</name>
            <value>hdfs://master:9000</value>
        </property>
    </configuration>
    $ vim hdfs-site.xml
    <configuration>
        <property>
            <name>dfs.namenode.secondary.http-address</name>
            <value>master:50090</value>
        </property>
        <property>
            <name>dfs.namenode.name.dir</name>
            <value>file:///flab/hadoop-2.6.0/hadoop-hdfs-name</value>
        </property>
        <property>
            <name>dfs.datanode.data.dir</name>
            <value>file:///flab/hadoop-2.6.0/hadoop-hdfs-data</value>
        </property>
        <property>
            <name>dfs.replication</name>
            <value>1</value>
        </property>
    </configuration>
    $ cp mapred-site.xml.template mapred-site.xml
    $ vim mapred-site.xml
    <configuration>
        <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
        </property>
    </configuration>
    $ vim yarn-site.xml
    <configuration>
        <property>
            <name>yarn.resourcemanager.hostname</name>
            <value>master</value>
        </property>
        <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
        </property>
    </configuration>

## 复制Hadoop
配置好后，将master上的Hadoop文件夹复制到各个slave节点上。建议参照[Science lab on CentOS](#)中vsftp安装环节关闭firewall。

    $ hostname
    master
    $ pwd
    /root
    $ scp -qr /flab/hadoop-2.6.0 root@slave1:/flab/
    $ scp -qr /etc/profile root@slave1:/etc/profile
    $ scp -qr /etc/hosts root@slave1:/etc/hosts
    $ scp -qr /flab/hadoop-2.6.0 root@slave2:/flab/
    $ scp -qr /etc/profile root@slave2:/etc/profile
    $ scp -qr /etc/hosts root@slave2:/etc/hosts
    # 停止firewall & 关闭selinux
    $ ssh root@slave1 systemctl stop firewalld.service // service iptables stop
    $ ssh root@slave1 systemctl disable firewalld.service chkconfig iptables off
    $ scp -qr /etc/selinux/config root@slave1:/etc/selinux/config
    $ ssh root@slave1 setenforce 0
    $ ssh root@slave2 systemctl stop firewalld.service // service iptables stop
    $ ssh root@slave2 systemctl disable firewalld.service chkconfig iptables off
    $ scp -qr /etc/selinux/config root@slave2:/etc/selinux/config
    $ ssh root@slave2 setenforce 0

## 运行Hadoop
在master节点上启动Hadoop。

    $ hostname
    master
    $ pwd
    /flab/hadoop-2.6.0
    $ bin/hdfs namenode -format #格式化文件系统
    $ sbin/start-dfs.sh #启动hdfs
    > jps@master: namenode secondarynamenode
    > jps@slave*: datanode
    $ sbin/start-yarn.sh #启动yarn
    > jps@master >> namenode secondarynamenode resourcemanage
    > jps@slave* >> datanode nodemanager
    > mapr >> http://192.168.0.106:8088
    > hdfs >> http://192.168.0.106:50070
    $ bin/hdfs dfsadmin –report #查看集群状态
    $ sbin/stop-all.sh #先别停止|还要测试

## 测试Hadoop
先简单的hdfs文件操作来测试hdfs运行情况，再执行WordCount实例检查MapReduct作业。

    $ hostname
    master
    $ pwd
    /root
    $ hdfs dfs -mkdir -p /test
    $ hdfs dfs -put test.txt /test
    $ hdfs dfs -ls -R /test
    $ hdfs dfs -cat /test/test.txt
    $ hdfs dfs -rm -f -R /test
    $ cd /flab/hadoop-2.6.0
    $ hdfs dfs -mkdir -p /count_in
    $ hdfs dfs -put readme.txt /count_in
    $ hadoop jar /flab/hadoop-2.6.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0.jar wordcount /count_in /count_out
    $ hdfs dfs -ls -R /count_out
    $ sbin/stop-all.sh #停止
OK，Hadoop 2.6.0的分布式环境搭建完成。