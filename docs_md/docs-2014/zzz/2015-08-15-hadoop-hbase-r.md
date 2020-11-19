title: 初探RHadoop and Hbase
date: 2015-08-15
tags: [R,rhdfs,rmr2,rhbase]
---
RHadoop是一款Hadoop和R语言的结合的产品，由RevolutionAnalytics公司开发，并将代码开源到github社区上面。RHadoop包含三个R包(rmr rhdfs rhbase)，分别是对应Hadoop系统架构中的MapReduce、HDFS和HBase三个部分。
>系统环境：CentOS 7 + JDK 1.7.0 + Hadoop 2.6.0 + Hbase 1.1.1 + R 3.0.3。

<!--more-->
## 准备
JDK与R的安装请参考[Science lab on CentOS](#)，Hadoop的安装请参考[Hadoop on CentOS](#)，并下载[rmr2_3.3.1 + rhdfs_1.0.8](https://github.com/RevolutionAnalytics/RHadoop/wiki/Downloads)。然后，还需要安装几个R依赖库`install.packages(c("Rcpp", "RJSONIO", "digest", "functional", "reshape2", "stringr", "plyr", "caTools", "rJava"))`。

## 安装hbase

    $ hostname
    master
    $ ll
    -rw-r--r--.  1 www www   102487389 Aug 31 21:24 hbase-1.1.1-bin.tar.gz
    $ tar -zxvf hbase-1.1.1-bin.tar.gz -C /flab
    $ mkdir -p /flab/hbase-1.1.1/hbase-zk-data
    $ echo $'export HBASE_HOME=/flab/hbase-1.1.1' >> /etc/profile
    $ echo $'export PATH=$PATH:$HBASE_HOME/bin' >> /etc/profile
    $ source /etc/profile #使配置文件生效
    $ vim /flab/hbase-1.1.1/conf/hbase-env.sh
    export JAVA_HOME=/flab/jdk1.7.0_80
    export HBASE_MANAGES_ZK=true
    $ vim /flab/hbase-1.1.1/conf/regionservers
    localhost -> master
    $ vim /flab/hbase-1.1.1/conf/hbase-site.xml
    <configuration>
        <property>
            <name>hbase.rootdir</name>
            <value>hdfs://master:9000/hbase</value>
        </property>
        <property>
            <name>hbase.cluster.distributed</name>
            <value>true</value>
        </property>
        <property>
            <name>hbase.master</name>
            <value>master:60000</value>
        </property>
        <property>
            <name>hbase.zookeeper.quorum</name>
            <value>master</value>
        </property>
        <property>
            <name>hbase.zookeeper.property.dataDir</name>
            <value>/flab/hbase-1.1.1/hbase-zk-data</value>
        </property>
    </configuration>

## 测试hbase

    $ /flab/hadoop-2.6.0/sbin/start-all.sh #启动hadoop
    $ /flab/hbase-1.1.1/bin/start-hbase.sh #启动hbase
    # 查看运行情况: http://192.168.0.106:16030/rs-status
    $ jps
    44427 HQuorumPeer
    44807 Jps
    44593 HRegionServer
    43800 SecondaryNameNode
    44480 HMaster
    43945 ResourceManager
    43657 NameNode
    $ /flab/hbase-1.1.1/bin/hbase shell #启动shell
    hbase(main):001:0> create 'student_shell','info'
    => Hbase::Table - student_shell
    hbase(main):002:0> list
    TABLE
    => ["student_shell"]
    hbase(main):003:0> put 'student_shell','mary','info:age','19'
    hbase(main):004:0> get 'student_shell','mary'
    COLUMN                                   CELL
     info:age                                 timestamp=1441272092798, value=19
    hbase(main):005:0> disable 'student_shell'
    hbase(main):006:0> drop 'student_shell'
    hbase(main):007:0> list
    TABLE
    => []
    hbase(main):008:0> exit

## 安装rhdfs&rmr2

    $ echo $'export HADOOP_HOME=/flab/hadoop-2.6.0' >> /etc/profile
    $ echo $'export HDFS_CMD=$HADOOP_HOME/bin/hdfs' >> /etc/profile
    $ echo $'export HADOOP_CMD=$HADOOP_HOME/bin/hadoop' >> /etc/profile
    $ echo $'export HADOOP_STREAMING=$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar' >> /etc/profile
    $ source /etc/profile
    $ hostname
    master
    $ ll
    -rw-r--r--.  1 www www       25105 Aug 31 21:24 rhdfs_1.0.8.tar.gz
    -rw-r--r--.  1 www www       63087 Aug 31 21:24 rmr2_3.3.1.tar.gz
    $ R CMD INSTALL rhdfs_1.0.8.tar.gz
    $ R CMD INSTALL rmr2_3.3.1.tar.gz

## 测试rhdfs
在测试开始前请保证Hadoop服务已准备就绪。启动`/flab/hadoop-2.6.0/sbin/start-all.sh`，停止`/flab/hadoop-2.6.0/sbin/stop-all.sh`。

    $ R
    > library(rhdfs)
    > hdfs.init()
    > hdfs.ls("/count_in")
      permission owner      group size          modtime                 file
    1 -rw-r--r--  root supergroup   75 2015-09-03 18:02 /count_in/readme.txt
    > hdfs.cat("/count_in/readme.txt")
    > q()

## 测试rmr2
在测试开始前请保证Hadoop服务已准备就绪。MapReduce只能访问HDFS文件系统，先要用to.dfs把数据存储到HDFS文件系统里。MapReduce的运算结果再用from.dfs函数从HDFS文件系统中取出。
>ERROR streaming.StreamJob: Job not successful. i'm sorry!

    $ R
    > library(rmr2)
    > dfs.ints = to.dfs(keyval(sample.int(5, 10, replace=T), 1:10))
    > from.dfs(dfs.ints())
    > func.map = function(k, v) keyval(v, 2*v)
    > func.rdu = function(k, v) keyval(k,sum(v))
    > dfs.calc = mapreduce(input=dfs.ints, map=func.map, reduce=func.rdu)
    > from.dfs(dfs.calc)


## 安装rhbase
rhbase是通过thrift调用HBase的，还需要安装thrift。
>thrift-0.9.2与rhbase_1.1.1没法协作，期待rhbase尽快更新版本吧！

    $ ll #安装thrift
    -rw-r--r--  1 www www   2336261 Aug 30 17:27 thrift-0.9.2.tar.gz
    $ yum -y install automake libtool flex bison pkgconfig gcc-c++ boost-devel libevent-devel zlib-devel python-devel ruby-devel
    $ tar -zxvf thrift-0.9.2.tar.gz
    $ cd /flab/thrift-0.9.2
    $ ./configure --with-lua=no --prefix=/flab/thrift-0.9.2
    $ make && make install
    $ echo $'export THRIFT_HOME=/flab/thrift-0.9.2' >> /etc/profile
    $ echo $'export PATH=$PATH:$THRIFT_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ thrift -version
    Thrift version 0.9.2
    $ /flab/hbase-1.1.1/bin/hbase-daemon.sh start thrift #启动ThriftServer
    $ jps
    $ /flab/hbase-1.1.1/bin/hbase-daemon.sh stop thrift #关闭ThriftServer
    $ ll #安装rhbase
    -rw-r--r--. 1 www www     61701 Aug 31 21:24 rhbase_1.2.1.tar.gz
    $ R CMD INSTALL rhbase_1.2.1.tar.gz

## 测试rhbase
测试rhbase前请先确认Hadoop&Hbase&Thrift均正确启动。

    $ R
    > library(rhbase)
    > hb.init(host=127.0.0.1, port=9090)
    > hb.new.table("student_rhbase", "info")
    > hb.list.tables()
    > hb.insert("student_rhbase", list(list("mary", "info:age", "24")))
    > hb.get('student_rhbase', 'mary')
    > hb.delete.table('student_rhbase')
    > hb.list.tables()
