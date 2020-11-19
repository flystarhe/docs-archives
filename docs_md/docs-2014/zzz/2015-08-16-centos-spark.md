title: Spark on CentOS
date: 2015-08-16
tags: [CentOS,Spark]
---
Apache Spark是一种与Hadoop相似的开源集群计算环境，Spark启用了内存分布数据集，除了能够提供交互式查询外，它还可以优化迭代工作负载。
>系统环境：CentOS 7 + JDK 1.7.0 + Scala 2.10.5 + Hadoop 2.6.0 + Spark 1.5.1。

<!--more-->
## 准备
做些必要的准备工作能省去不少麻烦。如：GCC系列之C编译器、C++编译器、Fortran编译器，辅助工具之make、vim、ssh，以及其它依赖关系。

    $ yum -y install gcc gcc-c++ gcc-gfortran make vim ssh libtool lapack lapack-devel blas blas-devel readline-devel libXt-devel zlib-devel libxml2-devel
    $ yum -y update
    $ hostname
    master
    $ cat /etc/hosts
    127.0.0.1   localhost localhost.localdomain
    ::1         localhost localhost.localdomain
    192.168.190.136  master    master.com
    192.168.190.138  slave1    slave1.com
    192.168.190.137  slave2    slave2.com
    $ mkdir -p /flab/abfs #lab目录

## 安装vsftp
全称是Very Secure FTP，一个安全、高速、稳定的FTP服务器。添加FTP账号www，指定`/home/www`为主目录，且默认不能登陆系统。

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

## 安装jdk & scala & hadoop & spark
执行`java -version`检查是否安装openJDK，或执行`rpm -qa | grep java`查看是否有openJDK相关的项，如果存在请先行卸载`rpm -e --nodeps xxx`。

    $ ll
    -rw-r--r-- 1 www www 195257604 Sep 25 13:29 hadoop-2.6.0.tar.gz
    -rw-r--r-- 1 www www 153530841 Sep 25 13:29 jdk-7u80-linux-x64.tar.gz
    -rw-r--r-- 1 www www  29924685 Sep 25 13:29 scala-2.10.5.tgz
    -rw-r--r-- 1 www www 280869269 Sep 25 13:29 spark-1.5.0-bin-hadoop2.6.tgz
    $ tar -zxvf jdk-7u80-linux-x64.tar.gz -C /flab
    $ tar -zxvf scala-2.10.5.tgz -C /flab
    $ tar -zxvf hadoop-2.6.0.tar.gz -C /flab
    $ tar -zxvf spark-1.5.0-bin-hadoop2.6.tgz -C /flab
    $ echo $'export JAVA_HOME=/flab/jdk1.7.0_80' >> /etc/profile
    $ echo $'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
    $ echo $'export CLASSPATH=.:$JAVA_HOME/lib:$JAVA_HOME/jre/lib' >> /etc/profile
    $ echo $'export SCALA_HOME=/flab/scala-2.10.5' >> /etc/profile
    $ echo $'export PATH=$PATH:$SCALA_HOME/bin' >> /etc/profile
    $ echo $'export HADOOP_HOME=/flab/hadoop-2.6.0' >> /etc/profile
    $ echo $'export PATH=$PATH:$HADOOP_HOME/bin' >> /etc/profile
    $ echo $'export SPARK_HOME=/flab/spark-1.5.0-bin-hadoop2.6' >> /etc/profile
    $ echo $'export PATH=$PATH:$SPARK_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ java -version
    $ scala -version

## 配置Hadoop
要修改的配置文件都在`$HADOOP_HOME/etc/hadoop`目录，涉及`hadoop-env.sh, yarn-env.sh, slaves, core-site.xml, hdfs-site.xml, mapred-site.xml, yarn-site.xml`。

    $ hostname
    master
    $ pwd
    /flab/hadoop-2.6.0/etc/hadoop
    $ mkdir -p /flab/abfs/hadoop-tmp /flab/abfs/hadoop-name /flab/abfs/hadoop-data
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
            <value>file:///flab/abfs/hadoop-tmp</value>
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
            <value>file:///flab/abfs/hadoop-name</value>
        </property>
        <property>
            <name>dfs.datanode.data.dir</name>
            <value>file:///flab/abfs/hadoop-data</value>
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

## 配置Spark
进入`$SPARK_HOME/conf`目录，按照模板创建`spark-env.sh`和`log4j.properties`，并编辑`slaves`。

    $ hostname
    master
    $ pwd
    /flab/spark-1.5.0-bin-hadoop2.6/conf
    $ cp spark-env.sh.template spark-env.sh
    $ vim spark-env.sh
    export JAVA_HOME=/flab/jdk1.7.0_80
    export SCALA_HOME=/flab/scala-2.10.5
    export PYSPARK_PYTHON=python3
    export HADOOP_HOME=/flab/hadoop-2.6.0
    export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
    export SPARK_MASTER_IP=master
    export SPARK_MASTER_PORT=8091
    export SPARK_MASTER_WEBUI_PORT=8092
    export SPARK_WORKER_PORT=8093
    export SPARK_WORKER_WEBUI_PORT=8094
    export MASTER=spark://master:8091
    $ cp log4j.properties.template log4j.properties
    $ cp slaves.template slaves
    $ vim slaves
    slave1
    slave2

## ssh无密码登录
假设已有3台软件环境雷同且能互连的机器`master:192.168.190.136; slave1:192.168.190.138; slave2:192.168.190.137;`。

    $ hostname
    master
    $ pwd
    /root
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

## 安装slaves
master的工作算是完成，剩下slaves的工作就简单多了，这里我们需要用到`scp`命令！
>请保证master能登录到slaves。

    $ hostname
    master
    $ scp -qr /flab root@slave1:/flab
    $ scp -qr /etc/profile root@slave1:/etc/profile
    $ scp -qr /etc/hosts root@slave1:/etc/hosts
    $ scp -qr /etc/selinux/config root@slave1:/etc/selinux/config
    $ ssh root@slave1 yum -y update
    $ ssh root@slave1 yum -y install gcc gcc-c++ gcc-gfortran make vim ssh libtool lapack lapack-devel blas blas-devel readline-devel libXt-devel zlib-devel libxml2-devel
    $ ssh root@slave1 systemctl stop firewalld.service
    $ ssh root@slave1 systemctl disable firewalld.service
    $ ssh root@slave1 setenforce 0
    $ ssh root@slave1 source /etc/profile
    $ scp -qr /flab root@slave2:/flab
    $ scp -qr /etc/profile root@slave2:/etc/profile
    $ scp -qr /etc/hosts root@slave2:/etc/hosts
    $ scp -qr /etc/selinux/config root@slave2:/etc/selinux/config
    $ ssh root@slave2 yum -y update
    $ ssh root@slave2 yum -y install gcc gcc-c++ gcc-gfortran make vim ssh libtool lapack lapack-devel blas blas-devel readline-devel libXt-devel zlib-devel libxml2-devel
    $ ssh root@slave2 systemctl stop firewalld.service
    $ ssh root@slave2 systemctl disable firewalld.service
    $ ssh root@slave2 setenforce 0
    $ ssh root@slave2 source /etc/profile

## 测试Hadoop
先简单的hdfs文件操作来测试hdfs运行情况，再执行WordCount实例检查MapReduct作业。

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
    > mapr >> http://192.168.190.136:8088
    > hdfs >> http://192.168.190.136:50070
    $ hdfs dfs -mkdir -p /test
    $ hdfs dfs -put README.txt /test
    $ hdfs dfs -ls -R /test
    $ hdfs dfs -cat /test/README.txt
    $ hdfs dfs -rm -f -R /test
    $ hdfs dfs -mkdir -p /count_in
    $ hdfs dfs -put README.txt /count_in
    $ hadoop jar /flab/hadoop-2.6.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0.jar wordcount /count_in /count_out
    $ hdfs dfs -ls -R /count_out
    $ sbin/stop-all.sh #停止

## 测试Spark

    $ /flab/hadoop-2.6.0/sbin/start-all.sh #启动服务
    $ /flab/spark-1.5.0-bin-hadoop2.6/sbin/start-all.sh #启动服务
    $ spark-shell --master spark://master:8091
    scala> val textFile = sc.textFile("file:///flab/hadoop-2.6.0/README.txt") #本地文件
    scala> val textFile = sc.textFile("/count_in/README.txt") #hdfs文件
    scala> val textFile = sc.textFile("hdfs://master:9000/count_in/README.txt") #hdfs文件
    scala> textFile.count() #统计文件行数
    scala> textFile.first() #输出首行内容
    scala> textFile.filter(line => line.contains("Hadoop")).count() #lines of contain Hadoop
    scala> textFile.map(line => line.split(" ").size).reduce((a, b) => if (a > b) a else b) #max of line words count
    scala> val wordCounts = textFile.flatMap(line => line.split(" ")).map(word => (word, 1)).reduceByKey((a, b) => a + b) #WordCount
    scala> wordCounts.collect()
    scala> :quit
    $ spark-submit --master spark://master:8091 --class org.apache.spark.examples.SparkPi /flab/spark-1.5.0-bin-hadoop2.6/lib/spark-examples-1.5.0-hadoop2.6.0.jar 100
    Pi is roughly 3.14082
    $ spark-submit --master spark://master:8091 /flab/spark-1.5.0-bin-hadoop2.6/examples/src/main/r/dataframe.R
    $ /flab/spark-1.5.0-bin-hadoop2.6/sbin/stop-all.sh #停止服务
    $ /flab/hadoop-2.6.0/sbin/stop-all.sh #停止服务

然后通过`master:8092`访问`SPARK_MASTER_WEBUI`，为了避免端口冲突，刻意修改了默认端口。如果用`sbin/start-all.sh`无法正常启动相关的进程，可以在`$SPARK_HOME/logs`目录下查看相关的错误信息。

## 号外. R on windows
Windows下的R安装没什么好讲的的。下载并安装JDK7，然后设置环境变量。如下：

    JAVA_HOME=D:\program\jdk
    PATH=%JAVA_HOME%\bin;$PATH;
    CLASSPATH=.;%JAVA_HOME%\lib;%JAVA_HOME%\jre\lib;

## 号外. R and Spark
首先启动R程序，安装rJava扩展，添加SparkR包到libPaths，编写测试脚本检查成效。

    > install.packages("rJava") #rJava
    > require(rJava)
    > .jinit() #start jvm
    > s = .jnew("java/lang/String", "Hello World!")
    > print(s) #print string object
    [1] "Java-Object{Hello World!}"
    > .jstrVal(s) #print string value
    [1] "Hello World!"
    > .libPaths(c("./lib", .libPaths())) #SparkR
    > library(SparkR)
    > sc <- sparkR.init(master="spark://master:8091")
    > sqlContext = sparkRSQL.init(sc)
    > loc.df = data.frame(name=c("John", "Smith", "Sarah"), age=c(19, 23, 18))
    > rdd.df = createDataFrame(sqlContext, loc.df)
    > printSchema(rdd.df)
    > sparkR.stop()

## 参考资料：
- [Spark Standalone Mode](http://spark.apache.org/docs/latest/spark-standalone.html)
- [Running Spark on Mesos](http://spark.apache.org/docs/latest/running-on-mesos.html)
- [Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)