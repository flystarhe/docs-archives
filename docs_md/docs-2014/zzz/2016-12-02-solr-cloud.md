title: Solr cloud and hdpsearch
date: 2016-12-02
tags: [Solr,Search,Hadoop,ZooKeeper]
---
SolrCloud是灵活的分布式搜索和索引，没有一个主节点分配节点，碎片和副本。相反，Solr的使用ZooKeeper来管理这些位置，这取决于配置文件和模式。查询和更新可以被发送到任何服务器。Solr将使用ZooKeeper数据库中的信息来找出哪些服务器需要处理请求。

<!--more-->
## zookeeper

```shell
[root@c01 solr]# ls zookeeper*
zookeeper-3.4.9.tar.gz
[root@c01 solr]# mkdir service1
[root@c01 solr]# tar -zxf zookeeper-3.4.9.tar.gz -C service1
[root@c01 solr]# cd service1/
[root@c01 service1]# mkdir data datalog logs
[root@c01 service1]# cd ..
[root@c01 solr]# ls service1/
data  datalog  logs  zookeeper-3.4.9
[root@c01 solr]# cp -r service1 service2
[root@c01 solr]# cp -r service1 service3
[root@c01 solr]# ls
service1  service2  service3  zookeeper-3.4.9.tar.gz
[root@c01 solr]# vim service1/data/myid #内容为1
1
[root@c01 solr]# vim service2/data/myid #内容为2
2
[root@c01 solr]# vim service3/data/myid #内容为3
3
[root@c01 solr]# vim service1/zookeeper-3.4.9/conf/zoo.cfg
tickTime=2000
initLimit=5
syncLimit=2
dataDir=/lab/solr/service1/data
dataLogDir=/lab/solr/service1/datalog
clientPort=2181
server.1=192.168.190.147:2888:3888
server.2=192.168.190.147:2889:3889
server.3=192.168.190.147:2890:3890
[root@c01 solr]# vim service2/zookeeper-3.4.9/conf/zoo.cfg
tickTime=2000
initLimit=5
syncLimit=2
dataDir=/lab/solr/service2/data
dataLogDir=/lab/solr/service2/datalog
clientPort=2182
server.1=192.168.190.147:2888:3888
server.2=192.168.190.147:2889:3889
server.3=192.168.190.147:2890:3890
[root@c01 solr]# vim service3/zookeeper-3.4.9/conf/zoo.cfg
tickTime=2000
initLimit=5
syncLimit=2
dataDir=/lab/solr/service3/data
dataLogDir=/lab/solr/service3/datalog
clientPort=2183
server.1=192.168.190.147:2888:3888
server.2=192.168.190.147:2889:3889
server.3=192.168.190.147:2890:3890
[root@c01 solr]# cd /lab/solr/service1/zookeeper-3.4.9/bin/
[root@c01 bin]# sh zkServer.sh start #启动
ZooKeeper JMX enabled by default
Using config: /lab/solr/service1/zookeeper-3.4.9/bin/../conf/zoo.cfg
Starting zookeeper ... STARTED
[root@c01 bin]# sh zkServer.sh status #查看状态/因为是集群，其他两台没起来，当然报错
ZooKeeper JMX enabled by default
Using config: /lab/solr/service1/zookeeper-3.4.9/bin/../conf/zoo.cfg
Error contacting service. It is probably not running.
[root@c01 bin]# cd /lab/solr/service2/zookeeper-3.4.9/bin/
[root@c01 bin]# sh zkServer.sh start #启动
ZooKeeper JMX enabled by default
Using config: /lab/solr/service2/zookeeper-3.4.9/bin/../conf/zoo.cfg
Starting zookeeper ... STARTED
[root@c01 bin]# sh zkServer.sh status #查看状态/因为是集群，其他两台没起来，当然报错
ZooKeeper JMX enabled by default
Using config: /lab/solr/service2/zookeeper-3.4.9/bin/../conf/zoo.cfg
Mode: leader
[root@c01 bin]# cd /lab/solr/service3/zookeeper-3.4.9/bin/
[root@c01 bin]# sh zkServer.sh start #启动
ZooKeeper JMX enabled by default
Using config: /lab/solr/service3/zookeeper-3.4.9/bin/../conf/zoo.cfg
Starting zookeeper ... STARTED
[root@c01 bin]# sh zkServer.sh status #查看状态/因为是集群，其他两台没起来，当然报错
ZooKeeper JMX enabled by default
Using config: /lab/solr/service3/zookeeper-3.4.9/bin/../conf/zoo.cfg
Mode: follower
```

## hadoop

```shell
[root@c01 solr]# tar -zxf hadoop-2.7.3.tar.gz
[root@c01 solr]# tar -zxf hadoop-2.7.3.tar.gz
[root@c01 solr]# vim /etc/hosts
192.168.190.147   ha0
[root@c01 solr]# vim hadoop-2.7.3/etc/hadoop/hadoop-env.sh
# The java implementation to use.
export JAVA_HOME=/lab/jdk1.8.0_111
[root@c01 solr]# vim hadoop-2.7.3/etc/hadoop/core-site.xml
<configuration>
<property>
  <name>fs.defaultFS</name>
  <value>hdfs://192.168.190.147:9000</value>
</property>
<property>
  <name>hadoop.tmp.dir</name>
  <value>/lab/solr/hadoop-tmp</value>
</property>
</configuration>
[root@c01 solr]# vim hadoop-2.7.3/etc/hadoop/hdfs-site.xml
<configuration>
<property>
  <name>dfs.replication</name>
  <value>1</value>
</property>
</configuration>
[root@c01 solr]# ssh 192.168.190.147 #验证ssh无密码登录
Last login: Thu Dec  1 18:18:33 2016 from ha0
[root@c01 ~]# exit
logout
Connection to 192.168.190.147 closed.
[root@c01 solr]# cd hadoop-2.7.3
[root@c01 hadoop-2.7.3]# ./bin/hdfs namenode -format #格式化文件系统
[root@c01 hadoop-2.7.3]# ./sbin/start-dfs.sh #启动hdfs
[root@c01 hadoop-2.7.3]# ./bin/hadoop fs -mkdir -p /lab
[root@c01 hadoop-2.7.3]# ./bin/hadoop fs -put /etc/hosts /lab/hosts.txt
[root@c01 hadoop-2.7.3]# ./bin/hadoop fs -ls -R / #http://192.168.190.147:50070/explorer.html
[root@c01 hadoop-2.7.3]# ./sbin/stop-dfs.sh #停止hdfs
```

我们已经安装了HDFS现在来配置启动YARN，然后运行一个WordCount程序：

```shell
[root@c01 hadoop-2.7.3]# cp etc/hadoop/mapred-site.xml.template etc/hadoop/mapred-site.xml
[root@c01 hadoop-2.7.3]# vim etc/hadoop/mapred-site.xml
<configuration>
<property>
  <name>mapreduce.framework.name</name>
  <value>yarn</value>
</property>
</configuration>
[root@c01 hadoop-2.7.3]# vim etc/hadoop/yarn-site.xml
<configuration>
<property>
  <name>yarn.nodemanager.aux-services</name>
  <value>mapreduce_shuffle</value>
</property>
</configuration>
[root@c01 hadoop-2.7.3]# ./sbin/start-yarn.sh #启动YARN
[root@c01 hadoop-2.7.3]# ./sbin/stop-yarn.sh #停止YARN
[root@c01 hadoop-2.7.3]# ./sbin/start-all.sh #启动全部
[root@c01 hadoop-2.7.3]# ./sbin/stop-all.sh #停止全部
```

用浏览器访问`http://192.168.190.147:8088/cluster`。现在我们的hdfs和yarn都运行成功了，我们可以运行一个WordCount的MR程序来测试我们的单机模式集群是否可以正常工作。

## solr-cloud
启动Solr的命令相当简单，如：
```shell
[root@c01 solr]# unzip solr-5.5.1.zip
[root@c01 solr]# cd solr-5.5.1
[root@c01 solr-5.5.1]# ./bin/solr start -help
[root@c01 solr-5.5.1]# ./bin/solr start -c -p 8983 -m 1g -z 192.168.190.147:2181,192.168.190.147:2182,192.168.190.147:2183 -Dsolr.directoryFactory=HdfsDirectoryFactory -Dsolr.lock.type=hdfs -Dsolr.hdfs.home=hdfs://192.168.190.147:9000/solr8983
[root@c01 solr-5.5.1]# ./bin/solr start -c -p 7574 -m 1g -z 192.168.190.147:2181,192.168.190.147:2182,192.168.190.147:2183 -Dsolr.directoryFactory=HdfsDirectoryFactory -Dsolr.lock.type=hdfs -Dsolr.hdfs.home=hdfs://192.168.190.147:9000/solr7574
[root@c01 solr-5.5.1]# ./bin/solr start -c -p 9527 -m 1g -z 192.168.190.147:2181,192.168.190.147:2182,192.168.190.147:2183 -Dsolr.directoryFactory=HdfsDirectoryFactory -Dsolr.lock.type=hdfs -Dsolr.hdfs.home=hdfs://192.168.190.147:9000/solr9527
```

1. `-c`参数告诉Solr的在SolrCloud模式启动
2. `-z`参数为ZooKeeper全体的连接字符串
3. `-Dsolr.directoryFactory`参数告诉Solr的所有索引应存放在HDFS
4. `-Dsolr.lock.type`参数指示索引将存储在HDFS
5. `-Dsolr.hdfs.home`指示Solr索引在HDFS的路径

>如果没有指定的ZooKeeper的连接字符串`-z`属性，Solr的将推出其嵌入式的ZooKeeper实例。这个实例有一个单一的ZooKeeper实例，因此没有提供故障转移，意味着并不用于生产。

注意，我们还没有定义`collection`和`configset`，有多少`shard`或`node`等这些属性在`collection`级别定义，当我们创建`collection`时会定义的。

也可以使用`-e`属性使用示例`collection`来启动Solr，如`./bin/solr start -e cloud`。这将启动一个交互式会话，并允许您定义`collection`，使用`configset`，以及`shard`和`replica`的数量。

启动成功后，在浏览器输入`http://192.168.190.147:8983/solr`访问管理控制台。

现在使用`./bin/solr`脚本来创建`collection`：
```shell
[root@c01 solr-5.5.1]# ./bin/solr create -help
[root@c01 solr-5.5.1]# ./bin/solr create -c SolrCollection -d data_driven_schema_configs -n mySolrConfigs -shards 2 -replicationFactor 2 -p 8983
```

1. `-c`参数提供创建`collection`的名称
2. `-d`参数提供要使用的`configset`的名称
3. `-n`参数提供在`ZooKeeper`的`configset`的名称
4. `-shards`参数提供`collection`分割`shard`的数量
5. `-replicationFactor`参数提供每个`shard`的副本数量
6. `-p`参数提供目标Solr实例的端口

我们再做个简单尝试，如：

    [root@c01 solr-5.5.1]# ./bin/solr create -c test01 -d test -n mySolrConfigset -shards 2 -replicationFactor 2 -p 8983

得到一个错误，没有找到叫test的`configset`目录。我们需要在`$SOLR_HOME\server\solr\configsets`目录下建立我们test的`configset`。简单起见，直接拷贝`basic_configs`：

    [root@c01 solr-5.5.1]# cp -r server/solr/configsets/basic_configs server/solr/configsets/test
    [root@c01 solr-5.5.1]# ./bin/solr create -c test01 -d test -n mySolrConfigset -shards 2 -replicationFactor 2 -p 8983

刷新管理控制台页面会看到一个`core selector`选择框，到此solrCloud部署并初始化成功，之后可以向这个collection导入数据进行查询。

    [root@c01 solr-5.5.1]# ./bin/solr stop -all

## hdpsearch

```shell
[root@c01 ~]# rpm --import http://public-repo-1.hortonworks.com/HDP-SOLR-2.3-100/repos/centos6/RPM-GPG-KEY/RPM-GPG-KEY-Jenkins
[root@c01 ~]# cd /etc/yum.repos.d/
[root@c01 ~]# wget http://public-repo-1.hortonworks.com/HDP-SOLR-2.3-100/repos/centos6/hdp-solr.repo
[root@c01 ~]# yum install lucidworks-hdpsearch
[root@c01 ~]# ls /opt/lucidworks-hdpsearch/
docs  hbase-indexer  hive  job  LICENSE.txt  pig  solr  spark  storm
```

使用hdpsearch，你应该是SolrCloud模式，在启动Solr时设置的。`Hadoop job jar`的工作分三个阶段，旨在采取原始内容和输出结果到Solr：

1. 从原始内容创建SequenceFiles
2. MapReduce工作，从原始内容中提取文本和元数据
3. MapReduce工作，使用SolrJ客户端发送提取的内容到Solr

索引CSV文件的例子：

```shell
[root@c01 hadoop-2.7.3]# ./bin/hadoop jar /opt/lucidworks-hdpsearch/job/solr-hadoop-job-2.2.2.jar
   com.lucidworks.hadoop.ingest.IngestJob
   -Dlww.commit.on.close=true -DcsvDelimiter=|
   -cls com.lucidworks.hadoop.ingest.CSVIngestMapper -c gettingstarted -i /data/CSV -of com.lucidworks.hadoop.io.LWMapRedOutputFormat -s http://localhost:8888/solr
```

## 参考资料：
- [Solr Quick Start](http://lucene.apache.org/solr/quickstart.html)
- [SolrCloud-5.2.1 集群部署](http://www.linuxidc.com/Linux/2016-02/128498.htm)
- [SolrCloud Reference Guide](https://cwiki.apache.org/confluence/display/solr/SolrCloud)
- [HDP Search Guide Install](https://doc.lucidworks.com/lucidworks-hdpsearch/2.5/Guide-Install-Manual.html)
- [HDP Search Connector User Guide](https://doc.lucidworks.com/lucidworks-hdpsearch/2.5/Guide-Jobs.html)