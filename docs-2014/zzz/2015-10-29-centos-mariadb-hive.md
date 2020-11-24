title: CentOS | Mariadb Hive
date: 2015-10-29
tags: [CentOS,搭建环境]
---
本文首先介绍CentOS下Mariadb安装配置及其在R中如何连接应用。然后介绍Hive的安装配置，并演示了`Hive CLI`、`Beeline`、`R-JDBC`及`SparkSQL`这4各种应用场景。

<!--more-->
## install mariadb
Mariadb是非常好的Mysql替代品，个人也比较青睐。安装Mysql请参考官方的[快速指南](http://dev.mysql.com/doc/mysql-yum-repo-quick-guide/en/)。

    $ cd /etc/yum.repos.d/
    $ vim /etc/yum.repos.d/MariaDB.repo
    # MariaDB 10.1 CentOS repository list - created 2016-11-16 04:20 UTC
    # http://downloads.mariadb.org/mariadb/repositories/
    [mariadb]
    name = MariaDB
    baseurl = http://yum.mariadb.org/10.1/centos6-amd64
    gpgkey=https://yum.mariadb.org/RPM-GPG-KEY-MariaDB
    gpgcheck=1
    $ yum -y install MariaDB-server MariaDB-client
    $ service mysql start
    $ service mysql status
    $ mysql -h master -u root -p
    MariaDB [(none)]> show databases;
    +--------------------+
    | Database           |
    +--------------------+
    | information_schema |
    | mysql              |
    | performance_schema |
    | test               |
    +--------------------+
    4 rows in set (0.00 sec)
    MariaDB [(none)]> use mysql
    MariaDB [mysql]> delete from user where user="";
    MariaDB [mysql]> update user set password=PASSWORD("root") where user="root";
    MariaDB [mysql]> grant all privileges on *.* to root@'%';
    MariaDB [mysql]> flush privileges;
    MariaDB [mysql]> quit
    Bye

## r connect mariadb(JDBC)
    require(RJDBC)
    drv <- JDBC("org.mariadb.jdbc.Driver", "E:/_scala/_lib/mariadb-java-client-1.2.3.jar", "`")
    con <- dbConnect(drv, "jdbc:mariadb://master:3306/flab?characterEncoding=UTF-8", "root", "root")
    dbListTables(con)
    if(dbExistsTable(con, "iris")) dbWriteTable(con, "iris", iris, overwrite=F, append=T) else dbWriteTable(con, "iris", iris)
    dbGetQuery(con, "select count(*) from iris")
    dat <- dbReadTable(con, "iris")
    dbDisconnect(con)

## install hive(lazy)
    $ tar -zxf apache-hive-1.2.1-bin.tar.gz -C /flab
    $ echo $'export HIVE_HOME=/flab/apache-hive-1.2.1-bin' >> /etc/profile
    $ echo $'export PATH=$PATH:$HIVE_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ cp $HIVE_HOME/lib/jline-2.12.jar $HADOOP_HOME/share/hadoop/yarn/lib/
    $ rm $HADOOP_HOME/share/hadoop/yarn/lib/jline-0.9.94.jar
    $ $HADOOP_HOME/sbin/start-all.sh
    $ hdfs dfs -mkdir -p  /tmp
    $ hdfs dfs -mkdir -p  /user/hive/warehouse
    $ hdfs dfs -chmod g+w /tmp
    $ hdfs dfs -chmod g+w /user/hive/warehouse

## using hive CLI
    $ $HIVE_HOME/bin/hive
    hive> use default;
    hive> drop table if exists test;
    hive> create table test(key int, val string) row format delimited fields terminated by ',' stored as textfile;
    hive> show tables;
    hive> describe test;
    hive> load data local inpath '/home/root/data.txt' into table test;
    hive> select * from test;
    hive> quit;
    $ cat /home/root/data.txt
    1,v1
    2,v2
    3,v3

## start HiveServer2
    $ hiveserver2
    # running in a new terminal

## beeline connect hive
    $ beeline
    beeline> !connect jdbc:hive2://master:10000/default
    Connecting to jdbc:hive2://master:10000/default
    Enter username for jdbc:hive2://master:10000/default: root
    Enter password for jdbc:hive2://master:10000/default: ****
    Connected to: Apache Hive (version 1.2.1)
    Driver: Spark Project Core (version 1.5.0)
    Transaction isolation: TRANSACTION_REPEATABLE_READ
    0: jdbc:hive2://master:10000/default> select * from test;
    +-----------+-------------+--+
    | test.key  | test.val    |
    +-----------+-------------+--+
    | 1         | v1          |
    +-----------+-------------+--+
    6 rows selected (5.042 seconds)
    0: jdbc:hive2://master:10000/default> !quit
    Closing: 0: jdbc:hive2://master:10000/default

## r connect hive(JDBC)
    require(RJDBC)
    drv <- JDBC("org.apache.hive.jdbc.HiveDriver", list.files("E:/jars-hive", pattern="jar$", full.names=T))
    con <- dbConnect(drv, "jdbc:hive2://master:10000/default", "root", "root")
    dbListTables(con)
    dbDisconnect(con)
    # from hive/build/dist/lib
    # need jars | hive-jdbc-1.2.1-standalone.jar
    # need jars | hive-jdbc-1.2.1.jar
    # need jars | hive-service-1.2.1.jar
    # need jars | libfb303-0.9.2.jar
    # need jars | libthrift-0.9.2.jar
    # and from hadoop
    # need jars | hadoop-common-2.6.0.jar

## spark connect hive(spark-shell or sparkR)
    # 为了让Spark能够连接到Hive的原有数据仓库
    # 我们需要将Hive中的hive-site.xml文件拷贝到Spark的conf目录下
    # 这样就可以通过这个配置文件找到Hive的元数据以及数据存放
    # 如果Hive的元数据存放在Mysql中需要准备好Mysql相关驱动
    # 如果Hive的元数据存放在Derby中需要准备好Derby相关驱动
    $ cp mysql-connector-java-5.1.37-bin.jar $SPARK_HOME/lib/
    $ cp mariadb-java-client-1.2.3.jar $SPARK_HOME/lib/
    $ cp $HIVE_HOME/lib/derby-10.10.2.0.jar $SPARK_HOME/lib/
    $ cp $HIVE_HOME/conf/hive-site.xml $SPARK_HOME/conf/
    $ spark-shell --jars $SPARK_HOME/lib/mariadb-java-client-1.2.3.jar
    scala> sqlContext.sql("show databases").collect().foreach(println)
    [default]
    scala> sqlContext.sql("use default")
    scala> sqlContext.sql("show tables").collect().foreach(println)
    [test,false]
    scala> sqlContext.sql("select * from test").collect().foreach(println)
    [1,v1]
    [2,v2]
    [3,v3]
    scala> sqlContext.sql("create table spark_lab(k int, v string)")
    scala> sqlContext.sql("show tables").collect().foreach(println)
    [spark_lab,false]
    [test,false]
    scala> :quit
    $ sparkR --jars $SPARK_HOME/lib/mariadb-java-client-1.2.3.jar
    > res <- sql(sqlContext, "show tables")
    > head(res)
    > res <- sql(sqlContext, "select * from test")
    > head(res)
    > q()

## compiling hadoop
    $ yum -y install cmake openssl–devel libtool automake autoconf zlib-devel lzo-devel
    $ tar -zxf apache-maven-3.3.3-bin.tar.gz -C /flab
    $ echo $'export MAVEN_HOME=/flab/apache-maven-3.3.3' >> /etc/profile
    $ echo $'export PATH=$PATH:$MAVEN_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ mvn -version
    $ tar -zxf protobuf-2.5.0.tar.gz
    $ cd protobuf-2.5.0
    $ ./configure --prefix=/flab/protobuf-2.5.0
    $ make && make install
    $ echo $'export PROTOBUF_HOME=/flab/protobuf-2.5.0' >> /etc/profile
    $ echo $'export PATH=$PATH:$PROTOBUF_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ protoc --version
    $ tar -zxf hadoop-2.6.0-src.tar.gz
    $ cd hadoop-2.6.0-src
    $ vim BUILDING.txt
    $ mvn clean package -Pdist,native -DskipTests -Dtar

## install hive(tail)
    $ tar -zxf apache-hive-1.2.1-bin.tar.gz -C /flab
    $ echo $'export HIVE_HOME=/flab/apache-hive-1.2.1-bin' >> /etc/profile
    $ echo $'export PATH=$PATH:$HIVE_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ cd /flab/abfs
    $ mkdir hive-iotmp hive-resources hive-log-query hive-log-operation
    $ cd $HIVE_HOME/conf
    $ cp hive-env.sh.template hive-env.sh
    $ cp hive-default.xml.template hive-site.xml
    $ vim hive-site.xml
    <property>
      <name>hive.exec.local.scratchdir</name>
      <value>/flab/abfs/hive-iotmp</value>
      <description>Local scratch space for Hive jobs</description>
    </property>
    <property>
      <name>hive.downloaded.resources.dir</name>
      <value>/flab/abfs/hive-resources</value>
    <description>Temporary local directory for added resources in the remote file system.</description>
    </property>
    <property>
      <name>hive.querylog.location</name>
      <value>/flab/abfs/hive-log-query</value>
      <description>Location of Hive run time structured log file</description>
    </property>
    <property>
      <name>hive.server2.logging.operation.log.location</name>
      <value>/flab/abfs/hive-log-operation</value>
    <description>Top level directory where operation logs are stored if logging functionality is enabled</description>
    </property>
    <property>
      <name>javax.jdo.option.ConnectionURL</name>
      <value>jdbc:mariadb://master:3306/hive?createDatabaseIfNotExist=true</value>
      <description>JDBC connect string for a JDBC metastore</description>
    </property>
    <property>
      <name>javax.jdo.option.ConnectionDriverName</name>
      <value>org.mariadb.jdbc.Driver</value>
      <description>Driver class name for a JDBC metastore</description>
    </property>
    <property>
      <name>javax.jdo.option.ConnectionUserName</name>
      <value>root</value>
        <description>Username to use against org.apache.spark.sql.AnalysisException</description>
    </property>
    <property>
      <name>javax.jdo.option.ConnectionPassword</name>
      <value>root</value>
      <description>password to use against metastore database</description>
    </property>
    $ cp $HIVE_HOME/lib/jline-2.12.jar $HADOOP_HOME/share/hadoop/yarn/lib/
    $ rm $HADOOP_HOME/share/hadoop/yarn/lib/jline-0.9.94.jar
    $ cp /home/www/mariadb-java-client-1.2.3.jar $HIVE_HOME/lib/
    $ $HADOOP_HOME/sbin/start-all.sh
    $ hdfs dfs -mkdir -p  /tmp
    $ hdfs dfs -mkdir -p  /user/hive/warehouse
    $ hdfs dfs -chmod g+w /tmp
    $ hdfs dfs -chmod g+w /user/hive/warehouse

## 参考资料：
- [GettingStarted - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/GettingStarted)
- [LanguageManual - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Cli)
- [Hive Tutorial - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/Tutorial)
- [Setting Up HiveServer2](https://cwiki.apache.org/confluence/display/Hive/Setting+Up+HiveServer2)
- [HiveServer2 Clients](https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients)
- [AdminManual Configuration](https://cwiki.apache.org/confluence/display/Hive/AdminManual+Configuration)
- [Hive on Spark: Getting Started](https://cwiki.apache.org/confluence/display/Hive/Hive+on+Spark:+Getting+Started)