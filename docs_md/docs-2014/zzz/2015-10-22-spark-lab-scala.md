title: Spark开发环境(Scala)
date: 2015-10-22
tags: [Spark,Scala]
---
Scala开发Spark应用首推的IDE非IDEA莫属。为图省心，建议在配置有`Hadoop+Spark`的Linux系统搭建Spark应用开发环境，因为在Windows下会有各种灵异事件。

<!--more-->
## Install idea
    $ ll
    -rwxrw-rw-. 1 root root 183304363 Sep 28 12:25 ideaIC-14.1.5.tar.gz
    $ tar -zxvf ideaIC-14.1.5.tar.gz -C /flab
    $ echo $'export IDEA_HOME=/flab/idea-IC-141.2735.5' >> /etc/profile
    $ echo $'export PATH=$PATH:$IDEA_HOME/bin' >> /etc/profile
    $ source /etc/profile
    $ idea.sh

## Setting idea
    Configure -> Plugins -> Intall JetBrains Plugins => 搜索并安装scala
    Configure -> Project Defaults -> Editor -> File Encodings => 全部设置为UTF-8

## Hello scala with Non-SBT
在IDEA欢迎界面，选择`New Project -> Scala -> Scala`，这里需要设置项目名称、存放路径、java-sdk、scala-version等，点击`Finish`即可打开新建的工程。

新建的工程现在还没有源文件，只有一个存放源文件的目录`src`以及存放工程其他信息的杂项，在src上右键创建包`com.flystarhe`以及类`hello`，当然我们这里需要创建的是入口类，即Object。键入如下代码，然后右键`Run 'hello'`就可以在控制台见到熟悉的内容。

    package com.flystarhe
    object hello {
      def main (args: Array[String]): Unit = {
        println("hello scala.")
      }
    }

## Hello scala with SBT
在IDEA欢迎界面，选择`New Project -> Scala -> SBT`，这里需要设置项目名称、存放路径、java-sdk、sbt-version、scala-version等，点击`Finish`即可打开新建的工程。SBT项目的目录结构类似：

    .
    |-- build.sbt
    |-- project
    |   `-- build.properties
    |-- src
    |   |-- main
    |   |   |-- java
    |   |   |   `-- <main Java sources>
    |   |   |-- resources
    |   |   |   `-- <files to include in main jar here>
    |   |   `-- scala
    |   |       `-- <main Scala sources>
    |   `-- test
    |       |-- java
    |       |   `-- <test Java sources>
    |       |-- resources
    |       |   `-- <files to include in test jar here>
    |       `-- scala
    |           `-- <test Scala sources>
    `-- target

## Import jar
打开`Project Structure`对话框，在`Project Settings`里选`Modules`，切换到`Dependencies`标签界面下，点击右边绿色的`+`，选择`JARs or directories...`，选择相应的jar包，点`OK`完成jar包添加。

## Program arguments
菜单栏依次操作`Run -> Edit Configurations...`打开Debug对话框，点击左上角`+`选择`Application`新建配置。修改`Name`值为项目名，`Main class`填写入口类名，`Program arguments`输入参数列表，选择`Use classpath of mod...`完成配置，然后执行`Run %Name%`即可。

## Create jar
Eclipse下打包jar很方便，IDEA下就没有那么快捷了，需要手动建立Artifact。打开`Project Structure`对话框，在`Project Settings`里选`Artifacts`，点击绿色的`+ -> JAR -> From modules with dependencies...`，弹出`Create JAR from Modules`对话框。选择`Module, Main Class, JAR files from libraries`，点`OK`回到`Project Structure`对话框。然后你可能需要检查`Name, Output directory, Class Path`等内容，确认无误后点击`Apply`应用配置。(当然也可以`+ -> JAR -> Empty`，自己按需定制)

配置完成后，就可以在菜单栏中选择`Build -> Build Artifacts...`，然后使用`Build or ReBuild`等命令打包了。打包完成后会在状态栏中显示如`Compilation completed successfully in 2s 334ms`的信息。命令窗口切换到`JAR`包输出目录，执行`java -jar your-jar-name.jar`。

## Spark word counts
开发Spark应用需要导入`spark-assembly-1.5.0-hadoop2.6.0.jar`包，在src上右键创建包`com.flystarhe`以及类`wc: Object`，键入如下代码：

    package com.flystarhe
    import org.apache.spark.{SparkConf, SparkContext}
    import org.apache.spark.SparkContext._
    object wc {
      def main(args: Array[String]) {
        if (args.length < 1) {
          System.err.println("Usage: <file>")
          System.exit(1)
        }
        val conf = new SparkConf()
        val sc = new SparkContext(conf)
        val lines = sc.textFile(args(0))
        lines.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_+_).collect().foreach(println)
        sc.stop()
      }
    }

完成`[Create jar: Artifacts]`配置，其中`JAR files from libraries`配置项选择`copy to the output directory and link via manifest`，并生成jar包。输出了目录可能会有很多jar包，我们只需上传与`Module`同名的jar包到服务器，如存放在`/root/wc_v1.0.jar`。保证`hdfs://master:9000/test or file:///root/test`有用来测试的文本文件。然后用`spark-submit`命令提交任务。

    $ spark-submit --master spark://master:8091 --name wc --class com.flystarhe.wc /root/wc_v1.0.jar hdfs://master:9000/test
    $ spark-submit --master spark://master:8091 --name wc --class com.flystarhe.wc /root/wc_v1.0.jar file:///root/test

## 参考资料：
- [SBT入门](http://www.scala-sbt.org/release/tutorial/zh-cn/index.html)
- [Spark Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)