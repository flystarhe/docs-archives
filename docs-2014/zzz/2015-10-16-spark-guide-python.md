title: Spark编程指南(Python)
date: 2015-10-16
tags: [Spark,Python]
---
本文包含 Spark 基本操作、向 Spark 传递函数、读取文本文件、保存和读取序列文件、RDD Transformations 与 Actions 函数介绍。

<!--more-->
## 基本操作
为了演示RDD的基本操作，请看以下的简单程序：

    lines = sc.textFile("data.txt")
    lineLengths = lines.map(lambda s: len(s))
    totalLength = lineLengths.reduce(lambda a, b: a + b)
    print totalLength

如果我们希望以后重复使用lineLengths，只需在reduce前加入下面这行代码：

    lineLengths.persist()

这条代码将使得lineLengths在第一次计算生成之后保存在内存中。

## 向Spark传递函数
Spark的API严重依赖于向驱动程序传递函数作为参数。有三种推荐的方法来传递函数作为参数。

- Lambda表达式，简单的函数可以直接写成一个lambda表达式。
- 代码很长的函数，在Spark的函数调用中本地用def定义。
- 模块中的顶级函数。

比如，传递一个无法转化为lambda表达式长函数，可以像以下代码这样：

    """MyScript.py"""
    if __name__ == "__main__":
        def myFunc(s):
            words = s.split(" ")
            return len(words)
        sc = SparkContext(...)
        sc.textFile("file.txt").map(myFunc)

## 读取文本文件
通过文本文件创建RDD要使用SparkContext的textFile方法。这个方法会使用一个文件的URI(本地文件路径、hdfs://、s3n://等)，然后读入这个文件建立一个文本行的集合。

    sc.textFile("data.txt").map(lambda s: len(s)).reduce(lambda a, b: a + b)

包括textFile在内的所有基于文件的Spark读入方法，都支持将文件夹、压缩文件、包含通配符的路径作为参数。比如，以下代码都是合法的：

    textFile("/my/directory")
    textFile("/my/directory/*.gz")
    textFile("/my/directory/*.txt")

`SparkContext.wholeTextFiles`能够读入包含多个小文本文件的目录，然后为每一个文件返回一个`(文件名, 内容)`对。这是与textFile方法为每一个文本行返回一条记录相对应的。

## 保存和读取序列文件
和文本文件类似，序列文件可以通过指定路径来保存与读取。键值类型都可以自行指定，但是对于标准可写类型可以不指定。

    rdd = sc.parallelize(range(1, 4)).map(lambda x: (x, "a" * x ))
    rdd.saveAsSequenceFile(“path/to/file”)
    sorted(sc.sequenceFile(“path/to/file”).collect())
    #> [(1, u’a’), (2, u’aa’), (3, u’aaa’)]

## RDD.Transformations
1. map(func) | 返回一个新的分布数据集，由原数据集元素经func处理后的结果组成；
2. filter(func) | 返回一个新的数据集，由传给func返回True的原数据集元素组成；
3. flatMap(func) | 类似map，但是每个传入元素可能有0或多个返回值，func可以返回一个序列而不是一个值；
4. mapParitions(func) | 类似map，但是RDD的每个分片都会分开独立运行，所以func的参数和返回值必须都是迭代器；
5. mapParitionsWithIndex(func) | 类似mapParitions，但是func有两个参数，第一个是分片的序号，第二个是迭代器。返回值还是迭代器；
6. sample(withReplacement, fraction, seed) | 使用提供的随机数种子取样；
7. union(otherDataset) | 返回新的数据集，包括原数据集和参数数据集的所有元素；
8. intersection(otherDataset) | 返回新的数据集，是两个集的交集；
9. distinct([numTasks]) | 返回新的数据集，包括原集中的不重复元素；
10. groupByKey([numTasks]) | 用于键值对RDD时返回`(K, V迭代器)`对的数据集；
11. aggregateByKey(zeroValue)(seqOp, combOp, [numTasks]) | 用于键值对RDD时返回`(K, U)`对集，对每一个Key的value进行聚集计算；
12. sortByKey([ascending], [numTasks]) | 用于键值对RDD时会返回RDD按键的顺序排序，升降序由第一个参数决定;
13. join(otherDataset, [numTasks]) | 用于键值对`(K, V)`和`(K, W)`RDD时返回`(K, (V, W))`对RDD；
14. cogroup(otherDataset, [numTasks]) | 用于两个键值对`(K, V)`和`(K, W)`RDD时返回`(K, (V迭代器， W迭代器))`对RDD；
15. cartesian(otherDataset) | 用于T和U类型RDD时返回`(T, U)`对类型键值对RDD；
16. pipe(command, [envVars]) | 通过shell命令管道处理每个RDD分片；
17. coalesce(numPartitions) | 把RDD的分片数量降低到参数大小；
18. repartition(numPartitions) | 重新打乱RDD中元素顺序并重新分片，数量由参数决定；
19. repartitionAndSortWithinPartitions(partitioner) | 按照参数给定的分片器重新分片，同时每个分片内部按照键排序。

## RDD.Actions
1. reduce(func) | 使用func进行聚集计算，func的参数是两个，返回值一个，两次func运行应当是完全解耦的，这样才能正确地并行运算；
2. collect() | 向驱动程序返回数据集的元素组成的数组；
3. count() | 返回数据集元素的数量；
4. first() | 返回数据集的第一个元素；
5. take(n) | 返回前n个元素组成的数组；
6. takeSample(withReplacement, num, [seed]) | 返回一个由原数据集中任意num个元素的数组；
7. takeOrder(n, [ordering]) | 返回排序后的前n个元素；
8. saveAsTextFile(path) | 将数据集的元素写成文本文件；
9. saveAsSequenceFile(path) | 将数据集的元素写成序列文件，这个API只能用于Java和Scala程序；
10. saveAsObjectFile(path) | 将数据集的元素使用Java的序列化特性写到文件中，这个API只能用于Java和Scala程序；
11. countByCount() | 只能用于键值对RDD，返回一个(K, int)；
12. hashmap，返回每个key的出现次数；
13. foreach(func) | 对数据集的每个元素执行func，通常用于完成一些带有副作用的函数，比如更新累加器或与外部存储交互等。

## 编写Spark应用
编写Spark应用与通过交互式控制台使用Spark类似，API是相同的。区别在于你需要访问的`SparkContext`它已经由`pyspark`自动加载好了。基本模板如下：

    ## my.py - execute with spark-submit
    ## Imports
    import sys
    from pyspark import SparkConf, SparkContext
    ## Constants
    DATE_FMT = "%y-%m-%d %a %H:%M"
    ## Functions
    def func():
        pass
    ## Main
    def main(sc):
        # Configure Spark
        conf = SparkConf().set("spark.ui.showConsoleProgress", "false")
        sc = SparkContext(master=sys.argv[1], appName=sys.argv[2], conf=conf)
        # Task codes
        # Quit Spark
        sc.stop()
    if __name__ == "__main__":
        main() if len(sys.argv) > 2 else print("args: master appName ..")

这个模板列出了一个Spark应用所需的东西：导入Python库，模块常量，用于调试和Spark UI的可识别的应用名称，还有作为驱动程序运行的一些主要分析方法。

## 参考资料：
- [Spark编程指南(Python版)](http://dataunion.org/14358.html)
- [Spark入门(Python版)](http://blog.jobbole.com/86232/)
- [Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html)
- [org.apache.spark.rdd.PartitionPruningRDD](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.PartitionPruningRDD)
- [org.apache.spark.rdd.RDD](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.RDD)