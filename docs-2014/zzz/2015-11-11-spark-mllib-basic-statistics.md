title: Spark MLlib之基础统计
date: 2015-11-11
tags: [Spark,MLlib]
---
MLlib支持概要统计，相关系数，分层抽样，假设检验，随机数生成和核密度估计等基础统计实现。

<!--more-->
## Summary statistics
MLlib通过`Statistics.colStats()`对`RDD[Vector]`进行列概要统计。`colStats()`返回`MultivariateStatisticalSummary`实例，包含了列式的最大值、最小值、均值、方差和非零值数量以及总数量。

    import org.apache.spark.mllib.linalg.Vector
    import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
    // an RDD of Vectors
    val dv1 = Vectors.dense(1,2,0,4)
    val dv2 = Vectors.dense(1,8,0,4)
    val observations: RDD[Vector] = sc.parallelize(Array(dv1, dv2))
    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean) // the mean value for each column
    println(summary.variance) // column-wise variance
    println(summary.numNonzeros) // number of nonzeros in each column

## Correlations
目前支持`Pearson’s`和`Spearman’s`相关性。通过`Statistics.corr()`计算系列之间的相关性。根据不同输入类型，两个`RDD[Double]`或一个`RDD[Vector]`，分别将输出一个实数或相关矩阵。

    import org.apache.spark.SparkContext
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.stat.Statistics
    val sc: SparkContext = ...
    val seriesX: RDD[Double] = sc.parallelize(Array(1,2,3))
    val seriesY: RDD[Double] = sc.parallelize(Array(1,3,2))
    // using Pearson's method. Enter "spearman" for Spearman's method.
    // Pearson's method will be used by default.
    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    // note that each Vector is a row and not a column
    val dv1 = Vectors.dense(1,2,0,4)
    val dv2 = Vectors.dense(1,8,0,4)
    val data: RDD[Vector] = sc.parallelize(Array(dv1, dv2))
    // using Pearson's method. Use "spearman" for Spearman's method.
    // Pearson's method will be used by default.
    val correlMatrix: Matrix = Statistics.corr(data, "pearson")

## Stratified sampling

    import org.apache.spark.SparkContext
    import org.apache.spark.SparkContext._
    import org.apache.spark.rdd.RDD
    import org.apache.spark.rdd.PairRDDFunctions
    val sc: SparkContext = ...
    // an RDD[(K, V)] of any key value pairs
    val data: RDD[(Int, Int)] = sc.parallelize(Array((1, 1), (1, 2), (2, 3), (2, 4), (2, 5)))
    // specify the exact fraction desired from each key
    val fractions: Map[Int, Double] = Map(1 -> 0.5, 2 -> 0.5)
    // Get an exact sample from each stratum
    val approxSample = data.sampleByKey(false, fractions)
    val exactSample = data.sampleByKeyExact(false, fractions)

## Random data generation
MLlib提供了均分分布，正态分布和泊松分布的随机数生成器。

    import org.apache.spark.SparkContext
    import org.apache.spark.mllib.random.RandomRDDs._
    val sc: SparkContext = ...
    // standard normal distribution `N(0, 1)`.
    val u = normalRDD(sc, 1000000L, 10)
    // Apply a transform to get a random double RDD following `N(1, 4)`.
    val v = u.map(x => 1.0 + 2.0 * x)

## Kernel density estimation
核密度估计在可视化经验概率分布和观察样本分布是非常有用的。

    import org.apache.spark.mllib.stat.KernelDensity
    import org.apache.spark.rdd.RDD
    import org.apache.spark.mllib.random.RandomRDDs._
    // an RDD of sample data
    val data: RDD[Double] = normalRDD(sc, 1000, 10)
    // for the Gaussian kernels
    val kd = new KernelDensity().setSample(data).setBandwidth(3.0)
    // Find density estimates for the given values
    val densities = kd.estimate(Array(-1.0, 2.0, 5.0))

## 参考资料：
- [MLlib - Basic Statistics](http://spark.apache.org/docs/latest/mllib-statistics.html)