title: Spark MLlib之数据类型
date: 2015-11-10
tags: [Spark,MLlib]
---
MLlib支持存储在单个机器的本地向量和矩阵，以及依靠多个RDD存储分布式矩阵。本地向量和矩阵作为公共接口的简单的数据模型。

<!--more-->
## Local vector
MLlib支持两种类型的本地向量：稠密和稀疏，分别对应`DenseVector`和`SparseVector`两种实现。本地向量的基类是`Vector`，推荐使用`Vectors`的工厂方法来创建本地向量。

    import org.apache.spark.mllib.linalg.{Vector, Vectors}
    // Create a dense vector (1.0, 0.0, 3.0).
    val dv1: Vector = Vectors.dense(1.0, 0.0, 3.0)
    // Create a sparse vector (1.0, 0.0, 3.0) by indices and values.
    val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
    // Create a sparse vector (1.0, 0.0, 3.0) by nonzero entries.
    val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))

>Note: Scala imports scala.collection.immutable.Vector by default, so you have to import org.apache.spark.mllib.linalg.Vector explicitly to use MLlib’s Vector.

## Labeled point
在MLlib中，`labeled point`用于监督学习算法，可以使用在回归和分类算法中，它存储为`double`类型，由`LabeledPoint`表示：

    import org.apache.spark.mllib.linalg.{Vector, Vectors}
    import org.apache.spark.mllib.regression.LabeledPoint
    // Create a labeled point with a positive label and a dense vector.
    val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
    // Create a labeled point with a negative label and a sparse vector.
    val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))

MLlib支持读取`LIBSVM`格式存储的数据。它的每一行代表一个标签，使用以下格式：

    label index1:value1 index2:value2 ...

`MLUtils.loadLibSVMFile`读取以`LIBSVM`格式存储的数据。

    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.util.MLUtils
    import org.apache.spark.rdd.RDD
    val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "file:///flab/spark-1.5.0-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")

## Local matrix
MLlib支持稠密矩阵和稀疏矩阵，分别对应`DenseMatrix`和`SparseMatrix`两种实现，采用列为主的顺序存储。本地矩阵的基类是`Matrix`，推荐使用`Matrices`的工厂方法来创建本地矩阵。

    import org.apache.spark.mllib.linalg.{Matrix, Matrices}
    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
    // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
    val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

## Distributed matrix
MLlib提供了三种类型的分布式矩阵，`RowMatrix`、`IndexedRowMatrix`和`CoordinateMatrix`。

### RowMatrix
`RowMatrix`每行作为一个本地向量，它的行索引是没有意义的。因为每一行是由本地向量表示，所以最大列数不能超过整数表示范围。

    import org.apache.spark.mllib.linalg.Vector
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    // an RDD of local vectors
    val rows: RDD[Vector] = ...
    // Create a RowMatrix from an RDD[Vector].
    val mat: RowMatrix = new RowMatrix(rows)
    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()
    // QR decomposition 
    val qrResult = mat.tallSkinnyQR(true)
    // Multivariate summary statistics
    import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
    val smm: MultivariateStatisticalSummary = mat.computeColumnSummaryStatistics()
    println(summy.mean)

### IndexedRowMatrix
`IndexedRowMatrix`类似于`RowMatrix`，区别在于它的行索引是有意义的。`IndexedRowMatrix`由`RDD[IndexedRow]`实例化，`IndexedRow`结构如`(Long, Vector)`。`IndexedRowMatrix`可以通过删除行索引的方式转化为`RowMatrix`。

    import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
    // an RDD of indexed rows
    val rows: RDD[IndexedRow] = ...
    // Create an IndexedRowMatrix from an RDD[IndexedRow].
    val mat: IndexedRowMatrix = new IndexedRowMatrix(rows)
    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()
    // Drop its row indices.
    val rowMat: RowMatrix = mat.toRowMatrix()

### CoordinateMatrix
它的每个条目是结构为`(i: Long, j: Long, value: Double)`的元组，适合于维度巨大的稀疏矩阵。`CoordinateMatrix`由`RDD[MatrixEntry]`实例化，`RDD[MatrixEntry]`结构如`(Long, Long, Double)`。

    import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, IndexedRowMatrix }
    // an RDD of matrix entries
    val entries: RDD[MatrixEntry] = ...
    // Create a CoordinateMatrix from an RDD[MatrixEntry].
    val mat: CoordinateMatrix = new CoordinateMatrix(entries)
    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()
    // Convert it to an IndexRowMatrix whose rows are sparse vectors.
    val indexedRowMatrix: IndexedRowMatrix  = mat.toIndexedRowMatrix()

### BlockMatrix
它的每个条目结构为`((Int, Int), Matrix)`，其中`(Int, Int)`是块的索引。

    import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
    // an RDD of (i, j, v) matrix entries
    val entries: RDD[MatrixEntry] = ...
    // Create a CoordinateMatrix from an RDD[MatrixEntry].
    val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)
    // Transform the CoordinateMatrix to a BlockMatrix
    val matA: BlockMatrix = coordMat.toBlockMatrix().cache()
    // Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
    // Nothing happens if it is valid.
    matA.validate()
    // Calculate A^T A.
    val ata = matA.transpose.multiply(matA)

## 参考资料：
- [MLlib - Data Types](http://spark.apache.org/docs/latest/mllib-data-types.html)