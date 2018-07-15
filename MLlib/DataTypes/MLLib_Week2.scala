// Databricks notebook source
// File contains sample declarations having valid as well as invalid scenarios

import org.apache.spark.mllib.linalg.{Vector, Vectors}

// COMMAND ----------

/* 
A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine. MLlib supports two types of local vectors: dense and sparse. A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0) can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector. 
*/

// COMMAND ----------

val dv: Vector = Vectors.dense(1, 0, 3)

// COMMAND ----------

val sv3: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))

// COMMAND ----------

val sv4: Vector = Vectors.sparse(3, Array(7, 2, 3), Array(1.0, 3.0,5))


// COMMAND ----------

val sv5: Vector = Vectors.sparse(3, Array(0, 2,5,6), Array(1.0, 3.0,5,5))


// COMMAND ----------

val sv1: Vector = Vectors.sparse(6, Array(0, 7, 8,9,9), Array(1.0, 3.0,4.0, 6, 7))

// COMMAND ----------

val sv2: Vector = Vectors.sparse(5, Seq((0, 4.0), (4, 3.0)))

// COMMAND ----------

val sv2: Vector = Vectors.sparse(5, Seq((4, 4.0), (1, 3.0)))

// COMMAND ----------

val sv6: Vector = Vectors.sparse(3, Seq((2,0.2), (2, 3.0)))

// COMMAND ----------

val sv7: Vector = Vectors.sparse(3, Seq( (1,0.2), (2, 3.0), (0,4.0)))

// COMMAND ----------

Results will be reordered in increasing order of indicies

// COMMAND ----------

val sv8: Vector = Vectors.sparse(4, Seq((3,0.2), (2, 3.0), (1,2.0)))

// COMMAND ----------

/*
A labeled point is a local vector, either dense or sparse, associated with a label/response. In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label, so we can use labeled points in both regression and classification. For binary classification, a label should be either 0 (negative) or 1 (positive). For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....
*/

// COMMAND ----------

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint


// COMMAND ----------

val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))

// COMMAND ----------

val pos = LabeledPoint(1, Vectors.dense(1.0, 0.0, 3.0))

// COMMAND ----------

val pos = LabeledPoint(-1.0, Vectors.dense(1.0, 0.0, 3.0))

// COMMAND ----------

val pos = LabeledPoint(-1.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))

// COMMAND ----------

val pos = LabeledPoint(-1.0, Vectors.sparse(4, Seq((3,0.2), (2, 3.0))))

// COMMAND ----------

val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))


// COMMAND ----------

/* 
A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine. MLlib supports dense matrices, whose entry values are stored in a single double array in column-major order, and sparse matrices, whose non-zero entry values are stored in the Compressed Sparse Column (CSC) format in column-major order. For example, the following dense matrix 
*/

// COMMAND ----------

import org.apache.spark.mllib.linalg.{Matrix, Matrices}

// COMMAND ----------

val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

// COMMAND ----------

val dm: Matrix = Matrices.dense(3,3, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))


// COMMAND ----------

val dm: Matrix = Matrices.dense(2,3, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))


// COMMAND ----------

val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0,7))

// COMMAND ----------

val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

// COMMAND ----------

val sm: Matrix = Matrices.sparse(4, 5,
Array(0, 2, 3, 6, 7, 8),
Array(0, 3, 1, 0, 2, 3, 2, 1),
Array(1.0, 14.0, 6.0, 2.0, 11.0, 16.0,12.0, 9.0))

// COMMAND ----------

import org.apache.spark.mllib.linalg.distributed.RowMatrix


// COMMAND ----------

/*
RowMatrix: “row-oriented distributed matrix without meaningful row indices” — RDD of sequence of Vector without rows indices 


A RowMatrix is a row-oriented distributed matrix without meaningful row indices, backed by an RDD of its rows, where each row is a local vector. Since each row is represented by a local vector, the number of columns is limited by the integer range but it should be much smaller in practice
*/

// COMMAND ----------

val rows: RDD[Vector] = sc.parallelize(Seq(

Vectors.dense(1.0, 0.0, 5.4, 0.0),
Vectors.dense(1.0, 0.0, 5.4, 0.0),
Vectors.dense(1.0, 0.0, 5.4, 0.0)

))
val rowMatrix: RowMatrix = new RowMatrix(rows)

// COMMAND ----------

// IndexedRowMatrix: like a RowMatrix but with meaningful row indices.

// COMMAND ----------

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}

// COMMAND ----------

val indexRows: RDD[IndexedRow] = sc.parallelize(Seq(
IndexedRow(0, Vectors.dense(1.0, 0.0, 5.4, 0.0)),
IndexedRow(2, Vectors.dense(1.0, 0.0, 5.4, 0.0))

))
val indexedRowMatrix: IndexedRowMatrix = new
IndexedRowMatrix(indexRows)

// COMMAND ----------

/*
CoordinateMatrix: elements values are explicit defined by using IndexedRow(row_index, col_index, value)

A CoordinateMatrix is a distributed matrix backed by an RDD of its entries. Each entry is a tuple of (i: Long, j: Long, value: Double), where i is the row index, j is the column index, and value is the entry value. A CoordinateMatrix should be used only when both dimensions of the matrix are huge and the matrix is very sparse.
*/

// COMMAND ----------

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

// COMMAND ----------

val coordinateMatrixEntry: RDD[MatrixEntry] = sc.parallelize(Seq(
MatrixEntry(0, 0, 1.2),
MatrixEntry(1, 0, 2.1),
MatrixEntry(6, 1, 3.7)
))
val coordinateMatrix: CoordinateMatrix = new
CoordinateMatrix(coordinateMatrixEntry)

// COMMAND ----------

/*
A BlockMatrix is a distributed matrix backed by an RDD of MatrixBlocks, where a MatrixBlock is a tuple of ((Int, Int), Matrix), where the (Int, Int) is the index of the block, and Matrix is the sub-matrix at the given index with size rowsPerBlock x colsPerBlock. BlockMatrix supports methods such as add and multiply with another BlockMatrix. BlockMatrix also has a helper function validate which can be used to check whether the BlockMatrix is set up properl
*/

// COMMAND ----------

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}

// COMMAND ----------

val blocks = sc.parallelize(Seq(

((0, 0), Matrices.dense(3, 2, Array(1, 2, 3, 4, 5, 6))),
((1, 0), Matrices.dense(3, 2, Array(7, 8, 9, 10, 11, 12)))

))
val blockMatrix: BlockMatrix = new BlockMatrix(blocks, 3, 2)

// COMMAND ----------


