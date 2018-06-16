// Databricks notebook source

/* 

The Dataset Description: The Data Set is generated from Audio sample analysis by Domain experts to decide the features that are required to perform logistic regression to determine the gender of the speaker

These are the features extracted from the sample voice recordings

The following acoustic properties of each voice are measured and included within the CSV:
meanfreq: mean frequency (in kHz)
sd: standard deviation of frequency
median: median frequency (in kHz)
Q25: first quantile (in kHz)/n
Q75: third quantile (in kHz)
IQR: interquantile range (in kHz)
skew: skewness (see note in specprop description)
kurt: kurtosis (see note in specprop description)
sp.ent: spectral entropy
sfm: spectral flatness
mode: mode frequency
centroid: frequency centroid (see specprop)
peakf: peak frequency (frequency with highest energy)
meanfun: average of fundamental frequency measured across acoustic signal
minfun: minimum fundamental frequency measured across acoustic signal
maxfun: maximum fundamental frequency measured across acoustic signal
meandom: average of dominant frequency measured across acoustic signal
mindom: minimum of dominant frequency measured across acoustic signal
maxdom: maximum of dominant frequency measured across acoustic signal
dfrange: range of dominant frequency measured across acoustic signal
modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
label: male or female

*/



// COMMAND ----------

//Reading the data from CSV files in spark using in built library in sprak 2.0 series
val gender_data = sqlContext.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/voice.csv")
val entire_data = gender_data.withColumnRenamed("sp.ent", "entropy").withColumnRenamed("label", "gender");
entire_data.printSchema();
entire_data.show();


// COMMAND ----------

// R formula simplifies the work greatly and simplifies our work of creating the vector assembling
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.functions._

val formula = new RFormula()
  .setFormula("gender ~ .")
  .setFeaturesCol("features")
  .setLabelCol("label")
val output = formula.fit(entire_data).transform(entire_data)
val required_data = output.select("features", "label");
required_data.select(avg($"label")).show();
required_data.show()

// male  - 0
//female - 1

// COMMAND ----------

/* Now we will divide the data into train and test data and apply logistic regression model on the train and test data.
And also we need to evaluate the predicted train and test data. So we will find out the accuracy metrics of both train and test data of created LR model
*/


import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD

// Dividing the data into train and test data
val Array(trainingData, testData) = required_data.randomSplit(Array(0.7, 0.3))


val lr = new LogisticRegression()
  .setMaxIter(10000)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Train model. This also runs the indexer.
val lrmodel = lr.fit(trainingData)

// Make predictions.

val lr_train_predictions = lrmodel.transform(trainingData)

val lr_test_predictions = lrmodel.transform(testData)

//lr_train_predictions.show()

//lr_test_predictions.show()

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrmodel.coefficients} Intercept: ${lrmodel.intercept}")


// evaluation of predicted train data of lr model
val lr_train_evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val lr_train_accuracy = lr_train_evaluator.evaluate(lr_train_predictions)
println("lr_train_error = " + (1.0 - lr_train_accuracy))

// evaluation of predicted test data of lr model
val lr_test_evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val lr_test_accuracy = lr_test_evaluator.evaluate(lr_test_predictions)
println("lr_test_error = " + (1.0 - lr_test_accuracy))

// COMMAND ----------

/* We can use multinomial family as a classification parameter on the train and test data.
And we will evaluate the predicted train and test data. So we will find out the accuracy metrics of both train and test data of created MLR model
*/

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD


// We can also use the multinomial family for binary classification
val mlr = new LogisticRegression()
  .setMaxIter(10000)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
  .setFamily("multinomial")


// Train model. This also runs the indexer.
val mlrmodel = mlr.fit(trainingData)

// Make predictions.

val mlr_train_predictions = mlrmodel.transform(trainingData)

val mlr_test_predictions = mlrmodel.transform(testData)

//mlr_train_predictions.show()

//mlr_test_predictions.show()

// Print the coefficients and intercepts for logistic regression with multinomial family
println(s"Multinomial coefficients: ${mlrmodel.coefficientMatrix}")
println(s"Multinomial intercepts: ${mlrmodel.interceptVector}")


// evaluation of predicted train data of mlr model
val mlr_train_evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val mlr_train_accuracy = mlr_train_evaluator.evaluate(mlr_train_predictions)
println("mlr_train_error = " + (1.0 - mlr_train_accuracy))


//evaluation of predicted test data of mlr model
val mlr_test_evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val mlr_test_accuracy = mlr_test_evaluator.evaluate(mlr_test_predictions)
println("mlr_test_error = " + (1.0 - mlr_test_accuracy))

// COMMAND ----------


