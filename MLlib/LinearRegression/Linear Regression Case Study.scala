// Databricks notebook source
// We need to predict the mpg based on other given features

// The Data is stored in the UCL Machine learning repository at the foloowing URL
// The following commaind download's the data and converts into test file
val html = scala.io.Source.fromURL("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data").mkString;
// Splitting the data by new line charecter and filter will filter out the empty lines
val list = html.split("\n").filter(_ != "");
// The missing Data in the data set is coded as "?" in the data set so to remove the lines containing ? we use the filter commaing again
val raw_data = sc.parallelize(list).filter(lines => !(lines.contains("?")));
// The Schema of the Data is described by case class 
case class cardata_one(mpg: Float, cylinders: Integer, displacement: Float,
					horsepower: Float, weight: Float, acceleration: Float,
					model_year:Integer,origin: Integer)
// THe Data Set is fixed width demilited and tab delimited so we are splitting the last colums which is use less for us
val raw_auto= raw_data.map(_.split("\t"))
// Casting Data to assign the coloumn types and column names
val auto_data = raw_auto.map(_.apply(0)).map(_.split(" ")).map(_.filter(_!="")).map(p => cardata_one(p(0).toFloat,                       p(1).toInt,p(2).toFloat,p(3).toFloat,p(4).toFloat,p(5).toFloat,p(6).toInt,p(7).toInt)).toDF();

// COMMAND ----------

// Functions in SQL provide functions like average minimum maximum 
import org.apache.spark.sql.functions._
auto_data.show();
auto_data.select(avg($"mpg")).show();

// COMMAND ----------

auto_data.printSchema();
// To apply Machine learning on the on the we need code the fature in form of a feature vector 
// Spark has vector assembler functions to provide the required format
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
val assembler = new VectorAssembler()
  .setInputCols(Array("cylinders", "displacement", "horsepower","weight","acceleration","model_year","origin"))
  .setOutputCol("features")

val output = assembler.transform(auto_data)

//println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
//output.select("features", "clicked").show(false)
output.show();

// COMMAND ----------

//To get better results we use Scaling the vectors to get better results 
// Spark Has standard Scaler to scalre the feture vectors
import org.apache.spark.ml.feature.StandardScaler
// Normalize each feature to have unit standard deviation.
// Compute summary statistics by fitting the StandardScaler.
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(true)
val scalerModel = scaler.fit(output)
val scaledData = scalerModel.transform(output)
scaledData.show()

// COMMAND ----------

// Finally for the training we select label and features coloums

val entire_data = scaledData.select($"mpg",$"scaledFeatures");
entire_data.show();
entire_data.printSchema();
val names = Seq("label","features");
val entire_data_DF = entire_data.toDF(names: _*)

// Split the data into train data and test data
val Array(trainingData, testData) = entire_data_DF.randomSplit(Array(0.7, 0.3))

//val final_traindata = trainingData.select('scaledFeatures).distinct().show()

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.Pipeline

val lr = new LinearRegression()
  .setMaxIter(10000)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
// Fit the model
// Chain indexer and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(scaler,lr))

// Train model. This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions.

val train_predictions = model.transform(trainingData)

val test_predictions = model.transform(testData)

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

train_predictions.select("prediction", "label","scaledFeatures").show(5)

test_predictions.select("prediction", "label","scaledFeatures").show(5)

// Select (prediction,true label) and compute train error.
val train_evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse_train = train_evaluator.evaluate(train_predictions)
println("Root Mean Squared Error (RMSE) on train data = " + rmse_train)

// Select (prediction, true label) and compute test error.
val test_evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse_test = test_evaluator.evaluate(test_predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse_test)

val diff = rmse_test - rmse_train

println("The difference in rmse of train and test data is =" + diff)

// COMMAND ----------


