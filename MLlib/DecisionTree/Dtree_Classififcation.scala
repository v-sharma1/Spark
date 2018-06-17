// Databricks notebook source
var wine_data = sqlContext.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("delimiter",";")
  .load("/FileStore/tables/winequality_white_dtree-f21ec.csv")
display(wine_data);

// COMMAND ----------

import org.apache.spark.sql.functions.{min, max}
wine_data.printSchema();

val col_names = wine_data.columns.slice(0,11)

for(coloum <- col_names){
  wine_data.agg(min(coloum),max(coloum)).show();
}

// COMMAND ----------

import org.apache.spark.ml.feature.QuantileDiscretizer
def discretizerFun (col: String, bucketNo: Int): 
 org.apache.spark.ml.feature.QuantileDiscretizer = {
  val discretizer = new QuantileDiscretizer()
  discretizer
  .setInputCol(col)
  .setOutputCol(s"${col}_result")
  .setNumBuckets(bucketNo)
}
val parts = 10;
val fixed_acidity = discretizerFun("fixed acidity", parts).fit(wine_data).transform(wine_data)
val volatile_acidity = discretizerFun("volatile acidity", parts).fit(fixed_acidity).transform(fixed_acidity)
val citric_acid = discretizerFun("citric acid", parts).fit(volatile_acidity).transform(volatile_acidity)
val residual_sugar = discretizerFun("residual sugar", parts).fit(citric_acid).transform(citric_acid)
val chlorides = discretizerFun("chlorides", parts).fit(residual_sugar).transform(residual_sugar)
val free_sulfur_dioxide = discretizerFun("free sulfur dioxide", parts).fit(chlorides).transform(chlorides)
val density = discretizerFun("density", parts).fit(free_sulfur_dioxide).transform(free_sulfur_dioxide)
val pH = discretizerFun("pH", parts).fit(density).transform(density)
val sulphates = discretizerFun("sulphates", parts).fit(pH).transform(pH)
val alcohol = discretizerFun("alcohol", parts).fit(sulphates).transform(sulphates)

// COMMAND ----------

alcohol.printSchema();
val cols = alcohol.columns.slice(11,22);

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val assembler = new VectorAssembler()
  .setInputCols(cols)
  .setOutputCol("features")

val wine_data_label = assembler.transform(alcohol)

// COMMAND ----------

wine_data_label.printSchema();

// COMMAND ----------

import org.apache.spark.ml.feature.MinMaxScaler
var i =1;
  val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

  // Compute summary statistics and generate MinMaxScalerModel
  val scalerModel = scaler.fit(wine_data_label);

  // rescale each feature to range [min, max].
   val scaledData = scalerModel.transform(wine_data_label);


println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
val train_data = scaledData.select("quality", "scaledFeatures")

// COMMAND ----------

train_data.printSchema()

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("quality")
  .setOutputCol("label")
  .fit(train_data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("scaledFeatures")
  .setOutputCol("features")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(train_data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = train_data.randomSplit(Array(0.8, 0.2))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val train_predictions = model.transform(trainingData)

val test_predictions = model.transform(testData)

// Select example rows to display.
train_predictions.select("predictedLabel", "label", "features").show(100)

test_predictions.select("predictedLabel", "label", "features").show(100)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val train_accuracy = evaluator.evaluate(train_predictions)
println(train_accuracy)
println("Train Error = " + (1.0 - train_accuracy))

val test_accuracy = evaluator.evaluate(test_predictions)
println(test_accuracy)
println("Test Error = " + (1.0 - test_accuracy))

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)
