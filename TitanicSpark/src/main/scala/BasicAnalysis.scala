import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.functions.{col, when}

object BasicAnalysis extends App {

  val spark = SparkSession.builder()
    .appName("TitanicSurvivalPrediction")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR") // We want to ignore all of the INFO and WARN messages.

  // Load the Titanic training dataset
  val trainData = spark.read.option("header", "true").csv("src/resources/data/train.csv")

  // Convert "Pclass" column to integer
  // Convert columns to the appropriate numeric type
  val data = trainData
    .withColumn("Age", trainData("Age").cast("double"))
    .withColumn("SibSp", trainData("SibSp").cast("double"))
    .withColumn("Parch", trainData("Parch").cast("double"))
    .withColumn("Fare", trainData("Fare").cast("double"))
    .withColumn("Pclass", trainData("Pclass").cast("int")) // Convert Pclass to integer

  // Handle categorical columns
  val categoricalCols = Array("Sex", "Embarked")
  val indexers = categoricalCols.map(col => new StringIndexer()
    .setInputCol(col)
    .setOutputCol(col + "_index")
    .setHandleInvalid("keep") // Handle NULL values by skipping them
    .fit(data))
  val indexedData = indexers.foldLeft(data)((df, indexer) => indexer.transform(df))

  // Assemble features
  val featureCols = Array("Pclass", "Sex_index", "Age", "SibSp", "Parch", "Fare", "Embarked_index")
  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
    .setHandleInvalid("keep") // Handle null values by skipping them
  val assembledData = assembler.transform(indexedData)

  val imputer = new Imputer()
    .setInputCols(Array("Age", "SibSp", "Parch", "Fare"))
    .setOutputCols(Array("Age_imputed", "SibSp_imputed", "Parch_imputed", "Fare_imputed"))
    .setStrategy("mean") // Use the mean value to impute missing values
    .setMissingValue(Double.NaN) // Handle NaN values explicitly

  val imputedData = imputer.fit(assembledData).transform(assembledData)
  val finalDataWithoutNaN = imputedData.na.drop()

  // Keep the 'Survived' column for model training
  val finalData = finalDataWithoutNaN.withColumn("label", assembledData("Survived").cast("double")).drop("Survived")

  // Train a Random Forest model
  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
  val rfModel = rf.fit(finalData)


  // Make predictions
  val predictions = rfModel.transform(finalData)

  // Evaluate the model
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")
  val accuracy = evaluator.evaluate(predictions)

  println(s"Accuracy: $accuracy")

  val testData = spark.read.option("header", "true").csv("src/resources/data/test.csv")

  val testDataPreprocessed = testData
    .withColumn("Age", testData("Age").cast("double"))
    .withColumn("SibSp", testData("SibSp").cast("double"))
    .withColumn("Parch", testData("Parch").cast("double"))
    .withColumn("Fare", testData("Fare").cast("double"))
    .withColumn("Pclass", testData("Pclass").cast("int"))

  val testDataIndexed = indexers.foldLeft(testDataPreprocessed)((df, indexer) => indexer.transform(df))
  val testDataAssembled = assembler.transform(testDataIndexed)

  // Make predictions on the test data
  val predictions_test = rfModel.transform(testDataAssembled)

  // Select only the PassengerId and prediction columns
  val result = predictions_test.select("PassengerId", "prediction")

  // Convert the prediction column to 0 or 1
  val resultWithSurvived = result.withColumn("Survived", when(col("prediction") === 1.0, 1).otherwise(0))

  // Write the result to a CSV file
  resultWithSurvived.select("PassengerId", "Survived")
    .coalesce(1) // Write to a single file
    .write
    .format("csv")
    .option("header", "true")
    .save("predicted_survival.csv")



}
