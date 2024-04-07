import BasicAnalysis.{accuracy, assembler, indexers, rfModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}


object TestSurvival extends App {

  val spark = SparkSession.builder()
    .appName("TitanicSurvivalPrediction")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR") // We want to ignore all of the INFO and WARN messages.



  val testData = spark.read.option("header", "true").csv("src/resources/data/test.csv")


  // Preprocess the test data
  val testDataPreprocessed = testData
    .withColumn("Age", testData("Age").cast("double"))
    .withColumn("SibSp", testData("SibSp").cast("double"))
    .withColumn("Parch", testData("Parch").cast("double"))
    .withColumn("Fare", testData("Fare").cast("double"))
    .withColumn("Pclass", testData("Pclass").cast("int"))

  val testDataIndexed = indexers.foldLeft(testDataPreprocessed)((df, indexer) => indexer.transform(df))
  val testDataAssembled = assembler.transform(testDataIndexed)

  val rfModel = RandomForestClassificationModel.load("src/resources/model/random_forest")


  // Make predictions on the test data
  val predictions_test = rfModel.transform(testDataAssembled)

  // Evaluate the model
  val evaluator_test = new BinaryClassificationEvaluator()
    .setLabelCol("Survived")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")
  val accuracy_test = evaluator_test.evaluate(predictions_test)

  println(s"Accuracy on test data: $accuracy")

}
