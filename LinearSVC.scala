// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer,VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
// $example off$
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.Row
object LinearSVC {
  def main(args: Array[String]): Unit = {
    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .master("local")
      .appName("RandomForestClassifierExample")
      .getOrCreate()
    import spark.implicits._

    val dataset_working = spark.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("data/mllib/working.csv")
      .select("EMP No","Gender","DOJ","Contract/Permanent","Reporting Line",
        "AHT","Attendance","Quality","CSAT","DSAT", "January","February","March",
        "April","May","June","July","August", "September","October","November","December")
      .withColumn("labelF",lit(1.0))

    val dataset_resign = spark.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("data/mllib/resigned.csv")
      .select("EMP No","Gender","DOJ","Contract/Permanent","Reporting Line",
        "AHT","Attendance","Quality","CSAT","DSAT", "January","February","March",
        "April","May","June","July","August", "September","October","November","December")
      .withColumn("labelF",lit(0.0))

    val dataset =dataset_resign.join(dataset_working, Seq("EMP No"), "leftanti").union(dataset_working)
    dataset.show()

    val new_data =dataset.withColumn("AHT_new", regexp_replace(dataset("AHT"),"%", "" ))
      .withColumn("Attendance_new", regexp_replace(dataset("Attendance"),"%", "" ))
      .withColumn("Quality_new", regexp_replace(dataset("Quality"),"%", "" ))
      .withColumn("CSAT_new", regexp_replace(dataset("CSAT"),"%", "" ))
      .withColumn("DSAT_new", regexp_replace(dataset("DSAT"),"%", "" ))
      .withColumn("January_new", regexp_replace(dataset("January"),"-", "0" ))
      .withColumn("February_new", regexp_replace(dataset("February"),"-", "0" ))
      .withColumn("March_new", regexp_replace(dataset("March"),"-", "0" ))
      .withColumn("April_new", regexp_replace(dataset("April"),"-", "0" ))
      .withColumn("May_new", regexp_replace(dataset("May"),"-", "0" ))
      .withColumn("June_new", regexp_replace(dataset("June"),"-", "0" ))
      .withColumn("July_new", regexp_replace(dataset("July"),"-", "0" ))
      .withColumn("August_new", regexp_replace(dataset("August"),"-", "0" ))
      .withColumn("September_new", regexp_replace(dataset("September"),"-", "0" ))
      .withColumn("October_new", regexp_replace(dataset("October"),"-", "0" ))
      .withColumn("November_new", regexp_replace(dataset("November"),"-", "0" ))
      .withColumn("December_new", regexp_replace(dataset("December"),"-", "0" ))
      .withColumn("DOJ_new",date_format(to_date(col("DOJ"), "MM/dd/yyyy"), "yyyy-MM-dd"))
      .withColumn("presentDay",lit(current_date()))
      .drop("DOJ","AHT","Attendance","Quality","CSAT","DSAT","January","February","March","April","May","June","July","August",
        "September","October","November","December")
      .withColumnRenamed("DOJ_new","DOJ")
      .withColumnRenamed("AHT_new","AHT")
      .withColumnRenamed("Attendance_new","Attendance")
      .withColumnRenamed("Quality_new","Quality")
      .withColumnRenamed("CSAT_new","CSAT")
      .withColumnRenamed("DSAT_new","DSAT")
      .withColumnRenamed("January_new", "January")
      .withColumnRenamed("February_new", "February")
      .withColumnRenamed("March_new", "March")
      .withColumnRenamed("April_new", "April")
      .withColumnRenamed("May_new", "May")
      .withColumnRenamed("June_new", "June")
      .withColumnRenamed("July_new", "July")
      .withColumnRenamed("August_new", "August")
      .withColumnRenamed("September_new", "September")
      .withColumnRenamed("October_new", "October")
      .withColumnRenamed("November_new", "November")
      .withColumnRenamed("December_new", "December")

    val renamed_data =new_data.withColumn("DOJ", new_data("DOJ").cast(DateType))
      .withColumn("AHT", new_data("AHT").cast(DoubleType))
      .withColumn("Attendance", new_data("Attendance").cast(DoubleType))
      .withColumn("Quality", new_data("Quality").cast(DoubleType))
      .withColumn("CSAT", new_data("CSAT").cast(DoubleType))
      .withColumn("DSAT", new_data("DSAT").cast(DoubleType))
      .withColumn("January",new_data( "January").cast(DoubleType))
      .withColumn("February", new_data("February").cast(DoubleType))
      .withColumn("March", new_data("March").cast(DoubleType))
      .withColumn("April",new_data( "April").cast(DoubleType))
      .withColumn("May", new_data("May").cast(DoubleType))
      .withColumn("June",new_data( "June").cast(DoubleType))
      .withColumn("July",new_data( "July").cast(DoubleType))
      .withColumn("August",new_data( "August").cast(DoubleType))
      .withColumn("September",new_data( "September").cast(DoubleType))
      .withColumn("October", new_data("October").cast(DoubleType))
      .withColumn("November",new_data( "November").cast(DoubleType))
      .withColumn("December", new_data("December").cast(DoubleType))
      .where("AHT is not null and Attendance is not null and Quality is not null " +
        "and CSAT is not null and DSAT is not null")
      .withColumn("datesBetween",datediff(col("presentDay"),col("DOJ"))/365)

    import org.apache.spark.sql.functions.{col, lit}
    // count non zero elements
    val performance_col=renamed_data.select("January","February","March",
      "April","May","June","July","August", "September","October","November","December")
    val count_non_zero = performance_col.columns.tail.map(x => when(col(x) =!= 0.0, 1).otherwise(0)).reduce(_ + _)
    val non_zero=performance_col.withColumn("non_zero_count", count_non_zero)
    // take the sum of all the column
    val marksColumns = Array(col("January"), col("February"), col("March"),
      col("April"), col("May"), col("June"), col("July"),
      col("August"), col("September"), col("October"), col("November"),
      col("December"))
    val sum = marksColumns.foldLeft(lit(0)){(x1, x2) => x1+x2}
    //calculate the average
    val avgColumns = Array (sum, count_non_zero)
    val avrg =avgColumns.reduceLeft((x,y) => x / y)
    //update the table
    val formated_data=renamed_data.withColumn("count", count_non_zero)
      .withColumn("sum", sum)
      .withColumn("avg", avrg)
      .drop("January","February","March",
        "April","May","June","July","August", "September","October","November","December")
      .where("avg is not null")

    val cat_feature=List("Gender","Contract/Permanent","Reporting Line")
    val encoded_features=cat_feature.flatMap{ name=>
      val indexer_cat=new StringIndexer()
        .setInputCol(name)
        .setOutputCol(name+"_cat")
        .fit(formated_data)
      Array(indexer_cat)
    }.toArray

    val pipeline_feature = new Pipeline().setStages(encoded_features)
    val stratified_data_cat = pipeline_feature.fit(formated_data).transform(formated_data)
      .drop("Gender","Contract/Permanent","Reporting Line")
      .withColumnRenamed("Gender_cat","Gender")
      .withColumnRenamed("Contract/Permanent_cat","Contract/Permanent")
      .withColumnRenamed("Reporting Line_cat","Reporting Line")
    stratified_data_cat.describe().show()
/////////////////////////////////////////////////////////////////////////////////////////////////////
    val stat = stratified_data_cat.describe("datesBetween", "AHT","Attendance","Quality","CSAT","DSAT","avg","Gender",
      "Contract/Permanent")
      .select(col("summary"), col("datesBetween").cast("double"), col("AHT").cast("double"),
        col("Attendance").cast("double"), col("Quality").cast("double"), col("CSAT").cast("double"),
        col("DSAT").cast("double"),col("avg").cast("double"), col("Contract/Permanent").cast("double"))
      .collect.map(_.toSeq.toArray).transpose.map(x => x.slice(1,3))

    stratified_data_cat.select(col("labelF"), col("datesBetween").minus(stat(1)(0)).divide(stat(1)(1)).as("datesBetween"),
      col("AHT").minus(stat(2)(0)).divide(stat(2)(1)).as("AHT"),col("Attendance").minus(stat(3)(0)).divide(stat(3)(1)).as("Attendance")).show()//,
//      col("Quality").minus(stat(4)(0)).divide(stat(4)(1)).as("Quality"), col("CSAT").minus(stat(5)(0)).divide(stat(5)(1)).as("CSAT"),
//      col("DSAT").minus(stat(6)(0)).divide(stat(6)(1)).as("DSAT"),col("avg").minus(stat(7)(0)).divide(stat(7)(1)).as("avg"),
//      col("Contract/Permanent").minus(stat(9)(0)).divide(stat(9)(1)).as("Contract/Permanent")).show()
///////////////////////////////////////////////////////////////////////////////////////////////////
    val assembler = new VectorAssembler()
      .setInputCols(Array("datesBetween","AHT","Attendance","Quality","CSAT","DSAT","avg","Gender",
        "Contract/Permanent"))
      .setOutputCol("indexedFeatures")
    val data1 = assembler.transform(stratified_data_cat)
    println("####################################################data with feature vector")
    data1.show(5)

    val labelAssembler = new StringIndexer()
      .setInputCol("labelF")
      .setOutputCol("label")
      .fit(data1)
    val labelIndexer=labelAssembler.transform(data1)
    //    labelIndexer.show(5)

    //whether include or not
    val FinalFeatures = new VectorIndexer()
      .setInputCol("indexedFeatures")
      .setOutputCol("features")
      .fit(labelIndexer)
    val featureIndexer=FinalFeatures.transform(labelIndexer)
    val dataFinal=featureIndexer.select("label","features")
    //    val FinalFeatures = featureIndexer.transform(Indexed)

    //    // Split the data into training and test sets (20% held out for testing).
    val Array(trainingData_org, testData) = dataFinal.randomSplit(Array(0.7, 0.3))
    val trainingData = trainingData_org.orderBy(rand())
    //    println("#############################################splitdata")
    //    trainingData.show(50)
    val lsvc = new LinearSVC()
      .setMaxIter(50)
      .setRegParam(0.5)

    // Fit the model
    val lsvcModel = lsvc.fit(trainingData)
    val predictions = lsvcModel.transform(testData)
    println("##############################################summary")
    predictions.show(2000,false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = ${(accuracy*100)}")
    println(s"Test Error = ${(1.0 - accuracy)}")

    // Print the coefficients and intercept for linear svc
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println