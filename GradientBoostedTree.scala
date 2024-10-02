// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer,VectorAssembler}
// $example off$
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.Row

object GradientBoostedTree {
  def main(args: Array[String]): Unit = {
      import org.apache.log4j._
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
      val spark = SparkSession
        .builder
        .master("local")
        .appName("GradientBoostedTree")
        .getOrCreate()
      import spark.implicits._

      val dataset_working = spark.read.format("csv")
        .option("inferSchema", "true")
        ////      .option("schema","AHT: double ,Attendance: double ,Quality: double ,CSAT: double,January: integer," +
        //      "February: integer,March: integer,April: integer,May: integer,June: integer,July: integer,August: integer," +
        //      "September: integer,October: integer,November: integer,December: integer")
        .option("header", "true")
        //      .schema(customSchema)
        .load("data_current.csv")
        .select("EMP No","Gender","DOJ","Contract/Permanent","Reporting Line",
          "AHT","Attendance","Quality","CSAT","DSAT", "January","February","March",
          "April","May","June","July","August", "September","October","November","December")
        .withColumn("labelF",lit(1.0))

      val dataset_resign = spark.read.format("csv")
        .option("inferSchema", "true")
        ////      .option("schema","AHT: double ,Attendance: double ,Quality: double ,CSAT: double,January: integer," +
        //      "February: integer,March: integer,April: integer,May: integer,June: integer,July: integer,August: integer," +
        //      "September: integer,October: integer,November: integer,December: integer")
        .option("header", "true")
        //      .schema(customSchema)
        .load("data_past.csv")
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

      //    renamed_data.show(1000)
      //    stratified_data_cat.printSchema()
      //    stratified_data_cat.collect.foreach(println)
      stratified_data_cat.describe().show()

      val assembler = new VectorAssembler()
        .setInputCols(Array("datesBetween","AHT","Attendance","Quality","CSAT","DSAT","avg","Gender",
          "Contract/Permanent"))
        .setOutputCol("indexedFeatures")
      val data = assembler.transform(stratified_data_cat)
      println("####################################################data with feature vector")
      data.show(5)

      val labelIndexer = new StringIndexer()
      .setInputCol("labelF")
      .setOutputCol("label")
      .fit(data)
    //    val labelIndexer=labelAssembler.transform(data)

    //whether include or not
    val indexedFeatures = new VectorIndexer()
      .setInputCol("indexedFeatures")
      .setOutputCol("Features")
      //      .setMaxCategories(4)
      .fit(data)
    //    val FinalFeatures = featureIndexer.transform(Indexed)
    //    // Split the data into training and test sets (20% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))
    //    val trainingData = trainingData_org.orderBy(rand())
    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setMaxIter(200)
    //      .setFeatureSubsetStrategy("auto")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, indexedFeatures, gbt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    // Make predictions.
    val predictions = model.transform(testData)
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(1000)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = ${accuracy}")

    val predictionAndLabelsqw =predictions.select("label","prediction").rdd
    val predictionAndLabels=predictionAndLabelsqw.map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"Precision of True= ${metrics.precision(1)}" )
    println(s"Precision of False=${metrics.precision(0)}")
    println(s"Recall of True  =  ${metrics.recall(1)}")
    println(s"Recall of False   ${metrics.recall(0)}")
    println(s"F-1 Score          ${metrics.accuracy}")
    println( s"Confusion Matrix\n ${metrics.confusionMatrix}")

    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    //    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
