package sparkTfIdf

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]): Unit = {
    // app args
    val dataPath = args(0)
    val nTop = args(1).toInt

    // create spark session
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("TF-IDF")
      .getOrCreate()

    // read data file
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)

    // preprocess text data
    val prepr = df
      .withColumn("id", monotonically_increasing_id())
      .withColumn("Preprocess", lower(col("Review")))
      .withColumn("Preprocess", regexp_replace(col("Preprocess"), "[^0-9a-zA-Z ]+", ""))
      .select(
        col("id"),
        explode(split(col("Preprocess"), " ")).alias("term"))
      .filter("term != ''")

    // fill term frequencies table
    val termFreqs = prepr
      .groupBy(col("id"), col("term"))
      .agg(count("*").alias("term_freq"))
      .as("termFreqs")

    // fill document frequencies table
    val docFreqs = termFreqs
      .groupBy(col("term"))
      .agg(count("*").alias("doc_freq"))
      .orderBy(desc("doc_freq"))
      .limit(nTop)
      .cache()
      .as("docFreqs")

    // calc TF-IDF table
    val tfIdf = termFreqs
      .join(docFreqs, col("docFreqs.term") === col("termFreqs.term"))
      .withColumn("tf_idf", round(col("term_freq")/col("doc_freq"), 7))
      .select(
        col("id"),
        col("termFreqs.term").as("term"),
        col("tf_idf"))
      .groupBy(col("id"))
      .pivot("term")
      .sum("tf_idf")
      .na.fill(0)
      .orderBy("id")
      // most frequent terms - first
      .select(docFreqs.select("term").rdd.map(r => r(0)).collect.toList.map(c => col(c.toString)): _*)

    tfIdf.show
  }
}
