package org.apache.spark.ml.made

import breeze.linalg.DenseVector
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, functions => F}


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001

  private def generateTestData(nRows: Int, seed: Int): DataFrame = {
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("x")

    var df = sqlc.range(0, nRows).select(
      F.rand(seed).alias("x1"),
      F.rand(seed + 1).alias("x2"),
      F.rand(seed + 2).alias("x3"),
    ).withColumn("y",
      F.col("x1")*1.5 + F.col("x2")*0.3 - F.col("x3")*0.7)

    df = assembler.transform(df).drop("x1", "x2", "x3")
    df
  }

  "Estimator" should "correct fit" in {
    val df: DataFrame = generateTestData(100, 9)

    val lr = new LinearRegression()
      .setFeaturesCol("x")
      .setLabelCol("y")
      .setPredictionCol("pred")
      .setIters(300)
      .setLearningRate(1.0)

    val model = lr.fit(df)

    model.weights.size should be(4)
    model.weights(0) should be(0.0 +- delta)
    model.weights(1) should be(1.5 +- delta)
    model.weights(2) should be(0.3 +- delta)
    model.weights(3) should be(-0.7 +- delta)
  }

  "Model" should "calculate correct y" in {
    // Arrange
    val df: DataFrame = generateTestData(10, 9)
    val weights = DenseVector(0.5, -0.1, 0.2)
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense)
      .setFeaturesCol("x")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    // Act
    val preds = model.transform(df).collect().map(_(0).asInstanceOf[Double])
    val trues = Seq(
      1.0991769626784262,
      0.8590851898057111,
      0.010868671563459331,
      0.5908384426382011,
      0.4715332877080067,
      -0.13127296440348998,
      -0.23832820009891204,
      0.558279162985744,
      1.3116902232591872
    )

    // Assert
    preds.length should be (10)
    for (i <- 0 until preds.length - 1) {
        preds(i) should be (trues(i) +- delta)
    }
  }
}
