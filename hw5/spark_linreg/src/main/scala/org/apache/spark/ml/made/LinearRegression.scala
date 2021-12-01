package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions.{monotonicallyIncreasingId, monotonically_increasing_id, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.expressions.Window


trait LinearRegressionParams extends PredictorParams
  with HasFeaturesCol with HasLabelCol with HasPredictionCol {
  val iters: IntParam = new IntParam(
    this,
    "iters",
    "Number of algorythm iterations")
  val lr: DoubleParam = new DoubleParam(
    this,
    "lr",
    "Regression learning rate")

  def setIters(value: Int): this.type = set(iters, value)
  def setLearningRate(value: Double): this.type = set(lr, value)
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  setDefault(iters -> 100, lr -> 0.1)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    checkColumn(schema, $(predictionCol), getPredictionCol)
    checkColumn(schema, $(labelCol), getLabelCol)
    checkColumn(schema, $(featuresCol), getFeaturesCol)
  }

  private def checkColumn(schema: StructType, col: String, getCol: String): StructType = {
    if (schema.fieldNames.contains(col)) {
      SchemaUtils.checkColumnType(schema, getCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getCol).copy(name = getCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  def calcGrad(feats: Vector, y: Double, weights: BreezeDenseVector[Double]): Vector = {
    val x = BreezeDenseVector.vertcat(BreezeDenseVector(1.0), feats.asBreeze.toDenseVector)
    val grad = x * (sum(x * weights) - y)
    Vectors.fromBreeze(grad)
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeats = MetadataUtils.getNumFeatures(dataset, $(featuresCol)) + 1
    var weights: BreezeDenseVector[Double] = BreezeDenseVector.zeros(numFeats)
    val calcGradUDF = udf[Vector, Vector, Double]((feats: Vector, y: Double) => calcGrad(feats, y, weights))

    for (_ <- 0 to $(iters)) { // epoch
      val gradCol = calcGradUDF(dataset($(featuresCol)), dataset($(labelCol)))
      val Row(Row(grad)) = dataset.select(Summarizer.metrics("mean").summary(gradCol)).first()
      weights = weights - $(lr) * grad.asInstanceOf[DenseVector].asBreeze.toDenseVector
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String, val weights: Vector)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  def this(weights: Vector) = this(Identifiable.randomUID("linearRegressionModel"), weights)

  def predict(features: Vector): Double = {
    val x = BreezeDenseVector.vertcat(BreezeDenseVector(1.0), features.asBreeze.toDenseVector)
    x.dot(weights.asBreeze.toDenseVector)
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val params = Tuple1(weights)
      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/vectors")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(weights.asBreeze.dot(x.asBreeze)))
      }
    )
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def copy(extra: ParamMap): LinearRegressionModel =
    copyValues(new LinearRegressionModel(uid, weights))
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/vectors")
      val weights = vectors.select(vectors("_1").as[Vector]).first()
      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}
