package linearRegression

import java.util.logging._
import breeze.linalg._
import breeze.numerics._


object Main {
  def main(args: Array[String]): Unit = {
    // app args
    val train_path = args(0)
    val test_path = args(1)
    val nSplits = args(2).toInt
    val epochs = args(3).toInt
    val lr= args(4).toDouble

    // setup logging
    val logger = Logger.getLogger("linear_regression")
    val handler = new FileHandler("app.log")
    handler.setFormatter(new SimpleFormatter())
    logger.addHandler(handler)

    // load data
    logger.info(f"Load data from: $train_path...")
    val D_train = Data.load(train_path)
    logger.info(f"Load data from: $test_path...")
    val D_test = Data.load(test_path)
    logger.info(f"Load data success")

    // split data
    logger.info(f"Splitting data into $nSplits train chunks...")
    val splits = Data.split(D_train, nSplits)

    // fitting CV models
    logger.info("Fitting model...")
    val models = new Array[LinearRegression](nSplits)
    for (k <- 0 until nSplits) {
      logger.info(f"  > fitting model #${k+1}...")
      val model = new LinearRegression(f"#${k + 1}", epochs, lr, logger)
      val train_data = splits.getTrainFor(k)
      val test_data = splits.getTestFor(k)
      model.fit(train_data.X, train_data.y)
      models(k) = model

      val preds = model.predict(test_data.X)
      val loss = sqrt(norm(preds - test_data.y))
      logger.info(f"  > validating model #${k+1}: RMSE=$loss%.2f")
    }

    // apply CV ensemble on test data
    logger.info("Apply CV scoring...")
    val X_test = D_test(::, 0 until D_test.cols - 1)
    val y_test = D_test(::, D_test.cols - 1)
    val results = new Array[DenseVector[Double]](nSplits)
    for (i <- 0 until nSplits) {
      val model = models(i)
      results(i) = model.predict(X_test)
    }
    val y_pred = results.reduce(_ + _).map(_ / nSplits)

    // saving results to file
    logger.info("Save predictions to file...")

    Data.save(y_pred, y_test, "preds.csv")

    logger.info("Done!")
  }
}
