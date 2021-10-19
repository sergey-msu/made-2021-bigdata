package linearRegression

import java.util.logging._
import breeze.linalg._
import breeze.numerics._


class LinearRegression(name: String, epochs: Int, lr: Double, logger: Logger) {
  var w: DenseVector[Double] = DenseVector.fill(1)(0)

  def fit(X: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    val n = X.rows
    val bias = DenseMatrix.fill[Double](n, 1)(1)
    val X_ = DenseMatrix.horzcat(X, bias)
    w = DenseVector.fill(X_.cols)(0)

    for (epoch <- 0 until epochs) {
      val grads = Range(0, n).map(i => X_(i, ::) * (X_(i, ::) * w - y(i)))
      val grad = sum(grads).t * 2.0 / n.toDouble
      w = w - lr * grad

      if (epoch % 10 == 0) {
        val loss = sqrt(norm(y - X_ * w))
        logger.info(f"    > model $name, epoch: $epoch, RMSE=$loss%.2f")
      }
    }
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val n = X.rows
    val bias = DenseMatrix.fill[Double](n, 1)(1)
    val X_ = DenseMatrix.horzcat(X, bias)
    X_ * w
  }


}
