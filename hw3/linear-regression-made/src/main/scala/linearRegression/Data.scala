package linearRegression

import java.io._
import breeze.linalg._


case class CrossValChunk(X: DenseMatrix[Double], y: DenseVector[Double]) {
}

class CrossValData(chunks: Array[CrossValChunk]) {
  def getTrainFor(k: Int): CrossValChunk = {
    val cvChunksX = new Array[DenseMatrix[Double]](chunks.length - 1)
    val cvChunksY = new Array[DenseVector[Double]](chunks.length - 1)
    var c = 0
    for (i <- chunks.indices) {
      if (i != k) {
        cvChunksX(c) = chunks(i).X
        cvChunksY(c) = chunks(i).y
        c = c + 1
      }
    }

    val X = DenseMatrix.vertcat(cvChunksX:_*)
    val y = DenseVector.vertcat(cvChunksY:_*)

    CrossValChunk(X, y)
  }

  def getTestFor(k: Int): CrossValChunk = {
    chunks(k)
  }
}


object Data {
  def load(path: String): DenseMatrix[Double] = {
    // https://www.kaggle.com/uciml/student-alcohol-consumption
    val csvFile: File = new File(path)
    val D: DenseMatrix[Double] = csvread(csvFile, skipLines = 1)

    D
  }

  def split(D: DenseMatrix[Double], nSplits : Int): CrossValData = {
    val n = D.rows
    val m = D.cols
    val nChunk = n / nSplits
    val result = new Array[CrossValChunk](nSplits)

    for (k <- 0 until nSplits) {
      val data = D(k*nChunk until min((k + 1)*nChunk, n), ::)
      val X = data(::, 0 until m - 1)
      val y = data(::, m - 1)
      result(k) = CrossValChunk(X, y)
    }

    new CrossValData(result)
  }

  def save(preds: DenseVector[Double], trues: DenseVector[Double], path: String): Unit = {
    val writer = new PrintWriter(new File(path))
    try {
      writer.write("true,pred\n")

      val n = preds.length
      for (i <- 0 until (n)) {
        val t = trues(i)
        val p = preds(i)
        writer.write(f"$t,$p\n")
      }
    } finally {
      writer.close()
    }
  }
}
