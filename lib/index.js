'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var Matrix = require('ml-matrix');
var Matrix__default = _interopDefault(Matrix);
require('ml-stat');
var ConfusionMatrix = _interopDefault(require('ml-confusion-matrix'));

/**
 * @private
 * Function that given vector, returns its norm
 * @param {Vector} X
 * @return {number} Norm of the vector
 */
function norm(X) {
  return Math.sqrt(X.clone().apply(pow2array).sum());
}

/**
 * @private
 * Function that pow 2 each element of a Matrix or a Vector,
 * used in the apply method of the Matrix object
 * @param {number} i - index i.
 * @param {number} j - index j.
 * @return {Matrix} The Matrix object modified at the index i, j.
 * */
function pow2array(i, j) {
  this.set(i, j, this.get(i, j) ** 2);
}

/**
 * @private
 * Function that initialize an array of matrices.
 * @param {Array} array
 * @param {boolean} isMatrix
 * @return {Array} array with the matrices initialized.
 */
function initializeMatrices(array, isMatrix) {
  if (isMatrix) {
    for (var i = 0; i < array.length; ++i) {
      for (var j = 0; j < array[i].length; ++j) {
        var elem = array[i][j];
        array[i][j] = elem !== null ? new Matrix__default(array[i][j]) : undefined;
      }
    }
  } else {
    for (i = 0; i < array.length; ++i) {
      array[i] = new Matrix__default(array[i]);
    }
  }

  return array;
}

/**
 * @private
 * Get total sum of square
 * @param {Array} x an array
 */
function tss(x) {
  return x.clone().mul(x.clone()).sum();
}

// from CV.kfold
function getFolds(features, k) {
  var N = features.length;
  var allIdx = new Array(N);
  for (var i = 0; i < N; i++) {
    allIdx[i] = i;
  }

  var l = Math.floor(N / k);
  // create random k-folds
  var current = [];
  var folds = [];
  while (allIdx.length) {
    var randi = Math.floor(Math.random() * allIdx.length);
    current.push(allIdx[randi]);
    allIdx.splice(randi, 1);
    if (current.length === l) {
      folds.push(current);
      current = [];
    }
  }
  if (current.length) folds.push(current);
  folds = folds.slice(0, k);

  let foldsIndex = folds.map((x, idx) => ({
    testIndex: x,
    trainIndex: [].concat(...folds.filter((el, idx2) => (idx2 !== idx)))
  }));
  return foldsIndex;
}

/**
 * @class PLS
 */
class PLS {
  /**
   * Constructor for Partial Least Squares (PLS)
   * @param {object} options
   * @param {number} [options.latentVectors] - Number of latent vector to get (if the algorithm doesn't find a good model below the tolerance)
   * @param {number} [options.tolerance=1e-5]
   * @param {boolean} [options.scale=true] - rescale dataset using mean.
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      this.meanX = model.meanX;
      this.stdDevX = model.stdDevX;
      this.meanY = model.meanY;
      this.stdDevY = model.stdDevY;
      this.PBQ = Matrix__default.checkMatrix(model.PBQ);
      this.R2X = model.R2X;
      this.scale = model.scale;
      this.scaleMethod = model.scaleMethod;
      this.tolerance = model.tolerance;
    } else {
      var {
        tolerance = 1e-5,
        scale = true,
      } = options;
      this.tolerance = tolerance;
      this.scale = scale;
      this.latentVectors = options.latentVectors;
    }
  }

  /**
   * Fits the model with the given data and predictions, in this function is calculated the
   * following outputs:
   *
   * T - Score matrix of X
   * P - Loading matrix of X
   * U - Score matrix of Y
   * Q - Loading matrix of Y
   * B - Matrix of regression coefficient
   * W - Weight matrix of X
   *
   * @param {Matrix|Array} trainingSet
   * @param {Matrix|Array} trainingValues
   */
  train(trainingSet, trainingValues) {
    trainingSet = Matrix__default.checkMatrix(trainingSet);
    trainingValues = Matrix__default.checkMatrix(trainingValues);

    if (trainingSet.length !== trainingValues.length) {
      throw new RangeError('The number of X rows must be equal to the number of Y rows');
    }

    this.meanX = trainingSet.mean('column');
    this.stdDevX = trainingSet.standardDeviation('column', { mean: this.meanX, unbiased: true });
    this.meanY = trainingValues.mean('column');
    this.stdDevY = trainingValues.standardDeviation('column', { mean: this.meanY, unbiased: true });

    if (this.scale) {
      trainingSet = trainingSet.clone().subRowVector(this.meanX).divRowVector(this.stdDevX);
      trainingValues = trainingValues.clone().subRowVector(this.meanY).divRowVector(this.stdDevY);
    }

    if (this.latentVectors === undefined) {
      this.latentVectors = Math.min(trainingSet.rows - 1, trainingSet.columns);
    }

    var rx = trainingSet.rows;
    var cx = trainingSet.columns;
    var ry = trainingValues.rows;
    var cy = trainingValues.columns;

    var ssqXcal = trainingSet.clone().mul(trainingSet).sum(); // for the r²
    var sumOfSquaresY = trainingValues.clone().mul(trainingValues).sum();

    var tolerance = this.tolerance;
    var n = this.latentVectors;
    var T = Matrix__default.zeros(rx, n);
    var P = Matrix__default.zeros(cx, n);
    var U = Matrix__default.zeros(ry, n);
    var Q = Matrix__default.zeros(cy, n);
    var B = Matrix__default.zeros(n, n);
    var W = P.clone();
    var k = 0;

    while (norm(trainingValues) > tolerance && k < n) {
      var transposeX = trainingSet.transpose();
      var transposeY = trainingValues.transpose();

      var tIndex = maxSumColIndex(trainingSet.clone().mul(trainingSet));
      var uIndex = maxSumColIndex(trainingValues.clone().mul(trainingValues));

      var t1 = trainingSet.getColumnVector(tIndex);
      var u = trainingValues.getColumnVector(uIndex);
      var t = Matrix__default.zeros(rx, 1);

      while (norm(t1.clone().sub(t)) > tolerance) {
        var w = transposeX.mmul(u);
        w.div(norm(w));
        t = t1;
        t1 = trainingSet.mmul(w);
        var q = transposeY.mmul(t1);
        q.div(norm(q));
        u = trainingValues.mmul(q);
      }

      t = t1;
      var num = transposeX.mmul(t);
      var den = t.transpose().mmul(t).get(0, 0);
      var p = num.div(den);
      var pnorm = norm(p);
      p.div(pnorm);
      t.mul(pnorm);
      w.mul(pnorm);

      num = u.transpose().mmul(t);
      den = t.transpose().mmul(t).get(0, 0);
      var b = num.div(den).get(0, 0);
      trainingSet.sub(t.mmul(p.transpose()));
      trainingValues.sub(t.clone().mul(b).mmul(q.transpose()));

      T.setColumn(k, t);
      P.setColumn(k, p);
      U.setColumn(k, u);
      Q.setColumn(k, q);
      W.setColumn(k, w);

      B.set(k, k, b);
      k++;
    }

    k--;
    T = T.subMatrix(0, T.rows - 1, 0, k);
    P = P.subMatrix(0, P.rows - 1, 0, k);
    U = U.subMatrix(0, U.rows - 1, 0, k);
    Q = Q.subMatrix(0, Q.rows - 1, 0, k);
    W = W.subMatrix(0, W.rows - 1, 0, k);
    B = B.subMatrix(0, k, 0, k);

    // TODO: review of R2Y
    // this.R2Y = t.transpose().mmul(t).mul(q[k][0]*q[k][0]).divS(ssqYcal)[0][0];
    //
    this.ssqYcal = sumOfSquaresY;
    this.E = trainingSet;
    this.F = trainingValues;
    this.T = T;
    this.P = P;
    this.U = U;
    this.Q = Q;
    this.W = W;
    this.B = B;
    this.PBQ = P.mmul(B).mmul(Q.transpose());
    this.R2X = t.transpose().mmul(t).mmul(p.transpose().mmul(p)).div(ssqXcal).get(0, 0);
  }

  /**
   * Predicts the behavior of the given dataset.
   * @param {Matrix|Array} dataset - data to be predicted.
   * @return {Matrix} - predictions of each element of the dataset.
   */
  predict(dataset) {
    var X = Matrix__default.checkMatrix(dataset);
    if (this.scale) {
      X = X.subRowVector(this.meanX).divRowVector(this.stdDevX);
    }
    var Y = X.mmul(this.PBQ);
    Y = Y.mulRowVector(this.stdDevY).addRowVector(this.meanY);
    return Y;
  }

  /**
   * Returns the explained variance on training of the PLS model
   * @return {number}
   */
  getExplainedVariance() {
    return this.R2X;
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    return {
      name: 'PLS',
      R2X: this.R2X,
      meanX: this.meanX,
      stdDevX: this.stdDevX,
      meanY: this.meanY,
      stdDevY: this.stdDevY,
      PBQ: this.PBQ,
      tolerance: this.tolerance,
      scale: this.scale,
    };
  }

  /**
   * Load a PLS model from a JSON Object
   * @param {object} model
   * @return {PLS} - PLS object from the given model
   */
  static load(model) {
    if (model.name !== 'PLS') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }
    return new PLS(true, model);
  }
}

/**
 * @private
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} data
 * @return {number} index of the maximum
 */
function maxSumColIndex(data) {
  return Matrix__default.rowVector(data.sum('column')).maxIndex()[0];
}

/**
 * @class KOPLS
 */
class KOPLS {
  /**
     * Constructor for Kernel-based Orthogonal Projections to Latent Structures (K-OPLS)
     * @param {object} options
     * @param {number} [options.predictiveComponents] - Number of predictive components to use.
     * @param {number} [options.orthogonalComponents] - Number of Y-Orthogonal components.
     * @param {Kernel} [options.kernel] - Kernel object to apply, see [ml-kernel](https://github.com/mljs/kernel).
     * @param {object} model - for load purposes.
     */
  constructor(options, model) {
    if (options === true) {
      this.trainingSet = new Matrix.Matrix(model.trainingSet);
      this.YLoadingMat = new Matrix.Matrix(model.YLoadingMat);
      this.SigmaPow = new Matrix.Matrix(model.SigmaPow);
      this.YScoreMat = new Matrix.Matrix(model.YScoreMat);
      this.predScoreMat = initializeMatrices(model.predScoreMat, false);
      this.YOrthLoadingVec = initializeMatrices(model.YOrthLoadingVec, false);
      this.YOrthEigen = model.YOrthEigen;
      this.YOrthScoreMat = initializeMatrices(model.YOrthScoreMat, false);
      this.toNorm = initializeMatrices(model.toNorm, false);
      this.TURegressionCoeff = initializeMatrices(model.TURegressionCoeff, false);
      this.kernelX = initializeMatrices(model.kernelX, true);
      this.kernel = model.kernel;
      this.orthogonalComp = model.orthogonalComp;
      this.predictiveComp = model.predictiveComp;
    } else {
      if (options.predictiveComponents === undefined) {
        throw new RangeError('no predictive components found!');
      }
      if (options.orthogonalComponents === undefined) {
        throw new RangeError('no orthogonal components found!');
      }
      if (options.kernel === undefined) {
        throw new RangeError('no kernel found!');
      }

      this.orthogonalComp = options.orthogonalComponents;
      this.predictiveComp = options.predictiveComponents;
      this.kernel = options.kernel;
    }
  }

  /**
     * Train the K-OPLS model with the given training set and labels.
     * @param {Matrix|Array} trainingSet
     * @param {Matrix|Array} trainingValues
     */
  train(trainingSet, trainingValues) {
    trainingSet = Matrix.Matrix.checkMatrix(trainingSet);
    trainingValues = Matrix.Matrix.checkMatrix(trainingValues);

    // to save and compute kernel with the prediction dataset.
    this.trainingSet = trainingSet.clone();

    var kernelX = this.kernel.compute(trainingSet);

    var Identity = Matrix.Matrix.eye(kernelX.rows, kernelX.rows, 1);
    var temp = kernelX;
    kernelX = new Array(this.orthogonalComp + 1);
    for (let i = 0; i < this.orthogonalComp + 1; i++) {
      kernelX[i] = new Array(this.orthogonalComp + 1);
    }
    kernelX[0][0] = temp;

    var result = new Matrix.SingularValueDecomposition(trainingValues.transpose().mmul(kernelX[0][0]).mmul(trainingValues), {
      computeLeftSingularVectors: true,
      computeRightSingularVectors: false
    });
    var YLoadingMat = result.leftSingularVectors;
    var Sigma = result.diagonalMatrix;

    YLoadingMat = YLoadingMat.subMatrix(0, YLoadingMat.rows - 1, 0, this.predictiveComp - 1);
    Sigma = Sigma.subMatrix(0, this.predictiveComp - 1, 0, this.predictiveComp - 1);

    var YScoreMat = trainingValues.mmul(YLoadingMat);

    var predScoreMat = new Array(this.orthogonalComp + 1);
    var TURegressionCoeff = new Array(this.orthogonalComp + 1);
    var YOrthScoreMat = new Array(this.orthogonalComp);
    var YOrthLoadingVec = new Array(this.orthogonalComp);
    var YOrthEigen = new Array(this.orthogonalComp);
    var YOrthScoreNorm = new Array(this.orthogonalComp);

    var SigmaPow = Matrix.Matrix.pow(Sigma, -0.5);
    // to avoid errors, check infinity
    SigmaPow.apply(function (i, j) {
      if (this.get(i, j) === Infinity) {
        this.set(i, j, 0);
      }
    });

    for (var i = 0; i < this.orthogonalComp; ++i) {
      predScoreMat[i] = kernelX[0][i].transpose().mmul(YScoreMat).mmul(SigmaPow);

      var TpiPrime = predScoreMat[i].transpose();
      TURegressionCoeff[i] = Matrix.inverse(TpiPrime.mmul(predScoreMat[i])).mmul(TpiPrime).mmul(YScoreMat);

      result = new Matrix.SingularValueDecomposition(TpiPrime.mmul(Matrix.Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime))).mmul(predScoreMat[i]), {
        computeLeftSingularVectors: true,
        computeRightSingularVectors: false
      });
      var CoTemp = result.leftSingularVectors;
      var SoTemp = result.diagonalMatrix;

      YOrthLoadingVec[i] = CoTemp.subMatrix(0, CoTemp.rows - 1, 0, 0);
      YOrthEigen[i] = SoTemp.get(0, 0);

      YOrthScoreMat[i] = Matrix.Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime)).mmul(predScoreMat[i]).mmul(YOrthLoadingVec[i]).mul(Math.pow(YOrthEigen[i], -0.5));

      var toiPrime = YOrthScoreMat[i].transpose();
      YOrthScoreNorm[i] = Matrix.Matrix.sqrt(toiPrime.mmul(YOrthScoreMat[i]));

      YOrthScoreMat[i] = YOrthScoreMat[i].divRowVector(YOrthScoreNorm[i]);

      var ITo = Matrix.Matrix.sub(Identity, YOrthScoreMat[i].mmul(YOrthScoreMat[i].transpose()));

      kernelX[0][i + 1] = kernelX[0][i].mmul(ITo);
      kernelX[i + 1][i + 1] = ITo.mmul(kernelX[i][i]).mmul(ITo);
    }

    var lastScoreMat = predScoreMat[this.orthogonalComp] = kernelX[0][this.orthogonalComp].transpose().mmul(YScoreMat).mmul(SigmaPow);

    var lastTpPrime = lastScoreMat.transpose();
    TURegressionCoeff[this.orthogonalComp] = Matrix.inverse(lastTpPrime.mmul(lastScoreMat)).mmul(lastTpPrime).mmul(YScoreMat);

    this.YLoadingMat = YLoadingMat;
    this.SigmaPow = SigmaPow;
    this.YScoreMat = YScoreMat;
    this.predScoreMat = predScoreMat;
    this.YOrthLoadingVec = YOrthLoadingVec;
    this.YOrthEigen = YOrthEigen;
    this.YOrthScoreMat = YOrthScoreMat;
    this.toNorm = YOrthScoreNorm;
    this.TURegressionCoeff = TURegressionCoeff;
    this.kernelX = kernelX;
  }

  /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix|Array} toPredict
     * @return {{y: Matrix, predScoreMat: Array<Matrix>, predYOrthVectors: Array<Matrix>}} predictions
     */
  predict(toPredict) {
    var KTestTrain = this.kernel.compute(toPredict, this.trainingSet);

    var temp = KTestTrain;
    KTestTrain = new Array(this.orthogonalComp + 1);
    for (let i = 0; i < this.orthogonalComp + 1; i++) {
      KTestTrain[i] = new Array(this.orthogonalComp + 1);
    }
    KTestTrain[0][0] = temp;

    var YOrthScoreVector = new Array(this.orthogonalComp);
    var predScoreMat = new Array(this.orthogonalComp);

    var i;
    for (i = 0; i < this.orthogonalComp; ++i) {
      predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);

      YOrthScoreVector[i] = Matrix.Matrix.sub(KTestTrain[i][i], predScoreMat[i].mmul(this.predScoreMat[i].transpose())).mmul(this.predScoreMat[i]).mmul(this.YOrthLoadingVec[i]).mul(Math.pow(this.YOrthEigen[i], -0.5));

      YOrthScoreVector[i] = YOrthScoreVector[i].divRowVector(this.toNorm[i]);

      var scoreMatPrime = this.YOrthScoreMat[i].transpose();
      KTestTrain[i + 1][0] = Matrix.Matrix.sub(KTestTrain[i][0], YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[0][i].transpose()));

      var p1 = Matrix.Matrix.sub(KTestTrain[i][0], KTestTrain[i][i].mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime));
      var p2 = YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[i][i]);
      var p3 = p2.mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime);

      KTestTrain[i + 1][i + 1] = p1.sub(p2).add(p3);
    }

    predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);
    var prediction = predScoreMat[i].mmul(this.TURegressionCoeff[i]).mmul(this.YLoadingMat.transpose());

    return {
      prediction: prediction,
      predScoreMat: predScoreMat,
      predYOrthVectors: YOrthScoreVector
    };
  }

  /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
  toJSON() {
    return {
      name: 'K-OPLS',
      YLoadingMat: this.YLoadingMat,
      SigmaPow: this.SigmaPow,
      YScoreMat: this.YScoreMat,
      predScoreMat: this.predScoreMat,
      YOrthLoadingVec: this.YOrthLoadingVec,
      YOrthEigen: this.YOrthEigen,
      YOrthScoreMat: this.YOrthScoreMat,
      toNorm: this.toNorm,
      TURegressionCoeff: this.TURegressionCoeff,
      kernelX: this.kernelX,
      trainingSet: this.trainingSet,
      orthogonalComp: this.orthogonalComp,
      predictiveComp: this.predictiveComp
    };
  }

  /**
     * Load a K-OPLS with the given model.
     * @param {object} model
     * @param {Kernel} kernel - kernel used on the model, see [ml-kernel](https://github.com/mljs/kernel).
     * @return {KOPLS}
     */
  static load(model, kernel) {
    if (model.name !== 'K-OPLS') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }

    if (!kernel) {
      throw new RangeError('You must provide a kernel for the model!');
    }

    model.kernel = kernel;
    return new KOPLS(true, model);
  }
}

let Utils = {};
Utils.norm = function norm(X) {
  return Math.sqrt(X.clone().apply(pow2array).sum());
};

/**
 * OPLS loop
 * @param {Array} x a matrix with features
 * @param {Array} y an array of labels (dependent variable)
 * @param {Object} options an object with options
 * @return {Object} an object with model (filteredX: err,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXOrtho: wOrtho,
    weightsPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY:)
 */
function oplsNIPALS(x, y, options = {}) {
  const {
    numberOSC = 100,
  } = options;

  let X = Matrix__default.checkMatrix(x.clone());
  let Y = Matrix__default.checkMatrix(y.clone());

  let u = Y.getColumnVector(0);

  let diff = 1;
  let t, c, w, uNew;
  for (let i = 0; i < numberOSC && diff > 1e-10; i++) {
    w = u.transpose().mmul(X).div(u.transpose().mmul(u).get(0, 0));
    w = w.transpose().div(norm(w));

    t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));// t_h paso 3

    // calc loading
    c = t.transpose().mmul(Y).div(t.transpose().mmul(t).get(0, 0));

    // calc new u and compare with one in previus iteration (stop criterion)
    uNew = Y.mmul(c.transpose());
    uNew = uNew.div(c.transpose().mmul(c).get(0, 0));

    if (i > 0) {
      diff = uNew.clone().sub(u).pow(2).sum() / uNew.clone().pow(2).sum();
    }

    u = uNew.clone();
  }

  // calc loadings
  let p = t.transpose().mmul(X).div(t.transpose().mmul(t).get(0, 0));

  let wOrtho = p.clone().sub(w.transpose().mmul(p.transpose()).div(w.transpose().mmul(w).get(0, 0)).mmul(w.transpose()));
  wOrtho.div(Utils.norm(wOrtho));

  // orthogonal scores
  let tOrtho = X.mmul(wOrtho.transpose()).div(wOrtho.mmul(wOrtho.transpose()).get(0, 0));

  // orthogonal loadings
  let pOrtho = tOrtho.transpose().mmul(X).div(tOrtho.transpose().mmul(tOrtho).get(0, 0));

  // filtered data
  let err = X.clone().sub(tOrtho.mmul(pOrtho));
  return { filteredX: err,
    weightsXOrtho: wOrtho,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY: c };
}

class OPLS {
  constructor(features, labels, options = {}) {
    if (features === true) {
      const opls = options;
      this.center = opls.center;
      this.scale = opls.scale;
      this.means = opls.means;
      this.stdevs = opls.stdevs;
      this.model = opls.model;
      this.tCV = this.tCV;
      this.tOrthCV = this.tOrthCV;
      return;
    }

    const {
      nComp = 3,
      center = true,
      scale = true,
      cvFolds = [],
    } = options;

    this.center = center;
    if (this.center) {
      this.means = features.mean('column');
    } else {
      this.stdevs = null;
    }
    this.scale = scale;
    if (this.scale) {
      this.stdevs = features.standardDeviation('column');
    } else {
      this.means = null;
    }

    if (typeof (labels[0]) === 'number') {
      console.warn('numeric labels: OPLS regression is used');
      var group = Matrix.Matrix
        .from1DArray(labels.length, 1, labels);
    } else if (typeof (labels[0]) === 'string') {
      console.warn('non-numeric labels: OPLS-DA is used');
    }

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    let folds;
    if (cvFolds.length > 0) {
      folds = cvFolds;
    } else {
      folds = getFolds(labels, 5);
    }

    let filteredXCV = [];
    let modelNC = [];
    let Q2 = [];
    let oplsNC = [];

    this.tCV = [];
    this.tOrthCV = [];
    this.model = [];

    for (var nc = 0; nc < nComp; nc++) {
      let yHatCV = new Matrix.Matrix(group.rows, 1);
      let tPredCV = new Matrix.Matrix(group.rows, 1);
      let scoresCV = new Matrix.Matrix(group.rows, 1);
      let oplsCV = [];

      let fold = 0;
      for (let f of folds) {
        let trainTest = this._getTrainTest(features, group, f);
        let testXk = trainTest.testFeatures;
        let Xk = trainTest.trainFeatures;
        let Yk = trainTest.trainLabels;

        // determine center and scale of training set
        let dataCenter = Xk.mean('column');
        let dataSD = Xk.standardDeviation('column');

        // center and scale training set
        if (center) {
          Xk.center('column');
          Yk.center('column');
        }

        if (scale) {
          Xk.scale('column');
          Yk.scale('column');
        }

        if (nc === 0) {
          oplsCV[fold] = oplsNIPALS(Xk, Yk);
        } else {
          oplsCV[fold] = oplsNIPALS(filteredXCV[fold], Yk);
        }
        filteredXCV[fold] = oplsCV[fold].filteredX;
        oplsNC[nc] = oplsCV;

        let plsCV = new Matrix.NIPALS(oplsCV[fold].filteredX, { Y: Yk });

        // scaling the test dataset with respect to the train
        testXk.center('column', { center: dataCenter });
        testXk.scale('column', { scale: dataSD });

        let Eh = testXk;
        // removing the orthogonal components from PLS
        let scores;
        for (let idx = 0; idx < nc + 1; idx++) {
          scores = Eh.clone().mmul(oplsNC[idx][fold].weightsXOrtho.transpose()); // ok
          Eh.sub(scores.clone().mmul(oplsNC[idx][fold].loadingsXOrtho));
        }

        // prediction
        let tPred = Eh.clone().mmul(plsCV.w.transpose());
        // this should be summed over ncomp (pls_prediction.R line 23)
        let yHat = tPred.clone().mmul(plsCV.betas); // ok

        // adding all prediction from all folds
        for (let i = 0; i < f.testIndex.length; i++) {
          yHatCV.setRow(f.testIndex[i], [yHat.get(i, 0)]);
          tPredCV.setRow(f.testIndex[i], [tPred.get(i, 0)]);
          scoresCV.setRow(f.testIndex[i], [scores.get(i, 0)]);
        }
        fold++;
      } // end of loop over folds

      this.tCV.push(tPredCV);
      this.tOrthCV.push(scoresCV);

      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implemented (check opls.R line 183) TODO
      let tssy = tss(group.center('column').scale('column'));
      let press = tss(group.clone().sub(yHatCV));
      let Q2y = 1 - (press / tssy);
      Q2.push(Q2y); // ok

      // calculate the R2y for the complete data
      if (nc === 0) {
        modelNC = this._predictAll(features, group);
      } else {
        modelNC = this._predictAll(modelNC.xRes,
          group,
          options = { scale: false, center: false });
      }
      // Deflated matrix for next compoment
      // Let last pls model for output

      modelNC.Q2y = Q2;
      this.model.push(modelNC);
      console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
    } // end of loop over nc

    // store scores from CV
    let tCV = this.tCV;
    let tOrthCV = this.tOrthCV;

    let m = this.model[nc - 1];
    let XOrth = m.XOrth;
    let FeaturesCS = features.center('column').scale('column');
    let labelsCS = group.center('column').scale('column');
    let Xres = FeaturesCS.clone().sub(XOrth);
    let plsCall = new Matrix.NIPALS(Xres, { Y: labelsCS });
    let E = Xres.clone().sub(plsCall.t.clone().mmul(plsCall.p));

    let R2x = this.model.map((x) => x.R2x);
    let R2y = this.model.map((x) => x.R2y);

    this.output = { Q2y: Q2, // ok
      R2x, // ok
      R2y, // ok
      tPred: m.plsC.t,
      pPred: m.plsC.p,
      wPred: m.plsC.w,
      betasPred: m.plsC.betas,
      Qpc: m.plsC.q,
      tCV, // ok
      tOrthCV, // ok
      tOrth: m.tOrth,
      pOrth: m.pOrth,
      wOrth: m.wOrth,
      XOrth,
      Yres: m.plsC.yResidual,
      E };
  }

  /**
   * get access to all the computed elements
   * Mainly for debug and testing
   * @return {Object} output object
   */
  getResults() {
    return this.output;
  }

  getScores() {
    let scoresX = this.tCV.map((x) => x.to1DArray());
    let scoresY = this.tOrthCV.map((x) => x.to1DArray());
    return { scoresX, scoresY };
  }

  /**
   * Load an OPLS model from JSON
   * @param {Object} model
   * @return {OPLS}
   */
  static load(model) {
    if (typeof model.name !== 'string') {
      throw new TypeError('model must have a name property');
    }
    if (model.name !== 'OPLS') {
      throw new RangeError(`invalid model: ${model.name}`);
    }
    return new OPLS(true, [], model);
  }

  /**
   * Export the current model to a JSON object
   * @return {Object} model
   */
  toJSON() {
    return {
      name: 'OPLS',
      center: this.center,
      scale: this.scale,
      means: this.means,
      stdevs: this.stdevs,
      model: this.model,
      tCV: this.tCV,
      tOrthCV: this.tOrthCV
    };
  }

  /**
   * Predict scores for new data
   * @param {Array} features a double array with X matrix
   * @param {Object} [options]
   * @param {Array} [options.trueLabel] an array with true values to compute confusion matrix
   * @param {Number} [options.nc] the number of components to be used
   * @return {Object} predictions
   */
  predict(features, options = {}) {
    var { trueLabels = [], nc = 1 } = options;
    let confusion = false;
    if (trueLabels.length > 0) {
      trueLabels = Matrix.Matrix.from1DArray(150, 1, trueLabels);
      confusion = true;
    }

    // scaling the test dataset with respect to the train
    if (this.center) {
      features.center('column', { center: this.means });
      if (confusion) {
        trueLabels.center('column', { center: this.means });
      }
    }
    if (this.scale) {
      features.scale('column', { scale: this.stdevs });
      if (confusion) {
        trueLabels.scale('column', { center: this.means });
      }
    }

    let Eh = features;
    // removing the orthogonal components from PLS
    let tOrth;
    let wOrth;
    let pOrth;
    let yHat;
    let tPred;

    for (let idx = 0; idx < nc; idx++) {
      wOrth = this.model[idx].wOrth.transpose();
      pOrth = this.model[idx].pOrth;
      tOrth = Eh.clone().mmul(wOrth);
      Eh.sub(tOrth.clone().mmul(pOrth));
      // prediction
      tPred = Eh.clone().mmul(this.model[idx].plsC.w.transpose());
      // this should be summed over ncomp (pls_prediction.R line 23)
      yHat = tPred.clone().mmul(this.model[idx].plsC.betas);
    }
    let confusionMatrix = [];
    if (confusion) {
      confusionMatrix = ConfusionMatrix
        .fromLabels(trueLabels.to1DArray(), yHat.to1DArray());
    }
    return { tPred,
      tOrth,
      yHat,
      confusionMatrix };
  }

  _predictAll(features, labels, options = {}) {
    // cannot use the global this.center here
    // since it is used in the NC loop and
    // centering and scaling should only be
    // performed once
    const { center = true,
      scale = true } = options;

    if (center) {
      features.center('column');
      labels.center('column');
    }

    if (scale) {
      features.scale('column');
      labels.scale('column');
      // reevaluate tssy and tssx after scaling
      this.tssy = tss(labels);
      this.tssx = tss(features);
    }

    let oplsC = oplsNIPALS(features, labels);
    let plsC = new Matrix.NIPALS(oplsC.filteredX, { Y: labels });

    let tPred = oplsC.filteredX.clone().mmul(plsC.w.transpose());
    let yHat = tPred.clone().mmul(plsC.betas);

    let rss = tss(labels.clone().sub(yHat));
    let R2y = 1 - (rss / this.tssy);

    let xEx = plsC.t.clone().mmul(plsC.p.clone());
    let rssx = tss(xEx);
    let R2x = (rssx / this.tssx);

    return { R2y,
      R2x,
      xRes: oplsC.filteredX,
      tOrth: oplsC.scoresXOrtho,
      pOrth: oplsC.loadingsXOrtho,
      wOrth: oplsC.weightsXOrtho,
      tPred: tPred,
      totalPred: yHat,
      XOrth: oplsC.scoresXOrtho.clone().mmul(oplsC.loadingsXOrtho),
      oplsC,
      plsC };
  }

  _getTrainTest(X, group, index) {
    let testFeatures = new Matrix.Matrix(index.testIndex.length, X.columns);
    let testLabels = new Matrix.Matrix(index.testIndex.length, 1);
    index.testIndex.forEach((el, idx) => {
      testFeatures.setRow(idx, X.getRow(el));
      testLabels.setRow(idx, group.getRow(el));
    });

    let trainFeatures = new Matrix.Matrix(index.trainIndex.length, X.columns);
    let trainLabels = new Matrix.Matrix(index.trainIndex.length, 1);
    index.trainIndex.forEach((el, idx) => {
      trainFeatures.setRow(idx, X.getRow(el));
      trainLabels.setRow(idx, group.getRow(el));
    });

    return ({
      trainFeatures,
      testFeatures,
      trainLabels,
      testLabels
    });
  }
}

exports.KOPLS = KOPLS;
exports.OPLS = OPLS;
exports.PLS = PLS;
