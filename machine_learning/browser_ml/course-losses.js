/* TensorFlow.js loss adapters used by the original Keras course.
 * All operations run within the active TF.js backend, including WebGPU.
 * Keep logits as logits; NEVER run softmax twice.
 */
(function (scope) {
  "use strict";
  const tf = scope.tf;
  if (!tf) throw new Error("Load TensorFlow.js before course-losses.js");

  function oneHot(yTrue, yPred) {
    const classes = yPred.shape[yPred.rank - 1];
    if (!Number.isInteger(classes) || classes < 2)
      throw new Error("Sparse categorical crossentropy requires >=2 classes");
    if (yTrue.rank === yPred.rank &&
        yTrue.shape.every((size, index) => size === yPred.shape[index])) {
      return yTrue;
    }
    return tf.oneHot(yTrue.flatten().toInt(), classes).reshape(yPred.shape);
  }

  // TF.js model.compile(loss=fn) passes truth and predictions as Tensor values.
  function sparseCategoricalCrossentropyFromLogits(yTrue, yPred) {
    return tf.tidy(() => tf.losses.softmaxCrossEntropy(oneHot(yTrue, yPred), yPred));
  }

  function sparseCategoricalCrossentropyProbabilities(yTrue, yPred) {
    return tf.tidy(() => {
      const probabilities = tf.clipByValue(yPred, 1e-7, 1.0);
      return tf.losses.softmaxCrossEntropy(oneHot(yTrue, yPred), tf.log(probabilities));
    });
  }

  function binaryCrossentropyFromLogits(yTrue, yPred) {
    return tf.tidy(() => tf.losses.sigmoidCrossEntropy(yTrue, yPred));
  }

  scope.mlCourseLosses = Object.freeze({
    sparseCategoricalCrossentropyFromLogits,
    sparseCategoricalCrossentropyProbabilities,
    binaryCrossentropyFromLogits,
  });
})(globalThis);
