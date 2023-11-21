import {
  tensor,
  tidy,
  zeros,
  ones,
  matMul,
  sub,
  transpose,
  moments,
  Tensor,
} from '@tensorflow/tfjs-node';
import _ from 'lodash';

/**
 * Logistic Regression Class
 * @class
 */
export default class LogisticRegression {
  /**
   * Constructor for Logistic Regression class
   * @constructor
   * @param {number[][]} features - Array of features
   * @param {number[]} labels - Array of labels
   * @param {object} options - Options for logistic regression
   */
  constructor(features, labels, options) {
    // Process features and labels
    this.features = this.processFeatures(tensor(features));
    this.labels = tensor(labels);
    this.costHistory = [];

    // Set default options and overwrite if provided
    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      decisionBoundary: 0.5,
      ...options,
    };

    // Initialize weights
    this.weights = zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  /**
   * Gradient Descent to optimize weights
   * @param {Tensor} features - Features Tensor
   * @param {Tensor} labels - Labels Tensor
   * @returns {Tensor} - Updated weights
   */
  gradientDescent(features, labels) {
    // Make predictions, calculate differences and slopes
    const currentGuesses = matMul(features, this.weights).softmax();
    const differences = sub(currentGuesses, labels);
    const slopes = matMul(transpose(features), differences).div(features.shape[0]);

    // Update weights based on learning rate
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   * Train the logistic regression model
   */
  train() {
    // Determine batch quantity for training iterations
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    // Perform gradient descent and update weights
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        this.weights = tidy(() => {
          const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
          const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

          return this.gradientDescent(featureSlice, labelSlice);
        });
      }

      // Record cost and optimize learning rate
      this.recordCost();
      this.optimizeLearningRate();
    }
  }

  /**
   * Predict using trained logistic regression model
   * @param {number[][]} observations - Array of observations
   * @returns {Tensor} - Predicted labels
   */
  predict(observations) {
    // Process features and make predictions
    const processedFeatures = this.processFeatures(tensor(observations));
    return processedFeatures.matMul(this.weights).softmax().argMax(1);
  }

  /**
   * Test the logistic regression model
   * @param {number[][]} testFeatures - Array of test features
   * @param {number[]} testLabels - Array of test labels
   * @returns {number} - Accuracy of the model
   */
  test(testFeatures, testLabels) {
    // Make predictions and compare with test labels to calculate accuracy
    const predictions = this.predict(testFeatures);
    const processedLabels = tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(processedLabels).sum().arraySync();
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /**
   * Process features for standardization
   * @param {Tensor} features - Features Tensor
   * @returns {Tensor} - Processed features
   */
  processFeatures(features) {
    // Standardize features or process based on mean and variance
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // Add bias unit to features
    return ones([features.shape[0], 1]).concat(features, 1);
  }

  /**
   * Standardize features
   * @param {Tensor} features - Features Tensor
   * @returns {Tensor} - Standardized features
   */
  standardize(features) {
    // Calculate mean and variance, standardize features
    const { mean, variance } = moments(features, 0);
    const filler = variance.cast('bool').logicalNot().cast('float32');

    // Save mean and variance for later use
    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  /**
   * Record cost for analysis
   */
  recordCost() {
    // Calculate cost and add to cost history
    const cost = tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();

      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log());

      return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).arraySync()[0][0];
    });

    this.costHistory.unshift(cost);
  }

  /**
   * Optimize learning rate based on cost history
   */
  optimizeLearningRate() {
    // Update learning rate based on cost history trends
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}
