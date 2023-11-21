import MNISTData from 'mnist-data';
import _ from 'lodash';

import LogisticRegression from './models/logistic-regression.js';

// Retrieve MNIST test & training data for digits 0 to 9
console.log('\n\n[+] Retrieving MNIST Data...');
const MNISTTrainingData = MNISTData.training(0, 60000);
const MNISTTestData = MNISTData.testing(0, 1500);

// Flatten image data: convert 2D array of ima s to 1D array of pixel intensities
console.log('[+] Flattening Image Data...');
const flattenedFeatures = MNISTTrainingData.images.values.map((image) => _.flatMap(image));
const flattenedTestFeatures = MNISTTestData.images.values.map((image) => _.flatMap(image));

// Encode labels: convert numeric labels (0-9) to one-hot encoded format
console.log('[+] Encoding Labels...');
const encodedLabels = MNISTTrainingData.labels.values.map((label) => {
  // Create an array of zeros with length equal to the number of possible labels (10 for digits 0-9)
  const encodedLabel = new Array(10).fill(0);
  // Set the index representing the label to 1 (one-hot encoding)
  encodedLabel[label] = 1;
  return encodedLabel;
});

const encodedTestLabels = MNISTTestData.labels.values.map((label) => {
  // Create an array of zeros with length equal to the number of possible labels (10 for digits 0-9)
  const encodedLabel = new Array(10).fill(0);
  // Set the index representing the label to 1 (one-hot encoding)
  encodedLabel[label] = 1;
  return encodedLabel;
});

console.log('[+] Initializing Logistic Regression...');

const regression = new LogisticRegression(flattenedFeatures, encodedLabels, {
  learningRate: 1,
  iterations: 50,
  batchSize: 300,
});

console.log('[+] Initiating Regression Training...');
regression.train();

console.log('[+] Calculating Model Accuracy...');
const modelAccuracy = regression.test(flattenedTestFeatures, encodedTestLabels);
console.log(`[+] Model Accuracy: ${modelAccuracy * 100}%\n\n`);
