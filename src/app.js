import MNISTData from 'mnist-data';
import _ from 'lodash';
import plot from 'node-remote-plot';

import LogisticRegression from './models/logistic-regression.js';

// Function to load and preprocess MNIST data
const loadData = () => {
  // Retrieve MNIST test & training data for digits 0 to 9
  console.log('\n\n[+] Retrieving MNIST Data...');
  const MNISTTrainingData = MNISTData.training(0, 60000);
  const MNISTTestData = MNISTData.testing(0, 10000);

  // Flatten image data: convert 2D array of images to 1D array of pixel intensities
  console.log('[+] Flattening Image Data...');
  const flattenedFeatures = MNISTTrainingData.images.values.map((image) => _.flatMap(image));
  const flattenedTestFeatures = MNISTTestData.images.values.map((image) => _.flatMap(image));

  // Encode labels: convert numeric labels (0-9) to one-hot encoded format
  console.log('[+] Encoding Labels...');
  const encodedLabels = MNISTTrainingData.labels.values.map((label) => {
    const encodedLabel = new Array(10).fill(0); // Create an array for one-hot encoding
    encodedLabel[label] = 1; // Set the index representing the label to 1 (one-hot encoding)
    return encodedLabel;
  });

  const encodedTestLabels = MNISTTestData.labels.values.map((label) => {
    const encodedLabel = new Array(10).fill(0); // Create an array for one-hot encoding
    encodedLabel[label] = 1; // Set the index representing the label to 1 (one-hot encoding)
    return encodedLabel;
  });

  // Return the loaded and preprocessed data
  return {
    features: flattenedFeatures,
    labels: encodedLabels,
    testFeatures: flattenedTestFeatures,
    testLabels: encodedTestLabels,
  };
};

// Load and preprocess the MNIST data
const { features, labels, testFeatures, testLabels } = loadData();

console.log('[+] Initializing Logistic Regression...');

// Initialize Logistic Regression model with preprocessed data
const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 300,
});

console.log('[+] Initiating Regression Training...');

// Train the Logistic Regression model
regression.train();

console.log('[+] Calculating Model Accuracy...');

// Test the trained model and calculate its accuracy
const modelAccuracy = regression.test(testFeatures, testLabels);
console.log(`[+] Model Accuracy: ${modelAccuracy * 100}%\n\n`);

// Plot cost history for visualization
plot({
  x: regression.costHistory.reverse(),
  name: 'cost_history',
  title: 'Cost History',
  xLabel: 'No of Iterations #',
  yLabel: 'Cost',
});
