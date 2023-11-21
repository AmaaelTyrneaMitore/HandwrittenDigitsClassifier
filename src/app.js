import { training } from 'mnist-data';
import _ from 'lodash';

// Retrieve MNIST training data for digits 0 to 9
const MNISTTrainingData = training(0, 10);

// Flatten image data: convert 2D array of images to 1D array of pixel intensities
const flattenedFeatures = MNISTTrainingData.images.values.map((image) => _.flatMap(image));

// Encode labels: convert numeric labels (0-9) to one-hot encoded format
const encodedLabels = MNISTTrainingData.labels.values.map((label) => {
  // Create an array of zeros with length equal to the number of possible labels (10 for digits 0-9)
  const encodedLabel = new Array(10).fill(0);

  // Set the index representing the label to 1 (one-hot encoding)
  encodedLabel[label] = 1;

  return encodedLabel;
});

console.log(encodedLabels);
