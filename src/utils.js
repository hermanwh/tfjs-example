import * as tf from "@tensorflow/tfjs";

import { shuffle } from "simple-statistics";

// Because the data from our react-csv-reader as a rather strange format,
// we have to parse it a bit before it can be
export function getDatasetByColumns(dataset) {
  const numberOfColumns = dataset[0].length;
  const columnsData = [];
  for (var i = 0; i < numberOfColumns; i++) {
    const column = dataset.map(x => x[i]);
    columnsData.push(column);
  }
  return columnsData;
}

export function getR2Score(predict, data) {
  data = data.map(x => Number(x));
  predict = predict.map(x => Number(x));

  var meanValue = 0;
  var SStot = 0;
  var SSres = 0;
  var rSquared = 0;

  for (var n = 0; n < data.length; n++) {
    meanValue += data[n];
  }
  meanValue = meanValue / data.length;

  for (var m = 0; m < data.length; m++) {
    SStot += Math.pow(data[m] - meanValue, 2);
    SSres += Math.pow(predict[m] - data[m], 2);
  }

  rSquared = 1 - SSres / SStot;

  return {
    meanValue: meanValue,
    SStot: SStot,
    SSres: SSres,
    rSquared: rSquared
  };
}

export function preprocessRemoveEmptyAndNull(data) {
  data = data.filter(x => !Object.values(x).some(y => y === "" || y == null));
  return data;
}

export function getFeatureTargetSplit(dataset, config) {
  const inputs = config.input;
  const targets = dataset.map(x => [Number(x[config.output[0]])]);
  let features = [];
  dataset.forEach(function(dataRow) {
    let row = [];
    inputs.forEach(function(inputName) {
      row.push(Number(dataRow[inputName]));
    });
    features.push(row);
  });
  return [features, targets];
}

export function getTestTrainSplit(features, targets, test_train_split) {
  const numberOfRows = features.length;
  const numberOfTest = Math.round(numberOfRows * test_train_split);
  const numberOfTrain = numberOfRows - numberOfTest;

  const x_train = features.slice(0, numberOfTrain - 1);
  const x_test = features.slice(numberOfTrain - 1);
  const y_train = targets.slice(0, numberOfTrain - 1);
  const y_test = targets.slice(numberOfTrain - 1);
  return [x_train, x_test, y_train, y_test];
}

export function convertToTensors(x_train, x_test, y_train, y_test) {
  const tensors = {};
  tensors.trainFeatures = tf.tensor2d(x_train);
  tensors.trainTargets = tf.tensor2d(y_train);
  tensors.testFeatures = tf.tensor2d(x_test);
  tensors.testTargets = tf.tensor2d(y_test);
  return tensors;
}

export function preprocess(data, sensorConfig, modelParams) {
  let newData = refactorRawData(data);
  newData = preprocessRemoveEmptyAndNull(newData);
  newData = shuffle(newData);
  let [features, targets] = getFeatureTargetSplit(newData, sensorConfig);
  return getTestTrainSplit(features, targets, modelParams.test_train_split);
}

export function refactorRawData(data) {
  let newData = [];
  const headers = data[0];
  data.slice(1, -1).forEach(function(values, index) {
    let result = {};
    headers.forEach((key, i) => (result[key] = values[i]));
    newData.push(result);
  });
  return newData;
}

export default convertToTensors;
