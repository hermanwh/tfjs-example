import * as tf from "@tensorflow/tfjs";

import {
  shuffle,
  max,
  min,
  mean,
  standardDeviation,
  sampleCorrelation
} from "simple-statistics";

export function getDatasetByColumns(dataset) {
  const numberOfColumns = dataset[0].length;
  const columnsData = [];
  for (var i = 0; i < numberOfColumns; i++) {
    const column = dataset.map(x => x[i]);
    columnsData.push(column);
  }
  return columnsData;
}

export function getCovarianceMatrix(dataset) {
  const columnData = getDatasetByColumns(dataset);
  const numberOfColumns = columnData.length;
  const covariances = [];
  for (var i = 0; i < numberOfColumns; i++) {
    const covariances_column_i = [];
    for (var j = 0; j < numberOfColumns; j++) {
      covariances_column_i.push(
        sampleCorrelation(columnData[i], columnData[j])
      );
    }
    covariances.push(covariances_column_i);
  }
  return covariances;
}

export function standardizeData(data) {
  const numberOfColumns = data[0].length;
  const numberOfRows = data.length;
  let meanvals = [];
  let stdvals = [];
  for (var k = 0; k < numberOfColumns; k++) {
    const col = data.map(x => x[k]);
    meanvals.push(mean(col));
    stdvals.push(standardDeviation(col));
  }
  const standardized = [];
  for (var i = 0; i < numberOfRows; i++) {
    const row = [];
    for (var j = 0; j < numberOfColumns; j++) {
      row.push((data[i][j] - meanvals[j]) / stdvals[j]);
    }
    standardized.push(row);
  }
  return standardized;
}

export function normalizeData(data) {
  const numberOfColumns = data[0].length;
  const numberOfRows = data.length;
  let maxvals = [];
  let minvals = [];
  for (var k = 0; k < numberOfColumns; k++) {
    const col = data.map(x => x[k]);
    maxvals.push(max(col));
    minvals.push(min(col));
  }
  const normalized = [];
  for (var i = 0; i < numberOfRows; i++) {
    const row = [];
    for (var j = 0; j < numberOfColumns; j++) {
      row.push((data[i][j] - minvals[j]) / (maxvals[j] - minvals[j]));
    }
    normalized.push(row);
  }
  return normalized;
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

export function shuffleData(data) {
  return shuffle(data);
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
  newData = shuffleData(newData);
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

export function getSequentialModel(
  numberOfUnits,
  inputSize,
  outputSize,
  activation,
  outputActivation
) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      kernelRegularizer: tf.regularizers.L1L2,
      units: numberOfUnits[0],
      activation: activation,
      inputShape: [inputSize]
    })
  );
  numberOfUnits.slice(1).forEach(layerUnits => {
    model.add(
      tf.layers.dense({
        kernelRegularizer: tf.regularizers.L1L2,
        units: layerUnits,
        activation: activation,
        inputShape: [layerUnits]
      })
    );
  });
  model.add(
    tf.layers.dense({ units: outputSize, activation: outputActivation })
  );
  return model;
}

export default convertToTensors;
