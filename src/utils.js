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
  let SSres_tot = 0;
  let SStot_tot = 0;
  for (let i = 0; i < data[0].length; i++) {
    let d = data.map(x => Number(x[i]));
    let p = predict.map(x => Number(x[i]));

    let SStot = 0;
    let SSres = 0;

    const meanValue =
      d.reduce(function(pv, cv) {
        return pv + cv;
      }, 0) / d.length;

    for (let m = 0; m < d.length; m++) {
      SStot += Math.pow(d[m] - meanValue, 2);
      SSres += Math.pow(p[m] - d[m], 2);
    }

    SSres_tot += SSres;
    SStot_tot += SStot;
  }

  let rSquared = 1 - SSres_tot / SStot_tot;

  return {
    SStot: SSres_tot,
    SSres: SStot_tot,
    rSquared: rSquared
  };
}

export function preprocessRemoveEmptyAndNull(data) {
  data = data.filter(x => !Object.values(x).some(y => y === "" || y == null));
  return data;
}

export function getFeatureTargetSplit(dataset, config) {
  const inputs = config.input;
  const outputs = config.output;
  let features = [];
  let targets = [];
  dataset.forEach(function(dataRow) {
    let featureRow = [];
    inputs.forEach(function(inputName) {
      featureRow.push(Number(dataRow[inputName]));
    });
    features.push(featureRow);

    let targetRow = [];
    outputs.forEach(function(outputName) {
      targetRow.push(Number(dataRow[outputName]));
    });
    targets.push(targetRow);
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
