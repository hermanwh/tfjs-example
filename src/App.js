import "./App.css";
import React, { useState } from "react";
import CSVReader from "react-csv-reader";
import AddSensor from "./AddSensor.js";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import {
  getR2Score,
  normalizeData,
  standardizeData,
  getCovarianceMatrix,
  getReducedDataset,
  shuffleData,
  fillConfigWithDataValues,
  shouldStandardize
} from "./statLib.js";

import {
  getFeatureTargetSplit,
  getTestTrainSplit,
  convertToTensors,
  getBasicModel,
  getComplexModel,
  getBasicModelWithRegularization,
  getComplexModelWithRegularization,
  preprocessData
} from "./utils.js";

const modelParams = {
  test_train_split: 0.2,
  activation: "relu",
  learningRate: 0.01,
  epochs: 10,
  optimizer: tf.train.adam(0.01),
  loss: "meanSquaredError",
  min_R2_score: 0.5,
  decent_R2_score: 0.8,
  max_mean_diff: 100,
  max_std_diff: 10,
  cov_limit: 0.9,
  max_iterations: 4
};

function App() {
  const [dataPoints, setDataPoints] = useState(null);
  const [sensorNames, setSensorNames] = useState(null);
  const [hasSelectedDataset, setHasSelectedDataset] = useState(false);
  const [sensorConfig, setSensorConfig] = useState(null);

  const [R2, setR2] = useState(-1000);
  const [trainingFailed, setTrainingFailed] = useState(false);
  const [hasTrained, setHasTrained] = useState(false);

  const selectDataset = data => {
    setHasSelectedDataset(true);
    setDataPoints(data);
    setSensorNames(data[0]);
  };

  console.log(sensorConfig);

  function addSensorFunc(sensor, type) {
    let config = [];
    let obj = {
      [sensor]: {
        name: sensor,
        type: type
      }
    };
    config["sensors"] =
      sensorConfig !== null ? { ...sensorConfig["sensors"], ...obj } : [];
    config["input"] = [];
    config["output"] = [];
    config["internal"] = [];
    Object.keys(config.sensors).forEach(x =>
      config[config.sensors[x].type].push(config.sensors[x].name)
    );
    setSensorConfig(config);
  }

  function organizeData(data) {
    let newData = [];
    const headers = data[0];
    console.log("headers: ", headers);
    data.slice(1, -1).forEach(function(values, index) {
      let result = {};
      headers.forEach((key, i) => (result[key] = values[i]));
      newData.push(result);
    });
    return newData;
  }

  async function performModelTraining() {
    console.log("Hei");
    console.log(dataPoints);

    let data = organizeData(dataPoints);

    console.log(data);

    fillConfigWithDataValues(data, sensorConfig);
    data = shuffleData(data);

    let [features, targets] = getFeatureTargetSplit(data, sensorConfig);

    console.log("features, processed", features);
    console.log("targets, processed", targets);

    const [x_train, x_test, y_train, y_test] = getTestTrainSplit(
      features,
      targets,
      modelParams.test_train_split
    );

    const tensors = convertToTensors(x_train, x_test, y_train, y_test);

    let model;
    let predictions;
    let tempR2 = 0;
    let trainCounter = 0;
    while (tempR2 < modelParams.min_R2_score) {
      model = await trainModel(
        tensors.trainFeatures,
        tensors.trainTargets,
        tensors.testFeatures,
        tensors.testTargets
      );
      console.log("Before predict: ", model);
      predictions = model.predict(tensors.testFeatures);
      //console.log("PREDICT", predictions);
      tempR2 = getR2Score(predictions.arraySync(), y_test).rSquared;
      console.log("R2 score", tempR2);
      setR2(tempR2);
      if (!hasTrained) {
        setHasTrained(true);
      }
      trainCounter += 1;
      if (
        trainCounter > modelParams.max_iterations &&
        !(tempR2 >= modelParams.min_R2_score)
      ) {
        setTrainingFailed(true);
        break;
      }
    }
  }

  async function trainModel(xTrain, yTrain, xTest, yTest) {
    console.log(xTrain);
    console.log(yTrain);

    let model = getComplexModel(xTrain.shape[1], yTrain.shape[1], modelParams);

    console.log("Model:", model);

    model.summary();
    model.compile({
      optimizer: modelParams.optimizer,
      loss: modelParams.loss
    });

    const lossContainer = document.getElementById("lossCanvas");
    const callbacks = tfvis.show.fitCallbacks(lossContainer, ["loss"], {
      callbacks: ["onEpochEnd"]
    });
    await model.fit(xTrain, yTrain, {
      epochs: modelParams.epochs,
      validationData: [xTest, yTest],
      callbacks: callbacks
    });
    return model;
  }

  return (
    <div className="App">
      <div className="step">
        <p>Step 1: Upload dataset in .csv format</p>
        <CSVReader cssClass="react-csv-input" onFileLoaded={selectDataset} />
      </div>

      <div className="step">
        <p>Step 2: Choose values for sensors</p>
        <table>
          <tbody>
            <tr>
              <th className="TableField">Sensor name</th>
              <th className="TableField">Input</th>
              <th className="TableField">Output</th>
              <th className="TableField">Exclude</th>
            </tr>
            {sensorNames &&
              sensorNames.map(sensor => (
                <AddSensor key={sensor} sensor={sensor} func={addSensorFunc} />
              ))}
          </tbody>
        </table>
      </div>

      <div className="step">
        <p>Step 3: Train model</p>
        <button className="react-csv-input" onClick={performModelTraining}>
          Init model training
        </button>
        <div>
          <h4>Loss</h4>
          <div className="canvases" id="lossCanvas"></div>
        </div>
      </div>
    </div>
  );
}

export default App;
