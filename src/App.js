import "./App.css";
import React, { useState } from "react";
import CSVReader from "react-csv-reader";
import { AddSensor, addSensorFunc } from "./AddSensor.js";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import {
  convertToTensors,
  getR2Score,
  preprocess,
  getSequentialModel
} from "./utils.js";

const modelParams = {
  batchSize: 128 * 2 * 2,
  test_train_split: 0.2,
  validation_split: 0.2,
  learningRate: 0.01,
  epochs: 200,
  optimizer: tf.train.adam(0.01),
  loss: "meanAbsoluteError",
  min_R2_score: 0.5,
  max_iterations: 4
};

function App() {
  const [dataPoints, setDataPoints] = useState(null);
  const [sensorNames, setSensorNames] = useState(null);
  const [hasSelectedDataset, setHasSelectedDataset] = useState(false);
  const [sensorConfig, setSensorConfig] = useState(null);

  const [R2, setR2] = useState(-1000);
  const [hasTrained, setHasTrained] = useState(false);

  const [isTraining, setIsTraining] = useState(false);

  const [processedData, setProcessedData] = useState([]);
  const [trainedModel, setTrainedModel] = useState(null);

  const selectDataset = data => {
    setHasSelectedDataset(true);
    setDataPoints(data);
    setSensorNames(data[0]);
  };

  async function trainModel(x_train, x_test, y_train, y_test) {
    setIsTraining(true);

    const tensors = convertToTensors(x_train, x_test, y_train, y_test);

    let model = await fitModel(tensors.trainFeatures, tensors.trainTargets);
    setTrainedModel(model);
    setR2(
      getR2Score(model.predict(tensors.testFeatures).arraySync(), y_test)
        .rSquared
    );

    setHasTrained(true);
    setIsTraining(false);
  }

  async function fitModel(xTrain, yTrain) {
    let model = getSequentialModel(
      [128],
      xTrain.shape[1],
      yTrain.shape[1],
      "relu",
      "linear"
    );

    console.log("Model:", model.summary());

    model.summary();
    model.compile({
      optimizer: modelParams.optimizer,
      loss: modelParams.loss
    });

    const lossContainer = document.getElementById("lossCanvas");
    const callbacks = tfvis.show.fitCallbacks(
      lossContainer,
      ["loss", "val_loss"],
      {
        callbacks: ["onEpochEnd", "onBatchEnd"]
      }
    );
    await model.fit(xTrain, yTrain, {
      batchSize: modelParams.batchSize,
      epochs: modelParams.epochs,
      validationSplit: modelParams.validation_split,
      callbacks: callbacks
    });
    return model;
  }

  async function performModelTraining() {
    const [x_train, x_test, y_train, y_test] = preprocess(
      dataPoints,
      sensorConfig,
      modelParams
    );
    setProcessedData([x_train, x_test, y_train, y_test]);
    trainModel(x_train, x_test, y_train, y_test);
  }

  return (
    <div className="App">
      <div className="step">
        <p>Step 1: Upload dataset in .csv format</p>
        <CSVReader cssClass="react-csv-input" onFileLoaded={selectDataset} />
      </div>

      <div className="step">
        <p>Step 2: Choose values for sensors</p>
        {sensorNames != null && (
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
                  <AddSensor
                    key={sensor}
                    sensor={sensor}
                    func={addSensorFunc}
                    sensorConfig={sensorConfig}
                    setFunc={setSensorConfig}
                  />
                ))}
            </tbody>
          </table>
        )}
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
        {hasTrained && (
          <div>
            <p>R-squared score: {R2.toFixed(5)}</p>
            <button
              className="buttonStyle"
              onClick={() => trainModel(...processedData)}
            >
              Retrain model
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
