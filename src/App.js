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
  batchSize: 128 * 2 * 2 * 2,
  test_train_split: 0.2,
  validation_split: 0.2,
  learningRate: 0.01,
  epochs: 10,
  optimizer: tf.train.adam(0.01),
  loss: "meanAbsoluteError",
  min_R2_score: 0.5,
  max_iterations: 4
};

function App() {
  const [step, setStep] = useState(1);

  const [dataPoints, setDataPoints] = useState(null);
  const [sensorNames, setSensorNames] = useState(null);
  const [hasSelectedDataset, setHasSelectedDataset] = useState(false);
  const [sensorConfig, setSensorConfig] = useState(null);

  const [R2, setR2] = useState(-1000);
  const [hasTrained, setHasTrained] = useState(false);

  const [isTraining, setIsTraining] = useState(false);

  const [processedData, setProcessedData] = useState([]);
  const [trainedModel, setTrainedModel] = useState(null);

  const [epochs, setEpochs] = useState(modelParams.epochs);
  const [batchsize, setBatchsize] = useState(modelParams.batchSize);

  const selectDataset = data => {
    setHasSelectedDataset(true);
    setDataPoints(data);
    setSensorNames(data[0]);
    setStep(2);
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
      [128, 128],
      xTrain.shape[1],
      yTrain.shape[1],
      "relu",
      "linear"
    );

    model.summary();
    model.compile({
      optimizer: modelParams.optimizer,
      loss: modelParams.loss
    });

    if (modelParams.batchSize > xTrain.shape[0]) {
      setBatchsize(32);
      setEpochs(30);
    }

    const lossContainer = document.getElementById("lossCanvas");
    const callbacks = tfvis.show.fitCallbacks(
      lossContainer,
      ["loss", "val_loss"],
      {
        callbacks: ["onEpochEnd", "onBatchEnd"]
      }
    );

    await model.fit(xTrain, yTrain, {
      batchSize:
        modelParams.batchSize > xTrain.shape[0] ? 32 : modelParams.batchSize,
      epochs: modelParams.batchSize > xTrain.shape[0] ? 30 : modelParams.epochs,
      validationSplit: modelParams.validation_split,
      callbacks: callbacks
    });
    return model;
  }

  async function performModelTraining() {
    setStep(3);
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
        <p className="midtext">
          This small web application shows an entry-level example of machine
          learning performed directly in the browser using{" "}
          <a href="https://www.tensorflow.org/js" target="_blank">
            TensorFlowJS
          </a>{" "}
          and{" "}
          <a href="https://reactjs.org/" target="_blank">
            React
          </a>
          . A dataset in{" "}
          <a
            href="https://en.wikipedia.org/wiki/Comma-separated_values"
            target="_blank"
          >
            .csv format
          </a>{" "}
          is parsed to a matrix of strings and converted to{" "}
          <a href="https://www.tensorflow.org/guide/tensor" target="_blank">
            tensors
          </a>
          . Input and output{" "}
          <a
            href="https://en.wikipedia.org/wiki/Feature_(machine_learning)"
            target="_blank"
          >
            features
          </a>{" "}
          may be adjusted by the user. A{" "}
          <a
            href="https://en.wikipedia.org/wiki/Artificial_neural_network"
            target="_blank"
          >
            neural network
          </a>{" "}
          model with sensible default{" "}
          <a
            href="https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)"
            target="_blank"
          >
            hyperparameters
          </a>{" "}
          and structure is trained.
          <br />
          <br />
          For testing, I prepared the following datasets which you are advised
          to use:
          <br />
          <a
            href="https://github.com/hermanwh/tfjs-example/blob/main/iris_flower_dataset_extended.csv"
            target="_blank"
          >
            Iris flower dataset
          </a>{" "}
          (direct download link:{" "}
          <a
            href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/hermanwh/tfjs-example/blob/main/iris_flower_dataset_extended.csv"
            target="_blank"
          >
            link1
          </a>
          )
          <br />
          <a
            href="https://github.com/hermanwh/tfjs-example/blob/main/mechanical_component_dataset.csv"
            target="_blank"
          >
            hydraulic jack system
          </a>{" "}
          (direct downloiad link:{" "}
          <a
            href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/hermanwh/tfjs-example/blob/main/mechanical_component_dataset.csv"
            target="_blank"
          >
            link2
          </a>
          ).
          <br />
          Download links from GitHub are created using{" "}
          <a href="https://minhaskamal.github.io/DownGit" target="_blank">
            https://minhaskamal.github.io/DownGit
          </a>
          .
          <br />
          <br />
          For implementation details, visit{" "}
          <a href="https://github.com/hermanwh/tfjs-example" target="_blank">
            the GitHub repository
          </a>
        </p>
      </div>

      <div className="step">
        {step > 1 ? (
          <p className="green">
            1. Upload dataset in .csv format &nbsp; &#10004;
          </p>
        ) : (
          <p>1. Upload dataset in .csv format</p>
        )}
        <p className="smalltext">
          The first row of the file should contain the column headers. The
          following rows should contain data values.
        </p>
        <CSVReader cssClass="react-csv-input" onFileLoaded={selectDataset} />
      </div>

      {step > 1 && (
        <div className="step">
          {step > 2 ? (
            <p className="green">
              2. Choose input and output features &nbsp; &#10004;
            </p>
          ) : (
            <p>2. Choose input and output features</p>
          )}
          <p className="smalltext">
            Any number of input features can be used to predict any number of
            output features. Irrelevant dataset columns, e.g. time or index
            columns, may be excluded. Features with no selected value are
            ignored.
          </p>
          {sensorNames != null && (
            <div>
              <table>
                <tbody>
                  <tr>
                    <th className="TableField">Feature name</th>
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
            </div>
          )}
        </div>
      )}

      {step > 1 && (
        <div className="step">
          {hasTrained ? (
            <p className="green">
              3. Train your neural network model &nbsp; &#10004;
            </p>
          ) : (
            <p>3. Train your neural network model</p>
          )}
          <button className="react-csv-input" onClick={performModelTraining}>
            Init model training
          </button>
        </div>
      )}

      <div className="step" style={{ paddingTop: "0px" }}>
        <div>
          {step > 2 && (
            <p className="smalltext">
              The model is being trained for {epochs} epochs in batches of{" "}
              {batchsize}. Epochs are the number of loops over the entirely
              training data. Batch size is the number of dataset rows used to
              calculate the stochastic gradient and perform weight updates. The
              figures below are updated when a batch or epoch is finished,
              respectively.
            </p>
          )}
          <div className="canvases" id="lossCanvas"></div>
        </div>
        {hasTrained && (
          <div>
            <p>R-squared score: {R2.toFixed(5)}</p>
            <p className="smalltext">
              The{" "}
              <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">
                R-squared metric
              </a>{" "}
              is a measure of how well observed outcomes are replicated by the
              model. 1.0 means perfect replication, while anything above 0.0
              indicates the model performed better than a mean predictor.
            </p>
            <button
              className="buttonStyle"
              onClick={() => trainModel(...processedData)}
            >
              Retrain model
            </button>

            <div className="step">
              <p>So what now?</p>
              <p className="smalltext">
                Good question! Usually, the trained model would be evaluated
                based on some metrics (R-squared is one such metric), and
                perhaps even using empirical analysis and expert knowledge. If
                deemed suitable, the model may be used to make predictions on
                new, future data samples. The difference between measured and
                predicted value can be used to derive fault models, e.g.
                indicating component health in a mechanical or industrial
                system. Using machine learning models in practice is a large and
                active area of research. You can read about an attempt at using
                neural network models for condition monitoring for the oil and
                gas industry in{" "}
                <a
                  href="https://github.com/hermanwh/master-thesis"
                  target="_black"
                >
                  my master thesis
                </a>
                .
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
