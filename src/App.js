import "./App.css";
import React, { useState, useEffect, useRef } from "react";
import CSVReader from "react-csv-reader";
import { AddSensor, addSensorFunc } from "./AddSensor.js";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { convertToTensors, getR2Score, preprocess } from "./utils.js";

// Some default machine learning model parameters
const modelParams = {
  // batch size is typically set in expontentials of 2, e.g. 32, 64, 128 etc.
  // Higher batch size gives more precise gradient approximations, at the cost of increased computation time
  batchSize: 128 * 2 * 2 * 2,
  // test_train_split = 0.2 means 20% of data will be set aside for testing the model
  test_train_split: 0.2,
  // validation_split = 0.2 means 20% of the training data will be used for validation during model training
  validation_split: 0.2,
  // the (initial) learning rate of the machine learning model, determines convergence rate and accuracy
  learningRate: 0.01,
  // number of entire loops over the training data
  epochs: 10,
  // the optimizer is the algorithm used to determine weight updates in the neural network
  optimizer: tf.train.adam(0.01),
  // model performance is said to increase if the loss metric decreases (usually)
  loss: "meanAbsoluteError"
};

function App() {
  // Some React state variables are used to keep track of various values

  const [step, setStep] = useState(1);

  const [dataPoints, setDataPoints] = useState(null);
  const [sensorNames, setSensorNames] = useState(null);
  const [sensorConfig, setSensorConfig] = useState(null);

  const [R2, setR2] = useState(-1000);
  const [hasTrained, setHasTrained] = useState(false);

  const [isTraining, setIsTraining] = useState(false);

  const [processedData, setProcessedData] = useState([]);
  const [trainedModel, setTrainedModel] = useState(null);

  const [csvError, setCsvError] = useState(false);

  const [epochs, setEpochs] = useState(modelParams.epochs);
  const [batchsize, setBatchsize] = useState(modelParams.batchSize);

  const selectDataset = data => {
    setCsvError(false);
    setDataPoints(data);
    setSensorNames(data[0]);
    setStep(2);
  };

  // Takes as parameters training and testing features (x) and targets (y)
  async function trainModel(x_train, x_test, y_train, y_test) {
    setIsTraining(true);

    // Converts arrays of data to tensors
    const tensors = convertToTensors(x_train, x_test, y_train, y_test);

    // Fits a machine learning model to the provided training data
    let model = await fitModel(tensors.trainFeatures, tensors.trainTargets);
    setTrainedModel(model);

    // calculates a metric which should indicate model performance
    setR2(
      getR2Score(model.predict(tensors.testFeatures).arraySync(), y_test)
        .rSquared
    );

    setHasTrained(true);
    setIsTraining(false);
  }

  // Defines a sequential neural network model based on the provided parameters:
  // - number of units: array of ints, number of units in each hidden layer
  // - inputSize: number of input features, equal to the input size of the first hidden layer
  // - outputSize: number of output features, equal to the output size of the final layer
  // - activation: activation func used in hidden layers
  // - outputActivation: activation func used in the output layer
  function getSequentialModel(
    numberOfUnits,
    inputSize,
    outputSize,
    activation,
    outputActivation
  ) {
    // define sequential model from tensorflow
    const model = tf.sequential();

    // add the first layer explicitly, with correct input size
    model.add(
      tf.layers.dense({
        kernelRegularizer: tf.regularizers.L1L2,
        units: numberOfUnits[0],
        activation: activation,
        inputShape: [inputSize]
      })
    );

    // add remaining hidden layers with equal input and output size
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

    // add output layer with correct output size
    model.add(
      tf.layers.dense({ units: outputSize, activation: outputActivation })
    );

    return model;
  }

  // Takes as parameters feature (x) and target (y) tensors
  async function fitModel(xTrain, yTrain) {
    // Define a sequential neural network model with the appropriate
    // input, hidden and output layer dimensions, with ReLU activation
    // internally and linear activation at the outputs
    let model = getSequentialModel(
      [128, 128],
      xTrain.shape[1],
      yTrain.shape[1],
      "relu",
      "linear"
    );

    // Print a model summary to console
    model.summary();
    // Compile the model
    model.compile({
      optimizer: modelParams.optimizer,
      loss: modelParams.loss
    });

    // If the provided dataset has very few rows (not desirable), we would like to
    // reduce the batch size and instead run additional epochs
    if (modelParams.batchSize > xTrain.shape[0]) {
      setBatchsize(32);
      setEpochs(30);
    }

    // A callback is performanced by the model each time it finishes an epoch or a batch
    // The callback targets a canvas element and plots the loss graphs
    const lossContainer = document.getElementById("lossCanvas");
    const callbacks = tfvis.show.fitCallbacks(
      lossContainer,
      ["loss", "val_loss"],
      {
        callbacks: ["onEpochEnd", "onBatchEnd"]
      }
    );

    // Fits the defined model to provided tensors
    await model.fit(xTrain, yTrain, {
      batchSize:
        modelParams.batchSize > xTrain.shape[0] ? 32 : modelParams.batchSize,
      epochs: modelParams.batchSize > xTrain.shape[0] ? 30 : modelParams.epochs,
      validationSplit: modelParams.validation_split,
      callbacks: callbacks
    });

    return model;
  }

  // General method which calls the previously defined functions
  async function performModelTraining() {
    setStep(3);

    // Preprocessing is performed to remove null-values and obtain the desired data structure
    const [x_train, x_test, y_train, y_test] = preprocess(
      dataPoints,
      sensorConfig,
      modelParams
    );
    setProcessedData([x_train, x_test, y_train, y_test]);
    trainModel(x_train, x_test, y_train, y_test);
  }

  // A ref is used to scroll the window downwards as the user performs actions
  const mainRef = useRef();
  useEffect(() => {
    if (mainRef.current) {
      mainRef.current.scrollIntoView({
        behavior: "smooth",
        block: "end",
        inline: "nearest"
      });
    }
  }, [step, isTraining, hasTrained, epochs, trainedModel, sensorConfig]);

  return (
    <div className="App" ref={mainRef}>
      <div className="step">
        <p style={{ marginBottom: "20px" }}>
          Machine learning demo using TensorFlowJS and React
        </p>
        {mainTextBody()}
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
          following rows should contain data values. In a program like Excel,
          this may look something like this:
        </p>
        {exampleTable()}
        <CSVReader
          cssClass="react-csv-input"
          onFileLoaded={selectDataset}
          onError={() => setCsvError(true)}
        />
        {csvError && <p>.csv upload failed, please try again</p>}
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
            output features. Irrelevant dataset columns, such as time or index
            columns, may be excluded. Features with no selected value are
            ignored. If you have provided a personal .csv file and features are
            not listed in the "feature names" column as expected, the provided
            file has the wrong formatting and cannot be used with this specific
            application, unfortunately.
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

      {step > 1 &&
        sensorConfig != null &&
        sensorConfig.input.length > 0 &&
        sensorConfig.output.length > 0 && (
          <div className="step">
            {hasTrained ? (
              <p className="green">
                3. Train your neural network model &nbsp; &#10004;
              </p>
            ) : (
              <p>3. Train your neural network model</p>
            )}
            <p className="smalltext">
              By default, the defined model is a{" "}
              <a
                href="https://en.wikipedia.org/wiki/Artificial_neural_network#Organization"
                target="_blank"
                rel="noopener noreferrer"
              >
                fully connected
              </a>{" "}
              neural network consisting of an input layer, two{" "}
              <a
                href="https://deepai.org/machine-learning-glossary-and-terms/hidden-layer-machine-learning"
                target="_blank"
                rel="noopener noreferrer"
              >
                hidden layers
              </a>{" "}
              of 128{" "}
              <a
                href="https://en.wikipedia.org/wiki/Artificial_neuron"
                target="_blank"
                rel="noopener noreferrer"
              >
                neurons or units
              </a>{" "}
              each, and an output layer. A{" "}
              <a
                href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)"
                target="_blank"
                rel="noopener noreferrer"
              >
                ReLU
              </a>{" "}
              <a
                href="https://en.wikipedia.org/wiki/Activation_function"
                target="_blank"
                rel="noopener noreferrer"
              >
                activation function
              </a>{" "}
              is used in the hidden neurons, while a linear activation is used
              at the output. The{" "}
              <a
                href="https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Adam optimizer
              </a>{" "}
              is used, as well as the{" "}
              <a
                href="https://en.wikipedia.org/wiki/Mean_absolute_error"
                target="_blank"
                rel="noopener noreferrer"
              >
                mean absolute error
              </a>{" "}
              metric.
            </p>
            <button className="react-csv-input" onClick={performModelTraining}>
              Start model training
            </button>
          </div>
        )}

      <div className="step" style={{ paddingTop: "0px" }}>
        <div>
          {step > 2 && (
            <p className="smalltext">
              The model is trained for {epochs} epochs in batches of {batchsize}
              . Epochs are the number of loops over the entirely training data.
              Batch size is the number of dataset rows used to calculate the
              stochastic gradient and perform weight updates. The figures below
              are updated with the calculated training and validation error
              metric when a batch or epoch is finished, respectively.
            </p>
          )}
          <div className="canvases" id="lossCanvas"></div>
        </div>
        {hasTrained && (
          <div>
            <p>R-squared score: {R2.toFixed(5)}</p>
            <p className="smalltext">
              The{" "}
              <a
                href="https://en.wikipedia.org/wiki/Coefficient_of_determination"
                target="_blank"
                rel="noopener noreferrer"
              >
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
                  target="_blank"
                  rel="noopener noreferrer"
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

function mainTextBody() {
  return (
    <p className="midtext">
      This small web application shows an entry-level example of machine
      learning performed directly in the browser using{" "}
      <a
        href="https://www.tensorflow.org/js"
        target="_blank"
        rel="noopener noreferrer"
      >
        TensorFlowJS
      </a>{" "}
      and{" "}
      <a href="https://reactjs.org/" target="_blank" rel="noopener noreferrer">
        React
      </a>
      . A dataset in{" "}
      <a
        href="https://en.wikipedia.org/wiki/Comma-separated_values"
        target="_blank"
        rel="noopener noreferrer"
      >
        .csv format
      </a>{" "}
      is parsed to a matrix of strings and converted to{" "}
      <a
        href="https://www.tensorflow.org/guide/tensor"
        target="_blank"
        rel="noopener noreferrer"
      >
        tensors
      </a>
      . Input and output{" "}
      <a
        href="https://en.wikipedia.org/wiki/Feature_(machine_learning)"
        target="_blank"
        rel="noopener noreferrer"
      >
        features
      </a>{" "}
      may be adjusted by the user. A{" "}
      <a
        href="https://en.wikipedia.org/wiki/Artificial_neural_network"
        target="_blank"
        rel="noopener noreferrer"
      >
        neural network
      </a>{" "}
      model with sensible default{" "}
      <a
        href="https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)"
        target="_blank"
        rel="noopener noreferrer"
      >
        hyperparameters
      </a>{" "}
      and structure is trained.
      <br />
      <br />
      For testing, I prepared the following datasets which you are advised to
      use:
      <br />
      <a
        href="https://github.com/hermanwh/tfjs-example/blob/main/iris_flower_dataset_extended.csv"
        target="_blank"
        rel="noopener noreferrer"
      >
        Iris flower dataset
      </a>{" "}
      (direct download link:{" "}
      <a
        href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/hermanwh/tfjs-example/blob/main/iris_flower_dataset_extended.csv"
        target="_blank"
        rel="noopener noreferrer"
      >
        link1
      </a>
      )
      <br />
      <a
        href="https://github.com/hermanwh/tfjs-example/blob/main/mechanical_component_dataset.csv"
        target="_blank"
        rel="noopener noreferrer"
      >
        Hydraulic jack system
      </a>{" "}
      (direct download link:{" "}
      <a
        href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/hermanwh/tfjs-example/blob/main/mechanical_component_dataset.csv"
        target="_blank"
        rel="noopener noreferrer"
      >
        link2
      </a>
      ).
      <br />
      Download links from GitHub are created using{" "}
      <a
        href="https://minhaskamal.github.io/DownGit"
        target="_blank"
        rel="noopener noreferrer"
      >
        https://minhaskamal.github.io/DownGit
      </a>
      .
      <br />
      <br />
      For implementation details, visit{" "}
      <a
        href="https://github.com/hermanwh/tfjs-example"
        target="_blank"
        rel="noopener noreferrer"
      >
        the GitHub repository
      </a>
    </p>
  );
}

function exampleTable() {
  return (
    <table className="noPad">
      <tbody>
        <tr>
          <td>sepal length</td>
          <td>sepal width</td>
          <td>petal length</td>
          <td>petal width</td>
        </tr>
        <tr>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.3</td>
          <td>0.3</td>
        </tr>
        <tr>
          <td>....</td>
          <td>....</td>
          <td>....</td>
          <td>....</td>
        </tr>
      </tbody>
    </table>
  );
}
