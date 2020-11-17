# View project on GitHub pages

Available at the following URL: https://hermanwh.github.io/tfjs-example/

# How to parse .csv file loaded using [react-csv-reader](https://www.npmjs.com/package/react-csv-reader)
NB: this guide is by no means optimal, and could certainly be improved through the use of more functional programming, e.g. map, filter etc.

Assuming your file is in .csv format, with the top row containing the column headers and the following rows containing values for each column. When uploading data using react-csv-reader, you can call a function as follows:

```javascript
const [dataPoints, setDataPoints] = useState(null);
...
const selectDataset = data => 
    // perform your actions, e.g. clearing your state variables
    // and binding the data:
    setDataPoints(data);
    setSensorNames(data[0]);
    setSensorConfig(null);
    setProcessedData([]);
  };
  
  ...
  
return (
  ...
  <CSVReader onFileLoaded={selectDataset} />
  ...
)
```

After loading your file from the frontend using react-csv-reader, your data object will have the following array structure:
```
[
  0: (4) ["sepallength", "sepalwidth", "petallength", "petalwidth"]
  1: (4) ["5.1", "3.5", "1.4", "0.2"]
  2: (4) ["4.9", "3.0", "1.4", "0.2"]
  3: (4) ["4.7", "3.2", "1.3", "0.2"]
  4: (4) ["4.6", "3.1", "1.5", "0.2"]
  ...
]
```

Passing this structue to the following method (top row is sliced because it contains the headers, bottom row is slices because it is typically just a newline):
```javascript
function refactorRawData(data) {
  let newData = [];
  const headers = data[0];
  data.slice(1, -1).forEach(function(values, index) {
    let result = {};
    headers.forEach((key, i) => (result[key] = values[i]));
    newData.push(result);
  });
  return newData;
}
```

we obtain an array of objects containing values for each of the dataset features:
```
[
  0: {sepallength: "5.1", sepalwidth: "3.5", petallength: "1.4", petalwidth: "0.2"}
  1: {sepallength: "4.9", sepalwidth: "3.0", petallength: "1.4", petalwidth: "0.2"}
  2: {sepallength: "4.7", sepalwidth: "3.2", petallength: "1.3", petalwidth: "0.2"}
  3: {sepallength: "4.6", sepalwidth: "3.1", petallength: "1.5", petalwidth: "0.2"}
  4: {sepallength: "5.0", sepalwidth: "3.6", petallength: "1.4", petalwidth: "0.2"}
  5: {sepallength: "5.4", sepalwidth: "3.9", petallength: "1.7", petalwidth: "0.4"}
  ...
]
```

At this point, perhaps you want to do additional preprocessing, such as removing all rows with null or empty values:
```javascript
function preprocessRemoveEmptyAndNull(data) {
  data = data.filter(x => !Object.values(x).some(y => y === "" || y == null));
  return data;
}
```

After defining a set of input and output features like this:
```
{
  input:["petallength", "petalwidth"]
  output: ["sepallength", "sepalwidth"]
}
```

the following function may be used to obtain arrays of arrays containing the input and output features in appropriate order:
```javascript
function getFeatureTargetSplit(dataset, config) {
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
```

So that if "sepalwidth" and "petallength" are chosen as input features, you end up with:
```
[
  0: (2) [3.5, 1.4]
  1: (2) [3, 1.4]
  2: (2) [3.2, 1.3]
  3: (2) [3.1, 1.5]
  4: (2) [3.6, 1.4]
  5: (2) [3.9, 1.7]
  ...
]
```

Furthermore, you may want to divide your data into training and testing data:
```javascript
function getTestTrainSplit(features, targets, test_train_split) {
  const numberOfRows = features.length;
  const numberOfTest = Math.round(numberOfRows * test_train_split);
  const numberOfTrain = numberOfRows - numberOfTest;

  const x_train = features.slice(0, numberOfTrain - 1);
  const x_test = features.slice(numberOfTrain - 1);
  const y_train = targets.slice(0, numberOfTrain - 1);
  const y_test = targets.slice(numberOfTrain - 1);
  return [x_train, x_test, y_train, y_test];
}
```

and finally into tensors:
```javascript
function convertToTensors(x_train, x_test, y_train, y_test) {
  const tensors = {};
  tensors.trainFeatures = tf.tensor2d(x_train);
  tensors.trainTargets = tf.tensor2d(y_train);
  tensors.testFeatures = tf.tensor2d(x_test);
  tensors.testTargets = tf.tensor2d(y_test);
  return tensors;
}
```

