import React, { useState } from "react";

import { Checkbox } from "@material-ui/core";

export function addSensorFunc(sensor, type, sensorConfig, setFunc) {
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
  setFunc(config);
}

export const AddSensor = props => {
  const [inputSensor, setInputSensor] = useState(false);
  const [outputSensor, setOutputSensor] = useState(false);
  const [internalSensor, setInternalSensor] = useState(false);

  const changeSensor = number => {
    switch (number) {
      case 0:
        setInputSensor(true);
        setOutputSensor(false);
        setInternalSensor(false);
        props.func(props.sensor, "input", props.sensorConfig, props.setFunc);
        // save something to store here?
        break;
      case 1:
        setOutputSensor(true);
        setInputSensor(false);
        setInternalSensor(false);
        props.func(props.sensor, "output", props.sensorConfig, props.setFunc);
        // save something to store here?
        break;
      case 2:
        setInternalSensor(true);
        setOutputSensor(false);
        setInputSensor(false);
        props.func(props.sensor, "internal", props.sensorConfig, props.setFunc);
        // save something to store here?
        break;
      default:
        break;
    }
  };

  return (
    <tr>
      <td>
        <p>{props.sensor}</p>
      </td>
      <td>
        <Checkbox
          color="default"
          onClick={() => changeSensor(0)}
          checked={inputSensor}
        />
      </td>
      <td>
        <Checkbox
          color="default"
          onClick={() => changeSensor(1)}
          checked={outputSensor}
        />
      </td>
      <td>
        <Checkbox
          color="default"
          onClick={() => changeSensor(2)}
          checked={internalSensor}
        />
      </td>
    </tr>
  );
};

export default AddSensor;
