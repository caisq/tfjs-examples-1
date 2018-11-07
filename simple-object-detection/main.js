/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const canvas = require('canvas');
const tf = require('@tensorflow/tfjs');
const synthesizer = require('./synthetic_images');
const fetch = require('node-fetch');
require('@tensorflow/tfjs-node-gpu');

global.fetch = fetch;

// Name prefixes of layers that will be unfrozen during fine-tuning.
const topLayerGroupNames = ['conv_pw_10', 'conv_pw_11'];

// Name of the layer that will become the top layer of the decapitated base.
const topLayerName =
    `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

async function loadDecapitatedMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const fineTuningLayers = [];
  const layer = mobilenet.getLayer(topLayerName);
  const decapitatedBase =
      tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  // Freeze the model's layers.
  for (const layer of decapitatedBase.layers) {
    layer.trainable = false;
    for (const groupName of topLayerGroupNames) {
      if (layer.name.indexOf(groupName) === 0) {
        fineTuningLayers.push(layer);
        break;
      }
    }
  }
  return {decapitatedBase, fineTuningLayers};
}

// const multiplier = tf.scalar(100);

function customLossFunction(yTrue, yPred) {
  return tf.tidy(() => {
    // const batchSize = yTrue.shape[0];
    // const boundingBoxDims = yTrue.shape[1] - 1;
    // TODO(cais): Use them.
    // const classTrue = yTrue.slice([0, 0], [batchSize, 1]);
    // const classPred = tf.sigmoid(yPred.slice([0, 0], [batchSize, 1]));
    // classTrue.print();
    // classPred.print();

    // const boundingBoxTrue = yTrue.slice([0, 1], [batchSize, boundingBoxDims]);
    // const boundingBoxPred = yPred.slice([0, 1], [batchSize, boundingBoxDims]);
    // const boundingBoxLoss =
    //     tf.metrics.meanAbsoluteError(boundingBoxTrue, boundingBoxPred);
    // return boundingBoxLoss;
    // return tf.metrics.meanAbsoluteError(yTrue, yPred);
    return tf.metrics.meanSquaredError(yTrue, yPred);
  });
}

async function buildObjectDetectionModel() {
  const {decapitatedBase, fineTuningLayers} = await loadDecapitatedMobilenet();
  // model.summary();  // DEBUG
  console.log(decapitatedBase.inputs[0].shape);  // DEBUG

  // newHead.add(tf.layers.dropout({rate: 0.5}));
  // newHead.add(tf.layers.dropout({rate: 0.25}));

  //   let y = decapitatedBase.outputs[0];
  //   y = tf.layers.flatten({inputShape:
  //   decapitatedBase.outputs[0].shape.slice(1)})
  //           .apply(y);
  //   y = tf.layers.dense({units: 200, activation: 'relu'}).apply(y);
  //   const zShape =
  //       tf.layers.dense({units: 1, activation: 'sigmoid', name: 'shapeOut'})
  //           .apply(y);
  //   const zBoundingBox = tf.layers.dense({units: 4, name:
  //   'boundsOut'}).apply(y); const model = tf.model(
  //       {inputs: decapitatedBase.inputs, outputs: [zShape, zBoundingBox]});
//   console.log(model.outputs[0].shape);  // DEBUG
//   console.log(model.outputs[1].shape);  // DEBUG

  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten(
      {inputShape: decapitatedBase.outputs[0].shape.slice(1)}));
  newHead.add(tf.layers.dense({
      units: 200, activation: 'relu', kernelInitializer: 'leCunNormal'}));
  newHead.add(tf.layers.dense({units: 5, kernelInitializer: 'leCunNormal'}));
  const newOutput = newHead.apply(decapitatedBase.outputs[0]);
  const model = tf.model({inputs: decapitatedBase.inputs, outputs: newOutput});

  // objDetectModel.summary();
  return {model, fineTuningLayers};
}

(async function main() {
  const canvasSize = 224;
  const numExamples = 10000;
  const numCircles = 10;
  const numLines = 10;
  const batchSize = 128;
  const initialTransferEpochs = 50;
  const fineTuningEpochs = 100;

  const synthDataCanvas = canvas.createCanvas(canvasSize, canvasSize);
  console.log(tf.version);  // DEBUG

  console.log(`Generating ${numExamples} training examples...`);  // DEBUG
  const synth =
      new synthesizer.ObjectDetectionImageSynthesizer(synthDataCanvas, tf);
  const {images, targets} =
      await synth.generateExampleBatch(numExamples, numCircles, numLines);

  //   const shapeTargets = targets.slice([0, 0], [numExamples, 1]);
  //   const boundingBoxTargets = targets.slice([0, 1], [numExamples, 4]);
  //   console.log(targets.shape);             // DEBUG
  //   console.log(shapeTargets.shape);        // DEBUG
  //   console.log(boundingBoxTargets.shape);  // DEBUG

  const {model, fineTuningLayers} = await buildObjectDetectionModel();
  console.log('fineTuningLayers:', fineTuningLayers.length);  // DEBUG
  //   model.compile({
  //     loss: ['binaryCrossentropy', 'meanAbsoluteError'],
  //     optimizer: 'rmsprop'
  //   });
  model.compile({loss: customLossFunction, optimizer: 'rmsprop'});
  //  tf.train.adam(5e-2)
  model.summary();

  //   await model.fit(images, [shapeTargets, boundingBoxTargets], {
  //     epochs: initialTransferEpochs,
  //     batchSize,
  //     validationSplit: 0.15,
  //   });
  await model.fit(images, targets, {
    epochs: initialTransferEpochs,
    batchSize,
    validationSplit: 0.15,
  });

  // Unfreeze layers for fine-tuning.
    for (const layer of fineTuningLayers) {
      layer.trainable = true;
    }
    // model.compile({
    //   loss: ['binaryCrossentropy', 'meanAbsoluteError'],
    //   optimizer: 'rmsprop'
    // });
    model.compile({
      loss: customLossFunction,
      optimizer: 'rmsprop'
    });
    //  tf.train.adam(5e-2)
    model.summary();

    // Do fine-tuning.
    await model.fit(images, targets, {
      epochs: fineTuningEpochs,
      batchSize: batchSize / 2,
      validationSplit: 0.15,
    });

    // Save model.
    await model.save('file://./dist/object_detection_model');
})();
