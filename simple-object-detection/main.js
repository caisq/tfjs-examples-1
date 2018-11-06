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
const synthesizer = require('./synthetic_data');
const fetch = require('node-fetch');
require('@tensorflow/tfjs-node-gpu');

global.fetch = fetch;

const topLayerGroupName = 'conv_pw_7';
const topLayerName = `${topLayerGroupName}_relu`;

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
    if (layer.name.indexOf(topLayerGroupName) === 0) {
      fineTuningLayers.push(layer);
    }
  }
  return {decapitatedBase, fineTuningLayers};
}

async function buildObjectDetectionModel() {
    const {decapitatedBase, fineTuningLayers} = await loadDecapitatedMobilenet();
    // model.summary();  // DEBUG
    console.log(decapitatedBase.inputs[0].shape);  // DEBUG    
  
    const newHead = tf.sequential();
    newHead.add(tf.layers.flatten(
        {inputShape: decapitatedBase.outputs[0].shape.slice(1)}));
    // newHead.add(tf.layers.dropout({rate: 0.5}));
    newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));    
    // newHead.add(tf.layers.dropout({rate: 0.25}));
    newHead.add(tf.layers.dense({units: 4}));
  
    const newHeadOutput = newHead.apply(decapitatedBase.outputs[0]);
    console.log('newHeadOutput.shape:', newHeadOutput.shape);
  
    const model =
        tf.model({inputs: decapitatedBase.inputs, outputs: newHeadOutput});

    // objDetectModel.summary();
    console.log(model.outputs[0].shape);  // DEBUG
    return {model, fineTuningLayers};
  }

(async function main() {
  const canvasSize = 224;
  const numExamples = 10000;
  const numCircles = 10;
  const numLines = 10;
  const batchSize = 128;
  const initialTransferEpochs = 40;
  const fineTuningEpochs = 40;

  const synthDataCanvas = canvas.createCanvas(canvasSize, canvasSize);
  console.log(tf.version);  // DEBUG

  console.log(`Generating ${numExamples} training examples...`);  // DEBUG
  const synth =
      new synthesizer.ObjectDetectionDataSynthesizer(synthDataCanvas, tf);
  const {images, boundingBoxes} = await synth.generateExampleBatch(
      numExamples, numCircles, numLines);

// DEBUG 
//   const slice = images.unstack()[numExamples -1].transpose([2, 0, 1]).unstack()[0];
//   const sliceData = slice.dataSync();
//   let str = ''
//   for (let i = 0; i < sliceData.length; ++i) {
//     if (i % canvasSize === 0) {
//       str += '\n';
//     }
//     str += sliceData[i] + ',';
//   }
//   console.log(str);
//   boundingBoxes.print();
// ~DEBUG
  const {model, fineTuningLayers} = await buildObjectDetectionModel();
  model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});
  console.log('fineTuningLayers.length:', fineTuningLayers.length);  // DEBUG
  model.summary();
  
  await model.fit(images, boundingBoxes, {
    epochs: initialTransferEpochs,
    batchSize,
    validationSplit: 0.15,
  });

  // Unfreeze layers for fine-tuning.
  for (const layer of fineTuningLayers) {
    layer.trainable = true;
  }
  model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});
  model.summary();

  // Do fine-tuning.
  await model.fit(images, boundingBoxes, {
    epochs: fineTuningEpochs,
    batchSize,
    validationSplit: 0.15,
  });

  // Save model.
  await model.save('file://./dist/object_detection_model');
})();
