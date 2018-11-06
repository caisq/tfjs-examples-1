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

import * as tf from '@tensorflow/tfjs';
import {generateExampleBatch} from './synthetic_data';

const trainButton = document.getElementById('train');

async function loadDecapitatedMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  const decapitatedBase = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output
  });
  // Freeze the model's layers.
  for (const layer of decapitatedBase.layers) {
    layer.trainable = false;
  }
  return decapitatedBase;
}

async function buildObjectDetectionModel() {
  const decapitatedBase = await loadDecapitatedMobilenet();
  // model.summary();  // DEBUG
  console.log(decapitatedBase.inputs[0].shape);  // DEBUG

  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten({
    inputShape: decapitatedBase.outputs[0].shape.slice(1)
  }));
  newHead.add(tf.layers.dense({units: 20, activation: 'relu'}));
  newHead.add(tf.layers.dense({units: 4}));

  const newHeadOutput = newHead.apply(decapitatedBase.outputs[0]);
  console.log('newHeadOutput.shape:', newHeadOutput.shape);

  const objDetectModel = tf.model({
    inputs: decapitatedBase.inputs,
    outputs: newHeadOutput
  });

  objDetectModel.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});
  // objDetectModel.summary();
  console.log(objDetectModel.outputs[0].shape);  // DEBUG
  return objDetectModel;
}

async function init() {
  const model = await buildObjectDetectionModel();
  console.log('model.outputs[0].shape:', model.outputs[0].shape);  // DEBUG

  const numTrainExamples = 64;
  const numValExamples = 20;

  const numCircles = 1;
  const numLines = 1;
  const {images: valImages, boundingBoxes: valBoundingBoxes} =
     await generateExampleBatch(numValExamples, numCircles, numLines);

  const iterations = 50;

    // boundingBoxes.print();

  trainButton.addEventListener('click', async () => {
    for (let i = 0; i < iterations; ++i) {
      console.log(`iterations ${i + 1} / ${iterations}`);

      const {images, boundingBoxes} =
          await generateExampleBatch(numTrainExamples, numCircles, numLines);
      let currentEpoch;
      await model.fit(images, boundingBoxes, {
        epochs: 1,
        batchSize: 16,
        validationData: [valImages, valBoundingBoxes],
        callbacks: {
          onEpochBegin: async (epoch) => {
            currentEpoch = epoch;
          },
          onBatchEnd: async (batch, logs) => {
            console.log(
                `  Epoch ${currentEpoch}, batch ${batch}: ` +
                `loss = ${logs.loss}`);
          },
          onEpochEnd: async (epoch, logs) => {
            console.log(
                `Epoch ${epoch}: loss = ${logs.loss}, ` +
                `val_loss = ${logs.val_loss}`);
          }
        }
      });
      tf.dispose([images, boundingBoxes]);
    }
  });
  // TODO(cais): Do evaluate().
}

init();
