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
import {ObjectDetectionDataSynthesizer} from './synthetic_data';

const testButton = document.getElementById('test');

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)';
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)';

function drawBoundingBox(canvas, trueBoundingBox, predictBoundigBox) {
  tf.util.assert(
      trueBoundingBox != null && trueBoundingBox.length === 4,
      `Expected boundingBoxArray to have less 4, ` +
      `but got ${trueBoundingBox} instead`);
  tf.util.assert(
      predictBoundigBox != null && predictBoundigBox.length === 4,
      `Expected boundingBoxArray to have less 4, ` +
      `but got ${trueBoundingBox} instead`);

  // Plot true bounding box.
  let left = trueBoundingBox[0];
  let right = trueBoundingBox[1];
  let top = trueBoundingBox[2];
  let bottom = trueBoundingBox[3];

  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = "15px Arial";
  ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.fillText('true', left, top);

  // Plot predicted bounding box.
  left = predictBoundigBox[0];
  right = predictBoundigBox[1];
  top = predictBoundigBox[2];
  bottom = predictBoundigBox[3];

  ctx.beginPath();
  ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = "15px Arial";
  ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.fillText('predicted', left, bottom);
}

async function init() {
  const canvas = document.getElementById('data-canvas');

  const model = await tf.loadModel('object_detection_model/model.json');
  model.summary();

  testButton.addEventListener('click', async () => {
    const synth = new ObjectDetectionDataSynthesizer(canvas, tf);
    const {images, boundingBoxes} =
        await synth.generateExampleBatch(1, 10, 10);

    tf.tidy(() => {
      const boundingBoxArray = boundingBoxes.dataSync();
      const out = model.predict(images);
      drawBoundingBox(canvas, boundingBoxArray, out.dataSync());
    });
  });

  // const model = await buildObjectDetectionModel();
  // console.log('model.outputs[0].shape:', model.outputs[0].shape);  // DEBUG

  // const numTrainExamples = 32;
  // const numValExamples = 20;

  // const numCircles = 1;
  // const numLines = 1;
  // const {images: valImages, boundingBoxes: valBoundingBoxes} =
  //     await generateExampleBatch(numValExamples, numCircles, numLines);

  // const iterations = 50;

  // // boundingBoxes.print();

  // trainButton.addEventListener('click', async () => {
  //   for (let i = 0; i < iterations; ++i) {
  //     console.log(`iterations ${i + 1} / ${iterations}`);

  //     const {images, boundingBoxes} =
  //         await generateExampleBatch(numTrainExamples, numCircles, numLines);
  //     let currentEpoch;
  //     await model.fit(images, boundingBoxes, {
  //       epochs: 1,
  //       batchSize: 16,
  //       validationData: [valImages, valBoundingBoxes],
  //       callbacks: {
  //         onEpochBegin: async (epoch) => {
  //           currentEpoch = epoch;
  //         },
  //         onBatchEnd: async (batch, logs) => {
  //           console.log(
  //               `  Epoch ${currentEpoch}, batch ${batch}: ` +
  //               `loss = ${logs.loss}`);
  //         },
  //         onEpochEnd: async (epoch, logs) => {
  //           console.log(
  //               `Epoch ${epoch}: loss = ${logs.loss}, ` +
  //               `val_loss = ${logs.val_loss}`);
  //         }
  //       }
  //     });
  //     tf.dispose([images, boundingBoxes]);
  //   }
  // });
  // TODO(cais): Do evaluate().
}

init();
