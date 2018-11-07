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
import {ObjectDetectionImageSynthesizer} from './synthetic_images';

const testButton = document.getElementById('test');

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)';
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)';

function drawBoundingBox(canvas, trueBoundingBox, predictBoundingBox) {
  tf.util.assert(
      trueBoundingBox != null && trueBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
          `but got ${trueBoundingBox} instead`);
  tf.util.assert(
      predictBoundingBox != null && predictBoundingBox.length === 4,
      `Expected boundingBoxArray to have length 4, ` +
          `but got ${trueBoundingBox} instead`);

  const w = canvas.width;
  const h = canvas.height;

  // Plot true bounding box.
  // let left = trueBoundingBox[0] * w;
  // let right = trueBoundingBox[1] * w;
  // let top = trueBoundingBox[2] * h;
  // let bottom = trueBoundingBox[3] * h;
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

  ctx.font = '15px Arial';
  ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.fillText('true', left, top);

  // Plot predicted bounding box.
  // left = predictBoundingBox[0] * w;
  // right = predictBoundingBox[1] * w;
  // top = predictBoundingBox[2] * h;
  // bottom = predictBoundingBox[3] * h;
  left = predictBoundingBox[0];
  right = predictBoundingBox[1];
  top = predictBoundingBox[2];
  bottom = predictBoundingBox[3];

  ctx.beginPath();
  ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = '15px Arial';
  ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.fillText('predicted', left, bottom);
}

async function init() {
  const canvas = document.getElementById('data-canvas');

  const model = await tf.loadModel('object_detection_model/model.json');
  model.summary();

  testButton.addEventListener('click', async () => {
    const synth = new ObjectDetectionImageSynthesizer(canvas, tf);
    const {images, targets} = await synth.generateExampleBatch(1, 10, 10);

    tf.tidy(() => {
      const boundingBoxArray = Array.from(targets.dataSync()).slice(1);      
      const t0 = tf.util.now();
      const modelOut = model.predict(images).dataSync();
      const tElapsed = tf.util.now() - t0;
      console.log(tElapsed);
      drawBoundingBox(canvas, boundingBoxArray, modelOut.slice(1));
    });
  });
}

init();
