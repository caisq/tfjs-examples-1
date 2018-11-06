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

let tf;

function generateRandomColorStyle() {
  const colorR = Math.round(Math.random() * 255);
  const colorG = Math.round(Math.random() * 255);
  const colorB = Math.round(Math.random() * 255);
  return `rgb(${colorR},${colorG},${colorB})`;
}

class ObjectDetectionDataSynthesizer {
  constructor(canvas, tensorFlow) {
    this.canvas = canvas;
    tf = tensorFlow;

    this.CIRCLE_RADIUS_MIN = 5;
    this.CIRCLE_RADIUS_MAX = 20;
    this.TRIANGLE_SIDE_MIN = 50;
    this.TRIANGLE_SIDE_MAX = 100;
  }

  async generateExample(numCircles, numLines) {
    const w = this.canvas.width;
    const h = this.canvas.height;

    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    // Draw circles.
    for (let i = 0; i < numCircles; ++i) {
      const centerX = w * Math.random();
      const centerY = h * Math.random();
      const radius = this.CIRCLE_RADIUS_MIN +
          (this.CIRCLE_RADIUS_MAX - this.CIRCLE_RADIUS_MIN) * Math.random();

      ctx.fillStyle = generateRandomColorStyle();
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw lines.
    for (let i = 0; i < numLines; ++i) {
      const x0 = Math.random() * w;
      const y0 = Math.random() * h;
      const x1 = Math.random() * w;
      const y1 = Math.random() * h;

      ctx.strokeStyle = generateRandomColorStyle();
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // Draw the detection target: an equilateral triangle.
    const side = this.TRIANGLE_SIDE_MIN +
        (this.TRIANGLE_SIDE_MAX - this.TRIANGLE_SIDE_MIN) * Math.random();
    const centerX = (w - side) * Math.random() + (side / 2);
    const centerY = (h - side) * Math.random() + (side / 2);

    const ctrToVertex = side / 2 / Math.cos(30 / 180 * Math.PI);
    const strToSide = ctrToVertex / 2;
    const topX = centerX;
    const topY = centerY - ctrToVertex;
    const leftX = centerX - side / 2;
    const leftY = centerY + strToSide;
    const rightX = centerX + side / 2;
    const rightY = leftY;

    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.moveTo(topX, topY);
    ctx.lineTo(leftX, leftY);
    ctx.lineTo(rightX, rightY);
    ctx.fill();

    return tf.tidy(() => {
      const imageTensor = tf.fromPixels(this.canvas);
      
      const boundingBoxTensor = tf.tensor1d([leftX, rightX, topY, leftY]);
      return {image: imageTensor, boundingBox: boundingBoxTensor};
    });
  }

  async generateExampleBatch(batchSize, numCircles, numLines) {
    const imageTensors = [];
    const boundingBoxTensors = [];
    for (let i = 0; i < batchSize; ++i) {
      const {image, boundingBox} =
          await this.generateExample(numCircles, numLines);
      imageTensors.push(image);
      boundingBoxTensors.push(boundingBox);
    }
    const images = tf.stack(imageTensors);
    const boundingBoxes = tf.stack(boundingBoxTensors);
    tf.dispose([imageTensors, boundingBoxTensors]);
    return {images, boundingBoxes};
  }
}

module.exports = {ObjectDetectionDataSynthesizer};
