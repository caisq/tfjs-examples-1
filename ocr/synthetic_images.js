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

/**
 * Module for synthesizing images to be used for training and testing the
 * simple object-detection model.
 *
 * This module is written in a way that can be used in both the Node.js-based
 * training pipeline (train.js) and the browser-based testing environment
 * (index.js).
 */

let tf;  // tensorflowjs module passed in for browser/node compatibility.

/**
 * Generate a random color style for canvas strokes and fills.
 *
 * @returns {string} Style string in the form of 'rgb(100,200,250)'.
 */
function generateRandomColorStyle() {
  const colorR = Math.round(Math.random() * 255);
  const colorG = Math.round(Math.random() * 255);
  const colorB = Math.round(Math.random() * 255);
  return `rgb(${colorR},${colorG},${colorB})`;
}

/**
 * Synthesizes images for optical character recognition.
 *
 * The synthesized imags consist of
 * - a white background
 * - a configurable number of circles of random radii and random color
 * - a configurable number of line segments of random starting and ending
 *   points and random color
 * - Target object: a rectangle or a triangle, with configurable probabilities.
 *   - If a rectangle, the side lengths are random and so is the color
 *   - If a triangle, it is always equilateral. The side length and the color
 *     is random and the triangle is rotated by a random angle.
 */
class OCRImageSynthesizer {
  /**
   * Constructor of OCRImageSynthesizer.
   *
   * TODO(cais): Revise doc string according to code.
   * TODO(cais): numChars can be made variable?
   *
   * @param {} canvas An HTML canvas object or node-canvas object.
   * @param {*} tensorFlow A tensorflow module passed in. This done for
   *   compatibility between browser and Node.js.
   */
  constructor(canvas, tensorFlow, numChars, locationNoiseLevel = 0.5) {
    this.ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789';
    this.CANDIDATE_FONT_NAMES = ['Arial', 'Courier New', 'Helvetica'];
    this.CANDIDATE_FONT_SIZES = [20, 25, 30];

    this.canvas = canvas;
    tf = tensorFlow;

    // Canvas dimensions.
    this.w = this.canvas.width;
    this.h = this.canvas.height;

    this.alphabetSize = this.ALPHABET.length;

    tf.util.assert(
        numChars > 0 && Number.isInteger(numChars),
        `Expected numChars to be a positive integer, but got ${numChars}`);
    this.numChars = numChars;

    tf.util.assert(
        locationNoiseLevel >= 0 && locationNoiseLevel <= 1,
        `Expected locationNoiseLevel to be >=0 and <=1, ` +
            `but got ${locationNoiseLevel}`);
    this.locationNoiseLevel = locationNoiseLevel;
  }

  generateString_() {
    return tf.tidy(() => {
      let string = '';
      const indices = [];
      for (let i = 0; i < this.numChars; ++i) {
        const index = Math.floor(Math.random() * this.alphabetSize);
        indices.push(index);
        string += this.ALPHABET[index];
      }
      return {
        string,
        target: tf.oneHot(tf.tensor1d(indices, 'int32'), this.alphabetSize)
      };
    });
  }

  /**
   * Generate a single image example.
   *
   * TODO(cais): Revise doc string according to code.
   *
   * @param {number} numCircles Number of circles (background object type)
   *   to include.
   * @param {number} numLines Number of line segments (backgrond object
   *   type) to include
   * @param {number} triangleProbability The probability of the target
   *   object being a triangle (instead of a rectangle). Must be a number
   *   >= 0 and <= 1. Default: 0.5.
   * @returns {Object} An object with the following fields:
   *   - image: A [w, h, 3]-shaped tensor for the pixel content of the image.
   *     w and h are the width and height of the canvas, respectively.
   *   - target: A [numChars, aphabetSize]-shaped tensor, one hot encoding
   *     of the character sequence in the image.
   */
  async generateExample() {
    const {string, target} = this.generateString_();
    console.log(`string = ${string}`);  // DEBUG

    // Determine the location of the characters, with noise added.
    const xSpacing = this.w / (1 + this.numChars);
    console.log(`xSpacing = ${xSpacing}`);  // DEBUG

    const xs = [];  // The x-coordinates of the characters.
    const ys = [];  // The y-coordinates
    for (let i = 0; i < string.length; ++i) {
      xs.push(xSpacing * (0.5 + i) +
          xSpacing * this.locationNoiseLevel * Math.random() / 2);
      ys.push(this.h * (0.5 + this.locationNoiseLevel * Math.random() / 2));
    }
    console.log(xs, ys);  // DEBUG

    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.w, this.h);  // Clear canvas.

    for (let i = 0; i < string.length; ++i) {
      // Draw font name and size randomly.
      const fontName = this.CANDIDATE_FONT_NAMES[Math.floor(
          Math.random() * this.CANDIDATE_FONT_NAMES.length)];
      const fontSize = this.CANDIDATE_FONT_SIZES[Math.floor(
          Math.random() * this.CANDIDATE_FONT_SIZES.length)];
      ctx.font = `${fontSize}px ${fontName}`;
      const x = xs[i];
      const y = ys[i];
      ctx.fillText(string[i], x, y);
    }

    // TODO(cais): Add noise to image.

    console.log('AAA');  // DEBUG
    return {image: tf.fromPixels(this.canvas), target};
  }

  // drawCircle(ctx, centerX, centerY, radius) {
  //   centerX = centerX == null ? this.w * Math.random() : centerX;
  //   centerY = centerY == null ? this.h * Math.random() : centerY;
  //   radius = radius == null ? this.CIRCLE_RADIUS_MIN +
  //           (this.CIRCLE_RADIUS_MAX - this.CIRCLE_RADIUS_MIN) * Math.random() :
  //                             radius;

  //   ctx.fillStyle = generateRandomColorStyle();
  //   ctx.beginPath();
  //   ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  //   ctx.fill();
  // }

  // drawLineSegment(ctx, x0, y0, x1, y1) {
  //   x0 = x0 == null ? Math.random() * this.w : x0;
  //   y0 = y0 == null ? Math.random() * this.h : y0;
  //   x1 = x1 == null ? Math.random() * this.w : x1;
  //   y1 = y1 == null ? Math.random() * this.h : y1;

  //   ctx.strokeStyle = generateRandomColorStyle();
  //   ctx.beginPath();
  //   ctx.moveTo(x0, y0);
  //   ctx.lineTo(x1, y1);
  //   ctx.stroke();
  // }

  // /**
  //  * Draw a rectangle.
  //  *
  //  * A rectangle is a target object in the simple object detection task here.
  //  * Therefore, its bounding box is returned.
  //  *
  //  * @param {} ctx  Canvas context.
  //  * @param {number} centerX Center x-coordinate of the triangle.
  //  * @param {number} centerY Center y-coordinate of the triangle.
  //  * @param {number} w Width of the rectangle.
  //  * @param {number} h Height of the rectangle.
  //  * @param {number} angle Angle that the triangle is rotated for, in radians.
  //  * @returns {[number, number, number, number]} Bounding box of the rectangle:
  //  *   [left, right, top bottom].
  //  */
  // drawRectangle(ctx, centerX, centerY, w, h) {
  //   ctx.moveTo(centerX - w / 2, centerY - h / 2);
  //   ctx.lineTo(centerX + w / 2, centerY - h / 2);
  //   ctx.lineTo(centerX + w / 2, centerY + h / 2);
  //   ctx.lineTo(centerX - w / 2, centerY + h / 2);

  //   return [centerX - w / 2, centerX + w / 2, centerY - h / 2, centerY + h / 2];
  // }

  // /**
  //  * Draw an equilateral triangle.
  //  *
  //  * A triangle are a target object in the simple object detection task here.
  //  * Therefore, its bounding box is returned.
  //  *
  //  * @param {} ctx  Canvas context.
  //  * @param {number} centerX Center x-coordinate of the triangle.
  //  * @param {number} centerY Center y-coordinate of the triangle.
  //  * @param {number} side Length of the side.
  //  * @param {number} angle Angle that the triangle is rotated for, in radians.
  //  * @returns {[number, number, number, number]} Bounding the triangle, with
  //  *   the rotation taken into account: [left, right, top bottom].
  //  */
  // drawTriangle(ctx, centerX, centerY, side, angle) {
  //   const ctrToVertex = side / 2 / Math.cos(30 / 180 * Math.PI);
  //   ctx.fillStyle = generateRandomColorStyle();
  //   ctx.beginPath();

  //   const alpha1 = angle + Math.PI / 2;
  //   const x1 = centerX + Math.cos(alpha1) * ctrToVertex;
  //   const y1 = centerY + Math.sin(alpha1) * ctrToVertex;
  //   const alpha2 = alpha1 + Math.PI / 3 * 2;
  //   const x2 = centerX + Math.cos(alpha2) * ctrToVertex;
  //   const y2 = centerY + Math.sin(alpha2) * ctrToVertex;
  //   const alpha3 = alpha2 + Math.PI / 3 * 2;
  //   const x3 = centerX + Math.cos(alpha3) * ctrToVertex;
  //   const y3 = centerY + Math.sin(alpha3) * ctrToVertex;

  //   ctx.moveTo(x1, y1);
  //   ctx.lineTo(x2, y2);
  //   ctx.lineTo(x3, y3);

  //   const xs = [x1, x2, x3];
  //   const ys = [y1, y2, y3];
  //   return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  // }

  // /**
  //  * Generate a number (i.e., batch) of examples.
  //  *
  //  * @param {number} batchSize Number of example image in the batch.
  //  * @param {number} numCircles Number of circles (background object type)
  //  *   to include.
  //  * @param {number} numLines Number of line segments (background object type)
  //  *   to include.
  //  * @returns {Object} An object with the following fields:
  //  *   - image: A [batchSize, w, h, 3]-shaped tensor for the pixel content of
  //  *     the image. w and h are the width and height of the canvas,
  //  *     respectively.
  //  *   - target: A [batchSize, 5]-shaped tensor. The first column is a 0-1
  //  *     indicator for whether the target is a triangle(0) or a rectangle (1).
  //  *     The remaning four columns are the bounding box of the shape:
  //  *     [left, right, top, bottom], in the unit of pixels.
  //  */
  // async generateExampleBatch(
  //     batchSize, numCircles, numLines, triangleProbability) {
  //   if (triangleProbability == null) {
  //     triangleProbability = 0.5;
  //   }
  //   const imageTensors = [];
  //   const targetTensors = [];
  //   for (let i = 0; i < batchSize; ++i) {
  //     const {image, target} =
  //         await this.generateExample(numCircles, numLines, triangleProbability);
  //     imageTensors.push(image);
  //     targetTensors.push(target);
  //   }
  //   const images = tf.stack(imageTensors);
  //   const targets = tf.stack(targetTensors);
  //   tf.dispose([imageTensors, targetTensors]);
  //   return {images, targets};
  // }
}

module.exports = {OCRImageSynthesizer};
