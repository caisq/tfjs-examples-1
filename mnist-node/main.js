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

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const argparse = require('argparse');
const ProgressBar = require('progress');

const data = require('./data');
const model = require('./model');

async function run(epochs, batchSize, modelSavePath) {
  await data.loadData();

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  model.summary();

  let currentEpoch;
  let progressBar;
  let epochBeginTime;
  let millisPerStep;
  let lossOrMetricTags;
  let lossOrMetricNames;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch =
      trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch =
      Math.ceil(numTrainExamplesPerEpoch / batchSize);
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    callbacks: {
      onEpochBegin: async (epoch) => {
        currentEpoch = epoch;
        epochBeginTime = tf.util.now();
      },
      onBatchEnd: async (batch, logs) => {
        if (batch === 0) {

          console.log(`Epoch ${currentEpoch + 1} / ${epochs}`);
          lossOrMetricTags = [];
          lossOrMetricNames = [];
          for (const key of Object.keys(logs)) {
            if (key !== 'batch' && key !== 'size') {
              lossOrMetricTags.push(`${key}Tag`);
              lossOrMetricNames.push(`${key}`);
            }
          }
          let progressBarSpec = ':bar: eta=:eta ';
          for (let i = 0; i < lossOrMetricTags.length; ++i) {
            progressBarSpec +=
                `:${lossOrMetricTags[i]}=:${lossOrMetricNames[i]}`;
            if (i < lossOrMetricTags.length - 1) {
              progressBarSpec += ' ';
            }
          }
          progressBar = new ProgressBar(
              progressBarSpec, {total: numTrainBatchesPerEpoch, head: `>`});
        }
        if (batch === numTrainBatchesPerEpoch - 1) {
          millisPerStep =
              (tf.util.now() - epochBeginTime) / numTrainExamplesPerEpoch;
        }
        const tickData = {};
        for (let i = 0; i < lossOrMetricNames.length; ++i) {
          tickData[lossOrMetricTags[i]] = lossOrMetricNames[i];
          tickData[lossOrMetricNames[i]] =
              logs[lossOrMetricNames[i]].toFixed(2);
        }
        progressBar.tick(tickData);
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(
            `Loss: ${logs.loss.toFixed(3)} (train), ` +
            `${logs.val_loss.toFixed(3)} (val); ` +
            `Accuracy: ${logs.acc.toFixed(3)} (train), ` +
            `${logs.val_acc.toFixed(3)} (val) ` +
            `(${millisPerStep.toFixed(2)} ms/step)`);
        await tf.nextFrame();
      }
    }
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: 'TensorFlow.js-Node MNIST Example.',
  addHelp: true
});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 20,
  help: 'Number of epochs to train the model for.'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 128,
  help: 'Batch size to be used during model training.'
})
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parseArgs();

run(args.epochs, args.batch_size, args.model_save_path);
