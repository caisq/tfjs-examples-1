/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

import {MnistDataset} from './data';
import {createModel} from './model';

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js Quantization Example: Training an MNIST Model',
    addHelp: true
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 20,
    help: 'Number of epochs to train the model for.'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during model training.'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: 'Validation split used for training.'
  });
  parser.addArgument('--modelSavePath', {
    type: 'string',
    defaultValue: './models/original',
    help: 'Path to which the model will be saved after training.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (requires CUDA-enabled ' +
    'GPU and supporting drivers and libraries.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    require('@tensorflow/tfjs-node-gpu');
  } else {
    require('@tensorflow/tfjs-node');
  }

  const mnistDataset = new MnistDataset();
  await mnistDataset.loadData();
  const {images: trainImages, labels: trainLabels} =
      mnistDataset.getTrainData();

  const model = createModel();
  model.summary();

  await model.fit(trainImages, trainLabels, {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit
  });

  const {images: testImages, labels: testLabels} = mnistDataset.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(6)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(6)}`);

  if (args.modelSavePath != null) {
    if (!fs.existsSync(path.dirname(args.modelSavePath))) {
      shelljs.mkdir('-p', path.dirname(args.modelSavePath));
    }
    await model.save(`file://${args.modelSavePath}`);
    console.log(`Saved model to path: ${args.modelSavePath}`);
  }
}

if (require.main === module) {
  main();
}
