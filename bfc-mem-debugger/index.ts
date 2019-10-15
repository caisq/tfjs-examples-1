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

import * as protobufjs from 'protobufjs';

// TODO(cais): Do not hardcode this.
interface Chunk {
  // uint64 address = 1;
  address: number;
  // int64 size = 2;
  size: number;
  // int64 requested_size = 3;
  requestedSize: number;
  // int32 bin = 4;
  bin: number;
  // string op_name = 5;
  opName: string;
  // uint64 freed_at_count = 6;
  freedAtCount: number;
  // uint64 action_count = 7;
  actionCount: number;
  // bool in_use = 8;
  inUse: boolean;
  // uint64 step_id = 9;
  stepId: number;
}

const openFileButton =
  document.getElementById('open-file') as HTMLButtonElement;
const fileInput =
  document.getElementById('file-input') as HTMLInputElement;

protobufjs.load('./bfc_memory_map.proto', (err, root) =>  {
  if (err) {
    throw err;
  }

  const MemDump = root.lookupType('MemoryDump');

  openFileButton.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (event) => {
    const files = event.target.files as FileList;
    if (files.length > 0) {
      const file = files[0];
      console.log(file.name);  // DEBUG
      const reader = new FileReader();
      reader.onload = () => {
        // console.log(reader.result);  // DEBUG
        const message = MemDump.decode(
            new Uint8Array(reader.result as ArrayBuffer)) as any;
        const chunks = message.chunk as Array<Chunk>;
        console.log('chunks:', chunks);  // DEBUG
        for (let i = 0; i < chunks.length; ++i) {
          const chunk = chunks[i];
          // if (chunk.freedAtCount > 0) {
          console.log(
              `chunk: address=${chunk.address - 140114718097408}, ` +
              `size=${(chunk.size / 1e3).toFixed(1)}k; ` +
              `requestedSize=${(chunk.requestedSize / 1e3).toFixed(1)}; ` +
              `opName=${chunk.opName}; ` +
              `inUse=${chunk.inUse}; ` +
              `actionCount=${chunk.actionCount}; `);
              // `freedAtCount=${chunk.freedAtCount}`);
                // `stepId=${chunk.stepId}`);  // DEBUG
          // }
        }
      };
      reader.readAsArrayBuffer(file);
    }
  });
});
