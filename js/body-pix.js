/**
 * @license
 * Copyright 2023 Google LLC.
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

(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-converter'), require('@tensorflow/tfjs-core')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-converter', '@tensorflow/tfjs-core'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global["body-pix"] = {}, global.tf, global.tf));
})(this, (function (exports, tfconv, tf) { 'use strict';

    function _interopNamespaceDefault(e) {
        var n = Object.create(null);
        if (e) {
            Object.keys(e).forEach(function (k) {
                if (k !== 'default') {
                    var d = Object.getOwnPropertyDescriptor(e, k);
                    Object.defineProperty(n, k, d.get ? d : {
                        enumerable: true,
                        get: function () { return e[k]; }
                    });
                }
            });
        }
        n.default = e;
        return Object.freeze(n);
    }

    var tfconv__namespace = /*#__PURE__*/_interopNamespaceDefault(tfconv);
    var tf__namespace = /*#__PURE__*/_interopNamespaceDefault(tf);

    /*! *****************************************************************************
    Copyright (c) Microsoft Corporation.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
    OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.
    ***************************************************************************** */
    /* global Reflect, Promise */

    var extendStatics = function(d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };

    function __extends(d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    }

    var __assign = function() {
        __assign = Object.assign || function __assign(t) {
            for (var s, i = 1, n = arguments.length; i < n; i++) {
                s = arguments[i];
                for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
            }
            return t;
        };
        return __assign.apply(this, arguments);
    };

    function __awaiter(thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    function __generator(thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (_) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
        }
    }

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
    /**
     * Takes the sigmoid of the part heatmap output and generates a 2d one-hot
     * tensor with ones where the part's score has the maximum value.
     *
     * @param partHeatmapScores
     */
    function toFlattenedOneHotPartMap(partHeatmapScores) {
        var numParts = partHeatmapScores.shape[2];
        var partMapLocations = tf__namespace.argMax(partHeatmapScores, 2);
        var partMapFlattened = tf__namespace.reshape(partMapLocations, [-1]);
        return tf__namespace.oneHot(partMapFlattened, numParts);
    }
    function clipByMask2d(image, mask) {
        return tf__namespace.mul(image, mask);
    }
    /**
     * Takes the sigmoid of the segmentation output, and generates a segmentation
     * mask with a 1 or 0 at each pixel where there is a person or not a person. The
     * segmentation threshold determines the threshold of a score for a pixel for it
     * to be considered part of a person.
     * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
     * @param segmentationThreshold The minimum that segmentation values must have
     * to be considered part of the person.  Affects the generation of the
     * segmentation mask and the clipping of the colored part image.
     *
     * @returns A segmentation mask with a 1 or 0 at each pixel where there is a
     * person or not a person.
     */
    function toMaskTensor(segmentScores, threshold) {
        return tf__namespace.tidy(function () {
            return tf__namespace.cast(tf__namespace.greater(segmentScores, tf__namespace.scalar(threshold)), 'int32');
        });
    }
    /**
     * Takes the sigmoid of the person and part map output, and returns a 2d tensor
     * of an image with the corresponding value at each pixel corresponding to the
     * part with the highest value. These part ids are clipped by the segmentation
     * mask. Wherever the a pixel is clipped by the segmentation mask, its value
     * will set to -1, indicating that there is no part in that pixel.
     * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
     * @param partHeatmapScores A 3d-tensor of the sigmoid of the part heatmap
     * output. The third dimension corresponds to the part.
     *
     * @returns A 2d tensor of an image with the corresponding value at each pixel
     * corresponding to the part with the highest value. These part ids are clipped
     * by the segmentation mask.  It will have values of -1 for pixels that are
     * outside of the body and do not have a corresponding part.
     */
    function decodePartSegmentation(segmentationMask, partHeatmapScores) {
        var _a = partHeatmapScores.shape, partMapHeight = _a[0], partMapWidth = _a[1], numParts = _a[2];
        return tf__namespace.tidy(function () {
            var flattenedMap = toFlattenedOneHotPartMap(partHeatmapScores);
            var partNumbers = tf__namespace.expandDims(tf__namespace.range(0, numParts, 1, 'int32'), 1);
            var partMapFlattened = tf__namespace.cast(tf__namespace.matMul(flattenedMap, partNumbers), 'int32');
            var partMap = tf__namespace.reshape(partMapFlattened, [partMapHeight, partMapWidth]);
            var partMapShiftedUpForClipping = tf__namespace.add(partMap, tf__namespace.scalar(1, 'int32'));
            return tf__namespace.sub(clipByMask2d(partMapShiftedUpForClipping, segmentationMask), tf__namespace.scalar(1, 'int32'));
        });
    }
    function decodeOnlyPartSegmentation(partHeatmapScores) {
        var _a = partHeatmapScores.shape, partMapHeight = _a[0], partMapWidth = _a[1], numParts = _a[2];
        return tf__namespace.tidy(function () {
            var flattenedMap = toFlattenedOneHotPartMap(partHeatmapScores);
            var partNumbers = tf__namespace.expandDims(tf__namespace.range(0, numParts, 1, 'int32'), 1);
            var partMapFlattened = tf__namespace.cast(tf__namespace.matMul(flattenedMap, partNumbers), 'int32');
            return tf__namespace.reshape(partMapFlattened, [partMapHeight, partMapWidth]);
        });
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
     * BodyPix supports using various convolution neural network models
     * (e.g. ResNet and MobileNetV1) as its underlying base model.
     * The following BaseModel interface defines a unified interface for
     * creating such BodyPix base models. Currently both MobileNet (in
     * ./mobilenet.ts) and ResNet (in ./resnet.ts) implements the BaseModel
     * interface. New base models that conform to the BaseModel interface can be
     * added to BodyPix.
     */
    var BaseModel = /** @class */ (function () {
        function BaseModel(model, outputStride) {
            this.model = model;
            this.outputStride = outputStride;
            var inputShape = this.model.inputs[0].shape;
            tf__namespace.util.assert((inputShape[1] === -1) && (inputShape[2] === -1), function () { return "Input shape [".concat(inputShape[1], ", ").concat(inputShape[2], "] ") +
                "must both be equal to or -1"; });
        }
        /**
         * Predicts intermediate Tensor representations.
         *
         * @param input The input RGB image of the base model.
         * A Tensor of shape: [`inputResolution`, `inputResolution`, 3].
         *
         * @return A dictionary of base model's intermediate predictions.
         * The returned dictionary should contains the following elements:
         * - heatmapScores: A Tensor3D that represents the keypoint heatmap scores.
         * - offsets: A Tensor3D that represents the offsets.
         * - displacementFwd: A Tensor3D that represents the forward displacement.
         * - displacementBwd: A Tensor3D that represents the backward displacement.
         * - segmentation: A Tensor3D that represents the segmentation of all
         * people.
         * - longOffsets: A Tensor3D that represents the long offsets used for
         * instance grouping.
         * - partHeatmaps: A Tensor3D that represents the body part segmentation.
         */
        BaseModel.prototype.predict = function (input) {
            var _this = this;
            return tf__namespace.tidy(function () {
                var asFloat = _this.preprocessInput(tf__namespace.cast(input, 'float32'));
                var asBatch = tf__namespace.expandDims(asFloat, 0);
                var results = _this.model.predict(asBatch);
                var results3d = results.map(function (y) { return tf__namespace.squeeze(y, [0]); });
                var namedResults = _this.nameOutputResults(results3d);
                return {
                    heatmapScores: tf__namespace.sigmoid(namedResults.heatmap),
                    offsets: namedResults.offsets,
                    displacementFwd: namedResults.displacementFwd,
                    displacementBwd: namedResults.displacementBwd,
                    segmentation: namedResults.segmentation,
                    partHeatmaps: namedResults.partHeatmaps,
                    longOffsets: namedResults.longOffsets,
                    partOffsets: namedResults.partOffsets
                };
            });
        };
        /**
         * Releases the CPU and GPU memory allocated by the model.
         */
        BaseModel.prototype.dispose = function () {
            this.model.dispose();
        };
        return BaseModel;
    }());

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    var MobileNet = /** @class */ (function (_super) {
        __extends(MobileNet, _super);
        function MobileNet() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        MobileNet.prototype.preprocessInput = function (input) {
            // Normalize the pixels [0, 255] to be between [-1, 1].
            return tf__namespace.tidy(function () { return tf__namespace.sub(tf__namespace.div(input, 127.5), 1.0); });
        };
        MobileNet.prototype.nameOutputResults = function (results) {
            var offsets = results[0], segmentation = results[1], partHeatmaps = results[2], longOffsets = results[3], heatmap = results[4], displacementFwd = results[5], displacementBwd = results[6], partOffsets = results[7];
            return {
                offsets: offsets,
                segmentation: segmentation,
                partHeatmaps: partHeatmaps,
                longOffsets: longOffsets,
                heatmap: heatmap,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
                partOffsets: partOffsets
            };
        };
        return MobileNet;
    }(BaseModel));

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
    var PART_NAMES = [
        'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
        'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
        'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
    ];
    var NUM_KEYPOINTS = PART_NAMES.length;
    var PART_IDS = PART_NAMES.reduce(function (result, jointName, i) {
        result[jointName] = i;
        return result;
    }, {});
    var CONNECTED_PART_NAMES = [
        ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
        ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
        ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
        ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
        ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
    ];
    /*
     * Define the skeleton. This defines the parent->child relationships of our
     * tree. Arbitrarily this defines the nose as the root of the tree, however
     * since we will infer the displacement for both parent->child and
     * child->parent, we can define the tree root as any node.
     */
    var POSE_CHAIN = [
        ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
        ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
        ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
        ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
        ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
        ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
        ['rightKnee', 'rightAnkle']
    ];
    CONNECTED_PART_NAMES.map(function (_a) {
        var jointNameA = _a[0], jointNameB = _a[1];
        return ([PART_IDS[jointNameA], PART_IDS[jointNameB]]);
    });

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
    function getScale(_a, _b, padding) {
        var height = _a[0], width = _a[1];
        var inputResolutionY = _b[0], inputResolutionX = _b[1];
        var padT = padding.top, padB = padding.bottom, padL = padding.left, padR = padding.right;
        var scaleY = inputResolutionY / (padT + padB + height);
        var scaleX = inputResolutionX / (padL + padR + width);
        return [scaleX, scaleY];
    }
    function getOffsetPoint(y, x, keypoint, offsets) {
        return {
            y: offsets.get(y, x, keypoint),
            x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
        };
    }
    function getImageCoords(part, outputStride, offsets) {
        var heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
        var _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets), y = _a.y, x = _a.x;
        return {
            x: part.heatmapX * outputStride + x,
            y: part.heatmapY * outputStride + y
        };
    }
    function clamp(a, min, max) {
        if (a < min) {
            return min;
        }
        if (a > max) {
            return max;
        }
        return a;
    }
    function squaredDistance(y1, x1, y2, x2) {
        var dy = y2 - y1;
        var dx = x2 - x1;
        return dy * dy + dx * dx;
    }
    function addVectors(a, b) {
        return { x: a.x + b.x, y: a.y + b.y };
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    function computeDistance(embedding, pose, minPartScore) {
        if (minPartScore === void 0) { minPartScore = 0.3; }
        var distance = 0.0;
        var numKpt = 0;
        for (var p = 0; p < embedding.length; p++) {
            if (pose.keypoints[p].score > minPartScore) {
                numKpt += 1;
                distance += Math.pow((embedding[p].x - pose.keypoints[p].position.x), 2) +
                    Math.pow((embedding[p].y - pose.keypoints[p].position.y), 2);
            }
        }
        if (numKpt === 0) {
            distance = Infinity;
        }
        else {
            distance = distance / numKpt;
        }
        return distance;
    }
    function convertToPositionInOuput(position, _a, _b, stride) {
        var padT = _a[0], padL = _a[1];
        var scaleX = _b[0], scaleY = _b[1];
        var y = Math.round(((padT + position.y + 1.0) * scaleY - 1.0) / stride);
        var x = Math.round(((padL + position.x + 1.0) * scaleX - 1.0) / stride);
        return { x: x, y: y };
    }
    function getEmbedding(location, keypointIndex, convertToPosition, outputResolutionX, longOffsets, refineSteps, _a) {
        var height = _a[0], width = _a[1];
        var newLocation = convertToPosition(location);
        var nn = newLocation.y * outputResolutionX + newLocation.x;
        var dy = longOffsets[NUM_KEYPOINTS * (2 * nn) + keypointIndex];
        var dx = longOffsets[NUM_KEYPOINTS * (2 * nn + 1) + keypointIndex];
        var y = location.y + dy;
        var x = location.x + dx;
        for (var t = 0; t < refineSteps; t++) {
            y = Math.min(y, height - 1);
            x = Math.min(x, width - 1);
            var newPos = convertToPosition({ x: x, y: y });
            var nn_1 = newPos.y * outputResolutionX + newPos.x;
            dy = longOffsets[NUM_KEYPOINTS * (2 * nn_1) + keypointIndex];
            dx = longOffsets[NUM_KEYPOINTS * (2 * nn_1 + 1) + keypointIndex];
            y = y + dy;
            x = x + dx;
        }
        return { x: x, y: y };
    }
    function matchEmbeddingToInstance(location, longOffsets, poses, numKptForMatching, _a, _b, outputResolutionX, _c, stride, refineSteps) {
        var padT = _a[0], padL = _a[1];
        var scaleX = _b[0], scaleY = _b[1];
        var height = _c[0], width = _c[1];
        var embed = [];
        var convertToPosition = function (pair) {
            return convertToPositionInOuput(pair, [padT, padL], [scaleX, scaleY], stride);
        };
        for (var keypointsIndex = 0; keypointsIndex < numKptForMatching; keypointsIndex++) {
            var embedding = getEmbedding(location, keypointsIndex, convertToPosition, outputResolutionX, longOffsets, refineSteps, [height, width]);
            embed.push(embedding);
        }
        var kMin = -1;
        var kMinDist = Infinity;
        for (var k = 0; k < poses.length; k++) {
            var dist = computeDistance(embed, poses[k]);
            if (dist < kMinDist) {
                kMin = k;
                kMinDist = dist;
            }
        }
        return kMin;
    }
    function getOutputResolution(_a, stride) {
        var inputResolutionY = _a[0], inputResolutionX = _a[1];
        var outputResolutionX = Math.round((inputResolutionX - 1.0) / stride + 1.0);
        var outputResolutionY = Math.round((inputResolutionY - 1.0) / stride + 1.0);
        return [outputResolutionX, outputResolutionY];
    }
    function decodeMultipleMasksCPU(segmentation, longOffsets, posesAboveScore, height, width, stride, _a, padding, refineSteps, numKptForMatching) {
        var inHeight = _a[0], inWidth = _a[1];
        if (numKptForMatching === void 0) { numKptForMatching = 5; }
        var dataArrays = posesAboveScore.map(function (x) { return new Uint8Array(height * width).fill(0); });
        var padT = padding.top, padL = padding.left;
        var _b = getScale([height, width], [inHeight, inWidth], padding), scaleX = _b[0], scaleY = _b[1];
        var outputResolutionX = getOutputResolution([inHeight, inWidth], stride)[0];
        for (var i = 0; i < height; i += 1) {
            for (var j = 0; j < width; j += 1) {
                var n = i * width + j;
                var prob = segmentation[n];
                if (prob === 1) {
                    var kMin = matchEmbeddingToInstance({ x: j, y: i }, longOffsets, posesAboveScore, numKptForMatching, [padT, padL], [scaleX, scaleY], outputResolutionX, [height, width], stride, refineSteps);
                    if (kMin >= 0) {
                        dataArrays[kMin][n] = 1;
                    }
                }
            }
        }
        return dataArrays;
    }
    function decodeMultiplePartMasksCPU(segmentation, longOffsets, partSegmentaion, posesAboveScore, height, width, stride, _a, padding, refineSteps, numKptForMatching) {
        var inHeight = _a[0], inWidth = _a[1];
        if (numKptForMatching === void 0) { numKptForMatching = 5; }
        var dataArrays = posesAboveScore.map(function (x) { return new Int32Array(height * width).fill(-1); });
        var padT = padding.top, padL = padding.left;
        var _b = getScale([height, width], [inHeight, inWidth], padding), scaleX = _b[0], scaleY = _b[1];
        var outputResolutionX = getOutputResolution([inHeight, inWidth], stride)[0];
        for (var i = 0; i < height; i += 1) {
            for (var j = 0; j < width; j += 1) {
                var n = i * width + j;
                var prob = segmentation[n];
                if (prob === 1) {
                    var kMin = matchEmbeddingToInstance({ x: j, y: i }, longOffsets, posesAboveScore, numKptForMatching, [padT, padL], [scaleX, scaleY], outputResolutionX, [height, width], stride, refineSteps);
                    if (kMin >= 0) {
                        dataArrays[kMin][n] = partSegmentaion[n];
                    }
                }
            }
        }
        return dataArrays;
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    function decodeMultipleMasksWebGl(segmentation, longOffsets, posesAboveScore, height, width, stride, _a, padding, refineSteps, minKptScore, maxNumPeople) {
        var inHeight = _a[0], inWidth = _a[1];
        // The height/width of the image/canvas itself.
        var _b = segmentation.shape, origHeight = _b[0], origWidth = _b[1];
        // The height/width of the output of the model.
        var _c = longOffsets.shape.slice(0, 2), outHeight = _c[0], outWidth = _c[1];
        var shapedLongOffsets = tf__namespace.reshape(longOffsets, [outHeight, outWidth, 2, NUM_KEYPOINTS]);
        // Make pose tensor of shape [MAX_NUM_PEOPLE, NUM_KEYPOINTS, 3] where
        // the last 3 coordinates correspond to the score, h and w coordinate of that
        // keypoint.
        var poseVals = new Float32Array(maxNumPeople * NUM_KEYPOINTS * 3).fill(0.0);
        for (var i = 0; i < posesAboveScore.length; i++) {
            var poseOffset = i * NUM_KEYPOINTS * 3;
            var pose = posesAboveScore[i];
            for (var kp = 0; kp < NUM_KEYPOINTS; kp++) {
                var keypoint = pose.keypoints[kp];
                var offset = poseOffset + kp * 3;
                poseVals[offset] = keypoint.score;
                poseVals[offset + 1] = keypoint.position.y;
                poseVals[offset + 2] = keypoint.position.x;
            }
        }
        var _d = getScale([height, width], [inHeight, inWidth], padding), scaleX = _d[0], scaleY = _d[1];
        var posesTensor = tf__namespace.tensor(poseVals, [maxNumPeople, NUM_KEYPOINTS, 3]);
        var padT = padding.top, padL = padding.left;
        var program = {
            variableNames: ['segmentation', 'longOffsets', 'poses'],
            outputShape: [origHeight, origWidth],
            userCode: "\n    int convertToPositionInOutput(int pos, int pad, float scale, int stride) {\n      return round(((float(pos + pad) + 1.0) * scale - 1.0) / float(stride));\n    }\n\n    float convertToPositionInOutputFloat(\n        int pos, int pad, float scale, int stride) {\n      return ((float(pos + pad) + 1.0) * scale - 1.0) / float(stride);\n    }\n\n    float dist(float x1, float y1, float x2, float y2) {\n      return pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0);\n    }\n\n    float sampleLongOffsets(float h, float w, int d, int k) {\n      float fh = fract(h);\n      float fw = fract(w);\n      int clH = int(ceil(h));\n      int clW = int(ceil(w));\n      int flH = int(floor(h));\n      int flW = int(floor(w));\n      float o11 = getLongOffsets(flH, flW, d, k);\n      float o12 = getLongOffsets(flH, clW, d, k);\n      float o21 = getLongOffsets(clH, flW, d, k);\n      float o22 = getLongOffsets(clH, clW, d, k);\n      float o1 = mix(o11, o12, fw);\n      float o2 = mix(o21, o22, fw);\n      return mix(o1, o2, fh);\n    }\n\n    int findNearestPose(int h, int w) {\n      float prob = getSegmentation(h, w);\n      if (prob < 1.0) {\n        return -1;\n      }\n\n      // Done(Tyler): convert from output space h/w to strided space.\n      float stridedH = convertToPositionInOutputFloat(\n        h, ".concat(padT, ", ").concat(scaleY, ", ").concat(stride, ");\n      float stridedW = convertToPositionInOutputFloat(\n        w, ").concat(padL, ", ").concat(scaleX, ", ").concat(stride, ");\n\n      float minDist = 1000000.0;\n      int iMin = -1;\n      for (int i = 0; i < ").concat(maxNumPeople, "; i++) {\n        float curDistSum = 0.0;\n        int numKpt = 0;\n        for (int k = 0; k < ").concat(NUM_KEYPOINTS, "; k++) {\n          float dy = sampleLongOffsets(stridedH, stridedW, 0, k);\n          float dx = sampleLongOffsets(stridedH, stridedW, 1, k);\n\n          float y = float(h) + dy;\n          float x = float(w) + dx;\n\n          for (int s = 0; s < ").concat(refineSteps, "; s++) {\n            int yRounded = round(min(y, float(").concat(height - 1.0, ")));\n            int xRounded = round(min(x, float(").concat(width - 1.0, ")));\n\n            float yStrided = convertToPositionInOutputFloat(\n              yRounded, ").concat(padT, ", ").concat(scaleY, ", ").concat(stride, ");\n            float xStrided = convertToPositionInOutputFloat(\n              xRounded, ").concat(padL, ", ").concat(scaleX, ", ").concat(stride, ");\n\n            float dy = sampleLongOffsets(yStrided, xStrided, 0, k);\n            float dx = sampleLongOffsets(yStrided, xStrided, 1, k);\n\n            y = y + dy;\n            x = x + dx;\n          }\n\n          float poseScore = getPoses(i, k, 0);\n          float poseY = getPoses(i, k, 1);\n          float poseX = getPoses(i, k, 2);\n          if (poseScore > ").concat(minKptScore, ") {\n            numKpt = numKpt + 1;\n            curDistSum = curDistSum + dist(x, y, poseX, poseY);\n          }\n        }\n        if (numKpt > 0 && curDistSum / float(numKpt) < minDist) {\n          minDist = curDistSum / float(numKpt);\n          iMin = i;\n        }\n      }\n      return iMin;\n    }\n\n    void main() {\n        ivec2 coords = getOutputCoords();\n        int nearestPose = findNearestPose(coords[0], coords[1]);\n        setOutput(float(nearestPose));\n      }\n  ")
        };
        var webglBackend = tf__namespace.backend();
        return webglBackend.compileAndRun(program, [segmentation, shapedLongOffsets, posesTensor]);
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    function toPersonKSegmentation(segmentation, k) {
        return tf__namespace.tidy(function () { return tf__namespace.cast(tf__namespace.equal(segmentation, tf__namespace.scalar(k)), 'int32'); });
    }
    function toPersonKPartSegmentation(segmentation, bodyParts, k) {
        return tf__namespace.tidy(function () { return tf__namespace.sub(tf__namespace.mul(tf__namespace.cast(tf__namespace.equal(segmentation, tf__namespace.scalar(k)), 'int32'), tf__namespace.add(bodyParts, 1)), 1); });
    }
    function isWebGlBackend() {
        return tf.getBackend() === 'webgl';
    }
    function decodePersonInstanceMasks(segmentation, longOffsets, poses, height, width, stride, _a, padding, minPoseScore, refineSteps, minKeypointScore, maxNumPeople) {
        var inHeight = _a[0], inWidth = _a[1];
        if (minPoseScore === void 0) { minPoseScore = 0.2; }
        if (refineSteps === void 0) { refineSteps = 8; }
        if (minKeypointScore === void 0) { minKeypointScore = 0.3; }
        if (maxNumPeople === void 0) { maxNumPeople = 10; }
        return __awaiter(this, void 0, void 0, function () {
            var posesAboveScore, personSegmentationsData, personSegmentations, segmentationsData, longOffsetsData;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        posesAboveScore = poses.filter(function (pose) { return pose.score >= minPoseScore; });
                        if (!isWebGlBackend()) return [3 /*break*/, 2];
                        personSegmentations = tf__namespace.tidy(function () {
                            var masksTensorInfo = decodeMultipleMasksWebGl(segmentation, longOffsets, posesAboveScore, height, width, stride, [inHeight, inWidth], padding, refineSteps, minKeypointScore, maxNumPeople);
                            var masksTensor = tf__namespace.engine().makeTensorFromDataId(masksTensorInfo.dataId, masksTensorInfo.shape, masksTensorInfo.dtype);
                            return posesAboveScore.map(function (_, k) { return toPersonKSegmentation(masksTensor, k); });
                        });
                        return [4 /*yield*/, Promise.all(personSegmentations.map(function (mask) { return mask.data(); }))];
                    case 1:
                        personSegmentationsData =
                            (_b.sent());
                        personSegmentations.forEach(function (x) { return x.dispose(); });
                        return [3 /*break*/, 5];
                    case 2: return [4 /*yield*/, segmentation.data()];
                    case 3:
                        segmentationsData = _b.sent();
                        return [4 /*yield*/, longOffsets.data()];
                    case 4:
                        longOffsetsData = _b.sent();
                        personSegmentationsData = decodeMultipleMasksCPU(segmentationsData, longOffsetsData, posesAboveScore, height, width, stride, [inHeight, inWidth], padding, refineSteps);
                        _b.label = 5;
                    case 5: return [2 /*return*/, personSegmentationsData.map(function (data, i) { return ({ data: data, pose: posesAboveScore[i], width: width, height: height }); })];
                }
            });
        });
    }
    function decodePersonInstancePartMasks(segmentation, longOffsets, partSegmentation, poses, height, width, stride, _a, padding, minPoseScore, refineSteps, minKeypointScore, maxNumPeople) {
        var inHeight = _a[0], inWidth = _a[1];
        if (minPoseScore === void 0) { minPoseScore = 0.2; }
        if (refineSteps === void 0) { refineSteps = 8; }
        if (minKeypointScore === void 0) { minKeypointScore = 0.3; }
        if (maxNumPeople === void 0) { maxNumPeople = 10; }
        return __awaiter(this, void 0, void 0, function () {
            var posesAboveScore, partSegmentationsByPersonData, partSegmentations, segmentationsData, longOffsetsData, partSegmentaionData;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        posesAboveScore = poses.filter(function (pose) { return pose.score >= minPoseScore; });
                        if (!isWebGlBackend()) return [3 /*break*/, 2];
                        partSegmentations = tf__namespace.tidy(function () {
                            var masksTensorInfo = decodeMultipleMasksWebGl(segmentation, longOffsets, posesAboveScore, height, width, stride, [inHeight, inWidth], padding, refineSteps, minKeypointScore, maxNumPeople);
                            var masksTensor = tf__namespace.engine().makeTensorFromDataId(masksTensorInfo.dataId, masksTensorInfo.shape, masksTensorInfo.dtype);
                            return posesAboveScore.map(function (_, k) {
                                return toPersonKPartSegmentation(masksTensor, partSegmentation, k);
                            });
                        });
                        return [4 /*yield*/, Promise.all(partSegmentations.map(function (x) { return x.data(); }))];
                    case 1:
                        partSegmentationsByPersonData =
                            (_b.sent());
                        partSegmentations.forEach(function (x) { return x.dispose(); });
                        return [3 /*break*/, 6];
                    case 2: return [4 /*yield*/, segmentation.data()];
                    case 3:
                        segmentationsData = _b.sent();
                        return [4 /*yield*/, longOffsets.data()];
                    case 4:
                        longOffsetsData = _b.sent();
                        return [4 /*yield*/, partSegmentation.data()];
                    case 5:
                        partSegmentaionData = _b.sent();
                        partSegmentationsByPersonData = decodeMultiplePartMasksCPU(segmentationsData, longOffsetsData, partSegmentaionData, posesAboveScore, height, width, stride, [inHeight, inWidth], padding, refineSteps);
                        _b.label = 6;
                    case 6: return [2 /*return*/, partSegmentationsByPersonData.map(function (data, k) { return ({ pose: posesAboveScore[k], data: data, height: height, width: width }); })];
                }
            });
        });
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    // algorithm based on Coursera Lecture from Algorithms, Part 1:
    // https://www.coursera.org/learn/algorithms-part1/lecture/ZjoSM/heapsort
    function half(k) {
        return Math.floor(k / 2);
    }
    var MaxHeap = /** @class */ (function () {
        function MaxHeap(maxSize, getElementValue) {
            this.priorityQueue = new Array(maxSize);
            this.numberOfElements = -1;
            this.getElementValue = getElementValue;
        }
        MaxHeap.prototype.enqueue = function (x) {
            this.priorityQueue[++this.numberOfElements] = x;
            this.swim(this.numberOfElements);
        };
        MaxHeap.prototype.dequeue = function () {
            var max = this.priorityQueue[0];
            this.exchange(0, this.numberOfElements--);
            this.sink(0);
            this.priorityQueue[this.numberOfElements + 1] = null;
            return max;
        };
        MaxHeap.prototype.empty = function () {
            return this.numberOfElements === -1;
        };
        MaxHeap.prototype.size = function () {
            return this.numberOfElements + 1;
        };
        MaxHeap.prototype.all = function () {
            return this.priorityQueue.slice(0, this.numberOfElements + 1);
        };
        MaxHeap.prototype.max = function () {
            return this.priorityQueue[0];
        };
        MaxHeap.prototype.swim = function (k) {
            while (k > 0 && this.less(half(k), k)) {
                this.exchange(k, half(k));
                k = half(k);
            }
        };
        MaxHeap.prototype.sink = function (k) {
            while (2 * k <= this.numberOfElements) {
                var j = 2 * k;
                if (j < this.numberOfElements && this.less(j, j + 1)) {
                    j++;
                }
                if (!this.less(k, j)) {
                    break;
                }
                this.exchange(k, j);
                k = j;
            }
        };
        MaxHeap.prototype.getValueAt = function (i) {
            return this.getElementValue(this.priorityQueue[i]);
        };
        MaxHeap.prototype.less = function (i, j) {
            return this.getValueAt(i) < this.getValueAt(j);
        };
        MaxHeap.prototype.exchange = function (i, j) {
            var t = this.priorityQueue[i];
            this.priorityQueue[i] = this.priorityQueue[j];
            this.priorityQueue[j] = t;
        };
        return MaxHeap;
    }());

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
    function scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores) {
        var _a = scores.shape, height = _a[0], width = _a[1];
        var localMaximum = true;
        var yStart = Math.max(heatmapY - localMaximumRadius, 0);
        var yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
        for (var yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            var xStart = Math.max(heatmapX - localMaximumRadius, 0);
            var xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
            for (var xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (scores.get(yCurrent, xCurrent, keypointId) > score) {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum) {
                break;
            }
        }
        return localMaximum;
    }
    /**
     * Builds a priority queue with part candidate positions for a specific image in
     * the batch. For this we find all local maxima in the score maps with score
     * values above a threshold. We create a single priority queue across all parts.
     */
    function buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores) {
        var _a = scores.shape, height = _a[0], width = _a[1], numKeypoints = _a[2];
        var queue = new MaxHeap(height * width * numKeypoints, function (_a) {
            var score = _a.score;
            return score;
        });
        for (var heatmapY = 0; heatmapY < height; ++heatmapY) {
            for (var heatmapX = 0; heatmapX < width; ++heatmapX) {
                for (var keypointId = 0; keypointId < numKeypoints; ++keypointId) {
                    var score = scores.get(heatmapY, heatmapX, keypointId);
                    // Only consider parts with score greater or equal to threshold as
                    // root candidates.
                    if (score < scoreThreshold) {
                        continue;
                    }
                    // Only consider keypoints whose score is maximum in a local window.
                    if (scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        queue.enqueue({ score: score, part: { heatmapY: heatmapY, heatmapX: heatmapX, id: keypointId } });
                    }
                }
            }
        }
        return queue;
    }

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
    var parentChildrenTuples = POSE_CHAIN.map(function (_a) {
        var parentJoinName = _a[0], childJoinName = _a[1];
        return ([PART_IDS[parentJoinName], PART_IDS[childJoinName]]);
    });
    var parentToChildEdges = parentChildrenTuples.map(function (_a) {
        var childJointId = _a[1];
        return childJointId;
    });
    var childToParentEdges = parentChildrenTuples.map(function (_a) {
        var parentJointId = _a[0];
        return parentJointId;
    });
    function getDisplacement(edgeId, point, displacements) {
        var numEdges = displacements.shape[2] / 2;
        return {
            y: displacements.get(point.y, point.x, edgeId),
            x: displacements.get(point.y, point.x, numEdges + edgeId)
        };
    }
    function getStridedIndexNearPoint(point, outputStride, height, width) {
        return {
            y: clamp(Math.round(point.y / outputStride), 0, height - 1),
            x: clamp(Math.round(point.x / outputStride), 0, width - 1)
        };
    }
    /**
     * We get a new keypoint along the `edgeId` for the pose instance, assuming
     * that the position of the `idSource` part is already known. For this, we
     * follow the displacement vector from the source to target part (stored in
     * the `i`-t channel of the displacement tensor). The displaced keypoint
     * vector is refined using the offset vector by `offsetRefineStep` times.
     */
    function traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets, outputStride, displacements, offsetRefineStep) {
        if (offsetRefineStep === void 0) { offsetRefineStep = 2; }
        var _a = scoresBuffer.shape, height = _a[0], width = _a[1];
        // Nearest neighbor interpolation for the source->target displacements.
        var sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypoint.position, outputStride, height, width);
        var displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);
        var displacedPoint = addVectors(sourceKeypoint.position, displacement);
        var targetKeypoint = displacedPoint;
        for (var i = 0; i < offsetRefineStep; i++) {
            var targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);
            var offsetPoint = getOffsetPoint(targetKeypointIndices.y, targetKeypointIndices.x, targetKeypointId, offsets);
            targetKeypoint = addVectors({
                x: targetKeypointIndices.x * outputStride,
                y: targetKeypointIndices.y * outputStride
            }, { x: offsetPoint.x, y: offsetPoint.y });
        }
        var targetKeyPointIndices = getStridedIndexNearPoint(targetKeypoint, outputStride, height, width);
        var score = scoresBuffer.get(targetKeyPointIndices.y, targetKeyPointIndices.x, targetKeypointId);
        return { position: targetKeypoint, part: PART_NAMES[targetKeypointId], score: score };
    }
    /**
     * Follows the displacement fields to decode the full pose of the object
     * instance given the position of a part that acts as root.
     *
     * @return An array of decoded keypoints and their scores for a single pose
     */
    function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd) {
        var numParts = scores.shape[2];
        var numEdges = parentToChildEdges.length;
        var instanceKeypoints = new Array(numParts);
        // Start a new detection instance at the position of the root.
        var rootPart = root.part, rootScore = root.score;
        var rootPoint = getImageCoords(rootPart, outputStride, offsets);
        instanceKeypoints[rootPart.id] = {
            score: rootScore,
            part: PART_NAMES[rootPart.id],
            position: rootPoint
        };
        // Decode the part positions upwards in the tree, following the backward
        // displacements.
        for (var edge = numEdges - 1; edge >= 0; --edge) {
            var sourceKeypointId = parentToChildEdges[edge];
            var targetKeypointId = childToParentEdges[edge];
            if (instanceKeypoints[sourceKeypointId] &&
                !instanceKeypoints[targetKeypointId]) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsBwd);
            }
        }
        // Decode the part positions downwards in the tree, following the forward
        // displacements.
        for (var edge = 0; edge < numEdges; ++edge) {
            var sourceKeypointId = childToParentEdges[edge];
            var targetKeypointId = parentToChildEdges[edge];
            if (instanceKeypoints[sourceKeypointId] &&
                !instanceKeypoints[targetKeypointId]) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsFwd);
            }
        }
        return instanceKeypoints;
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    function withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, _a, keypointId) {
        var x = _a.x, y = _a.y;
        return poses.some(function (_a) {
            var keypoints = _a.keypoints;
            var correspondingKeypoint = keypoints[keypointId].position;
            return squaredDistance(y, x, correspondingKeypoint.y, correspondingKeypoint.x) <=
                squaredNmsRadius;
        });
    }
    /* Score the newly proposed object instance without taking into account
     * the scores of the parts that overlap with any previously detected
     * instance.
     */
    function getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints) {
        var notOverlappedKeypointScores = instanceKeypoints.reduce(function (result, _a, keypointId) {
            var position = _a.position, score = _a.score;
            if (!withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, position, keypointId)) {
                result += score;
            }
            return result;
        }, 0.0);
        return notOverlappedKeypointScores /= instanceKeypoints.length;
    }
    // A point (y, x) is considered as root part candidate if its score is a
    // maximum in a window |y - y'| <= kLocalMaximumRadius, |x - x'| <=
    // kLocalMaximumRadius.
    var kLocalMaximumRadius = 1;
    /**
     * Detects multiple poses and finds their parts from part scores and
     * displacement vectors. It returns up to `maxDetections` object instance
     * detections in decreasing root score order. It works as follows: We first
     * create a priority queue with local part score maxima above
     * `scoreThreshold`, considering all parts at the same time. Then we
     * iteratively pull the top  element of the queue (in decreasing score order)
     * and treat it as a root candidate for a new object instance. To avoid
     * duplicate detections, we reject the root candidate if it is within a disk
     * of `nmsRadius` pixels from the corresponding part of a previously detected
     * instance, which is a form of part-based non-maximum suppression (NMS). If
     * the root candidate passes the NMS check, we start a new object instance
     * detection, treating the corresponding part as root and finding the
     * positions of the remaining parts by following the displacement vectors
     * along the tree-structured part graph. We assign to the newly detected
     * instance a score equal to the sum of scores of its parts which have not
     * been claimed by a previous instance (i.e., those at least `nmsRadius`
     * pixels away from the corresponding part of all previously detected
     * instances), divided by the total number of parts `numParts`.
     *
     * @param heatmapScores 3-D tensor with shape `[height, width, numParts]`.
     * The value of heatmapScores[y, x, k]` is the score of placing the `k`-th
     * object part at position `(y, x)`.
     *
     * @param offsets 3-D tensor with shape `[height, width, numParts * 2]`.
     * The value of [offsets[y, x, k], offsets[y, x, k + numParts]]` is the
     * short range offset vector of the `k`-th  object part at heatmap
     * position `(y, x)`.
     *
     * @param displacementsFwd 3-D tensor of shape
     * `[height, width, 2 * num_edges]`, where `num_edges = num_parts - 1` is the
     * number of edges (parent-child pairs) in the tree. It contains the forward
     * displacements between consecutive part from the root towards the leaves.
     *
     * @param displacementsBwd 3-D tensor of shape
     * `[height, width, 2 * num_edges]`, where `num_edges = num_parts - 1` is the
     * number of edges (parent-child pairs) in the tree. It contains the backward
     * displacements between consecutive part from the root towards the leaves.
     *
     * @param outputStride The output stride that was used when feed-forwarding
     * through the PoseNet model.  Must be 32, 16, or 8.
     *
     * @param maxPoseDetections Maximum number of returned instance detections per
     * image.
     *
     * @param scoreThreshold Only return instance detections that have root part
     * score greater or equal to this value. Defaults to 0.5.
     *
     * @param nmsRadius Non-maximum suppression part distance. It needs to be
     * strictly positive. Two parts suppress each other if they are less than
     * `nmsRadius` pixels away. Defaults to 20.
     *
     * @return An array of poses and their scores, each containing keypoints and
     * the corresponding keypoint scores.
     */
    function decodeMultiplePoses(scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer, outputStride, maxPoseDetections, scoreThreshold, nmsRadius) {
        if (scoreThreshold === void 0) { scoreThreshold = 0.5; }
        if (nmsRadius === void 0) { nmsRadius = 20; }
        var poses = [];
        var queue = buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, scoresBuffer);
        var squaredNmsRadius = nmsRadius * nmsRadius;
        // Generate at most maxDetections object instances per image in
        // decreasing root part score order.
        while (poses.length < maxPoseDetections && !queue.empty()) {
            // The top element in the queue is the next root candidate.
            var root = queue.dequeue();
            // Part-based non-maximum suppression: We reject a root candidate if it
            // is within a disk of `nmsRadius` pixels from the corresponding part of
            // a previously detected instance.
            var rootImageCoords = getImageCoords(root.part, outputStride, offsetsBuffer);
            if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root.part.id)) {
                continue;
            }
            // Start a new detection instance at the position of the root.
            var keypoints = decodePose(root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer, displacementsBwdBuffer);
            var score = getInstanceScore(poses, squaredNmsRadius, keypoints);
            poses.push({ keypoints: keypoints, score: score });
        }
        return poses;
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     * https://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     * =============================================================================
     */
    var imageNetMean = [-123.15, -115.90, -103.06];
    var ResNet = /** @class */ (function (_super) {
        __extends(ResNet, _super);
        function ResNet() {
            return _super !== null && _super.apply(this, arguments) || this;
        }
        ResNet.prototype.preprocessInput = function (input) {
            return tf__namespace.add(input, imageNetMean);
        };
        ResNet.prototype.nameOutputResults = function (results) {
            var displacementBwd = results[0], displacementFwd = results[1], heatmap = results[2], longOffsets = results[3], offsets = results[4], partHeatmaps = results[5], segmentation = results[6], partOffsets = results[7];
            return {
                offsets: offsets,
                segmentation: segmentation,
                partHeatmaps: partHeatmaps,
                longOffsets: longOffsets,
                heatmap: heatmap,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
                partOffsets: partOffsets
            };
        };
        return ResNet;
    }(BaseModel));

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     * https://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     * =============================================================================
     */
    var RESNET50_BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/';
    var MOBILENET_BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/';
    // The BodyPix 2.0 ResNet50 models use the latest TensorFlow.js 1.0 model
    // format.
    function resNet50SavedModel(stride, quantBytes) {
        var graphJson = "model-stride".concat(stride, ".json");
        // quantBytes=4 corresponding to the non-quantized full-precision SavedModel.
        if (quantBytes === 4) {
            return RESNET50_BASE_URL + "float/" + graphJson;
        }
        else {
            return RESNET50_BASE_URL + "quant".concat(quantBytes, "/") + graphJson;
        }
    }
    // The BodyPix 2.0 MobileNetV1 models use the latest TensorFlow.js 1.0 model
    // format.
    function mobileNetSavedModel(stride, multiplier, quantBytes) {
        var toStr = { 1.0: '100', 0.75: '075', 0.50: '050' };
        var graphJson = "model-stride".concat(stride, ".json");
        // quantBytes=4 corresponding to the non-quantized full-precision SavedModel.
        if (quantBytes === 4) {
            return MOBILENET_BASE_URL + "float/".concat(toStr[multiplier], "/") + graphJson;
        }
        else {
            return MOBILENET_BASE_URL + "quant".concat(quantBytes, "/").concat(toStr[multiplier], "/") +
                graphJson;
        }
    }

    /**
     * @license
     * Copyright 2020 Google Inc. All Rights Reserved.
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
     *
     * =============================================================================
     */
    var _a;
    function getSizeFromImageLikeElement(input) {
        if ('offsetHeight' in input && input.offsetHeight !== 0
            && 'offsetWidth' in input && input.offsetWidth !== 0) {
            return [input.offsetHeight, input.offsetWidth];
        }
        else if (input.height != null && input.width != null) {
            return [input.height, input.width];
        }
        else {
            throw new Error("HTMLImageElement must have height and width attributes set.");
        }
    }
    function getSizeFromVideoElement(input) {
        if (input.hasAttribute('height') && input.hasAttribute('width')) {
            // Prioritizes user specified height and width.
            // We can't test the .height and .width properties directly,
            // because they evaluate to 0 if unset.
            return [input.height, input.width];
        }
        else {
            return [input.videoHeight, input.videoWidth];
        }
    }
    function getInputSize(input) {
        if ((typeof (HTMLCanvasElement) !== 'undefined' &&
            input instanceof HTMLCanvasElement) ||
            (typeof (OffscreenCanvas) !== 'undefined' &&
                input instanceof OffscreenCanvas) ||
            (typeof (HTMLImageElement) !== 'undefined' &&
                input instanceof HTMLImageElement)) {
            return getSizeFromImageLikeElement(input);
        }
        else if (typeof (ImageData) !== 'undefined' && input instanceof ImageData) {
            return [input.height, input.width];
        }
        else if (typeof (HTMLVideoElement) !== 'undefined' &&
            input instanceof HTMLVideoElement) {
            return getSizeFromVideoElement(input);
        }
        else if (input instanceof tf__namespace.Tensor) {
            return [input.shape[0], input.shape[1]];
        }
        else {
            throw new Error("error: Unknown input type: ".concat(input, "."));
        }
    }
    function isValidInputResolution(resolution, outputStride) {
        return (resolution - 1) % outputStride === 0;
    }
    function toValidInputResolution(inputResolution, outputStride) {
        if (isValidInputResolution(inputResolution, outputStride)) {
            return inputResolution;
        }
        return Math.floor(inputResolution / outputStride) * outputStride + 1;
    }
    var INTERNAL_RESOLUTION_STRING_OPTIONS = {
        low: 'low',
        medium: 'medium',
        high: 'high',
        full: 'full'
    };
    var INTERNAL_RESOLUTION_PERCENTAGES = (_a = {},
        _a[INTERNAL_RESOLUTION_STRING_OPTIONS.low] = 0.25,
        _a[INTERNAL_RESOLUTION_STRING_OPTIONS.medium] = 0.5,
        _a[INTERNAL_RESOLUTION_STRING_OPTIONS.high] = 0.75,
        _a[INTERNAL_RESOLUTION_STRING_OPTIONS.full] = 1.0,
        _a);
    var MIN_INTERNAL_RESOLUTION = 0.1;
    var MAX_INTERNAL_RESOLUTION = 2.0;
    function toInternalResolutionPercentage(internalResolution) {
        if (typeof internalResolution === 'string') {
            var result = INTERNAL_RESOLUTION_PERCENTAGES[internalResolution];
            tf__namespace.util.assert(typeof result === 'number', function () { return "string value of inputResolution must be one of ".concat(Object.values(INTERNAL_RESOLUTION_STRING_OPTIONS)
                .join(','), " but was ").concat(internalResolution, "."); });
            return result;
        }
        else {
            tf__namespace.util.assert(typeof internalResolution === 'number' &&
                internalResolution <= MAX_INTERNAL_RESOLUTION &&
                internalResolution >= MIN_INTERNAL_RESOLUTION, function () {
                return "inputResolution must be a string or number between ".concat(MIN_INTERNAL_RESOLUTION, " and ").concat(MAX_INTERNAL_RESOLUTION, ", but ") +
                    "was ".concat(internalResolution);
            });
            return internalResolution;
        }
    }
    function toInputResolutionHeightAndWidth(internalResolution, outputStride, _a) {
        var inputHeight = _a[0], inputWidth = _a[1];
        var internalResolutionPercentage = toInternalResolutionPercentage(internalResolution);
        return [
            toValidInputResolution(inputHeight * internalResolutionPercentage, outputStride),
            toValidInputResolution(inputWidth * internalResolutionPercentage, outputStride)
        ];
    }
    function toInputTensor(input) {
        // TODO: tf.browser.fromPixels types to support OffscreenCanvas
        // @ts-ignore
        return input instanceof tf__namespace.Tensor ? input : tf__namespace.browser.fromPixels(input);
    }
    function resizeAndPadTo(imageTensor, _a, flipHorizontal) {
        var targetH = _a[0], targetW = _a[1];
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        var _b = imageTensor.shape, height = _b[0], width = _b[1];
        var targetAspect = targetW / targetH;
        var aspect = width / height;
        var resizeW;
        var resizeH;
        var padL;
        var padR;
        var padT;
        var padB;
        if (aspect > targetAspect) {
            // resize to have the larger dimension match the shape.
            resizeW = targetW;
            resizeH = Math.ceil(resizeW / aspect);
            var padHeight = targetH - resizeH;
            padL = 0;
            padR = 0;
            padT = Math.floor(padHeight / 2);
            padB = targetH - (resizeH + padT);
        }
        else {
            resizeH = targetH;
            resizeW = Math.ceil(targetH * aspect);
            var padWidth = targetW - resizeW;
            padL = Math.floor(padWidth / 2);
            padR = targetW - (resizeW + padL);
            padT = 0;
            padB = 0;
        }
        var resizedAndPadded = tf__namespace.tidy(function () {
            // resize to have largest dimension match image
            var resized;
            if (flipHorizontal) {
                resized = tf__namespace.image.resizeBilinear(tf__namespace.reverse(imageTensor, 1), [resizeH, resizeW]);
            }
            else {
                resized = tf__namespace.image.resizeBilinear(imageTensor, [resizeH, resizeW]);
            }
            var padded = tf__namespace.pad3d(resized, [[padT, padB], [padL, padR], [0, 0]]);
            return padded;
        });
        return { resizedAndPadded: resizedAndPadded, paddedBy: [[padT, padB], [padL, padR]] };
    }
    function scaleAndCropToInputTensorShape(tensor, _a, _b, _c, applySigmoidActivation) {
        var inputTensorHeight = _a[0], inputTensorWidth = _a[1];
        var resizedAndPaddedHeight = _b[0], resizedAndPaddedWidth = _b[1];
        var _d = _c[0], padT = _d[0], padB = _d[1], _e = _c[1], padL = _e[0], padR = _e[1];
        if (applySigmoidActivation === void 0) { applySigmoidActivation = false; }
        return tf__namespace.tidy(function () {
            var inResizedAndPadded = tf__namespace.image.resizeBilinear(tensor, [resizedAndPaddedHeight, resizedAndPaddedWidth], true);
            if (applySigmoidActivation) {
                inResizedAndPadded = tf__namespace.sigmoid(inResizedAndPadded);
            }
            return removePaddingAndResizeBack(inResizedAndPadded, [inputTensorHeight, inputTensorWidth], [[padT, padB], [padL, padR]]);
        });
    }
    function removePaddingAndResizeBack(resizedAndPadded, _a, _b) {
        var originalHeight = _a[0], originalWidth = _a[1];
        var _c = _b[0], padT = _c[0], padB = _c[1], _d = _b[1], padL = _d[0], padR = _d[1];
        return tf__namespace.tidy(function () {
            var batchedImage = tf__namespace.expandDims(resizedAndPadded);
            return tf__namespace.squeeze(tf__namespace.image
                .cropAndResize(batchedImage, [[
                    padT / (originalHeight + padT + padB - 1.0),
                    padL / (originalWidth + padL + padR - 1.0),
                    (padT + originalHeight - 1.0) /
                        (originalHeight + padT + padB - 1.0),
                    (padL + originalWidth - 1.0) / (originalWidth + padL + padR - 1.0)
                ]], [0], [originalHeight, originalWidth]), [0]);
        });
    }
    function padAndResizeTo(input, _a) {
        var targetH = _a[0], targetW = _a[1];
        var _b = getInputSize(input), height = _b[0], width = _b[1];
        var targetAspect = targetW / targetH;
        var aspect = width / height;
        var _c = [0, 0, 0, 0], padT = _c[0], padB = _c[1], padL = _c[2], padR = _c[3];
        if (aspect < targetAspect) {
            // pads the width
            padT = 0;
            padB = 0;
            padL = Math.round(0.5 * (targetAspect * height - width));
            padR = Math.round(0.5 * (targetAspect * height - width));
        }
        else {
            // pads the height
            padT = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
            padB = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
            padL = 0;
            padR = 0;
        }
        var resized = tf__namespace.tidy(function () {
            var imageTensor = toInputTensor(input);
            imageTensor = tf__namespace.pad3d(imageTensor, [[padT, padB], [padL, padR], [0, 0]]);
            return tf__namespace.image.resizeBilinear(imageTensor, [targetH, targetW]);
        });
        return { resized: resized, padding: { top: padT, left: padL, right: padR, bottom: padB } };
    }
    function toTensorBuffers3D(tensors) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, Promise.all(tensors.map(function (tensor) { return tensor.buffer(); }))];
            });
        });
    }
    function scalePose(pose, scaleY, scaleX, offsetY, offsetX) {
        if (offsetY === void 0) { offsetY = 0; }
        if (offsetX === void 0) { offsetX = 0; }
        return {
            score: pose.score,
            keypoints: pose.keypoints.map(function (_a) {
                var score = _a.score, part = _a.part, position = _a.position;
                return ({
                    score: score,
                    part: part,
                    position: {
                        x: position.x * scaleX + offsetX,
                        y: position.y * scaleY + offsetY
                    }
                });
            })
        };
    }
    function scalePoses(poses, scaleY, scaleX, offsetY, offsetX) {
        if (offsetY === void 0) { offsetY = 0; }
        if (offsetX === void 0) { offsetX = 0; }
        if (scaleX === 1 && scaleY === 1 && offsetY === 0 && offsetX === 0) {
            return poses;
        }
        return poses.map(function (pose) { return scalePose(pose, scaleY, scaleX, offsetY, offsetX); });
    }
    function flipPoseHorizontal(pose, imageWidth) {
        return {
            score: pose.score,
            keypoints: pose.keypoints.map(function (_a) {
                var score = _a.score, part = _a.part, position = _a.position;
                return ({
                    score: score,
                    part: part,
                    position: { x: imageWidth - 1 - position.x, y: position.y }
                });
            })
        };
    }
    function flipPosesHorizontal(poses, imageWidth) {
        if (imageWidth <= 0) {
            return poses;
        }
        return poses.map(function (pose) { return flipPoseHorizontal(pose, imageWidth); });
    }
    function scaleAndFlipPoses(poses, _a, _b, padding, flipHorizontal) {
        var height = _a[0], width = _a[1];
        var inputResolutionHeight = _b[0], inputResolutionWidth = _b[1];
        var scaleY = (height + padding.top + padding.bottom) / (inputResolutionHeight);
        var scaleX = (width + padding.left + padding.right) / (inputResolutionWidth);
        var scaledPoses = scalePoses(poses, scaleY, scaleX, -padding.top, -padding.left);
        if (flipHorizontal) {
            return flipPosesHorizontal(scaledPoses, width);
        }
        else {
            return scaledPoses;
        }
    }

    /**
     * @license
     * Copyright 2019 Google Inc. All Rights Reserved.
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
    var APPLY_SIGMOID_ACTIVATION = true;
    var FLIP_POSES_AFTER_SCALING = false;
    // The default configuration for loading MobileNetV1 based BodyPix.
    //
    // (And for references, the default configuration for loading ResNet
    // based PoseNet is also included).
    //
    // ```
    // const RESNET_CONFIG = {
    //   architecture: 'ResNet50',
    //   outputStride: 32,
    //   quantBytes: 4,
    // } as ModelConfig;
    // ```
    var MOBILENET_V1_CONFIG = {
        architecture: 'MobileNetV1',
        outputStride: 16,
        quantBytes: 4,
        multiplier: 0.75,
    };
    var VALID_ARCHITECTURE = ['MobileNetV1', 'ResNet50'];
    var VALID_STRIDE = {
        'MobileNetV1': [8, 16, 32],
        'ResNet50': [32, 16]
    };
    var VALID_MULTIPLIER = {
        'MobileNetV1': [0.50, 0.75, 1.0],
        'ResNet50': [1.0]
    };
    var VALID_QUANT_BYTES = [1, 2, 4];
    function validateModelConfig(config) {
        config = config || MOBILENET_V1_CONFIG;
        if (config.architecture == null) {
            config.architecture = 'MobileNetV1';
        }
        if (VALID_ARCHITECTURE.indexOf(config.architecture) < 0) {
            throw new Error("Invalid architecture ".concat(config.architecture, ". ") +
                "Should be one of ".concat(VALID_ARCHITECTURE));
        }
        if (config.outputStride == null) {
            config.outputStride = 16;
        }
        if (VALID_STRIDE[config.architecture].indexOf(config.outputStride) < 0) {
            throw new Error("Invalid outputStride ".concat(config.outputStride, ". ") +
                "Should be one of ".concat(VALID_STRIDE[config.architecture], " ") +
                "for architecture ".concat(config.architecture, "."));
        }
        if (config.multiplier == null) {
            config.multiplier = 1.0;
        }
        if (VALID_MULTIPLIER[config.architecture].indexOf(config.multiplier) < 0) {
            throw new Error("Invalid multiplier ".concat(config.multiplier, ". ") +
                "Should be one of ".concat(VALID_MULTIPLIER[config.architecture], " ") +
                "for architecture ".concat(config.architecture, "."));
        }
        if (config.quantBytes == null) {
            config.quantBytes = 4;
        }
        if (VALID_QUANT_BYTES.indexOf(config.quantBytes) < 0) {
            throw new Error("Invalid quantBytes ".concat(config.quantBytes, ". ") +
                "Should be one of ".concat(VALID_QUANT_BYTES, " ") +
                "for architecture ".concat(config.architecture, "."));
        }
        return config;
    }
    var PERSON_INFERENCE_CONFIG = {
        flipHorizontal: false,
        internalResolution: 'medium',
        segmentationThreshold: 0.7,
        maxDetections: 10,
        scoreThreshold: 0.4,
        nmsRadius: 20,
    };
    var MULTI_PERSON_INSTANCE_INFERENCE_CONFIG = {
        flipHorizontal: false,
        internalResolution: 'medium',
        segmentationThreshold: 0.7,
        maxDetections: 10,
        scoreThreshold: 0.4,
        nmsRadius: 20,
        minKeypointScore: 0.3,
        refineSteps: 10
    };
    function validatePersonInferenceConfig(config) {
        var segmentationThreshold = config.segmentationThreshold, maxDetections = config.maxDetections, scoreThreshold = config.scoreThreshold, nmsRadius = config.nmsRadius;
        if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
            throw new Error("segmentationThreshold ".concat(segmentationThreshold, ". ") +
                "Should be in range [0.0, 1.0]");
        }
        if (maxDetections <= 0) {
            throw new Error("Invalid maxDetections ".concat(maxDetections, ". ") +
                "Should be > 0");
        }
        if (scoreThreshold < 0.0 || scoreThreshold > 1.0) {
            throw new Error("Invalid scoreThreshold ".concat(scoreThreshold, ". ") +
                "Should be in range [0.0, 1.0]");
        }
        if (nmsRadius <= 0) {
            throw new Error("Invalid nmsRadius ".concat(nmsRadius, "."));
        }
    }
    function validateMultiPersonInstanceInferenceConfig(config) {
        var segmentationThreshold = config.segmentationThreshold, maxDetections = config.maxDetections, scoreThreshold = config.scoreThreshold, nmsRadius = config.nmsRadius, minKeypointScore = config.minKeypointScore, refineSteps = config.refineSteps;
        if (segmentationThreshold < 0.0 || segmentationThreshold > 1.0) {
            throw new Error("segmentationThreshold ".concat(segmentationThreshold, ". ") +
                "Should be in range [0.0, 1.0]");
        }
        if (maxDetections <= 0) {
            throw new Error("Invalid maxDetections ".concat(maxDetections, ". ") +
                "Should be > 0");
        }
        if (scoreThreshold < 0.0 || scoreThreshold > 1.0) {
            throw new Error("Invalid scoreThreshold ".concat(scoreThreshold, ". ") +
                "Should be in range [0.0, 1.0]");
        }
        if (nmsRadius <= 0) {
            throw new Error("Invalid nmsRadius ".concat(nmsRadius, "."));
        }
        if (minKeypointScore < 0 || minKeypointScore > 1) {
            throw new Error("Invalid minKeypointScore ".concat(minKeypointScore, ".") +
                "Should be in range [0.0, 1.0]");
        }
        if (refineSteps <= 0 || refineSteps > 20) {
            throw new Error("Invalid refineSteps ".concat(refineSteps, ".") +
                "Should be in range [1, 20]");
        }
    }
    var BodyPix = /** @class */ (function () {
        function BodyPix(net) {
            this.baseModel = net;
        }
        BodyPix.prototype.predictForPersonSegmentation = function (input) {
            var _a = this.baseModel.predict(input), segmentation = _a.segmentation, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
            return {
                segmentLogits: segmentation,
                heatmapScores: heatmapScores,
                offsets: offsets,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
            };
        };
        BodyPix.prototype.predictForPersonSegmentationAndPart = function (input) {
            var _a = this.baseModel.predict(input), segmentation = _a.segmentation, partHeatmaps = _a.partHeatmaps, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
            return {
                segmentLogits: segmentation,
                partHeatmapLogits: partHeatmaps,
                heatmapScores: heatmapScores,
                offsets: offsets,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
            };
        };
        BodyPix.prototype.predictForMultiPersonInstanceSegmentationAndPart = function (input) {
            var _a = this.baseModel.predict(input), segmentation = _a.segmentation, longOffsets = _a.longOffsets, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd, partHeatmaps = _a.partHeatmaps;
            return {
                segmentLogits: segmentation,
                longOffsets: longOffsets,
                heatmapScores: heatmapScores,
                offsets: offsets,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
                partHeatmaps: partHeatmaps
            };
        };
        /**
         * Given an image with people, returns a dictionary of all intermediate
         * tensors including: 1) a binary array with 1 for the pixels that are part of
         * the person, and 0 otherwise, 2) heatmapScores, 3) offsets, and 4) paddings.
         *
         * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
         * The input image to feed through the network.
         *
         * @param internalResolution Defaults to 'medium'. The internal resolution
         * that the input is resized to before inference. The larger the
         * internalResolution the more accurate the model at the cost of slower
         * prediction times. Available values are 'low', 'medium', 'high', 'full', or
         * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
         * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
         *
         * @param segmentationThreshold The minimum that segmentation values must have
         * to be considered part of the person. Affects the generation of the
         * segmentation mask.
         *
         * @return A dictionary containing `segmentation`, `heatmapScores`, `offsets`,
         * and `padding`:
         * - `segmentation`: A 2d Tensor with 1 for the pixels that are part of the
         * person, and 0 otherwise. The width and height correspond to the same
         * dimensions of the input image.
         * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
         * pose estimation decoding.
         * - `offsets`: A 3d Tensor of the keypoint offsets used by pose
         * estimation decoding.
         * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement used
         * by pose estimation decoding.
         * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
         * by pose estimation decoding.
         * - `padding`: The padding (unit pixels) being applied to the input image
         * before it is fed into the model.
         */
        BodyPix.prototype.segmentPersonActivation = function (input, internalResolution, segmentationThreshold) {
            var _this = this;
            if (segmentationThreshold === void 0) { segmentationThreshold = 0.5; }
            var _a = getInputSize(input), height = _a[0], width = _a[1];
            var internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(internalResolution, this.baseModel.outputStride, [height, width]);
            var _b = padAndResizeTo(input, internalResolutionHeightAndWidth), resized = _b.resized, padding = _b.padding;
            var _c = tf__namespace.tidy(function () {
                var _a = _this.predictForPersonSegmentation(resized), segmentLogits = _a.segmentLogits, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
                var _b = resized.shape, resizedHeight = _b[0], resizedWidth = _b[1];
                var scaledSegmentScores = scaleAndCropToInputTensorShape(segmentLogits, [height, width], [resizedHeight, resizedWidth], [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                return {
                    segmentation: toMaskTensor(tf__namespace.squeeze(scaledSegmentScores), segmentationThreshold),
                    heatmapScores: heatmapScores,
                    offsets: offsets,
                    displacementFwd: displacementFwd,
                    displacementBwd: displacementBwd,
                };
            }), segmentation = _c.segmentation, heatmapScores = _c.heatmapScores, offsets = _c.offsets, displacementFwd = _c.displacementFwd, displacementBwd = _c.displacementBwd;
            resized.dispose();
            return {
                segmentation: segmentation,
                heatmapScores: heatmapScores,
                offsets: offsets,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
                padding: padding,
                internalResolutionHeightAndWidth: internalResolutionHeightAndWidth
            };
        };
        /**
         * Given an image with many people, returns a PersonSegmentation dictionary
         * that contains the segmentation mask for all people and a single pose.
         *
         * Note: The segmentation mask returned by this method covers all people but
         * the pose works well for one person. If you want to estimate instance-level
         * multiple person segmentation & pose for each person, use
         * `segmentMultiPerson` instead.
         *
         * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
         * The input image to feed through the network.
         *
         * @param config PersonInferenceConfig object that contains
         * parameters for the BodyPix inference using person decoding.
         *
         * @return A SemanticPersonSegmentation dictionary that contains height,
         * width, the flattened binary segmentation mask and the poses for all people.
         * The width and height correspond to the same dimensions of the input image.
         * - `height`: The height of the segmentation data in pixel unit.
         * - `width`: The width of the segmentation data in pixel unit.
         * - `data`: The flattened Uint8Array of segmentation data. 1 means the pixel
         * belongs to a person and 0 means the pixel doesn't belong to a person. The
         * size of the array is equal to `height` x `width` in row-major order.
         * - `allPoses`: The 2d poses of all people.
         */
        BodyPix.prototype.segmentPerson = function (input, config) {
            if (config === void 0) { config = PERSON_INFERENCE_CONFIG; }
            return __awaiter(this, void 0, void 0, function () {
                var _a, segmentation, heatmapScores, offsets, displacementFwd, displacementBwd, padding, internalResolutionHeightAndWidth, _b, height, width, result, tensorBuffers, scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, poses;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            config = __assign(__assign({}, PERSON_INFERENCE_CONFIG), config);
                            validatePersonInferenceConfig(config);
                            _a = this.segmentPersonActivation(input, config.internalResolution, config.segmentationThreshold), segmentation = _a.segmentation, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd, padding = _a.padding, internalResolutionHeightAndWidth = _a.internalResolutionHeightAndWidth;
                            _b = segmentation.shape, height = _b[0], width = _b[1];
                            return [4 /*yield*/, segmentation.data()];
                        case 1:
                            result = _c.sent();
                            segmentation.dispose();
                            return [4 /*yield*/, toTensorBuffers3D([heatmapScores, offsets, displacementFwd, displacementBwd])];
                        case 2:
                            tensorBuffers = _c.sent();
                            scoresBuf = tensorBuffers[0], offsetsBuf = tensorBuffers[1], displacementsFwdBuf = tensorBuffers[2], displacementsBwdBuf = tensorBuffers[3];
                            poses = decodeMultiplePoses(scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, this.baseModel.outputStride, config.maxDetections, config.scoreThreshold, config.nmsRadius);
                            poses = scaleAndFlipPoses(poses, [height, width], internalResolutionHeightAndWidth, padding, FLIP_POSES_AFTER_SCALING);
                            heatmapScores.dispose();
                            offsets.dispose();
                            displacementFwd.dispose();
                            displacementBwd.dispose();
                            return [2 /*return*/, { height: height, width: width, data: result, allPoses: poses }];
                    }
                });
            });
        };
        /**
         * Given an image with multiple people, returns an *array* of
         * PersonSegmentation object. Each element in the array corresponding to one
         * of the people in the input image. In other words, it predicts
         * instance-level multiple person segmentation & pose for each person.
         *
         * The model does standard ImageNet pre-processing before inferring through
         * the model. The image pixels should have values [0-255].
         *
         * @param input
         * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
         * image to feed through the network.
         *
         * @param config MultiPersonInferenceConfig object that contains
         * parameters for the BodyPix inference using multi-person decoding.
         *
         * @return An array of PersonSegmentation object, each containing a width,
         * height, a binary array (1 for the pixels that are part of the
         * person, and 0 otherwise) and 2D pose. The array size corresponds to the
         * number of pixels in the image. The width and height correspond to the
         * dimensions of the image the binary array is shaped to, which are the same
         * dimensions of the input image.
         */
        BodyPix.prototype.segmentMultiPerson = function (input, config) {
            if (config === void 0) { config = MULTI_PERSON_INSTANCE_INFERENCE_CONFIG; }
            return __awaiter(this, void 0, void 0, function () {
                var _a, height, width, internalResolutionHeightAndWidth, _b, resized, padding, _c, segmentation, longOffsets, heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw, tensorBuffers, scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, poses, instanceMasks;
                var _this = this;
                return __generator(this, function (_d) {
                    switch (_d.label) {
                        case 0:
                            config = __assign(__assign({}, MULTI_PERSON_INSTANCE_INFERENCE_CONFIG), config);
                            validateMultiPersonInstanceInferenceConfig(config);
                            _a = getInputSize(input), height = _a[0], width = _a[1];
                            internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(config.internalResolution, this.baseModel.outputStride, [height, width]);
                            _b = padAndResizeTo(input, internalResolutionHeightAndWidth), resized = _b.resized, padding = _b.padding;
                            _c = tf__namespace.tidy(function () {
                                var _a = _this.predictForMultiPersonInstanceSegmentationAndPart(resized), segmentLogits = _a.segmentLogits, longOffsets = _a.longOffsets, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
                                var scaledSegmentScores = scaleAndCropToInputTensorShape(segmentLogits, [height, width], internalResolutionHeightAndWidth, [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                                var scaledLongOffsets;
                                {
                                    scaledLongOffsets = longOffsets;
                                }
                                var segmentation = toMaskTensor(tf__namespace.squeeze(scaledSegmentScores), config.segmentationThreshold);
                                return {
                                    segmentation: segmentation,
                                    longOffsets: scaledLongOffsets,
                                    heatmapScoresRaw: heatmapScores,
                                    offsetsRaw: offsets,
                                    displacementFwdRaw: displacementFwd,
                                    displacementBwdRaw: displacementBwd,
                                };
                            }), segmentation = _c.segmentation, longOffsets = _c.longOffsets, heatmapScoresRaw = _c.heatmapScoresRaw, offsetsRaw = _c.offsetsRaw, displacementFwdRaw = _c.displacementFwdRaw, displacementBwdRaw = _c.displacementBwdRaw;
                            return [4 /*yield*/, toTensorBuffers3D([heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw])];
                        case 1:
                            tensorBuffers = _d.sent();
                            scoresBuf = tensorBuffers[0], offsetsBuf = tensorBuffers[1], displacementsFwdBuf = tensorBuffers[2], displacementsBwdBuf = tensorBuffers[3];
                            poses = decodeMultiplePoses(scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, this.baseModel.outputStride, config.maxDetections, config.scoreThreshold, config.nmsRadius);
                            poses = scaleAndFlipPoses(poses, [height, width], internalResolutionHeightAndWidth, padding, FLIP_POSES_AFTER_SCALING);
                            return [4 /*yield*/, decodePersonInstanceMasks(segmentation, longOffsets, poses, height, width, this.baseModel.outputStride, internalResolutionHeightAndWidth, padding, config.scoreThreshold, config.refineSteps, config.minKeypointScore, config.maxDetections)];
                        case 2:
                            instanceMasks = _d.sent();
                            resized.dispose();
                            segmentation.dispose();
                            longOffsets.dispose();
                            heatmapScoresRaw.dispose();
                            offsetsRaw.dispose();
                            displacementFwdRaw.dispose();
                            displacementBwdRaw.dispose();
                            return [2 /*return*/, instanceMasks];
                    }
                });
            });
        };
        /**
         * Given an image with many people, returns a dictionary containing: height,
         * width, a tensor with a part id from 0-24 for the pixels that are
         * part of a corresponding body part, and -1 otherwise. This does standard
         * ImageNet pre-processing before inferring through the model.  The image
         * should pixels should have values [0-255].
         *
         * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
         * The input image to feed through the network.
         *
         * @param internalResolution Defaults to 'medium'. The internal resolution
         * percentage that the input is resized to before inference. The larger the
         * internalResolution the more accurate the model at the cost of slower
         * prediction times. Available values are 'low', 'medium', 'high', 'full', or
         * a percentage value between 0 and 1. The values 'low', 'medium', 'high', and
         * 'full' map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
         *
         * @param segmentationThreshold The minimum that segmentation values must have
         * to be considered part of the person.  Affects the clipping of the colored
         * part image.
         *
         * @return  A dictionary containing `partSegmentation`, `heatmapScores`,
         * `offsets`, and `padding`:
         * - `partSegmentation`: A 2d Tensor with a part id from 0-24 for
         * the pixels that are part of a corresponding body part, and -1 otherwise.
         * - `heatmapScores`: A 3d Tensor of the keypoint heatmaps used by
         * single-person pose estimation decoding.
         * - `offsets`: A 3d Tensor of the keypoint offsets used by single-person pose
         * estimation decoding.
         * - `displacementFwd`: A 3d Tensor of the keypoint forward displacement
         * used by pose estimation decoding.
         * - `displacementBwd`: A 3d Tensor of the keypoint backward displacement used
         * by pose estimation decoding.
         * - `padding`: The padding (unit pixels) being applied to the input image
         * before it is fed into the model.
         */
        BodyPix.prototype.segmentPersonPartsActivation = function (input, internalResolution, segmentationThreshold) {
            var _this = this;
            if (segmentationThreshold === void 0) { segmentationThreshold = 0.5; }
            var _a = getInputSize(input), height = _a[0], width = _a[1];
            var internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(internalResolution, this.baseModel.outputStride, [height, width]);
            var _b = padAndResizeTo(input, internalResolutionHeightAndWidth), resized = _b.resized, padding = _b.padding;
            var _c = tf__namespace.tidy(function () {
                var _a = _this.predictForPersonSegmentationAndPart(resized), segmentLogits = _a.segmentLogits, partHeatmapLogits = _a.partHeatmapLogits, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
                var _b = resized.shape, resizedHeight = _b[0], resizedWidth = _b[1];
                var scaledSegmentScores = scaleAndCropToInputTensorShape(segmentLogits, [height, width], [resizedHeight, resizedWidth], [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                var scaledPartHeatmapScore = scaleAndCropToInputTensorShape(partHeatmapLogits, [height, width], [resizedHeight, resizedWidth], [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                var segmentation = toMaskTensor(tf__namespace.squeeze(scaledSegmentScores), segmentationThreshold);
                return {
                    partSegmentation: decodePartSegmentation(segmentation, scaledPartHeatmapScore),
                    heatmapScores: heatmapScores,
                    offsets: offsets,
                    displacementFwd: displacementFwd,
                    displacementBwd: displacementBwd,
                };
            }), partSegmentation = _c.partSegmentation, heatmapScores = _c.heatmapScores, offsets = _c.offsets, displacementFwd = _c.displacementFwd, displacementBwd = _c.displacementBwd;
            resized.dispose();
            return {
                partSegmentation: partSegmentation,
                heatmapScores: heatmapScores,
                offsets: offsets,
                displacementFwd: displacementFwd,
                displacementBwd: displacementBwd,
                padding: padding,
                internalResolutionHeightAndWidth: internalResolutionHeightAndWidth
            };
        };
        /**
         * Given an image with many people, returns a PartSegmentation dictionary that
         * contains the body part segmentation mask for all people and a single pose.
         *
         * Note: The body part segmentation mask returned by this method covers all
         * people but the pose works well when there is one person. If you want to
         * estimate instance-level multiple person body part segmentation & pose for
         * each person, use `segmentMultiPersonParts` instead.
         *
         * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
         * The input image to feed through the network.
         *
         * @param config PersonInferenceConfig object that contains
         * parameters for the BodyPix inference using single person decoding.
         *
         * @return A SemanticPartSegmentation dictionary that contains height, width,
         * the flattened binary segmentation mask and the pose for the person. The
         * width and height correspond to the same dimensions of the input image.
         * - `height`: The height of the person part segmentation data in pixel unit.
         * - `width`: The width of the person part segmentation data in pixel unit.
         * - `data`: The flattened Int32Array of person part segmentation data with a
         * part id from 0-24 for the pixels that are part of a corresponding body
         * part, and -1 otherwise. The size of the array is equal to `height` x
         * `width` in row-major order.
         * - `allPoses`: The 2d poses of all people.
         */
        BodyPix.prototype.segmentPersonParts = function (input, config) {
            if (config === void 0) { config = PERSON_INFERENCE_CONFIG; }
            return __awaiter(this, void 0, void 0, function () {
                var _a, partSegmentation, heatmapScores, offsets, displacementFwd, displacementBwd, padding, internalResolutionHeightAndWidth, _b, height, width, data, tensorBuffers, scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, poses;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            config = __assign(__assign({}, PERSON_INFERENCE_CONFIG), config);
                            validatePersonInferenceConfig(config);
                            _a = this.segmentPersonPartsActivation(input, config.internalResolution, config.segmentationThreshold), partSegmentation = _a.partSegmentation, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd, padding = _a.padding, internalResolutionHeightAndWidth = _a.internalResolutionHeightAndWidth;
                            _b = partSegmentation.shape, height = _b[0], width = _b[1];
                            return [4 /*yield*/, partSegmentation.data()];
                        case 1:
                            data = _c.sent();
                            partSegmentation.dispose();
                            return [4 /*yield*/, toTensorBuffers3D([heatmapScores, offsets, displacementFwd, displacementBwd])];
                        case 2:
                            tensorBuffers = _c.sent();
                            scoresBuf = tensorBuffers[0], offsetsBuf = tensorBuffers[1], displacementsFwdBuf = tensorBuffers[2], displacementsBwdBuf = tensorBuffers[3];
                            poses = decodeMultiplePoses(scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, this.baseModel.outputStride, config.maxDetections, config.scoreThreshold, config.nmsRadius);
                            poses = scaleAndFlipPoses(poses, [height, width], internalResolutionHeightAndWidth, padding, FLIP_POSES_AFTER_SCALING);
                            heatmapScores.dispose();
                            offsets.dispose();
                            displacementFwd.dispose();
                            displacementBwd.dispose();
                            return [2 /*return*/, { height: height, width: width, data: data, allPoses: poses }];
                    }
                });
            });
        };
        /**
         * Given an image with multiple people, returns an *array* of PartSegmentation
         * object. Each element in the array corresponding to one
         * of the people in the input image. In other words, it predicts
         * instance-level multiple person body part segmentation & pose for each
         * person.
         *
         * This does standard ImageNet pre-processing before inferring through
         * the model. The image pixels should have values [0-255].
         *
         * @param input
         * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) The input
         * image to feed through the network.
         *
         * @param config MultiPersonInferenceConfig object that contains
         * parameters for the BodyPix inference using multi-person decoding.
         *
         * @return An array of PartSegmentation object, each containing a width,
         * height, a flattened array (with part id from 0-24 for the pixels that are
         * part of a corresponding body part, and -1 otherwise) and 2D pose. The width
         * and height correspond to the dimensions of the image. Each flattened part
         * segmentation array size is equal to `height` x `width`.
         */
        BodyPix.prototype.segmentMultiPersonParts = function (input, config) {
            if (config === void 0) { config = MULTI_PERSON_INSTANCE_INFERENCE_CONFIG; }
            return __awaiter(this, void 0, void 0, function () {
                var _a, height, width, internalResolutionHeightAndWidth, _b, resized, padding, _c, segmentation, longOffsets, heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw, partSegmentation, tensorBuffers, scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, poses, instanceMasks;
                var _this = this;
                return __generator(this, function (_d) {
                    switch (_d.label) {
                        case 0:
                            config = __assign(__assign({}, MULTI_PERSON_INSTANCE_INFERENCE_CONFIG), config);
                            validateMultiPersonInstanceInferenceConfig(config);
                            _a = getInputSize(input), height = _a[0], width = _a[1];
                            internalResolutionHeightAndWidth = toInputResolutionHeightAndWidth(config.internalResolution, this.baseModel.outputStride, [height, width]);
                            _b = padAndResizeTo(input, internalResolutionHeightAndWidth), resized = _b.resized, padding = _b.padding;
                            _c = tf__namespace.tidy(function () {
                                var _a = _this.predictForMultiPersonInstanceSegmentationAndPart(resized), segmentLogits = _a.segmentLogits, longOffsets = _a.longOffsets, heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd, partHeatmaps = _a.partHeatmaps;
                                // decoding with scaling.
                                var scaledSegmentScores = scaleAndCropToInputTensorShape(segmentLogits, [height, width], internalResolutionHeightAndWidth, [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                                // decoding with scaling.
                                var scaledPartSegmentationScores = scaleAndCropToInputTensorShape(partHeatmaps, [height, width], internalResolutionHeightAndWidth, [[padding.top, padding.bottom], [padding.left, padding.right]], APPLY_SIGMOID_ACTIVATION);
                                var scaledLongOffsets = longOffsets;
                                var segmentation = toMaskTensor(tf__namespace.squeeze(scaledSegmentScores), config.segmentationThreshold);
                                var partSegmentation = decodeOnlyPartSegmentation(scaledPartSegmentationScores);
                                return {
                                    segmentation: segmentation,
                                    longOffsets: scaledLongOffsets,
                                    heatmapScoresRaw: heatmapScores,
                                    offsetsRaw: offsets,
                                    displacementFwdRaw: displacementFwd,
                                    displacementBwdRaw: displacementBwd,
                                    partSegmentation: partSegmentation
                                };
                            }), segmentation = _c.segmentation, longOffsets = _c.longOffsets, heatmapScoresRaw = _c.heatmapScoresRaw, offsetsRaw = _c.offsetsRaw, displacementFwdRaw = _c.displacementFwdRaw, displacementBwdRaw = _c.displacementBwdRaw, partSegmentation = _c.partSegmentation;
                            return [4 /*yield*/, toTensorBuffers3D([heatmapScoresRaw, offsetsRaw, displacementFwdRaw, displacementBwdRaw])];
                        case 1:
                            tensorBuffers = _d.sent();
                            scoresBuf = tensorBuffers[0], offsetsBuf = tensorBuffers[1], displacementsFwdBuf = tensorBuffers[2], displacementsBwdBuf = tensorBuffers[3];
                            poses = decodeMultiplePoses(scoresBuf, offsetsBuf, displacementsFwdBuf, displacementsBwdBuf, this.baseModel.outputStride, config.maxDetections, config.scoreThreshold, config.nmsRadius);
                            poses = scaleAndFlipPoses(poses, [height, width], internalResolutionHeightAndWidth, padding, FLIP_POSES_AFTER_SCALING);
                            return [4 /*yield*/, decodePersonInstancePartMasks(segmentation, longOffsets, partSegmentation, poses, height, width, this.baseModel.outputStride, internalResolutionHeightAndWidth, padding, config.scoreThreshold, config.refineSteps, config.minKeypointScore, config.maxDetections)];
                        case 2:
                            instanceMasks = _d.sent();
                            resized.dispose();
                            segmentation.dispose();
                            longOffsets.dispose();
                            heatmapScoresRaw.dispose();
                            offsetsRaw.dispose();
                            displacementFwdRaw.dispose();
                            displacementBwdRaw.dispose();
                            partSegmentation.dispose();
                            return [2 /*return*/, instanceMasks];
                    }
                });
            });
        };
        BodyPix.prototype.dispose = function () {
            this.baseModel.dispose();
        };
        return BodyPix;
    }());
    /**
     * Loads the MobileNet BodyPix model.
     */
    function loadMobileNet(config) {
        return __awaiter(this, void 0, void 0, function () {
            var outputStride, quantBytes, multiplier, url, graphModel, mobilenet;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        outputStride = config.outputStride;
                        quantBytes = config.quantBytes;
                        multiplier = config.multiplier;
                        if (tf__namespace == null) {
                            throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please " +
                                "also include @tensorflow/tfjs on the page before using this\n        model.");
                        }
                        url = mobileNetSavedModel(outputStride, multiplier, quantBytes);
                        return [4 /*yield*/, tfconv__namespace.loadGraphModel(config.modelUrl || url)];
                    case 1:
                        graphModel = _a.sent();
                        mobilenet = new MobileNet(graphModel, outputStride);
                        return [2 /*return*/, new BodyPix(mobilenet)];
                }
            });
        });
    }
    /**
     * Loads the ResNet BodyPix model.
     */
    function loadResNet(config) {
        return __awaiter(this, void 0, void 0, function () {
            var outputStride, quantBytes, url, graphModel, resnet;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        outputStride = config.outputStride;
                        quantBytes = config.quantBytes;
                        if (tf__namespace == null) {
                            throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please " +
                                "also include @tensorflow/tfjs on the page before using this\n        model.");
                        }
                        url = resNet50SavedModel(outputStride, quantBytes);
                        return [4 /*yield*/, tfconv__namespace.loadGraphModel(config.modelUrl || url)];
                    case 1:
                        graphModel = _a.sent();
                        resnet = new ResNet(graphModel, outputStride);
                        return [2 /*return*/, new BodyPix(resnet)];
                }
            });
        });
    }
    /**
     * Loads the BodyPix model instance from a checkpoint, with the ResNet
     * or MobileNet architecture. The model to be loaded is configurable using the
     * config dictionary ModelConfig. Please find more details in the
     * documentation of the ModelConfig.
     *
     * @param config ModelConfig dictionary that contains parameters for
     * the BodyPix loading process. Please find more details of each parameters
     * in the documentation of the ModelConfig interface. The predefined
     * `MOBILENET_V1_CONFIG` and `RESNET_CONFIG` can also be used as references
     * for defining your customized config.
     */
    function load(config) {
        if (config === void 0) { config = MOBILENET_V1_CONFIG; }
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                config = validateModelConfig(config);
                if (config.architecture === 'ResNet50') {
                    return [2 /*return*/, loadResNet(config)];
                }
                else if (config.architecture === 'MobileNetV1') {
                    return [2 /*return*/, loadMobileNet(config)];
                }
                else {
                    return [2 /*return*/, null];
                }
            });
        });
    }

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
    // method copied from bGlur in https://codepen.io/zhaojun/pen/zZmRQe
    function cpuBlur(canvas, image, blur) {
        var ctx = canvas.getContext('2d');
        var sum = 0;
        var delta = 5;
        var alphaLeft = 1 / (2 * Math.PI * delta * delta);
        var step = blur < 3 ? 1 : 2;
        for (var y = -blur; y <= blur; y += step) {
            for (var x = -blur; x <= blur; x += step) {
                var weight = alphaLeft * Math.exp(-(x * x + y * y) / (2 * delta * delta));
                sum += weight;
            }
        }
        for (var y = -blur; y <= blur; y += step) {
            for (var x = -blur; x <= blur; x += step) {
                ctx.globalAlpha = alphaLeft *
                    Math.exp(-(x * x + y * y) / (2 * delta * delta)) / sum * blur;
                ctx.drawImage(image, x, y);
            }
        }
        ctx.globalAlpha = 1;
    }

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
    var offScreenCanvases = {};
    function isSafari() {
        return (/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
    }
    function assertSameDimensions(_a, _b, nameA, nameB) {
        var widthA = _a.width, heightA = _a.height;
        var widthB = _b.width, heightB = _b.height;
        if (widthA !== widthB || heightA !== heightB) {
            throw new Error("error: dimensions must match. ".concat(nameA, " has dimensions ").concat(widthA, "x").concat(heightA, ", ").concat(nameB, " has dimensions ").concat(widthB, "x").concat(heightB));
        }
    }
    function flipCanvasHorizontal(canvas) {
        var ctx = canvas.getContext('2d');
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);
    }
    function drawWithCompositing(ctx, image, compositeOperation) {
        ctx.globalCompositeOperation = compositeOperation;
        ctx.drawImage(image, 0, 0);
    }
    function createOffScreenCanvas() {
        if (typeof document !== 'undefined') {
            return document.createElement('canvas');
        }
        else if (typeof OffscreenCanvas !== 'undefined') {
            return new OffscreenCanvas(0, 0);
        }
        else {
            throw new Error('Cannot create a canvas in this context');
        }
    }
    function ensureOffscreenCanvasCreated(id) {
        if (!offScreenCanvases[id]) {
            offScreenCanvases[id] = createOffScreenCanvas();
        }
        return offScreenCanvases[id];
    }
    function drawAndBlurImageOnCanvas(image, blurAmount, canvas) {
        var height = image.height, width = image.width;
        var ctx = canvas.getContext('2d');
        canvas.width = width;
        canvas.height = height;
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        if (isSafari()) {
            cpuBlur(canvas, image, blurAmount);
        }
        else {
            // tslint:disable:no-any
            ctx.filter = "blur(".concat(blurAmount, "px)");
            ctx.drawImage(image, 0, 0, width, height);
        }
        ctx.restore();
    }
    function drawAndBlurImageOnOffScreenCanvas(image, blurAmount, offscreenCanvasName) {
        var canvas = ensureOffscreenCanvasCreated(offscreenCanvasName);
        if (blurAmount === 0) {
            renderImageToCanvas(image, canvas);
        }
        else {
            drawAndBlurImageOnCanvas(image, blurAmount, canvas);
        }
        return canvas;
    }
    function renderImageToCanvas(image, canvas) {
        var width = image.width, height = image.height;
        canvas.width = width;
        canvas.height = height;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, width, height);
    }
    /**
     * Draw an image on a canvas
     */
    function renderImageDataToCanvas(image, canvas) {
        canvas.width = image.width;
        canvas.height = image.height;
        var ctx = canvas.getContext('2d');
        ctx.putImageData(image, 0, 0);
    }
    function renderImageDataToOffScreenCanvas(image, canvasName) {
        var canvas = ensureOffscreenCanvasCreated(canvasName);
        renderImageDataToCanvas(image, canvas);
        return canvas;
    }
    /**
     * Given the output from estimating multi-person segmentation, generates an
     * image with foreground and background color at each pixel determined by the
     * corresponding binary segmentation value at the pixel from the output.  In
     * other words, pixels where there is a person will be colored with foreground
     * color and where there is not a person will be colored with background color.
     *
     * @param personOrPartSegmentation The output from
     * `segmentPerson`, `segmentMultiPerson`,
     * `segmentPersonParts` or `segmentMultiPersonParts`. They can
     * be SemanticPersonSegmentation object, an array of PersonSegmentation object,
     * SemanticPartSegmentation object, or an array of PartSegmentation object.
     *
     * @param foreground Default to {r:0, g:0, b:0, a: 0}. The foreground color
     * (r,g,b,a) for visualizing pixels that belong to people.
     *
     * @param background Default to {r:0, g:0, b:0, a: 255}. The background color
     * (r,g,b,a) for visualizing pixels that don't belong to people.
     *
     * @param drawContour Default to false. Whether to draw the contour around each
     * person's segmentation mask or body part mask.
     *
     * @param foregroundIds Default to [1]. The integer values that represent
     * foreground. For person segmentation, 1 is the foreground. For body part
     * segmentation, it can be a subset of all body parts ids.
     *
     * @returns An ImageData with the same width and height of
     * all the PersonSegmentation in multiPersonSegmentation, with opacity and
     * transparency at each pixel determined by the corresponding binary
     * segmentation value at the pixel from the output.
     */
    function toMask(personOrPartSegmentation, foreground, background, drawContour, foregroundIds) {
        if (foreground === void 0) { foreground = {
            r: 0,
            g: 0,
            b: 0,
            a: 0
        }; }
        if (background === void 0) { background = {
            r: 0,
            g: 0,
            b: 0,
            a: 255
        }; }
        if (drawContour === void 0) { drawContour = false; }
        if (foregroundIds === void 0) { foregroundIds = [1]; }
        if (Array.isArray(personOrPartSegmentation) &&
            personOrPartSegmentation.length === 0) {
            return null;
        }
        var multiPersonOrPartSegmentation;
        if (!Array.isArray(personOrPartSegmentation)) {
            multiPersonOrPartSegmentation = [personOrPartSegmentation];
        }
        else {
            multiPersonOrPartSegmentation = personOrPartSegmentation;
        }
        var _a = multiPersonOrPartSegmentation[0], width = _a.width, height = _a.height;
        var bytes = new Uint8ClampedArray(width * height * 4);
        function drawStroke(bytes, row, column, width, radius, color) {
            if (color === void 0) { color = { r: 0, g: 255, b: 255, a: 255 }; }
            for (var i = -radius; i <= radius; i++) {
                for (var j = -radius; j <= radius; j++) {
                    if (i !== 0 && j !== 0) {
                        var n = (row + i) * width + (column + j);
                        bytes[4 * n + 0] = color.r;
                        bytes[4 * n + 1] = color.g;
                        bytes[4 * n + 2] = color.b;
                        bytes[4 * n + 3] = color.a;
                    }
                }
            }
        }
        function isSegmentationBoundary(segmentationData, row, column, width, foregroundIds, radius) {
            if (foregroundIds === void 0) { foregroundIds = [1]; }
            if (radius === void 0) { radius = 1; }
            var numberBackgroundPixels = 0;
            for (var i = -radius; i <= radius; i++) {
                var _loop_2 = function (j) {
                    if (i !== 0 && j !== 0) {
                        var n_1 = (row + i) * width + (column + j);
                        if (!foregroundIds.some(function (id) { return id === segmentationData[n_1]; })) {
                            numberBackgroundPixels += 1;
                        }
                    }
                };
                for (var j = -radius; j <= radius; j++) {
                    _loop_2(j);
                }
            }
            return numberBackgroundPixels > 0;
        }
        for (var i = 0; i < height; i += 1) {
            var _loop_1 = function (j) {
                var n = i * width + j;
                bytes[4 * n + 0] = background.r;
                bytes[4 * n + 1] = background.g;
                bytes[4 * n + 2] = background.b;
                bytes[4 * n + 3] = background.a;
                var _loop_3 = function (k) {
                    if (foregroundIds.some(function (id) { return id === multiPersonOrPartSegmentation[k].data[n]; })) {
                        bytes[4 * n] = foreground.r;
                        bytes[4 * n + 1] = foreground.g;
                        bytes[4 * n + 2] = foreground.b;
                        bytes[4 * n + 3] = foreground.a;
                        var isBoundary = isSegmentationBoundary(multiPersonOrPartSegmentation[k].data, i, j, width, foregroundIds);
                        if (drawContour && i - 1 >= 0 && i + 1 < height && j - 1 >= 0 &&
                            j + 1 < width && isBoundary) {
                            drawStroke(bytes, i, j, width, 1);
                        }
                    }
                };
                for (var k = 0; k < multiPersonOrPartSegmentation.length; k++) {
                    _loop_3(k);
                }
            };
            for (var j = 0; j < width; j += 1) {
                _loop_1(j);
            }
        }
        return new ImageData(bytes, width, height);
    }
    var RAINBOW_PART_COLORS = [
        [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
        [238, 67, 149], [255, 78, 125], [255, 94, 99], [255, 115, 75],
        [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
        [175, 240, 91], [135, 245, 87], [96, 247, 96], [64, 243, 115],
        [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
        [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
    ];
    /**
     * Given the output from person body part segmentation (or multi-person
     * instance body part segmentation) and an array of colors indexed by part id,
     * generates an image with the corresponding color for each part at each pixel,
     * and white pixels where there is no part.
     *
     * @param partSegmentation The output from segmentPersonParts
     * or segmentMultiPersonParts. The former is a SemanticPartSegmentation
     * object and later is an array of PartSegmentation object.
     *
     * @param partColors A multi-dimensional array of rgb colors indexed by
     * part id.  Must have 24 colors, one for every part.
     *
     * @returns An ImageData with the same width and height of all the element in
     * multiPersonPartSegmentation, with the corresponding color for each part at
     * each pixel, and black pixels where there is no part.
     */
    function toColoredPartMask(partSegmentation, partColors) {
        if (partColors === void 0) { partColors = RAINBOW_PART_COLORS; }
        if (Array.isArray(partSegmentation) && partSegmentation.length === 0) {
            return null;
        }
        var multiPersonPartSegmentation;
        if (!Array.isArray(partSegmentation)) {
            multiPersonPartSegmentation = [partSegmentation];
        }
        else {
            multiPersonPartSegmentation = partSegmentation;
        }
        var _a = multiPersonPartSegmentation[0], width = _a.width, height = _a.height;
        var bytes = new Uint8ClampedArray(width * height * 4);
        for (var i = 0; i < height * width; ++i) {
            // invert mask.  Invert the segmentation mask.
            var j = i * 4;
            bytes[j + 0] = 255;
            bytes[j + 1] = 255;
            bytes[j + 2] = 255;
            bytes[j + 3] = 255;
            for (var k = 0; k < multiPersonPartSegmentation.length; k++) {
                var partId = multiPersonPartSegmentation[k].data[i];
                if (partId !== -1) {
                    var color = partColors[partId];
                    if (!color) {
                        throw new Error("No color could be found for part id ".concat(partId));
                    }
                    bytes[j + 0] = color[0];
                    bytes[j + 1] = color[1];
                    bytes[j + 2] = color[2];
                    bytes[j + 3] = 255;
                }
            }
        }
        return new ImageData(bytes, width, height);
    }
    var CANVAS_NAMES = {
        blurred: 'blurred',
        blurredMask: 'blurred-mask',
        mask: 'mask',
        lowresPartMask: 'lowres-part-mask',
    };
    /**
     * Given an image and a maskImage of type ImageData, draws the image with the
     * mask on top of it onto a canvas.
     *
     * @param canvas The canvas to be drawn onto.
     *
     * @param image The original image to apply the mask to.
     *
     * @param maskImage An ImageData containing the mask.  Ideally this should be
     * generated by toMask or toColoredPartMask.
     *
     * @param maskOpacity The opacity of the mask when drawing it on top of the
     * image. Defaults to 0.7. Should be a float between 0 and 1.
     *
     * @param maskBlurAmount How many pixels to blur the mask by. Defaults to 0.
     * Should be an integer between 0 and 20.
     *
     * @param flipHorizontal If the result should be flipped horizontally.  Defaults
     * to false.
     */
    function drawMask(canvas, image, maskImage, maskOpacity, maskBlurAmount, flipHorizontal) {
        if (maskOpacity === void 0) { maskOpacity = 0.7; }
        if (maskBlurAmount === void 0) { maskBlurAmount = 0; }
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        var _a = getInputSize(image), height = _a[0], width = _a[1];
        canvas.width = width;
        canvas.height = height;
        var ctx = canvas.getContext('2d');
        ctx.save();
        if (flipHorizontal) {
            flipCanvasHorizontal(canvas);
        }
        ctx.drawImage(image, 0, 0);
        ctx.globalAlpha = maskOpacity;
        if (maskImage) {
            assertSameDimensions({ width: width, height: height }, maskImage, 'image', 'mask');
            var mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);
            var blurredMask = drawAndBlurImageOnOffScreenCanvas(mask, maskBlurAmount, CANVAS_NAMES.blurredMask);
            ctx.drawImage(blurredMask, 0, 0, width, height);
        }
        ctx.restore();
    }
    /**
     * Given an image and a maskImage of type ImageData, draws the image with the
     * pixelated mask on top of it onto a canvas.
     *
     * @param canvas The canvas to be drawn onto.
     *
     * @param image The original image to apply the mask to.
     *
     * @param maskImage An ImageData containing the mask.  Ideally this should be
     * generated by toColoredPartMask.
     *
     * @param maskOpacity The opacity of the mask when drawing it on top of the
     * image. Defaults to 0.7. Should be a float between 0 and 1.
     *
     * @param maskBlurAmount How many pixels to blur the mask by. Defaults to 0.
     * Should be an integer between 0 and 20.
     *
     * @param flipHorizontal If the result should be flipped horizontally.  Defaults
     * to false.
     *
     * @param pixelCellWidth The width of each pixel cell. Default to 10 px.
     */
    function drawPixelatedMask(canvas, image, maskImage, maskOpacity, maskBlurAmount, flipHorizontal, pixelCellWidth) {
        if (maskOpacity === void 0) { maskOpacity = 0.7; }
        if (maskBlurAmount === void 0) { maskBlurAmount = 0; }
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        if (pixelCellWidth === void 0) { pixelCellWidth = 10.0; }
        var _a = getInputSize(image), height = _a[0], width = _a[1];
        assertSameDimensions({ width: width, height: height }, maskImage, 'image', 'mask');
        var mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);
        var blurredMask = drawAndBlurImageOnOffScreenCanvas(mask, maskBlurAmount, CANVAS_NAMES.blurredMask);
        canvas.width = blurredMask.width;
        canvas.height = blurredMask.height;
        var ctx = canvas.getContext('2d');
        ctx.save();
        if (flipHorizontal) {
            flipCanvasHorizontal(canvas);
        }
        var offscreenCanvas = ensureOffscreenCanvasCreated(CANVAS_NAMES.lowresPartMask);
        var offscreenCanvasCtx = offscreenCanvas
            .getContext('2d');
        offscreenCanvas.width = blurredMask.width * (1.0 / pixelCellWidth);
        offscreenCanvas.height = blurredMask.height * (1.0 / pixelCellWidth);
        offscreenCanvasCtx.drawImage(blurredMask, 0, 0, blurredMask.width, blurredMask.height, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreenCanvas, 0, 0, offscreenCanvas.width, offscreenCanvas.height, 0, 0, canvas.width, canvas.height);
        // Draws vertical grid lines that are `pixelCellWidth` apart from each other.
        for (var i = 0; i < offscreenCanvas.width; i++) {
            ctx.beginPath();
            ctx.strokeStyle = '#ffffff';
            ctx.moveTo(pixelCellWidth * i, 0);
            ctx.lineTo(pixelCellWidth * i, canvas.height);
            ctx.stroke();
        }
        // Draws horizontal grid lines that are `pixelCellWidth` apart from each
        // other.
        for (var i = 0; i < offscreenCanvas.height; i++) {
            ctx.beginPath();
            ctx.strokeStyle = '#ffffff';
            ctx.moveTo(0, pixelCellWidth * i);
            ctx.lineTo(canvas.width, pixelCellWidth * i);
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0 - maskOpacity;
        ctx.drawImage(image, 0, 0, blurredMask.width, blurredMask.height);
        ctx.restore();
    }
    function createPersonMask(multiPersonSegmentation, edgeBlurAmount) {
        var backgroundMaskImage = toMask(multiPersonSegmentation, { r: 0, g: 0, b: 0, a: 255 }, { r: 0, g: 0, b: 0, a: 0 });
        var backgroundMask = renderImageDataToOffScreenCanvas(backgroundMaskImage, CANVAS_NAMES.mask);
        if (edgeBlurAmount === 0) {
            return backgroundMask;
        }
        else {
            return drawAndBlurImageOnOffScreenCanvas(backgroundMask, edgeBlurAmount, CANVAS_NAMES.blurredMask);
        }
    }
    /**
     * Given a personSegmentation and an image, draws the image with its background
     * blurred onto the canvas.
     *
     * @param canvas The canvas to draw the background-blurred image onto.
     *
     * @param image The image to blur the background of and draw.
     *
     * @param personSegmentation A SemanticPersonSegmentation or an array of
     * PersonSegmentation object.
     *
     * @param backgroundBlurAmount How many pixels in the background blend into each
     * other.  Defaults to 3. Should be an integer between 1 and 20.
     *
     * @param edgeBlurAmount How many pixels to blur on the edge between the person
     * and the background by.  Defaults to 3. Should be an integer between 0 and 20.
     *
     * @param flipHorizontal If the output should be flipped horizontally.  Defaults
     * to false.
     */
    function drawBokehEffect(canvas, image, multiPersonSegmentation, backgroundBlurAmount, edgeBlurAmount, flipHorizontal) {
        if (backgroundBlurAmount === void 0) { backgroundBlurAmount = 3; }
        if (edgeBlurAmount === void 0) { edgeBlurAmount = 3; }
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        var blurredImage = drawAndBlurImageOnOffScreenCanvas(image, backgroundBlurAmount, CANVAS_NAMES.blurred);
        canvas.width = blurredImage.width;
        canvas.height = blurredImage.height;
        var ctx = canvas.getContext('2d');
        if (Array.isArray(multiPersonSegmentation) &&
            multiPersonSegmentation.length === 0) {
            ctx.drawImage(blurredImage, 0, 0);
            return;
        }
        var personMask = createPersonMask(multiPersonSegmentation, edgeBlurAmount);
        ctx.save();
        if (flipHorizontal) {
            flipCanvasHorizontal(canvas);
        }
        // draw the original image on the final canvas
        var _a = getInputSize(image), height = _a[0], width = _a[1];
        ctx.drawImage(image, 0, 0, width, height);
        // "destination-in" - "The existing canvas content is kept where both the
        // new shape and existing canvas content overlap. Everything else is made
        // transparent."
        // crop what's not the person using the mask from the original image
        drawWithCompositing(ctx, personMask, 'destination-in');
        // "destination-over" - "The existing canvas content is kept where both the
        // new shape and existing canvas content overlap. Everything else is made
        // transparent."
        // draw the blurred background on top of the original image where it doesn't
        // overlap.
        drawWithCompositing(ctx, blurredImage, 'destination-over');
        ctx.restore();
    }
    function createBodyPartMask(multiPersonPartSegmentation, bodyPartIdsToMask, edgeBlurAmount) {
        var backgroundMaskImage = toMask(multiPersonPartSegmentation, { r: 0, g: 0, b: 0, a: 0 }, { r: 0, g: 0, b: 0, a: 255 }, true, bodyPartIdsToMask);
        var backgroundMask = renderImageDataToOffScreenCanvas(backgroundMaskImage, CANVAS_NAMES.mask);
        if (edgeBlurAmount === 0) {
            return backgroundMask;
        }
        else {
            return drawAndBlurImageOnOffScreenCanvas(backgroundMask, edgeBlurAmount, CANVAS_NAMES.blurredMask);
        }
    }
    /**
     * Given a personSegmentation and an image, draws the image with its background
     * blurred onto the canvas.
     *
     * @param canvas The canvas to draw the background-blurred image onto.
     *
     * @param image The image to blur the background of and draw.
     *
     * @param partSegmentation A SemanticPartSegmentation or an array of
     * PartSegmentation object.
     *
     * @param bodyPartIdsToBlur Default to [0, 1] (left-face and right-face). An
     * array of body part ids to blur. Each must be one of the 24 body part ids.
     *
     * @param backgroundBlurAmount How many pixels in the background blend into each
     * other.  Defaults to 3. Should be an integer between 1 and 20.
     *
     * @param edgeBlurAmount How many pixels to blur on the edge between the person
     * and the background by.  Defaults to 3. Should be an integer between 0 and 20.
     *
     * @param flipHorizontal If the output should be flipped horizontally.  Defaults
     * to false.
     */
    function blurBodyPart(canvas, image, partSegmentation, bodyPartIdsToBlur, backgroundBlurAmount, edgeBlurAmount, flipHorizontal) {
        if (bodyPartIdsToBlur === void 0) { bodyPartIdsToBlur = [0, 1]; }
        if (backgroundBlurAmount === void 0) { backgroundBlurAmount = 3; }
        if (edgeBlurAmount === void 0) { edgeBlurAmount = 3; }
        if (flipHorizontal === void 0) { flipHorizontal = false; }
        var blurredImage = drawAndBlurImageOnOffScreenCanvas(image, backgroundBlurAmount, CANVAS_NAMES.blurred);
        canvas.width = blurredImage.width;
        canvas.height = blurredImage.height;
        var ctx = canvas.getContext('2d');
        if (Array.isArray(partSegmentation) && partSegmentation.length === 0) {
            ctx.drawImage(blurredImage, 0, 0);
            return;
        }
        var bodyPartMask = createBodyPartMask(partSegmentation, bodyPartIdsToBlur, edgeBlurAmount);
        ctx.save();
        if (flipHorizontal) {
            flipCanvasHorizontal(canvas);
        }
        // draw the original image on the final canvas
        var _a = getInputSize(image), height = _a[0], width = _a[1];
        ctx.drawImage(image, 0, 0, width, height);
        // "destination-in" - "The existing canvas content is kept where both the
        // new shape and existing canvas content overlap. Everything else is made
        // transparent."
        // crop what's not the person using the mask from the original image
        drawWithCompositing(ctx, bodyPartMask, 'destination-in');
        // "destination-over" - "The existing canvas content is kept where both the
        // new shape and existing canvas content overlap. Everything else is made
        // transparent."
        // draw the blurred background on top of the original image where it doesn't
        // overlap.
        drawWithCompositing(ctx, blurredImage, 'destination-over');
        ctx.restore();
    }

    /**
     * @license
     * Copyright 2020 Google Inc. All Rights Reserved.
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
     *
     * =============================================================================
     */
    var PART_CHANNELS = [
        'left_face',
        'right_face',
        'left_upper_arm_front',
        'left_upper_arm_back',
        'right_upper_arm_front',
        'right_upper_arm_back',
        'left_lower_arm_front',
        'left_lower_arm_back',
        'right_lower_arm_front',
        'right_lower_arm_back',
        'left_hand',
        'right_hand',
        'torso_front',
        'torso_back',
        'left_upper_leg_front',
        'left_upper_leg_back',
        'right_upper_leg_front',
        'right_upper_leg_back',
        'left_lower_leg_front',
        'left_lower_leg_back',
        'right_lower_leg_front',
        'right_lower_leg_back',
        'left_feet',
        'right_feet'
    ];

    /** @license See the LICENSE file. */
    // This code is auto-generated, do not modify this file!
    var version = '2.2.1';

    exports.BodyPix = BodyPix;
    exports.PART_CHANNELS = PART_CHANNELS;
    exports.blurBodyPart = blurBodyPart;
    exports.drawBokehEffect = drawBokehEffect;
    exports.drawMask = drawMask;
    exports.drawPixelatedMask = drawPixelatedMask;
    exports.flipPoseHorizontal = flipPoseHorizontal;
    exports.load = load;
    exports.resizeAndPadTo = resizeAndPadTo;
    exports.scaleAndCropToInputTensorShape = scaleAndCropToInputTensorShape;
    exports.toColoredPartMask = toColoredPartMask;
    exports.toMask = toMask;
    exports.version = version;

}));
//# sourceMappingURL=body-pix.umd.js.map
