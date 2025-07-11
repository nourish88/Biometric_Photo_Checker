/**
    * @license
    * Copyright 2020 Google LLC. All Rights Reserved.
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
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-core'), require('@tensorflow/tfjs-converter')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-core', '@tensorflow/tfjs-converter'], factory) :
    (global = global || self, factory(global.facemesh = {}, global.tf, global.tf));
}(this, (function (exports, tf, tfconv) { 'use strict';

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
        * Copyright 2020 Google LLC. All Rights Reserved.
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
    const disposeBox=t=>{t.startEndTensor.dispose(),t.startPoint.dispose(),t.endPoint.dispose();},createBox=t=>({startEndTensor:t,startPoint:tf.slice(t,[0,0],[-1,2]),endPoint:tf.slice(t,[0,2],[-1,2])}),scaleBox=(t,o)=>{const s=tf.mul(t.startPoint,o),e=tf.mul(t.endPoint,o),i=tf.concat2d([s,e],1);return createBox(i)},ANCHORS_CONFIG={strides:[8,16],anchors:[2,6]},NUM_LANDMARKS=6;function generateAnchors(t,o,s){const e=[];for(let i=0;i<s.strides.length;i++){const n=s.strides[i],a=Math.floor((o+n-1)/n),r=Math.floor((t+n-1)/n),c=s.anchors[i];for(let t=0;t<a;t++){const o=n*(t+.5);for(let t=0;t<r;t++){const s=n*(t+.5);for(let t=0;t<c;t++)e.push([s,o]);}}}return e}function decodeBounds(t,o,s){const e=tf.slice(t,[0,1],[-1,2]),i=tf.add(e,o),n=tf.slice(t,[0,3],[-1,2]),a=tf.div(n,s),r=tf.div(i,s),c=tf.div(a,2),l=tf.sub(r,c),d=tf.add(r,c),h=tf.mul(l,s),p=tf.mul(d,s);return tf.concat2d([h,p],1)}function getInputTensorDimensions(t){return t instanceof tf.Tensor?[t.shape[0],t.shape[1]]:[t.height,t.width]}function flipFaceHorizontal(t,o){let s,e,i;if(t.topLeft instanceof tf.Tensor&&t.bottomRight instanceof tf.Tensor){const[n,a]=tf.tidy(()=>[tf.concat([tf.sub(o-1,t.topLeft.slice(0,1)),t.topLeft.slice(1,1)]),tf.concat([tf.sub(o-1,t.bottomRight.slice(0,1)),t.bottomRight.slice(1,1)])]);s=n,e=a,null!=t.landmarks&&(i=tf.tidy(()=>{const s=tf.sub(tf.tensor1d([o-1,0]),t.landmarks),e=tf.tensor1d([1,-1]);return tf.mul(s,e)}));}else {const[n,a]=t.topLeft,[r,c]=t.bottomRight;s=[o-1-n,a],e=[o-1-r,c],null!=t.landmarks&&(i=t.landmarks.map(t=>[o-1-t[0],t[1]]));}const n={topLeft:s,bottomRight:e};return null!=i&&(n.landmarks=i),null!=t.probability&&(n.probability=t.probability instanceof tf.Tensor?t.probability.clone():t.probability),n}function scaleBoxFromPrediction(t,o){return tf.tidy(()=>{let s;return s=t.hasOwnProperty("box")?t.box:t,scaleBox(s,o).startEndTensor.squeeze()})}class BlazeFaceModel{constructor(t,o,s,e,i,n){this.blazeFaceModel=t,this.width=o,this.height=s,this.maxFaces=e,this.anchorsData=generateAnchors(o,s,ANCHORS_CONFIG),this.anchors=tf.tensor2d(this.anchorsData),this.inputSizeData=[o,s],this.inputSize=tf.tensor1d([o,s]),this.iouThreshold=i,this.scoreThreshold=n;}async getBoundingBoxes(t,o,s=!0){const[e,i,n]=tf.tidy(()=>{const o=t.resizeBilinear([this.width,this.height]),s=tf.mul(tf.sub(o.div(255),.5),2),e=this.blazeFaceModel.predict(s).squeeze(),i=decodeBounds(e,this.anchors,this.inputSize),n=tf.slice(e,[0,0],[-1,1]);return [e,i,tf.sigmoid(n).squeeze()]}),a=console.warn;console.warn=(()=>{});const r=tf.image.nonMaxSuppression(i,n,this.maxFaces,this.iouThreshold,this.scoreThreshold);console.warn=a;const c=await r.array();r.dispose();let l=c.map(t=>tf.slice(i,[t,0],[1,-1]));o||(l=await Promise.all(l.map(async t=>{const o=await t.array();return t.dispose(),o})));const d=t.shape[1],h=t.shape[2];let p;p=o?tf.div([h,d],this.inputSize):[h/this.inputSizeData[0],d/this.inputSizeData[1]];const u=[];for(let t=0;t<l.length;t++){const i=l[t],a=tf.tidy(()=>{const a=createBox(i instanceof tf.Tensor?i:tf.tensor2d(i));if(!s)return a;const r=c[t];let l;return l=o?this.anchors.slice([r,0],[1,2]):this.anchorsData[r],{box:a,landmarks:tf.slice(e,[r,NUM_LANDMARKS-1],[1,-1]).squeeze().reshape([NUM_LANDMARKS,-1]),probability:tf.slice(n,[r],[1]),anchor:l}});u.push(a);}return i.dispose(),n.dispose(),e.dispose(),{boxes:u,scaleFactor:p}}async estimateFaces(t,o=!1,s=!1,e=!0){const[,i]=getInputTensorDimensions(t),n=tf.tidy(()=>(t instanceof tf.Tensor||(t=tf.browser.fromPixels(t)),t.toFloat().expandDims(0))),{boxes:a,scaleFactor:r}=await this.getBoundingBoxes(n,o,e);return n.dispose(),o?a.map(t=>{const o=scaleBoxFromPrediction(t,r);let n={topLeft:o.slice([0],[2]),bottomRight:o.slice([2],[2])};if(e){const{landmarks:o,probability:s,anchor:e}=t,i=o.add(e).mul(r);n.landmarks=i,n.probability=s;}return s&&(n=flipFaceHorizontal(n,i)),n}):Promise.all(a.map(async t=>{const o=scaleBoxFromPrediction(t,r);let n;if(e){const[s,e,i]=await Promise.all([t.landmarks,o,t.probability].map(async t=>t.array())),a=t.anchor,[c,l]=r,d=s.map(t=>[(t[0]+a[0])*c,(t[1]+a[1])*l]);n={topLeft:e.slice(0,2),bottomRight:e.slice(2),landmarks:d,probability:i},disposeBox(t.box),t.landmarks.dispose(),t.probability.dispose();}else {const t=await o.array();n={topLeft:t.slice(0,2),bottomRight:t.slice(2)};}return o.dispose(),s&&(n=flipFaceHorizontal(n,i)),n}))}}const BLAZEFACE_MODEL_URL="https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1";async function load({maxFaces:t=10,inputWidth:o=128,inputHeight:s=128,iouThreshold:e=.3,scoreThreshold:i=.75}={}){const n=await tfconv.loadGraphModel(BLAZEFACE_MODEL_URL,{fromTFHub:!0});return new BlazeFaceModel(n,o,s,t,e,i)}

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    var MESH_ANNOTATIONS = {
        silhouette: [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ],
        lipsUpperOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        lipsLowerOuter: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        lipsUpperInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        lipsLowerInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        rightEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
        rightEyeLower0: [33, 7, 163, 144, 145, 153, 154, 155, 133],
        rightEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
        rightEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
        rightEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
        rightEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
        rightEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],
        rightEyebrowUpper: [156, 70, 63, 105, 66, 107, 55, 193],
        rightEyebrowLower: [35, 124, 46, 53, 52, 65],
        leftEyeUpper0: [466, 388, 387, 386, 385, 384, 398],
        leftEyeLower0: [263, 249, 390, 373, 374, 380, 381, 382, 362],
        leftEyeUpper1: [467, 260, 259, 257, 258, 286, 414],
        leftEyeLower1: [359, 255, 339, 254, 253, 252, 256, 341, 463],
        leftEyeUpper2: [342, 445, 444, 443, 442, 441, 413],
        leftEyeLower2: [446, 261, 448, 449, 450, 451, 452, 453, 464],
        leftEyeLower3: [372, 340, 346, 347, 348, 349, 350, 357, 465],
        leftEyebrowUpper: [383, 300, 293, 334, 296, 336, 285, 417],
        leftEyebrowLower: [265, 353, 276, 283, 282, 295],
        midwayBetweenEyes: [168],
        noseTip: [1],
        noseBottom: [2],
        noseRightCorner: [98],
        noseLeftCorner: [327],
        rightCheek: [205],
        leftCheek: [425]
    };

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    function scaleBoxCoordinates(box, factor) {
        var startPoint = [box.startPoint[0] * factor[0], box.startPoint[1] * factor[1]];
        var endPoint = [box.endPoint[0] * factor[0], box.endPoint[1] * factor[1]];
        return { startPoint: startPoint, endPoint: endPoint };
    }
    function getBoxSize(box) {
        return [
            Math.abs(box.endPoint[0] - box.startPoint[0]),
            Math.abs(box.endPoint[1] - box.startPoint[1])
        ];
    }
    function getBoxCenter(box) {
        return [
            box.startPoint[0] + (box.endPoint[0] - box.startPoint[0]) / 2,
            box.startPoint[1] + (box.endPoint[1] - box.startPoint[1]) / 2
        ];
    }
    function cutBoxFromImageAndResize(box, image, cropSize) {
        var h = image.shape[1];
        var w = image.shape[2];
        var boxes = [[
                box.startPoint[1] / h, box.startPoint[0] / w, box.endPoint[1] / h,
                box.endPoint[0] / w
            ]];
        return tf.image.cropAndResize(image, boxes, [0], cropSize);
    }
    /**
     * Enlarges the box by the provided factor.
     * @param box An object with startPoint and endPoint properties describing the
     * outlines of the box to be enlarged.
     * @param factor optional The enlargement factor. Defaults to 1.5
     */
    function enlargeBox(box, factor) {
        if (factor === void 0) { factor = 1.5; }
        var center = getBoxCenter(box);
        var size = getBoxSize(box);
        var newHalfSize = [factor * size[0] / 2, factor * size[1] / 2];
        var startPoint = [center[0] - newHalfSize[0], center[1] - newHalfSize[1]];
        var endPoint = [center[0] + newHalfSize[0], center[1] + newHalfSize[1]];
        return { startPoint: startPoint, endPoint: endPoint, landmarks: box.landmarks };
    }

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    /**
     * Normalizes the provided angle to the range -pi to pi.
     * @param angle The angle in radians to be normalized.
     */
    function normalizeRadians(angle) {
        return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
    }
    /**
     * Computes the angle of rotation between two anchor points.
     * @param point1 First anchor point
     * @param point2 Second anchor point
     */
    function computeRotation(point1, point2) {
        var radians = Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
        return normalizeRadians(radians);
    }
    function buildTranslationMatrix(x, y) {
        return [[1, 0, x], [0, 1, y], [0, 0, 1]];
    }
    function dot(v1, v2) {
        var product = 0;
        for (var i = 0; i < v1.length; i++) {
            product += v1[i] * v2[i];
        }
        return product;
    }
    function getColumnFrom2DArr(arr, columnIndex) {
        var column = [];
        for (var i = 0; i < arr.length; i++) {
            column.push(arr[i][columnIndex]);
        }
        return column;
    }
    function multiplyTransformMatrices(mat1, mat2) {
        var product = [];
        var size = mat1.length;
        for (var row = 0; row < size; row++) {
            product.push([]);
            for (var col = 0; col < size; col++) {
                product[row].push(dot(mat1[row], getColumnFrom2DArr(mat2, col)));
            }
        }
        return product;
    }
    function buildRotationMatrix(rotation, center) {
        var cosA = Math.cos(rotation);
        var sinA = Math.sin(rotation);
        var rotationMatrix = [[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]];
        var translationMatrix = buildTranslationMatrix(center[0], center[1]);
        var translationTimesRotation = multiplyTransformMatrices(translationMatrix, rotationMatrix);
        var negativeTranslationMatrix = buildTranslationMatrix(-center[0], -center[1]);
        return multiplyTransformMatrices(translationTimesRotation, negativeTranslationMatrix);
    }
    function invertTransformMatrix(matrix) {
        var rotationComponent = [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]];
        var translationComponent = [matrix[0][2], matrix[1][2]];
        var invertedTranslation = [
            -dot(rotationComponent[0], translationComponent),
            -dot(rotationComponent[1], translationComponent)
        ];
        return [
            rotationComponent[0].concat(invertedTranslation[0]),
            rotationComponent[1].concat(invertedTranslation[1]), [0, 0, 1]
        ];
    }
    function rotatePoint(homogeneousCoordinate, rotationMatrix) {
        return [
            dot(homogeneousCoordinate, rotationMatrix[0]),
            dot(homogeneousCoordinate, rotationMatrix[1])
        ];
    }

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    var UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;
    var LANDMARKS_COUNT = 468;
    var MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [1, 168];
    var BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES = [3, 2];
    // The Pipeline coordinates between the bounding box and skeleton models.
    var Pipeline = /** @class */ (function () {
        function Pipeline(boundingBoxDetector, meshDetector, meshWidth, meshHeight, maxContinuousChecks, maxFaces) {
            // An array of facial bounding boxes.
            this.regionsOfInterest = [];
            this.runsWithoutFaceDetector = 0;
            this.boundingBoxDetector = boundingBoxDetector;
            this.meshDetector = meshDetector;
            this.meshWidth = meshWidth;
            this.meshHeight = meshHeight;
            this.maxContinuousChecks = maxContinuousChecks;
            this.maxFaces = maxFaces;
        }
        Pipeline.prototype.transformRawCoords = function (rawCoords, box, angle, rotationMatrix) {
            var _this = this;
            var boxSize = getBoxSize({ startPoint: box.startPoint, endPoint: box.endPoint });
            var scaleFactor = [boxSize[0] / this.meshWidth, boxSize[1] / this.meshHeight];
            var coordsScaled = rawCoords.map(function (coord) {
                return [
                    scaleFactor[0] * (coord[0] - _this.meshWidth / 2),
                    scaleFactor[1] * (coord[1] - _this.meshHeight / 2), coord[2]
                ];
            });
            var coordsRotationMatrix = buildRotationMatrix(angle, [0, 0]);
            var coordsRotated = coordsScaled.map(function (coord) {
                var rotated = rotatePoint(coord, coordsRotationMatrix);
                return rotated.concat([coord[2]]);
            });
            var inverseRotationMatrix = invertTransformMatrix(rotationMatrix);
            var boxCenter = getBoxCenter({ startPoint: box.startPoint, endPoint: box.endPoint }).concat([
                1
            ]);
            var originalBoxCenter = [
                dot(boxCenter, inverseRotationMatrix[0]),
                dot(boxCenter, inverseRotationMatrix[1])
            ];
            return coordsRotated.map(function (coord) { return ([
                coord[0] + originalBoxCenter[0],
                coord[1] + originalBoxCenter[1], coord[2]
            ]); });
        };
        /**
         * Returns an array of predictions for each face in the input.
         *
         * @param input - tensor of shape [1, H, W, 3].
         */
        Pipeline.prototype.predict = function (input) {
            return __awaiter(this, void 0, void 0, function () {
                var returnTensors, annotateFace, _a, boxes, scaleFactor_1, scaledBoxes;
                var _this = this;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            if (!this.shouldUpdateRegionsOfInterest()) return [3 /*break*/, 2];
                            returnTensors = false;
                            annotateFace = true;
                            return [4 /*yield*/, this.boundingBoxDetector.getBoundingBoxes(input, returnTensors, annotateFace)];
                        case 1:
                            _a = _b.sent(), boxes = _a.boxes, scaleFactor_1 = _a.scaleFactor;
                            if (boxes.length === 0) {
                                this.regionsOfInterest = [];
                                return [2 /*return*/, null];
                            }
                            scaledBoxes = boxes.map(function (prediction) {
                                var predictionBoxCPU = {
                                    startPoint: prediction.box.startPoint.squeeze().arraySync(),
                                    endPoint: prediction.box.endPoint.squeeze().arraySync()
                                };
                                var scaledBox = scaleBoxCoordinates(predictionBoxCPU, scaleFactor_1);
                                var enlargedBox = enlargeBox(scaledBox);
                                return __assign({}, enlargedBox, { landmarks: prediction.landmarks.arraySync() });
                            });
                            boxes.forEach(function (box) {
                                if (box != null && box.startPoint != null) {
                                    box.startEndTensor.dispose();
                                    box.startPoint.dispose();
                                    box.endPoint.dispose();
                                }
                            });
                            this.updateRegionsOfInterest(scaledBoxes);
                            this.runsWithoutFaceDetector = 0;
                            return [3 /*break*/, 3];
                        case 2:
                            this.runsWithoutFaceDetector++;
                            _b.label = 3;
                        case 3: return [2 /*return*/, tf.tidy(function () {
                                return _this.regionsOfInterest.map(function (box, i) {
                                    var angle;
                                    // The facial bounding box landmarks could come either from blazeface
                                    // (if we are using a fresh box), or from the mesh model (if we are
                                    // reusing an old box).
                                    var boxLandmarksFromMeshModel = box.landmarks.length === LANDMARKS_COUNT;
                                    if (boxLandmarksFromMeshModel) {
                                        var indexOfNose = MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[0], indexOfForehead = MESH_MODEL_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[1];
                                        angle = computeRotation(box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
                                    }
                                    else {
                                        var indexOfNose = BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[0], indexOfForehead = BLAZEFACE_KEYPOINTS_LINE_OF_SYMMETRY_INDICES[1];
                                        angle = computeRotation(box.landmarks[indexOfNose], box.landmarks[indexOfForehead]);
                                    }
                                    var faceCenter = getBoxCenter({ startPoint: box.startPoint, endPoint: box.endPoint });
                                    var faceCenterNormalized = [faceCenter[0] / input.shape[2], faceCenter[1] / input.shape[1]];
                                    var rotatedImage = tf.image.rotateWithOffset(input, angle, 0, faceCenterNormalized);
                                    var rotationMatrix = buildRotationMatrix(-angle, faceCenter);
                                    var boxCPU = { startPoint: box.startPoint, endPoint: box.endPoint };
                                    var face = cutBoxFromImageAndResize(boxCPU, rotatedImage, [
                                        _this.meshHeight, _this.meshWidth
                                    ]).div(255);
                                    // The first returned tensor represents facial contours, which are
                                    // included in the coordinates.
                                    var _a = _this.meshDetector.predict(face), flag = _a[1], coords = _a[2];
                                    var coordsReshaped = tf.reshape(coords, [-1, 3]);
                                    var rawCoords = coordsReshaped.arraySync();
                                    var transformedCoordsData = _this.transformRawCoords(rawCoords, box, angle, rotationMatrix);
                                    var transformedCoords = tf.tensor2d(transformedCoordsData);
                                    var landmarksBox = _this.calculateLandmarksBoundingBox(transformedCoordsData);
                                    _this.regionsOfInterest[i] = __assign({}, landmarksBox, { landmarks: transformedCoords.arraySync() });
                                    var prediction = {
                                        coords: coordsReshaped,
                                        scaledCoords: transformedCoords,
                                        box: landmarksBox,
                                        flag: flag.squeeze()
                                    };
                                    return prediction;
                                });
                            })];
                    }
                });
            });
        };
        // Updates regions of interest if the intersection over union between
        // the incoming and previous regions falls below a threshold.
        Pipeline.prototype.updateRegionsOfInterest = function (boxes) {
            for (var i = 0; i < boxes.length; i++) {
                var box = boxes[i];
                var previousBox = this.regionsOfInterest[i];
                var iou = 0;
                if (previousBox && previousBox.startPoint) {
                    var _a = box.startPoint, boxStartX = _a[0], boxStartY = _a[1];
                    var _b = box.endPoint, boxEndX = _b[0], boxEndY = _b[1];
                    var _c = previousBox.startPoint, previousBoxStartX = _c[0], previousBoxStartY = _c[1];
                    var _d = previousBox.endPoint, previousBoxEndX = _d[0], previousBoxEndY = _d[1];
                    var xStartMax = Math.max(boxStartX, previousBoxStartX);
                    var yStartMax = Math.max(boxStartY, previousBoxStartY);
                    var xEndMin = Math.min(boxEndX, previousBoxEndX);
                    var yEndMin = Math.min(boxEndY, previousBoxEndY);
                    var intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
                    var boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
                    var previousBoxArea = (previousBoxEndX - previousBoxStartX) *
                        (previousBoxEndY - boxStartY);
                    iou = intersection / (boxArea + previousBoxArea - intersection);
                }
                if (iou < UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
                    this.regionsOfInterest[i] = box;
                }
            }
            this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
        };
        Pipeline.prototype.clearRegionOfInterest = function (index) {
            if (this.regionsOfInterest[index] != null) {
                this.regionsOfInterest = this.regionsOfInterest.slice(0, index).concat(this.regionsOfInterest.slice(index + 1));
            }
        };
        Pipeline.prototype.shouldUpdateRegionsOfInterest = function () {
            var roisCount = this.regionsOfInterest.length;
            var noROIs = roisCount === 0;
            if (this.maxFaces === 1 || noROIs) {
                return noROIs;
            }
            return roisCount !== this.maxFaces &&
                this.runsWithoutFaceDetector >= this.maxContinuousChecks;
        };
        Pipeline.prototype.calculateLandmarksBoundingBox = function (landmarks) {
            var xs = landmarks.map(function (d) { return d[0]; });
            var ys = landmarks.map(function (d) { return d[1]; });
            var startPoint = [Math.min.apply(Math, xs), Math.min.apply(Math, ys)];
            var endPoint = [Math.max.apply(Math, xs), Math.max.apply(Math, ys)];
            var box = { startPoint: startPoint, endPoint: endPoint };
            return enlargeBox({ startPoint: box.startPoint, endPoint: box.endPoint });
        };
        return Pipeline;
    }());

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    var UV_COORDS = [
        [0.499976992607117, 0.652534008026123],
        [0.500025987625122, 0.547487020492554],
        [0.499974012374878, 0.602371990680695],
        [0.482113003730774, 0.471979022026062],
        [0.500150978565216, 0.527155995368958],
        [0.499909996986389, 0.498252987861633],
        [0.499523013830185, 0.40106201171875],
        [0.289712011814117, 0.380764007568359],
        [0.499954998493195, 0.312398016452789],
        [0.499987006187439, 0.269918978214264],
        [0.500023007392883, 0.107050001621246],
        [0.500023007392883, 0.666234016418457],
        [0.5000159740448, 0.679224014282227],
        [0.500023007392883, 0.692348003387451],
        [0.499976992607117, 0.695277988910675],
        [0.499976992607117, 0.70593398809433],
        [0.499976992607117, 0.719385027885437],
        [0.499976992607117, 0.737019002437592],
        [0.499967992305756, 0.781370997428894],
        [0.499816000461578, 0.562981009483337],
        [0.473773002624512, 0.573909997940063],
        [0.104906998574734, 0.254140973091125],
        [0.365929991006851, 0.409575998783112],
        [0.338757991790771, 0.41302502155304],
        [0.311120003461838, 0.409460008144379],
        [0.274657994508743, 0.389131009578705],
        [0.393361985683441, 0.403706014156342],
        [0.345234006643295, 0.344011008739471],
        [0.370094001293182, 0.346076011657715],
        [0.319321990013123, 0.347265005111694],
        [0.297903001308441, 0.353591024875641],
        [0.24779200553894, 0.410809993743896],
        [0.396889001131058, 0.842755019664764],
        [0.280097991228104, 0.375599980354309],
        [0.106310002505779, 0.399955987930298],
        [0.2099249958992, 0.391353011131287],
        [0.355807989835739, 0.534406006336212],
        [0.471751004457474, 0.65040397644043],
        [0.474155008792877, 0.680191993713379],
        [0.439785003662109, 0.657229006290436],
        [0.414617002010345, 0.66654098033905],
        [0.450374007225037, 0.680860996246338],
        [0.428770989179611, 0.682690978050232],
        [0.374971002340317, 0.727805018424988],
        [0.486716985702515, 0.547628998756409],
        [0.485300987958908, 0.527395009994507],
        [0.257764995098114, 0.314490020275116],
        [0.401223003864288, 0.455172002315521],
        [0.429818987846375, 0.548614978790283],
        [0.421351999044418, 0.533740997314453],
        [0.276895999908447, 0.532056987285614],
        [0.483370006084442, 0.499586999416351],
        [0.33721199631691, 0.282882988452911],
        [0.296391993761063, 0.293242990970612],
        [0.169294998049736, 0.193813979625702],
        [0.447580009698868, 0.302609980106354],
        [0.392390012741089, 0.353887975215912],
        [0.354490011930466, 0.696784019470215],
        [0.067304998636246, 0.730105042457581],
        [0.442739009857178, 0.572826027870178],
        [0.457098007202148, 0.584792017936707],
        [0.381974011659622, 0.694710969924927],
        [0.392388999462128, 0.694203019142151],
        [0.277076005935669, 0.271932005882263],
        [0.422551989555359, 0.563233017921448],
        [0.385919004678726, 0.281364023685455],
        [0.383103013038635, 0.255840003490448],
        [0.331431001424789, 0.119714021682739],
        [0.229923993349075, 0.232002973556519],
        [0.364500999450684, 0.189113974571228],
        [0.229622006416321, 0.299540996551514],
        [0.173287004232407, 0.278747975826263],
        [0.472878992557526, 0.666198015213013],
        [0.446828007698059, 0.668527007102966],
        [0.422762006521225, 0.673889994621277],
        [0.445307999849319, 0.580065965652466],
        [0.388103008270264, 0.693961024284363],
        [0.403039008378983, 0.706539988517761],
        [0.403629004955292, 0.693953037261963],
        [0.460041999816895, 0.557139039039612],
        [0.431158006191254, 0.692366003990173],
        [0.452181994915009, 0.692366003990173],
        [0.475387006998062, 0.692366003990173],
        [0.465828001499176, 0.779190003871918],
        [0.472328990697861, 0.736225962638855],
        [0.473087012767792, 0.717857003211975],
        [0.473122000694275, 0.704625964164734],
        [0.473033010959625, 0.695277988910675],
        [0.427942007780075, 0.695277988910675],
        [0.426479011774063, 0.703539967536926],
        [0.423162013292313, 0.711845993995667],
        [0.4183090031147, 0.720062971115112],
        [0.390094995498657, 0.639572978019714],
        [0.013953999616206, 0.560034036636353],
        [0.499913990497589, 0.58014702796936],
        [0.413199990987778, 0.69539999961853],
        [0.409626007080078, 0.701822996139526],
        [0.468080013990402, 0.601534962654114],
        [0.422728985548019, 0.585985004901886],
        [0.463079988956451, 0.593783974647522],
        [0.37211999297142, 0.47341400384903],
        [0.334562003612518, 0.496073007583618],
        [0.411671012639999, 0.546965003013611],
        [0.242175996303558, 0.14767599105835],
        [0.290776997804642, 0.201445996761322],
        [0.327338010072708, 0.256527006626129],
        [0.399509996175766, 0.748921036720276],
        [0.441727995872498, 0.261676013469696],
        [0.429764986038208, 0.187834024429321],
        [0.412198007106781, 0.108901023864746],
        [0.288955003023148, 0.398952007293701],
        [0.218936994671822, 0.435410976409912],
        [0.41278201341629, 0.398970007896423],
        [0.257135003805161, 0.355440020561218],
        [0.427684992551804, 0.437960982322693],
        [0.448339998722076, 0.536936044692993],
        [0.178560003638268, 0.45755398273468],
        [0.247308000922203, 0.457193970680237],
        [0.286267012357712, 0.467674970626831],
        [0.332827985286713, 0.460712015628815],
        [0.368755996227264, 0.447206974029541],
        [0.398963987827301, 0.432654976844788],
        [0.476410001516342, 0.405806005001068],
        [0.189241006970406, 0.523923993110657],
        [0.228962004184723, 0.348950982093811],
        [0.490725994110107, 0.562400996685028],
        [0.404670000076294, 0.485132992267609],
        [0.019469000399113, 0.401564002037048],
        [0.426243007183075, 0.420431017875671],
        [0.396993011236191, 0.548797011375427],
        [0.266469985246658, 0.376977026462555],
        [0.439121007919312, 0.51895797252655],
        [0.032313998788595, 0.644356966018677],
        [0.419054001569748, 0.387154996395111],
        [0.462783008813858, 0.505746960639954],
        [0.238978996872902, 0.779744982719421],
        [0.198220998048782, 0.831938028335571],
        [0.107550002634525, 0.540755033493042],
        [0.183610007166862, 0.740257024765015],
        [0.134409993886948, 0.333683013916016],
        [0.385764002799988, 0.883153975009918],
        [0.490967005491257, 0.579378008842468],
        [0.382384985685349, 0.508572995662689],
        [0.174399003386497, 0.397670984268188],
        [0.318785011768341, 0.39623498916626],
        [0.343364000320435, 0.400596976280212],
        [0.396100014448166, 0.710216999053955],
        [0.187885001301765, 0.588537991046906],
        [0.430987000465393, 0.944064974784851],
        [0.318993002176285, 0.898285031318665],
        [0.266247987747192, 0.869701027870178],
        [0.500023007392883, 0.190576016902924],
        [0.499976992607117, 0.954452991485596],
        [0.366169989109039, 0.398822009563446],
        [0.393207013607025, 0.39553701877594],
        [0.410373002290726, 0.391080021858215],
        [0.194993004202843, 0.342101991176605],
        [0.388664990663528, 0.362284004688263],
        [0.365961998701096, 0.355970978736877],
        [0.343364000320435, 0.355356991291046],
        [0.318785011768341, 0.35834002494812],
        [0.301414996385574, 0.363156020641327],
        [0.058132998645306, 0.319076001644135],
        [0.301414996385574, 0.387449026107788],
        [0.499987989664078, 0.618434011936188],
        [0.415838003158569, 0.624195992946625],
        [0.445681989192963, 0.566076993942261],
        [0.465844005346298, 0.620640993118286],
        [0.49992299079895, 0.351523995399475],
        [0.288718998432159, 0.819945991039276],
        [0.335278987884521, 0.852819979190826],
        [0.440512001514435, 0.902418971061707],
        [0.128294005990028, 0.791940987110138],
        [0.408771991729736, 0.373893976211548],
        [0.455606997013092, 0.451801002025604],
        [0.499877005815506, 0.908990025520325],
        [0.375436991453171, 0.924192011356354],
        [0.11421000212431, 0.615022003650665],
        [0.448662012815475, 0.695277988910675],
        [0.4480200111866, 0.704632043838501],
        [0.447111994028091, 0.715808033943176],
        [0.444831997156143, 0.730794012546539],
        [0.430011987686157, 0.766808986663818],
        [0.406787008047104, 0.685672998428345],
        [0.400738000869751, 0.681069016456604],
        [0.392399996519089, 0.677703022956848],
        [0.367855995893478, 0.663918972015381],
        [0.247923001646996, 0.601333022117615],
        [0.452769994735718, 0.420849978923798],
        [0.43639200925827, 0.359887003898621],
        [0.416164010763168, 0.368713974952698],
        [0.413385987281799, 0.692366003990173],
        [0.228018000721931, 0.683571994304657],
        [0.468268007040024, 0.352671027183533],
        [0.411361992359161, 0.804327011108398],
        [0.499989002943039, 0.469825029373169],
        [0.479153990745544, 0.442654013633728],
        [0.499974012374878, 0.439637005329132],
        [0.432112008333206, 0.493588984012604],
        [0.499886006116867, 0.866917014122009],
        [0.49991300702095, 0.821729004383087],
        [0.456548988819122, 0.819200992584229],
        [0.344549000263214, 0.745438992977142],
        [0.37890899181366, 0.574010014533997],
        [0.374292999505997, 0.780184984207153],
        [0.319687992334366, 0.570737957954407],
        [0.357154995203018, 0.604269981384277],
        [0.295284003019333, 0.621580958366394],
        [0.447750002145767, 0.862477004528046],
        [0.410986006259918, 0.508723020553589],
        [0.31395098567009, 0.775308012962341],
        [0.354128003120422, 0.812552988529205],
        [0.324548006057739, 0.703992962837219],
        [0.189096003770828, 0.646299958229065],
        [0.279776990413666, 0.71465802192688],
        [0.1338230073452, 0.682700991630554],
        [0.336768001317978, 0.644733011722565],
        [0.429883986711502, 0.466521978378296],
        [0.455527991056442, 0.548622965812683],
        [0.437114000320435, 0.558896005153656],
        [0.467287987470627, 0.529924988746643],
        [0.414712011814117, 0.335219979286194],
        [0.37704598903656, 0.322777986526489],
        [0.344107985496521, 0.320150971412659],
        [0.312875986099243, 0.32233202457428],
        [0.283526003360748, 0.333190023899078],
        [0.241245999932289, 0.382785975933075],
        [0.102986000478268, 0.468762993812561],
        [0.267612010240555, 0.424560010433197],
        [0.297879010438919, 0.433175981044769],
        [0.333433985710144, 0.433878004550934],
        [0.366427004337311, 0.426115989685059],
        [0.396012008190155, 0.416696012020111],
        [0.420121014118195, 0.41022801399231],
        [0.007561000064015, 0.480777025222778],
        [0.432949006557465, 0.569517970085144],
        [0.458638995885849, 0.479089021682739],
        [0.473466008901596, 0.545744001865387],
        [0.476087987422943, 0.563830018043518],
        [0.468472003936768, 0.555056989192963],
        [0.433990985155106, 0.582361996173859],
        [0.483518004417419, 0.562983989715576],
        [0.482482999563217, 0.57784903049469],
        [0.42645001411438, 0.389798998832703],
        [0.438998997211456, 0.39649498462677],
        [0.450067013502121, 0.400434017181396],
        [0.289712011814117, 0.368252992630005],
        [0.276670008897781, 0.363372981548309],
        [0.517862021923065, 0.471948027610779],
        [0.710287988185883, 0.380764007568359],
        [0.526226997375488, 0.573909997940063],
        [0.895093023777008, 0.254140973091125],
        [0.634069979190826, 0.409575998783112],
        [0.661242008209229, 0.41302502155304],
        [0.688880026340485, 0.409460008144379],
        [0.725341975688934, 0.389131009578705],
        [0.606630027294159, 0.40370500087738],
        [0.654766023159027, 0.344011008739471],
        [0.629905998706818, 0.346076011657715],
        [0.680678009986877, 0.347265005111694],
        [0.702096998691559, 0.353591024875641],
        [0.75221198797226, 0.410804986953735],
        [0.602918028831482, 0.842862963676453],
        [0.719901978969574, 0.375599980354309],
        [0.893692970275879, 0.399959981441498],
        [0.790081977844238, 0.391354024410248],
        [0.643998026847839, 0.534487962722778],
        [0.528249025344849, 0.65040397644043],
        [0.525849997997284, 0.680191040039062],
        [0.560214996337891, 0.657229006290436],
        [0.585384011268616, 0.66654098033905],
        [0.549625992774963, 0.680860996246338],
        [0.57122802734375, 0.682691991329193],
        [0.624852001667023, 0.72809898853302],
        [0.513050019741058, 0.547281980514526],
        [0.51509702205658, 0.527251958847046],
        [0.742246985435486, 0.314507007598877],
        [0.598631024360657, 0.454979002475739],
        [0.570338010787964, 0.548575043678284],
        [0.578631997108459, 0.533622980117798],
        [0.723087012767792, 0.532054007053375],
        [0.516445994377136, 0.499638974666595],
        [0.662801027297974, 0.282917976379395],
        [0.70362401008606, 0.293271005153656],
        [0.830704987049103, 0.193813979625702],
        [0.552385985851288, 0.302568018436432],
        [0.607609987258911, 0.353887975215912],
        [0.645429015159607, 0.696707010269165],
        [0.932694971561432, 0.730105042457581],
        [0.557260990142822, 0.572826027870178],
        [0.542901992797852, 0.584792017936707],
        [0.6180260181427, 0.694710969924927],
        [0.607590973377228, 0.694203019142151],
        [0.722943007946014, 0.271963000297546],
        [0.577413976192474, 0.563166975975037],
        [0.614082992076874, 0.281386971473694],
        [0.616907000541687, 0.255886018276215],
        [0.668509006500244, 0.119913995265961],
        [0.770092010498047, 0.232020974159241],
        [0.635536015033722, 0.189248979091644],
        [0.77039098739624, 0.299556016921997],
        [0.826722025871277, 0.278755009174347],
        [0.527121007442474, 0.666198015213013],
        [0.553171992301941, 0.668527007102966],
        [0.577238023281097, 0.673889994621277],
        [0.554691970348358, 0.580065965652466],
        [0.611896991729736, 0.693961024284363],
        [0.59696102142334, 0.706539988517761],
        [0.596370995044708, 0.693953037261963],
        [0.539958000183105, 0.557139039039612],
        [0.568841993808746, 0.692366003990173],
        [0.547818005084991, 0.692366003990173],
        [0.52461302280426, 0.692366003990173],
        [0.534089982509613, 0.779141008853912],
        [0.527670979499817, 0.736225962638855],
        [0.526912987232208, 0.717857003211975],
        [0.526877999305725, 0.704625964164734],
        [0.526966989040375, 0.695277988910675],
        [0.572058022022247, 0.695277988910675],
        [0.573521018028259, 0.703539967536926],
        [0.57683801651001, 0.711845993995667],
        [0.581691026687622, 0.720062971115112],
        [0.609944999217987, 0.639909982681274],
        [0.986046016216278, 0.560034036636353],
        [0.5867999792099, 0.69539999961853],
        [0.590372025966644, 0.701822996139526],
        [0.531915009021759, 0.601536989212036],
        [0.577268004417419, 0.585934996604919],
        [0.536915004253387, 0.593786001205444],
        [0.627542972564697, 0.473352015018463],
        [0.665585994720459, 0.495950996875763],
        [0.588353991508484, 0.546862006187439],
        [0.757824003696442, 0.14767599105835],
        [0.709249973297119, 0.201507985591888],
        [0.672684013843536, 0.256581008434296],
        [0.600408971309662, 0.74900496006012],
        [0.55826598405838, 0.261672019958496],
        [0.570303976535797, 0.187870979309082],
        [0.588165998458862, 0.109044015407562],
        [0.711045026779175, 0.398952007293701],
        [0.781069993972778, 0.435405015945435],
        [0.587247014045715, 0.398931980133057],
        [0.742869973182678, 0.355445981025696],
        [0.572156012058258, 0.437651991844177],
        [0.55186802148819, 0.536570012569427],
        [0.821442008018494, 0.457556009292603],
        [0.752701997756958, 0.457181990146637],
        [0.71375697851181, 0.467626988887787],
        [0.66711300611496, 0.460672974586487],
        [0.631101012229919, 0.447153985500336],
        [0.6008620262146, 0.432473003864288],
        [0.523481011390686, 0.405627012252808],
        [0.810747981071472, 0.523926019668579],
        [0.771045982837677, 0.348959028720856],
        [0.509127020835876, 0.562718033790588],
        [0.595292985439301, 0.485023975372314],
        [0.980530977249146, 0.401564002037048],
        [0.573499977588654, 0.420000016689301],
        [0.602994978427887, 0.548687994480133],
        [0.733529984951019, 0.376977026462555],
        [0.560611009597778, 0.519016981124878],
        [0.967685997486115, 0.644356966018677],
        [0.580985009670258, 0.387160003185272],
        [0.537728011608124, 0.505385041236877],
        [0.760966002941132, 0.779752969741821],
        [0.801778972148895, 0.831938028335571],
        [0.892440974712372, 0.54076099395752],
        [0.816350996494293, 0.740260004997253],
        [0.865594983100891, 0.333687007427216],
        [0.614073991775513, 0.883246004581451],
        [0.508952975273132, 0.579437971115112],
        [0.617941975593567, 0.508316040039062],
        [0.825608015060425, 0.397674977779388],
        [0.681214988231659, 0.39623498916626],
        [0.656635999679565, 0.400596976280212],
        [0.603900015354156, 0.710216999053955],
        [0.81208598613739, 0.588539004325867],
        [0.56801301240921, 0.944564998149872],
        [0.681007981300354, 0.898285031318665],
        [0.733752012252808, 0.869701027870178],
        [0.633830010890961, 0.398822009563446],
        [0.606792986392975, 0.39553701877594],
        [0.589659988880157, 0.391062021255493],
        [0.805015981197357, 0.342108011245728],
        [0.611334979534149, 0.362284004688263],
        [0.634037971496582, 0.355970978736877],
        [0.656635999679565, 0.355356991291046],
        [0.681214988231659, 0.35834002494812],
        [0.698584973812103, 0.363156020641327],
        [0.941866993904114, 0.319076001644135],
        [0.698584973812103, 0.387449026107788],
        [0.584177017211914, 0.624107003211975],
        [0.554318010807037, 0.566076993942261],
        [0.534153997898102, 0.62064003944397],
        [0.711217999458313, 0.819975018501282],
        [0.664629995822906, 0.852871000766754],
        [0.559099972248077, 0.902631998062134],
        [0.871706008911133, 0.791940987110138],
        [0.591234028339386, 0.373893976211548],
        [0.544341027736664, 0.451583981513977],
        [0.624562978744507, 0.924192011356354],
        [0.88577002286911, 0.615028977394104],
        [0.551338016986847, 0.695277988910675],
        [0.551980018615723, 0.704632043838501],
        [0.552887976169586, 0.715808033943176],
        [0.555167973041534, 0.730794012546539],
        [0.569944024085999, 0.767035007476807],
        [0.593203008174896, 0.685675978660583],
        [0.599261999130249, 0.681069016456604],
        [0.607599973678589, 0.677703022956848],
        [0.631937980651855, 0.663500010967255],
        [0.752032995223999, 0.601315021514893],
        [0.547226011753082, 0.420395016670227],
        [0.563543975353241, 0.359827995300293],
        [0.583841025829315, 0.368713974952698],
        [0.586614012718201, 0.692366003990173],
        [0.771915018558502, 0.683578014373779],
        [0.531597018241882, 0.352482974529266],
        [0.588370978832245, 0.804440975189209],
        [0.52079701423645, 0.442565023899078],
        [0.567984998226166, 0.493479013442993],
        [0.543282985687256, 0.819254994392395],
        [0.655317008495331, 0.745514988899231],
        [0.621008992195129, 0.574018001556396],
        [0.625559985637665, 0.78031200170517],
        [0.680198013782501, 0.570719003677368],
        [0.64276397228241, 0.604337990283966],
        [0.704662978649139, 0.621529996395111],
        [0.552012026309967, 0.862591981887817],
        [0.589071989059448, 0.508637011051178],
        [0.685944974422455, 0.775357007980347],
        [0.645735025405884, 0.812640011310577],
        [0.675342977046967, 0.703978002071381],
        [0.810858011245728, 0.646304965019226],
        [0.72012197971344, 0.714666962623596],
        [0.866151988506317, 0.682704985141754],
        [0.663187026977539, 0.644596993923187],
        [0.570082008838654, 0.466325998306274],
        [0.544561982154846, 0.548375964164734],
        [0.562758982181549, 0.558784961700439],
        [0.531987011432648, 0.530140042304993],
        [0.585271000862122, 0.335177004337311],
        [0.622952997684479, 0.32277899980545],
        [0.655896008014679, 0.320163011550903],
        [0.687132000923157, 0.322345972061157],
        [0.716481983661652, 0.333200991153717],
        [0.758756995201111, 0.382786989212036],
        [0.897013008594513, 0.468769013881683],
        [0.732392013072968, 0.424547016620636],
        [0.70211398601532, 0.433162987232208],
        [0.66652500629425, 0.433866024017334],
        [0.633504986763, 0.426087975502014],
        [0.603875994682312, 0.416586995124817],
        [0.579657971858978, 0.409945011138916],
        [0.992439985275269, 0.480777025222778],
        [0.567192018032074, 0.569419980049133],
        [0.54136598110199, 0.478899002075195],
        [0.526564002037048, 0.546118021011353],
        [0.523913025856018, 0.563830018043518],
        [0.531529009342194, 0.555056989192963],
        [0.566035985946655, 0.582329034805298],
        [0.51631098985672, 0.563053965568542],
        [0.5174720287323, 0.577877044677734],
        [0.573594987392426, 0.389806985855103],
        [0.560697972774506, 0.395331978797913],
        [0.549755990505219, 0.399751007556915],
        [0.710287988185883, 0.368252992630005],
        [0.723330020904541, 0.363372981548309]
    ];

    /**
     * @license
     * Copyright 2020 Google LLC. All Rights Reserved.
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
    var FACEMESH_GRAPHMODEL_PATH = 'https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1';
    var MESH_MODEL_INPUT_WIDTH = 192;
    var MESH_MODEL_INPUT_HEIGHT = 192;
    /**
     * Load the model.
     *
     * @param options - a configuration object with the following properties:
     *  - `maxContinuousChecks` How many frames to go without running the bounding
     * box detector. Only relevant if maxFaces > 1. Defaults to 5.
     *  - `detectionConfidence` Threshold for discarding a prediction. Defaults to
     * 0.9.
     *  - `maxFaces` The maximum number of faces detected in the input. Should be
     * set to the minimum number for performance. Defaults to 10.
     *  - `iouThreshold` A float representing the threshold for deciding whether
     * boxes overlap too much in non-maximum suppression. Must be between [0, 1].
     * Defaults to 0.3.
     *  - `scoreThreshold` A threshold for deciding when to remove boxes based
     * on score in non-maximum suppression. Defaults to 0.75.
     */
    function load$1(_a) {
        var _b = _a === void 0 ? {} : _a, _c = _b.maxContinuousChecks, maxContinuousChecks = _c === void 0 ? 5 : _c, _d = _b.detectionConfidence, detectionConfidence = _d === void 0 ? 0.9 : _d, _e = _b.maxFaces, maxFaces = _e === void 0 ? 10 : _e, _f = _b.iouThreshold, iouThreshold = _f === void 0 ? 0.3 : _f, _g = _b.scoreThreshold, scoreThreshold = _g === void 0 ? 0.75 : _g;
        return __awaiter(this, void 0, void 0, function () {
            var _h, blazeFace, blazeMeshModel, faceMesh;
            return __generator(this, function (_j) {
                switch (_j.label) {
                    case 0: return [4 /*yield*/, Promise.all([
                            loadDetectorModel(maxFaces, iouThreshold, scoreThreshold), loadMeshModel()
                        ])];
                    case 1:
                        _h = _j.sent(), blazeFace = _h[0], blazeMeshModel = _h[1];
                        faceMesh = new FaceMesh(blazeFace, blazeMeshModel, maxContinuousChecks, detectionConfidence, maxFaces);
                        return [2 /*return*/, faceMesh];
                }
            });
        });
    }
    function loadDetectorModel(maxFaces, iouThreshold, scoreThreshold) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, load({ maxFaces: maxFaces, iouThreshold: iouThreshold, scoreThreshold: scoreThreshold })];
            });
        });
    }
    function loadMeshModel() {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, tfconv.loadGraphModel(FACEMESH_GRAPHMODEL_PATH, { fromTFHub: true })];
            });
        });
    }
    function getInputTensorDimensions$1(input) {
        return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
            [input.height, input.width];
    }
    function flipFaceHorizontal$1(face, imageWidth) {
        if (face.mesh instanceof tf.Tensor) {
            var _a = tf.tidy(function () {
                var subtractBasis = tf.tensor1d([imageWidth - 1, 0, 0]);
                var multiplyBasis = tf.tensor1d([1, -1, 1]);
                return tf.tidy(function () {
                    return [
                        tf.concat([
                            tf.sub(imageWidth - 1, face.boundingBox.topLeft.slice(0, 1)),
                            face.boundingBox.topLeft.slice(1, 1)
                        ]),
                        tf.concat([
                            tf.sub(imageWidth - 1, face.boundingBox.bottomRight.slice(0, 1)),
                            face.boundingBox.bottomRight.slice(1, 1)
                        ]),
                        tf.sub(subtractBasis, face.mesh).mul(multiplyBasis),
                        tf.sub(subtractBasis, face.scaledMesh).mul(multiplyBasis)
                    ];
                });
            }), topLeft = _a[0], bottomRight = _a[1], mesh = _a[2], scaledMesh = _a[3];
            return Object.assign({}, face, { boundingBox: { topLeft: topLeft, bottomRight: bottomRight }, mesh: mesh, scaledMesh: scaledMesh });
        }
        return Object.assign({}, face, {
            boundingBox: {
                topLeft: [
                    imageWidth - 1 - face.boundingBox.topLeft[0],
                    face.boundingBox.topLeft[1]
                ],
                bottomRight: [
                    imageWidth - 1 - face.boundingBox.bottomRight[0],
                    face.boundingBox.bottomRight[1]
                ]
            },
            mesh: (face.mesh).map(function (coord) {
                var flippedCoord = coord.slice(0);
                flippedCoord[0] = imageWidth - 1 - coord[0];
                return flippedCoord;
            }),
            scaledMesh: face.scaledMesh.map(function (coord) {
                var flippedCoord = coord.slice(0);
                flippedCoord[0] = imageWidth - 1 - coord[0];
                return flippedCoord;
            })
        });
    }
    var FaceMesh = /** @class */ (function () {
        function FaceMesh(blazeFace, blazeMeshModel, maxContinuousChecks, detectionConfidence, maxFaces) {
            this.pipeline = new Pipeline(blazeFace, blazeMeshModel, MESH_MODEL_INPUT_WIDTH, MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, maxFaces);
            this.detectionConfidence = detectionConfidence;
        }
        FaceMesh.getAnnotations = function () {
            return MESH_ANNOTATIONS;
        };
        /**
         * Returns an array of UV coordinates for the 468 facial keypoint vertices in
         * mesh_map.jpg. Can be used to map textures to the facial mesh.
         */
        FaceMesh.getUVCoords = function () {
            return UV_COORDS;
        };
        /**
         * Returns an array of faces in an image.
         *
         * @param input The image to classify. Can be a tensor, DOM element image,
         * video, or canvas.
         * @param returnTensors (defaults to `false`) Whether to return tensors as
         * opposed to values.
         * @param flipHorizontal Whether to flip/mirror the facial keypoints
         * horizontally. Should be true for videos that are flipped by default (e.g.
         * webcams).
         *
         * @return An array of AnnotatedPrediction objects.
         */
        FaceMesh.prototype.estimateFaces = function (input, returnTensors, flipHorizontal) {
            if (returnTensors === void 0) { returnTensors = false; }
            if (flipHorizontal === void 0) { flipHorizontal = false; }
            return __awaiter(this, void 0, void 0, function () {
                var _a, width, image, predictions, savedWebglPackDepthwiseConvFlag;
                var _this = this;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            _a = getInputTensorDimensions$1(input), width = _a[1];
                            image = tf.tidy(function () {
                                if (!(input instanceof tf.Tensor)) {
                                    input = tf.browser.fromPixels(input);
                                }
                                return input.toFloat().expandDims(0);
                            });
                            if (!(tf.getBackend() === 'webgl')) return [3 /*break*/, 2];
                            savedWebglPackDepthwiseConvFlag = tf.env().get('WEBGL_PACK_DEPTHWISECONV');
                            tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
                            return [4 /*yield*/, this.pipeline.predict(image)];
                        case 1:
                            predictions = _b.sent();
                            tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
                            return [3 /*break*/, 4];
                        case 2: return [4 /*yield*/, this.pipeline.predict(image)];
                        case 3:
                            predictions = _b.sent();
                            _b.label = 4;
                        case 4:
                            image.dispose();
                            if (predictions != null && predictions.length > 0) {
                                return [2 /*return*/, Promise.all(predictions.map(function (prediction, i) { return __awaiter(_this, void 0, void 0, function () {
                                        var coords, scaledCoords, box, flag, tensorsToRead, tensorValues, flagValue, annotatedPrediction_1, _a, coordsArr, coordsArrScaled, annotatedPrediction, annotations, key;
                                        var _this = this;
                                        return __generator(this, function (_b) {
                                            switch (_b.label) {
                                                case 0:
                                                    coords = prediction.coords, scaledCoords = prediction.scaledCoords, box = prediction.box, flag = prediction.flag;
                                                    tensorsToRead = [flag];
                                                    if (!returnTensors) {
                                                        tensorsToRead = tensorsToRead.concat([coords, scaledCoords]);
                                                    }
                                                    return [4 /*yield*/, Promise.all(tensorsToRead.map(function (d) { return __awaiter(_this, void 0, void 0, function () { return __generator(this, function (_a) {
                                                            return [2 /*return*/, d.array()];
                                                        }); }); }))];
                                                case 1:
                                                    tensorValues = _b.sent();
                                                    flagValue = tensorValues[0];
                                                    flag.dispose();
                                                    if (flagValue < this.detectionConfidence) {
                                                        this.pipeline.clearRegionOfInterest(i);
                                                    }
                                                    if (returnTensors) {
                                                        annotatedPrediction_1 = {
                                                            faceInViewConfidence: flagValue,
                                                            mesh: coords,
                                                            scaledMesh: scaledCoords,
                                                            boundingBox: {
                                                                topLeft: tf.tensor1d(box.startPoint),
                                                                bottomRight: tf.tensor1d(box.endPoint)
                                                            }
                                                        };
                                                        if (flipHorizontal) {
                                                            return [2 /*return*/, flipFaceHorizontal$1(annotatedPrediction_1, width)];
                                                        }
                                                        return [2 /*return*/, annotatedPrediction_1];
                                                    }
                                                    _a = tensorValues.slice(1), coordsArr = _a[0], coordsArrScaled = _a[1];
                                                    scaledCoords.dispose();
                                                    coords.dispose();
                                                    annotatedPrediction = {
                                                        faceInViewConfidence: flagValue,
                                                        boundingBox: { topLeft: box.startPoint, bottomRight: box.endPoint },
                                                        mesh: coordsArr,
                                                        scaledMesh: coordsArrScaled
                                                    };
                                                    if (flipHorizontal) {
                                                        annotatedPrediction =
                                                            flipFaceHorizontal$1(annotatedPrediction, width);
                                                    }
                                                    annotations = {};
                                                    for (key in MESH_ANNOTATIONS) {
                                                        annotations[key] = MESH_ANNOTATIONS[key].map(function (index) { return annotatedPrediction.scaledMesh[index]; });
                                                    }
                                                    annotatedPrediction['annotations'] = annotations;
                                                    return [2 /*return*/, annotatedPrediction];
                                            }
                                        });
                                    }); }))];
                            }
                            return [2 /*return*/, []];
                    }
                });
            });
        };
        return FaceMesh;
    }());

    exports.FaceMesh = FaceMesh;
    exports.load = load$1;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
