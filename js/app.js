let model;
let bodyPixModel;
const fileInput = document.getElementById("fileInput");
const inputImage = document.getElementById("inputImage");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");
const choosePhotoBtn = document.getElementById("choosePhotoBtn");
const progressBar = document.getElementById("progressBar");

let lastLoadedPhotoURL = null;
let analyzing = false;
let modelLoaded = false;

// Initialize UI
if (inputImage) inputImage.style.display = "none";
if (resultDiv) resultDiv.innerHTML = "";

// Enhanced background analysis with anomaly detection
async function analyzeBackground(imageElement) {
  if (!bodyPixModel) {
    console.log("BodyPix model not loaded");
    return null;
  }

  try {
    console.log("Starting enhanced background analysis...");
    const canvas = document.createElement("canvas");
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageElement, 0, 0);

    const segmentation = await bodyPixModel.segmentPerson(canvas, {
      flipHorizontal: false,
      internalResolution: "medium",
      segmentationThreshold: 0.5,
      maxDetections: 1,
      scoreThreshold: 0.3,
      nmsRadius: 20,
    });

    const { width, height, data } = segmentation;
    const expandedMask = new Uint8Array(width * height);
    const marginX = Math.max(Math.floor(width * 0.08), 10);
    const marginY = Math.max(Math.floor(height * 0.08), 10);

    // Copy and expand person mask
    for (let i = 0; i < data.length; i++) {
      expandedMask[i] = data[i];
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (data[idx] === 1) {
          for (let dy = -marginY; dy <= marginY; dy++) {
            for (let dx = -marginX; dx <= marginX; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                const nIdx = ny * width + nx;
                expandedMask[nIdx] = 1;
              }
            }
          }
        }
      }
    }

    const imageData = ctx.getImageData(0, 0, width, height);
    const pixels = imageData.data;
    let backgroundPixels = [];
    let backgroundPositions = [];
    let personPixelCount = 0;

    // Collect background pixels with their positions
    for (let i = 0; i < expandedMask.length; i++) {
      if (expandedMask[i] === 1) {
        personPixelCount++;
      } else {
        const pixelIdx = i * 4;
        const r = pixels[pixelIdx];
        const g = pixels[pixelIdx + 1];
        const b = pixels[pixelIdx + 2];
        const a = pixels[pixelIdx + 3];
        if (a > 0) {
          backgroundPixels.push({ r, g, b });
          backgroundPositions.push({
            x: i % width,
            y: Math.floor(i / width),
            r,
            g,
            b,
          });
        }
      }
    }

    if (backgroundPixels.length < 100) {
      return {
        isPlain: false,
        confidence: 0,
        reason: "Yeterli arka plan alanı bulunamadı",
        avgColor: null,
        variance: 0,
        anomalies: [],
      };
    }

    // Calculate average background color
    const avgR =
      backgroundPixels.reduce((sum, p) => sum + p.r, 0) /
      backgroundPixels.length;
    const avgG =
      backgroundPixels.reduce((sum, p) => sum + p.g, 0) /
      backgroundPixels.length;
    const avgB =
      backgroundPixels.reduce((sum, p) => sum + p.b, 0) /
      backgroundPixels.length;

    // Calculate variance
    let totalVariance = 0;
    for (const pixel of backgroundPixels) {
      const dr = pixel.r - avgR;
      const dg = pixel.g - avgG;
      const db = pixel.b - avgB;
      totalVariance += dr * dr + dg * dg + db * db;
    }
    const variance = totalVariance / backgroundPixels.length;

    // ANOMALY DETECTION - Pixel by pixel analysis
    const anomalies = [];
    const anomalyThreshold = 50; // Color difference threshold
    const clusterRadius = 5; // Minimum cluster size to consider as anomaly

    console.log(
      `Analyzing ${backgroundPositions.length} background pixels for anomalies...`
    );

    // Group anomalous pixels into clusters
    const anomalousPixels = [];

    for (const pos of backgroundPositions) {
      const colorDiff = Math.sqrt(
        Math.pow(pos.r - avgR, 2) +
          Math.pow(pos.g - avgG, 2) +
          Math.pow(pos.b - avgB, 2)
      );

      if (colorDiff > anomalyThreshold) {
        anomalousPixels.push({
          x: pos.x,
          y: pos.y,
          r: pos.r,
          g: pos.g,
          b: pos.b,
          diff: colorDiff,
        });
      }
    }

    // Cluster anomalous pixels
    const clusters = [];
    const visited = new Set();

    for (let i = 0; i < anomalousPixels.length; i++) {
      if (visited.has(i)) continue;

      const cluster = [];
      const queue = [i];
      visited.add(i);

      while (queue.length > 0) {
        const currentIdx = queue.shift();
        const current = anomalousPixels[currentIdx];
        cluster.push(current);

        // Find nearby anomalous pixels
        for (let j = 0; j < anomalousPixels.length; j++) {
          if (visited.has(j)) continue;

          const other = anomalousPixels[j];
          const distance = Math.sqrt(
            Math.pow(current.x - other.x, 2) + Math.pow(current.y - other.y, 2)
          );

          if (distance <= clusterRadius) {
            visited.add(j);
            queue.push(j);
          }
        }
      }

      if (cluster.length >= 3) {
        // Minimum cluster size
        clusters.push(cluster);
      }
    }

    // Create anomaly regions
    for (const cluster of clusters) {
      const xs = cluster.map((p) => p.x);
      const ys = cluster.map((p) => p.y);
      const avgDiff =
        cluster.reduce((sum, p) => sum + p.diff, 0) / cluster.length;
      const avgClusterR =
        cluster.reduce((sum, p) => sum + p.r, 0) / cluster.length;
      const avgClusterG =
        cluster.reduce((sum, p) => sum + p.g, 0) / cluster.length;
      const avgClusterB =
        cluster.reduce((sum, p) => sum + p.b, 0) / cluster.length;

      anomalies.push({
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys),
        pixelCount: cluster.length,
        avgColorDiff: Math.round(avgDiff),
        avgColor: {
          r: Math.round(avgClusterR),
          g: Math.round(avgClusterG),
          b: Math.round(avgClusterB),
        },
      });
    }

    console.log(
      `Found ${anomalies.length} anomaly clusters with ${anomalousPixels.length} total anomalous pixels`
    );

    // Determine if background is acceptable
    const avgBrightness = (avgR + avgG + avgB) / 3;
    const isLightBackground = avgBrightness > 200;
    const isLowVariance = variance < 800;
    const hasSignificantAnomalies =
      anomalies.length > 0 &&
      anomalies.some((a) => a.pixelCount > 10 && a.avgColorDiff > 60);

    const isPlainBackground = isLowVariance && !hasSignificantAnomalies;

    // Calculate confidence score
    const brightnessScore = Math.min(100, (avgBrightness / 255) * 100);
    const uniformityScore = Math.max(0, 100 - variance / 10);
    const anomalyPenalty = hasSignificantAnomalies ? 30 : 0;
    const confidence = Math.max(
      0,
      Math.round((brightnessScore + uniformityScore) / 2 - anomalyPenalty)
    );

    let reason = "Uygun";
    if (!isLightBackground && hasSignificantAnomalies) {
      reason = "Arka plan koyu ve desenli/kirli";
    } else if (!isLightBackground) {
      reason = "Arka plan çok koyu";
    } else if (hasSignificantAnomalies) {
      reason = `Arka planda ${anomalies.length} farklı renk bölgesi tespit edildi`;
    } else if (!isLowVariance) {
      reason = "Arka plan çok karmaşık/desenli";
    }

    return {
      isPlain: isLightBackground && isPlainBackground,
      confidence: confidence,
      avgColor: {
        r: Math.round(avgR),
        g: Math.round(avgG),
        b: Math.round(avgB),
      },
      variance: Math.round(variance),
      brightness: Math.round(avgBrightness),
      reason: reason,
      backgroundPixelCount: backgroundPixels.length,
      personPixelCount: personPixelCount,
      anomalies: anomalies,
      anomalousPixelCount: anomalousPixels.length,
      anomalousPixels: anomalousPixels, // Add anomalousPixels to the return object
    };
  } catch (error) {
    console.error("Background analysis error:", error);
    return {
      isPlain: false,
      confidence: 0,
      reason: `Analiz hatası: ${error.message}`,
      avgColor: null,
      variance: 0,
      anomalies: [],
    };
  }
}

// Function to draw anomaly visualization
function drawAnomalyVisualization(imageElement, backgroundAnalysis) {
  if (
    !backgroundAnalysis ||
    !backgroundAnalysis.anomalies ||
    backgroundAnalysis.anomalies.length === 0
  ) {
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = imageElement.naturalWidth;
  canvas.height = imageElement.naturalHeight;
  const ctx = canvas.getContext("2d");

  // Draw original image
  ctx.drawImage(imageElement, 0, 0);

  // Draw anomaly rectangles
  ctx.strokeStyle = "#ff0000";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(255, 0, 0, 0.2)";

  backgroundAnalysis.anomalies.forEach((anomaly, index) => {
    const width = anomaly.maxX - anomaly.minX;
    const height = anomaly.maxY - anomaly.minY;

    // Draw rectangle
    ctx.fillRect(anomaly.minX, anomaly.minY, width, height);
    ctx.strokeRect(anomaly.minX, anomaly.minY, width, height);

    // Draw label
    ctx.fillStyle = "#ff0000";
    ctx.font = "12px Arial";
    ctx.fillText(`${index + 1}`, anomaly.minX + 2, anomaly.minY + 14);
    ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
  });

  return canvas;
}

// Function to draw face detection visualization
function drawFaceVisualization(imageElement, predictions) {
  if (!predictions || predictions.length === 0) {
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = imageElement.naturalWidth;
  canvas.height = imageElement.naturalHeight;
  const ctx = canvas.getContext("2d");

  // Draw original image
  ctx.drawImage(imageElement, 0, 0);

  predictions.forEach((prediction, index) => {
    const keypoints = prediction.scaledMesh;
    const xs = keypoints.map((pt) => pt[0]);
    const ys = keypoints.map((pt) => pt[1]);

    const minX = Math.max(0, Math.min(...xs));
    const maxX = Math.min(canvas.width, Math.max(...xs));
    const minY = Math.max(0, Math.min(...ys));
    const maxY = Math.min(canvas.height, Math.max(...ys));

    const faceWidth = maxX - minX;
    const faceHeight = maxY - minY;
    const faceHeightPercentage = Math.round((faceHeight / canvas.height) * 100);

    // Draw face bounding box
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.strokeRect(minX, minY, faceWidth, faceHeight);

    // Draw face center
    const faceCenterX = (minX + maxX) / 2;
    const faceCenterY = (minY + maxY) / 2;
    ctx.fillStyle = "#00ff00";
    ctx.beginPath();
    ctx.arc(faceCenterX, faceCenterY, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Draw image center
    const imageCenterX = canvas.width / 2;
    const imageCenterY = canvas.height / 2;
    ctx.fillStyle = "#ff0000";
    ctx.beginPath();
    ctx.arc(imageCenterX, imageCenterY, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Draw center line
    ctx.strokeStyle = "#ff0000";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, imageCenterY);
    ctx.lineTo(canvas.width, imageCenterY);
    ctx.moveTo(imageCenterX, 0);
    ctx.lineTo(imageCenterX, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw key facial points
    ctx.fillStyle = "#0000ff";
    const keyFacialPoints = [
      { index: 33, label: "L Eye" }, // Left eye
      { index: 263, label: "R Eye" }, // Right eye
      { index: 1, label: "Nose" }, // Nose tip
      { index: 13, label: "U Lip" }, // Upper lip
      { index: 14, label: "L Lip" }, // Lower lip
      { index: 10, label: "Forehead" }, // Forehead
      { index: 152, label: "Chin" }, // Chin
    ];

    keyFacialPoints.forEach((point) => {
      if (keypoints[point.index]) {
        const [x, y] = keypoints[point.index];
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();

        // Label
        ctx.fillStyle = "#0000ff";
        ctx.font = "10px Arial";
        ctx.fillText(point.label, x + 5, y - 5);
      }
    });

    // Draw info box
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(10, 10, 250, 120);

    ctx.fillStyle = "#ffffff";
    ctx.font = "12px Arial";
    ctx.fillText(`Yüz ${index + 1} Detayları:`, 15, 25);
    ctx.fillText(`Boyut: ${faceWidth}x${faceHeight} px`, 15, 40);
    ctx.fillText(`Yükseklik: ${faceHeightPercentage}% (hedef: 50-80%)`, 15, 55);
    ctx.fillText(
      `Pozisyon: (${Math.round(minX)}, ${Math.round(minY)})`,
      15,
      70
    );
    ctx.fillText(
      `Merkez: (${Math.round(faceCenterX)}, ${Math.round(faceCenterY)})`,
      15,
      85
    );
    ctx.fillText(
      `Resim merkezi: (${Math.round(imageCenterX)}, ${Math.round(
        imageCenterY
      )})`,
      15,
      100
    );

    // Calculate and show offset
    const xOffset = (faceCenterX - imageCenterX) / imageCenterX;
    const yOffset = (faceCenterY - imageCenterY) / imageCenterY;
    const xPercentage = Math.round(Math.abs(xOffset) * 100);
    const yPercentage = Math.round(Math.abs(yOffset) * 100);

    ctx.fillText(`Sapma: X:${xPercentage}%, Y:${yPercentage}%`, 15, 115);
  });

  return canvas;
}

// Display results function
// Remove old processImage and displayResults functions and all resultDiv result display logic
// (No code for processImage, displayResults, or resultDiv result display remains)

// Process image function
async function processImage() {
  if (!model) {
    resultDiv.innerHTML = `
      <div class="result-fail">
        <span>Model yüklenmedi. Lütfen bekleyin.</span>
      </div>
    `;

    return;
  }

  if (analyzing) return;
  analyzing = true;
  showSpinner("Analiz ediliyor...");

  try {
    console.log("Starting face detection...");
    const predictions = await model.estimateFaces(inputImage);
    console.log("Face detection complete, predictions:", predictions);

    // Create face visualization
    const faceCanvas = drawFaceVisualization(inputImage, predictions);

    let backgroundAnalysis = null;
    if (bodyPixModel) {
      console.log("Starting background analysis...");
      backgroundAnalysis = await analyzeBackground(inputImage);
      console.log("Background analysis complete");
    }

    // Create anomaly visualization
    const anomalyCanvas = backgroundAnalysis
      ? drawAnomalyVisualization(inputImage, backgroundAnalysis)
      : null;

    hideSpinner();
    // Display results
    // The old displayResults function is removed, so this block is no longer needed.
    // The new UI flow handles result display via showResults and resultPanel/resultList.
  } catch (err) {
    console.error("Analysis error:", err);
    hideSpinner();
    resultDiv.innerHTML = `
      <div class="result-fail">
        <span>Analiz hatası: ${err.message}</span>
      </div>
    `;
  } finally {
    analyzing = false;
  }
}

// Load models function
async function loadModel() {
  try {
    console.log("Checking libraries:", {
      tf: typeof tf,
      facemesh: typeof facemesh,
      bodyPix: typeof bodyPix,
    });

    if (typeof tf === "undefined") throw new Error("TensorFlow.js yüklenemedi");
    if (typeof facemesh === "undefined")
      throw new Error("Facemesh modeli yüklenemedi");
    if (typeof bodyPix === "undefined")
      throw new Error("BodyPix modeli yüklenemedi");

    // Show loading progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 5;
      if (progress > 85) clearInterval(interval);
      if (progressBar) progressBar.style.width = `${Math.min(progress, 90)}%`;
    }, 200);

    if (loadingDiv) {
      loadingDiv.style.display = "block";
      const loadingTextDiv = document.getElementById("loadingText");
      if (loadingTextDiv)
        loadingTextDiv.textContent = "Yapay zeka modelleri yükleniyor...";
    }

    console.log("Loading models...");

    // Load both models in parallel
    const [faceModel, bodyModel] = await Promise.all([
      facemesh.load(),
      bodyPix.load({
        architecture: "MobileNetV1",
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2,
      }),
    ]);

    model = faceModel;
    bodyPixModel = bodyModel;
    modelLoaded = true;

    console.log("Both models loaded successfully");

    // Complete progress bar
    if (progressBar) progressBar.style.width = "100%";

    setTimeout(() => {
      if (loadingDiv) {
        loadingDiv.style.display = "none";
      }
    }, 1000);

    if (fileInput) fileInput.disabled = false;
  } catch (e) {
    console.error("Model loading error:", e);
    modelLoaded = false;
    if (loadingDiv) {
      loadingDiv.style.display = "block";
      const loadingTextDiv = document.getElementById("loadingText");
      if (loadingTextDiv)
        loadingTextDiv.textContent = `AI modelleri yüklenemedi: ${e.message}`;
    }
    if (fileInput) fileInput.disabled = true;
  }
}

// Utility functions
function resetUI() {
  analyzing = false;
  if (resultDiv) resultDiv.innerHTML = "";
  if (inputImage) inputImage.style.display = "none";
  showSpinner("Fotoğraf hazırlanıyor...");
}

function showSpinner(text) {
  if (loadingDiv) {
    loadingDiv.style.display = "flex";
    loadingDiv.innerHTML = `
      <div class="loader"></div>
      <div>${text}</div>
    `;
  }
}

function hideSpinner() {
  if (loadingDiv) {
    loadingDiv.style.display = "none";
  }
}

// Utility: Resize image to max height and return canvas
function getResizedCanvas(image, maxHeight = 480) {
  const scale = maxHeight / image.naturalHeight;
  const width = Math.round(image.naturalWidth * scale);
  const height = Math.round(image.naturalHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, width, height);
  return { canvas, scale };
}

// Utility: Map landmarks from resized canvas back to original image
function mapLandmarksToOriginal(landmarks, scale) {
  return landmarks.map((pt) => [pt[0] / scale, pt[1] / scale, pt[2] / scale]);
}

// Event listeners
if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (lastLoadedPhotoURL) URL.revokeObjectURL(lastLoadedPhotoURL);
    lastLoadedPhotoURL = URL.createObjectURL(file);

    if (inputImage) {
      inputImage.src = lastLoadedPhotoURL;
      inputImage.onload = () => {
        if (!analyzing && modelLoaded) {
          // Only process if not analyzing and model is loaded
          // The new UI flow handles the analysis button click
        }
      };
      inputImage.style.display = "block";
    }
  });
}

// === ICAO Biyometrik Fotoğraf Kontrolü Uygulaması ===
// UI ve temel akış refaktörü, ICAO kontrolleri için fonksiyon yer tutucuları ile

// --- DOM Elementleri ---
const analyzeBtn = document.getElementById("analyzeBtn");
const resultCanvas = document.getElementById("resultCanvas");
const resultPanel = document.getElementById("resultPanel");
const resultList = document.getElementById("resultList");

let selectedImage = null;
let imageBitmap = null;

// --- Fotoğraf Seçildiğinde ---
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  inputImage.src = url;
  inputImage.style.display = "block";
  resultCanvas.style.display = "none";
  resultPanel.style.display = "none";
  analyzeBtn.style.display = "block";
  selectedImage = file;
  // Görüntüyü bitmap olarak da sakla (analiz için)
  imageBitmap = await createImageBitmap(file);
});

// --- 'Analiz Et' Butonuna Basıldığında ---
analyzeBtn.addEventListener("click", async () => {
  if (!selectedImage || !imageBitmap) return;
  // Yükleniyor göster
  loadingDiv.style.display = "block";
  const loadingTextDiv = document.getElementById("loadingText");
  if (loadingTextDiv)
    loadingTextDiv.textContent = "Yapay zeka analiz yapıyor...";
  analyzeBtn.disabled = true;
  resultPanel.style.display = "none";
  resultCanvas.style.display = "none";
  resultList.innerHTML = "";

  let results = [];
  let backgroundAnalysis = null;
  let facePredictions = [];
  let faceCoverageResult = null;
  let faceCanvas = null;

  try {
    // --- BACKGROUND CHECK (run first so overlays are available) ---
    if (!bodyPixModel) {
      backgroundAnalysis = null;
      results.push({
        passed: false,
        message: "AI arka plan modeli yüklenmedi.",
        info: "Arka plan analizi için AI modeli gereklidir.",
      });
    } else {
      backgroundAnalysis = await analyzeBackground(inputImage);
      console.log("[DEBUG] Background analysis:", backgroundAnalysis);
    }

    // --- FACE DETECTION & COVERAGE CHECK ---
    if (!model) {
      results.push({
        passed: false,
        message: "AI yüz modeli yüklenmedi.",
        info: "Yüz tespiti için AI modeli gereklidir.",
      });
    } else {
      // Resize image before facemesh
      const { canvas: resizedCanvas, scale } = getResizedCanvas(
        inputImage,
        480
      );
      console.log(
        `[DEBUG] Original size: ${inputImage.naturalWidth}x${inputImage.naturalHeight}, Resized: ${resizedCanvas.width}x${resizedCanvas.height}, Scale: ${scale}`
      );
      facePredictions = await model.estimateFaces(resizedCanvas);
      console.log("[DEBUG] Face predictions (resized):", facePredictions);
      if (facePredictions.length === 0) {
        results.push({
          passed: false,
          message: "Yüz tespit edilemedi.",
          info: "Fotoğrafta yüz algılanamadı. Lütfen yüzünüz net ve doğrudan kameraya bakıyor olsun.",
        });
      } else if (facePredictions.length > 1) {
        results.push({
          passed: false,
          message: `Birden fazla yüz tespit edildi (${facePredictions.length}). Sadece bir kişi olmalı.`,
          info: "Fotoğrafta yalnızca bir kişi olmalıdır.",
        });
      } else {
        // Tek yüz bulundu
        // Map landmarks back to original image size
        const keypointsResized = facePredictions[0].scaledMesh;
        const keypoints = mapLandmarksToOriginal(keypointsResized, scale);
        // ICAO: Estimate head top above minY
        const chinY = keypoints[152][1];
        const minY = Math.min(...keypoints.map((pt) => pt[1]));
        const faceHeightRaw = chinY - minY;
        const estimatedHeadTop = minY - 0.25 * faceHeightRaw;
        const faceHeight = chinY - estimatedHeadTop;
        const imgHeight = inputImage.naturalHeight;
        const faceCoverageRatio = faceHeight / imgHeight;
        const percent = Math.round(faceCoverageRatio * 100);
        console.log(
          `[DEBUG] Face chinY: ${chinY}, minY: ${minY}, estimatedHeadTop: ${estimatedHeadTop}, faceHeight: ${faceHeight}, Image height: ${imgHeight}, Ratio: ${percent}%`
        );
        if (percent >= 70 && percent <= 80) {
          faceCoverageResult = {
            passed: true,
            message: `Yüz oranı uygun (%${percent}).`,
            info: "Yüz, fotoğrafın %70-%80'ini kaplamalıdır (çene ile baş üstü arası, baş üstü tahmini).",
          };
        } else {
          faceCoverageResult = {
            passed: false,
            message: `Yüz oranı uygun değil (%${percent}). ICAO: %70-%80 arası olmalı.`,
            info: "Yüz, fotoğrafın %70-%80'ini kaplamalıdır (çene ile baş üstü arası, baş üstü tahmini).",
          };
        }
        results.push(faceCoverageResult);

        // --- ICAO CHECK: EYES OPEN ---
        // Left eye: 159 (top), 145 (bottom); Right eye: 386 (top), 374 (bottom)
        const leftEyeTop = keypoints[159][1];
        const leftEyeBot = keypoints[145][1];
        const rightEyeTop = keypoints[386][1];
        const rightEyeBot = keypoints[374][1];
        const leftEyeOpenness = leftEyeBot - leftEyeTop;
        const rightEyeOpenness = rightEyeBot - rightEyeTop;
        const minEyeOpenness = Math.max(2, faceHeight * 0.015); // at least 2px for small faces
        let eyesOpenPassed = true;
        let eyeDetails = [];
        if (leftEyeOpenness <= minEyeOpenness) {
          eyesOpenPassed = false;
          eyeDetails.push("sol göz");
        }
        if (rightEyeOpenness <= minEyeOpenness) {
          eyesOpenPassed = false;
          eyeDetails.push("sağ göz");
        }
        console.log(
          `[DEBUG] Left eye openness: ${leftEyeOpenness}, Right eye openness: ${rightEyeOpenness}, Min: ${minEyeOpenness}`
        );
        if (eyesOpenPassed) {
          results.push({
            passed: true,
            message: "Gözler açık ve görünür.",
            info: "Her iki göz de açık ve net görünür olmalıdır.",
          });
        } else {
          results.push({
            passed: false,
            message: `Gözler yeterince açık değil (${eyeDetails.join(", ")}).`,
            info: "Her iki göz de açık ve net görünür olmalıdır.",
          });
        }

        // --- ICAO CHECK: NEUTRAL EXPRESSION (MOUTH CLOSED) ---
        // Upper lip: 13, Lower lip: 14
        const upperLip = keypoints[13][1];
        const lowerLip = keypoints[14][1];
        const lipDistance = lowerLip - upperLip;
        const maxLipDistance = faceHeight * 0.06;
        console.log(
          `[DEBUG] Lip distance: ${lipDistance}, Max allowed: ${maxLipDistance}`
        );
        if (lipDistance < maxLipDistance) {
          results.push({
            passed: true,
            message: "Nötr ifade (ağız kapalı).",
            info: "Ağız kapalı ve yüz ifadesi nötr olmalıdır.",
          });
        } else {
          const lipOpenPercentage = Math.round(
            (lipDistance / faceHeight) * 100
          );
          results.push({
            passed: false,
            message: `Ağız çok açık (%${lipOpenPercentage} yüz yüksekliği) - nötr ifade gerekli.`,
            info: "Ağız kapalı ve yüz ifadesi nötr olmalıdır.",
          });
        }

        // --- ICAO CHECK: FACE CENTERED ---
        // Calculate face center (mean of all landmarks)
        const xs = keypoints.map((pt) => pt[0]);
        const ys = keypoints.map((pt) => pt[1]);
        const faceCenterX = xs.reduce((a, b) => a + b, 0) / xs.length;
        const faceCenterY = ys.reduce((a, b) => a + b, 0) / ys.length;
        const imageCenterX = inputImage.naturalWidth / 2;
        const imageCenterY = inputImage.naturalHeight / 2;
        const xOffset = (faceCenterX - imageCenterX) / imageCenterX;
        const yOffset = (faceCenterY - imageCenterY) / imageCenterY;
        const xPercentage = Math.round(Math.abs(xOffset) * 100);
        const yPercentage = Math.round(Math.abs(yOffset) * 100);
        const xTolerance = 0.1; // 10%
        const yTolerance = 0.1; // 10%
        let centeredPassed =
          Math.abs(xOffset) <= xTolerance && Math.abs(yOffset) <= yTolerance;
        let centerDetails = [];
        let centerAdvice = "";
        if (!centeredPassed) {
          if (Math.abs(xOffset) > xTolerance) {
            centerDetails.push(`X: %${xPercentage}`);
            if (faceCenterX < imageCenterX) {
              centerAdvice += "Yüz çok solda (sağ boşluk fazla). ";
            } else if (faceCenterX > imageCenterX) {
              centerAdvice += "Yüz çok sağda (sol boşluk fazla). ";
            }
          }
          if (Math.abs(yOffset) > yTolerance) {
            centerDetails.push(`Y: %${yPercentage}`);
            if (faceCenterY > imageCenterY) {
              centerAdvice += "Yüz çok aşağıda (üst boşluk fazla). ";
            } else if (faceCenterY < imageCenterY) {
              centerAdvice += "Yüz çok yukarıda (alt boşluk fazla). ";
            }
          }
        }
        centerAdvice = centerAdvice.trim();
        console.log(
          `[DEBUG] Face center: (${faceCenterX}, ${faceCenterY}), Image center: (${imageCenterX}, ${imageCenterY}), X offset: ${xOffset}, Y offset: ${yOffset}`
        );
        if (centeredPassed) {
          results.push({
            passed: true,
            message: "Yüz fotoğrafın ortasında.",
            info: "Yüz, fotoğrafın yatay ve dikey olarak ortasında olmalıdır (sapma ≤ %10).",
          });
        } else {
          results.push({
            passed: false,
            message: `Yüz ortalanmamış (${centerDetails.join(", ")}).`,
            info: `Yüz, fotoğrafın yatay ve dikey olarak ortasında olmalıdır (sapma ≤ %10). ${centerAdvice}`.trim(),
          });
        }

        // --- ICAO CHECK: NO HEAD TILT (LOOKING DIRECTLY AT CAMERA) ---
        // Left eye: 33, Right eye: 263
        const leftEye = keypoints[33];
        const rightEye = keypoints[263];
        let tiltAngle = 0;
        if (leftEye && rightEye) {
          const dx = rightEye[0] - leftEye[0];
          const dy = rightEye[1] - leftEye[1];
          tiltAngle = Math.atan2(dy, dx) * (180 / Math.PI);
          const absTilt = Math.abs(tiltAngle);
          console.log(`[DEBUG] Eye tilt angle: ${tiltAngle}°`);
          if (absTilt < 5) {
            results.push({
              passed: true,
              message: "Baş eğik değil (düz bakış).",
              info: "Baş, kameraya düz bakacak şekilde olmalı (eğim < 5°).",
            });
          } else {
            results.push({
              passed: false,
              message: `Baş eğik (~${tiltAngle.toFixed(
                1
              )}°). Düz bakış gerekli!`,
              info: "Baş, kameraya düz bakacak şekilde olmalı (eğim < 5°).",
            });
          }
        }

        // --- ICAO CHECK: SHARP FOCUS (NOT BLURRY) ---
        // Use face bounding box
        const minX = Math.max(0, Math.min(...keypoints.map((pt) => pt[0])));
        const maxX = Math.min(
          inputImage.naturalWidth,
          Math.max(...keypoints.map((pt) => pt[0]))
        );
        const minYBox = Math.max(0, Math.min(...keypoints.map((pt) => pt[1])));
        const maxYBox = Math.min(
          inputImage.naturalHeight,
          Math.max(...keypoints.map((pt) => pt[1]))
        );
        const faceW = Math.max(1, maxX - minX);
        const faceH = Math.max(1, maxYBox - minYBox);
        if (faceW < 5 || faceH < 5) {
          results.push({
            passed: false,
            message: "Yüz bölgesi çok küçük, netlik kontrolü yapılamadı.",
            info: "Yüzün netliği değerlendirilemedi. Fotoğrafı daha yakın çekin veya daha yüksek çözünürlükte yükleyin.",
          });
        } else {
          // Extract face region from image
          const tmpCanvas = document.createElement("canvas");
          tmpCanvas.width = faceW;
          tmpCanvas.height = faceH;
          const tmpCtx = tmpCanvas.getContext("2d");
          tmpCtx.drawImage(
            inputImage,
            minX,
            minYBox,
            faceW,
            faceH,
            0,
            0,
            faceW,
            faceH
          );
          const faceImageData = tmpCtx.getImageData(0, 0, faceW, faceH);
          // Convert to grayscale
          const gray = [];
          for (let i = 0; i < faceImageData.data.length; i += 4) {
            const r = faceImageData.data[i];
            const g = faceImageData.data[i + 1];
            const b = faceImageData.data[i + 2];
            gray.push(0.299 * r + 0.587 * g + 0.114 * b);
          }
          // Compute Laplacian
          function laplacianVariance(gray, w, h) {
            let sum = 0,
              sumSq = 0,
              count = 0;
            const kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0];
            for (let y = 1; y < h - 1; y++) {
              for (let x = 1; x < w - 1; x++) {
                let lap = 0;
                let idx = 0;
                for (let ky = -1; ky <= 1; ky++) {
                  for (let kx = -1; kx <= 1; kx++) {
                    const px = x + kx;
                    const py = y + ky;
                    lap += gray[py * w + px] * kernel[idx++];
                  }
                }
                sum += lap;
                sumSq += lap * lap;
                count++;
              }
            }
            if (count === 0) return NaN;
            const mean = sum / count;
            return sumSq / count - mean * mean;
          }
          const sharpness = laplacianVariance(gray, faceW, faceH);
          console.log(
            `[DEBUG] Face sharpness (variance of Laplacian): ${sharpness}`
          );
          const sharpnessThreshold = 40; // Less strict, more realistic for real photos
          if (!isFinite(sharpness) || isNaN(sharpness)) {
            results.push({
              passed: false,
              message: "Yüz netliği hesaplanamadı.",
              info: "Yüzün netliği değerlendirilemedi. Fotoğrafı daha yakın çekin veya daha yüksek çözünürlükte yükleyin.",
            });
          } else if (sharpness > sharpnessThreshold) {
            results.push({
              passed: true,
              message: `Yüz net (odak iyi). (Keskinlik: ${sharpness.toFixed(
                1
              )})`,
              info: "Yüz net ve odakta olmalıdır. Bulanık fotoğraflar kabul edilmez.",
            });
          } else {
            results.push({
              passed: false,
              message: `Yüz bulanık veya odak kötü. (Keskinlik: ${sharpness.toFixed(
                1
              )})`,
              info: "Yüz net ve odakta olmalıdır. Fotoğrafı daha net çekin veya yeniden tarayın.",
            });
          }
        }

        // Görseli çiz (draw all landmarks as blue dots for debug)
        faceCanvas = document.createElement("canvas");
        faceCanvas.width = inputImage.naturalWidth;
        faceCanvas.height = inputImage.naturalHeight;
        const ctx = faceCanvas.getContext("2d");
        ctx.drawImage(inputImage, 0, 0, faceCanvas.width, faceCanvas.height);
        // Draw all landmarks
        ctx.fillStyle = "#1976d2";
        keypoints.forEach((pt) => {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 2, 0, 2 * Math.PI);
          ctx.fill();
        });
        // Draw estimated head top as a red line
        ctx.strokeStyle = "#e53935";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, estimatedHeadTop);
        ctx.lineTo(faceCanvas.width, estimatedHeadTop);
        ctx.stroke();
        ctx.fillStyle = "#e53935";
        ctx.beginPath();
        ctx.arc(faceCanvas.width / 2, estimatedHeadTop, 4, 0, 2 * Math.PI);
        ctx.fill();
        // Draw chin as a red line
        ctx.strokeStyle = "#e53935";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, chinY);
        ctx.lineTo(faceCanvas.width, chinY);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(faceCanvas.width / 2, chinY, 4, 0, 2 * Math.PI);
        ctx.fill();
        // Draw face center as green cross
        ctx.strokeStyle = "#43a047";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(faceCenterX - 10, faceCenterY);
        ctx.lineTo(faceCenterX + 10, faceCenterY);
        ctx.moveTo(faceCenterX, faceCenterY - 10);
        ctx.lineTo(faceCenterX, faceCenterY + 10);
        ctx.stroke();
        // Draw image center as orange cross
        ctx.strokeStyle = "#ff9800";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(imageCenterX - 10, imageCenterY);
        ctx.lineTo(imageCenterX + 10, imageCenterY);
        ctx.moveTo(imageCenterX, imageCenterY - 10);
        ctx.lineTo(imageCenterX, imageCenterY + 10);
        ctx.stroke();
        // Draw background anomaly regions if any
        if (
          backgroundAnalysis &&
          backgroundAnalysis.anomalies &&
          backgroundAnalysis.anomalies.length > 0
        ) {
          ctx.save();
          ctx.strokeStyle = "#ff0000";
          ctx.lineWidth = 2;
          ctx.fillStyle = "rgba(255,0,0,0.18)";
          backgroundAnalysis.anomalies.forEach((anomaly, idx) => {
            const width = anomaly.maxX - anomaly.minX;
            const height = anomaly.maxY - anomaly.minY;
            ctx.fillRect(anomaly.minX, anomaly.minY, width, height);
            ctx.strokeRect(anomaly.minX, anomaly.minY, width, height);
            // Label
            ctx.fillStyle = "#ff0000";
            ctx.font = "bold 14px Arial";
            ctx.fillText(`${idx + 1}`, anomaly.minX + 4, anomaly.minY + 18);
            ctx.fillStyle = "rgba(255,0,0,0.18)";
          });
          // Draw actual anomaly pixels as red dots
          if (
            backgroundAnalysis.anomalousPixels &&
            backgroundAnalysis.anomalousPixels.length > 0
          ) {
            ctx.fillStyle = "#ff0000";
            backgroundAnalysis.anomalousPixels.forEach((p) => {
              ctx.beginPath();
              ctx.arc(p.x, p.y, 1.5, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
          ctx.restore();
        }
        // Optionally, draw bounding box and key points as before
        // (You can add more debug drawing here if needed)
      }
    }

    // --- BACKGROUND CHECK (results only) ---
    if (backgroundAnalysis) {
      if (backgroundAnalysis.isPlain) {
        results.push({
          passed: true,
          message: `Arka plan uygun (güven: %${backgroundAnalysis.confidence}, parlaklık: ${backgroundAnalysis.brightness}/255)`,
          info: "Arka plan beyaz ve düz olmalıdır.",
        });
      } else {
        let advice = "";
        if (backgroundAnalysis.brightness < 200)
          advice += " - daha açık arka plan kullanın";
        if (backgroundAnalysis.variance > 800)
          advice += " - daha düz/tek renk arka plan kullanın";
        if (
          backgroundAnalysis.anomalies &&
          backgroundAnalysis.anomalies.length > 0
        )
          advice += ` - ${backgroundAnalysis.anomalies.length} farklı renk bölgesi temizleyin`;
        results.push({
          passed: false,
          message: `Arka plan uygun değil - ${backgroundAnalysis.reason}${advice}`,
          info: "Arka plan beyaz ve düz olmalıdır.",
        });
      }
    }
  } catch (err) {
    results.push({
      passed: false,
      message: `Analiz hatası: ${err.message}`,
      info: "Beklenmeyen bir hata oluştu.",
    });
  }

  // Sonuçları göster
  showResults(results);

  // Sonuç görselini göster (öncelik: yüz varsa yüz çizimi, yoksa arka plan anomalisi, yoksa orijinal)
  if (faceCanvas) {
    resultCanvas.width = faceCanvas.width;
    resultCanvas.height = faceCanvas.height;
    const ctx = resultCanvas.getContext("2d");
    ctx.drawImage(faceCanvas, 0, 0);
    resultCanvas.style.display = "block";
  } else if (
    backgroundAnalysis &&
    backgroundAnalysis.anomalies &&
    backgroundAnalysis.anomalies.length > 0
  ) {
    const anomalyCanvas = drawAnomalyVisualization(
      inputImage,
      backgroundAnalysis
    );
    if (anomalyCanvas) {
      resultCanvas.width = anomalyCanvas.width;
      resultCanvas.height = anomalyCanvas.height;
      const ctx = resultCanvas.getContext("2d");
      ctx.drawImage(anomalyCanvas, 0, 0);
      resultCanvas.style.display = "block";
    }
  } else {
    drawResultImage();
  }

  loadingDiv.style.display = "none";
  analyzeBtn.disabled = false;
});

// --- Sonuçları Listele ---
function showResults(results) {
  resultPanel.style.display = "block";
  resultList.innerHTML = "";
  results.forEach((r) => {
    const li = document.createElement("li");
    li.innerHTML = `<div style='font-size:1.08em;font-weight:600;margin-bottom:0.18em;'>${
      r.message
    }</div><div style='font-size:1em;opacity:0.85;margin-top:4px;'>${
      r.info || ""
    }</div>`;
    if (
      r.message.includes("hesaplanamadı") ||
      r.message.includes("kontrolü yapılamadı")
    ) {
      li.style.color = "#888";
      li.style.fontWeight = "400";
    } else {
      li.style.color = r.passed ? "#2e7d32" : "#c62828";
      li.style.fontWeight = r.passed ? "500" : "600";
    }
    li.style.marginBottom = "1em";
    resultList.appendChild(li);
  });
}

// --- Sonuç Görselini Çiz (şimdilik orijinal) ---
function drawResultImage() {
  if (!imageBitmap) return;
  resultCanvas.width = imageBitmap.width;
  resultCanvas.height = imageBitmap.height;
  const ctx = resultCanvas.getContext("2d");
  ctx.drawImage(imageBitmap, 0, 0);
  resultCanvas.style.display = "block";
}

// --- ICAO Kontrolleri için Fonksiyon Yer Tutucuları ---
// TODO: Her bir ICAO kuralı için fonksiyonlar ekle
// function checkColor(imageData) { ... }
// function checkNeutralExpression(landmarks) { ... }
// function checkEyesOpen(landmarks) { ... }
// function checkBackground(imageData, mask) { ... }
// function checkFaceCoverage(landmarks, imageDims) { ... }
// function checkSharpness(imageData) { ... }
// function checkCentered(landmarks, imageDims) { ... }
// function checkGlasses(landmarks, imageData) { ... }
// ...

// --- Başlatıcı (gerekirse) ---
window.initializeApp = function () {
  // Gerekli başlatma işlemleri burada yapılabilir
};

// Auto-initialize if libraries are already loaded
if (
  typeof tf !== "undefined" &&
  typeof facemesh !== "undefined" &&
  typeof bodyPix !== "undefined"
) {
  console.log("Libraries already loaded, auto-initializing...");
  loadModel();
}
