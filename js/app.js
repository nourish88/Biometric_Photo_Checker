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
function displayResults(
  predictions,
  imgW,
  imgH,
  backgroundAnalysis,
  faceCanvas,
  anomalyCanvas
) {
  let checks = [];
  let passed = true;

  // Face detection checks
  if (predictions.length === 1) {
    checks.push({ ok: true, text: "Tek bir yüz tespit edildi" });

    const keypoints = predictions[0].scaledMesh;
    const xs = keypoints.map((pt) => pt[0]);
    const ys = keypoints.map((pt) => pt[1]);

    const minX = Math.max(0, Math.min(...xs));
    const maxX = Math.min(imgW, Math.max(...xs));
    const minY = Math.max(0, Math.min(...ys));
    const maxY = Math.min(imgH, Math.max(...ys));

    const faceWidth = maxX - minX;
    const faceHeight = maxY - minY;
    const faceHeightPercentage = (faceHeight / imgH) * 100;

    console.log("Face detection details:", {
      imageSize: `${imgW}x${imgH}`,
      faceSize: `${faceWidth}x${faceHeight}`,
      faceHeightPercentage: faceHeightPercentage,
      faceBounds: { minX, maxX, minY, maxY },
      keypointCount: keypoints.length,
    });

    // Face size check with detailed info
    if (faceHeightPercentage >= 50 && faceHeightPercentage <= 80) {
      checks.push({
        ok: true,
        text: `Yüz boyutu uygun (${Math.round(
          faceHeight
        )}px = yüksekliğin %${Math.round(faceHeightPercentage)}'i)`,
      });
    } else {
      let sizeAdvice = "";
      if (faceHeightPercentage < 50) {
        sizeAdvice = " - daha yakın çekin";
      } else if (faceHeightPercentage > 80) {
        sizeAdvice = " - daha uzak çekin";
      }

      checks.push({
        ok: false,
        text: `Yüz boyutu uygun değil (${Math.round(
          faceHeight
        )}px = %${Math.round(
          faceHeightPercentage
        )}, ICAO: 50-80%)${sizeAdvice}`,
      });
      passed = false;
    }

    // Face positioning with 10% tolerance
    const faceCenterX = (minX + maxX) / 2;
    const faceCenterY = (minY + maxY) / 2;
    const imageCenterX = imgW / 2;
    const imageCenterY = imgH / 2;

    const xOffset = (faceCenterX - imageCenterX) / imageCenterX;
    const yOffset = (faceCenterY - imageCenterY) / imageCenterY;

    const xPercentage = Math.round(Math.abs(xOffset) * 100);
    const yPercentage = Math.round(Math.abs(yOffset) * 100);

    // 10% tolerance for both X and Y
    const xTolerance = 0.1;
    const yTolerance = 0.1;

    // Determine direction
    let xDirection = "";
    let yDirection = "";

    if (Math.abs(xOffset) > xTolerance) {
      xDirection = xOffset > 0 ? "sağda" : "solda";
    }

    if (Math.abs(yOffset) > yTolerance) {
      yDirection = yOffset > 0 ? "aşağıda" : "yukarıda";
    }

    if (Math.abs(xOffset) <= xTolerance && Math.abs(yOffset) <= yTolerance) {
      checks.push({
        ok: true,
        text: `Yüz pozisyonu merkezi (sapma: X:%${xPercentage}, Y:%${yPercentage})`,
      });
    } else {
      let positionText = "Yüz merkezi değil - ";
      let details = [];

      if (Math.abs(xOffset) > xTolerance) {
        details.push(`%${xPercentage} ${xDirection}`);
      }

      if (Math.abs(yOffset) > yTolerance) {
        details.push(`%${yPercentage} ${yDirection}`);
      }

      positionText += details.join(", ");

      checks.push({
        ok: false,
        text: positionText,
      });
      passed = false;
    }

    // Eye alignment with angle measurement
    const leftEye = keypoints[33];
    const rightEye = keypoints[263];
    if (leftEye && rightEye) {
      const eyeDx = Math.abs(leftEye[1] - rightEye[1]);
      const eyeDistance = Math.sqrt(
        Math.pow(rightEye[0] - leftEye[0], 2) +
          Math.pow(rightEye[1] - leftEye[1], 2)
      );
      const tiltAngle = Math.round(
        Math.atan2(eyeDx, eyeDistance) * (180 / Math.PI)
      );

      if (eyeDx < faceHeight * 0.08) {
        checks.push({
          ok: true,
          text: `Düz bakış (eğim: ~${tiltAngle}°)`,
        });
      } else {
        checks.push({
          ok: false,
          text: `Baş eğik (~${tiltAngle}°) - düz bakış gerekli`,
        });
        passed = false;
      }
    }

    // Eyes open with measurement
    const leftEyeTop = keypoints[159][1];
    const leftEyeBot = keypoints[145][1];
    const rightEyeTop = keypoints[386][1];
    const rightEyeBot = keypoints[374][1];

    const leftEyeOpenness = leftEyeBot - leftEyeTop;
    const rightEyeOpenness = rightEyeBot - rightEyeTop;
    const minEyeOpenness = faceHeight * 0.015;

    if (leftEyeOpenness > minEyeOpenness && rightEyeOpenness > minEyeOpenness) {
      checks.push({
        ok: true,
        text: "Gözler açık ve görünür",
      });
    } else {
      let eyeDetails = [];
      if (leftEyeOpenness <= minEyeOpenness) eyeDetails.push("sol göz");
      if (rightEyeOpenness <= minEyeOpenness) eyeDetails.push("sağ göz");

      checks.push({
        ok: false,
        text: `Gözler yeterince açık değil (${eyeDetails.join(", ")})`,
      });
      passed = false;
    }

    // Neutral expression with lip measurement
    const upperLip = keypoints[13][1];
    const lowerLip = keypoints[14][1];
    const lipDistance = lowerLip - upperLip;
    const maxLipDistance = faceHeight * 0.06;

    if (lipDistance < maxLipDistance) {
      checks.push({
        ok: true,
        text: "Nötr ifade (ağız kapalı)",
      });
    } else {
      const lipOpenPercentage = Math.round((lipDistance / faceHeight) * 100);
      checks.push({
        ok: false,
        text: `Ağız çok açık (%${lipOpenPercentage} yüz yüksekliği) - nötr ifade gerekli`,
      });
      passed = false;
    }
  } else if (predictions.length === 0) {
    checks.push({
      ok: false,
      text: "Yüz tespit edilemedi - daha iyi aydınlatma deneyin",
    });
    passed = false;
  } else {
    checks.push({
      ok: false,
      text: `${predictions.length} yüz tespit edildi - sadece tek kişi olmalı`,
    });
    passed = false;
  }

  // Background analysis with anomaly detection
  if (backgroundAnalysis && backgroundAnalysis.avgColor !== null) {
    const hasAnomalies =
      backgroundAnalysis.anomalies && backgroundAnalysis.anomalies.length > 0;

    if (backgroundAnalysis.isPlain) {
      checks.push({
        ok: true,
        text: `🤖 AI: Arka plan uygun (güven: %${backgroundAnalysis.confidence}, parlaklık: ${backgroundAnalysis.brightness}/255)`,
      });
    } else {
      let bgAdvice = "";
      if (backgroundAnalysis.brightness < 200) {
        bgAdvice = " - daha açık arka plan kullanın";
      }
      if (backgroundAnalysis.variance > 800) {
        bgAdvice += " - daha düz/tek renk arka plan kullanın";
      }
      if (hasAnomalies) {
        bgAdvice += ` - ${backgroundAnalysis.anomalies.length} farklı renk bölgesi temizleyin`;
      }

      checks.push({
        ok: false,
        text: `🤖 AI: Arka plan uygun değil - ${backgroundAnalysis.reason}${bgAdvice}`,
      });
      passed = false;
    }

    // Add anomaly warning if detected
    if (hasAnomalies) {
      const significantAnomalies = backgroundAnalysis.anomalies.filter(
        (a) => a.pixelCount > 10
      );
      if (significantAnomalies.length > 0) {
        checks.push({
          ok: false,
          text: `⚠️ Arka planda ${significantAnomalies.length} farklı renk bölgesi: kalem izi, gölge veya nesne olabilir`,
        });
        passed = false;
      }
    }
  } else {
    checks.push({
      ok: true,
      text: "⚠️ Manuel kontrol: Arka plan düz ve açık renk mi? (AI analizi başarısız)",
    });
  }

  // Manual checks
  checks.push({
    ok: true,
    text: "⚠️ Manuel: Aydınlatma uniform, gölge yok mu?",
  });
  checks.push({
    ok: true,
    text: "⚠️ Manuel: Fotoğraf net ve keskin odakta mı?",
  });
  checks.push({ ok: true, text: "⚠️ Manuel: Cilt tonu doğal mı?" });
  checks.push({
    ok: true,
    text: "⚠️ Manuel: Gözlük varsa çerçeve ince ve şeffaf mı?",
  });

  // Display results
  resultDiv.innerHTML = `
    <div class="section-title">
      <svg viewBox="0 0 24 24">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
      </svg>
      <span>ICAO Analiz Sonuçları</span>
    </div>
    <ul class="checklist">
      ${checks
        .map(
          (c) => `
        <li>
          <div class="check-icon ${c.ok ? "check-yes" : "check-no"}">
            ${c.ok ? "✓" : "✗"}
          </div>
          <div>${c.text}</div>
        </li>
      `
        )
        .join("")}
    </ul>
    <div class="${passed ? "result-success" : "result-fail"}">
      <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
        ${
          passed
            ? '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>'
            : '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>'
        }
      </svg>
      <span>${
        passed ? "Kontroller başarılı!" : "ICAO standartlarına uygun değil"
      }</span>
    </div>
    ${
      backgroundAnalysis && backgroundAnalysis.avgColor
        ? `
      <div class="note">
        <b>🤖 AI Arka Plan Detayları:</b><br>
        <div style="display:flex;align-items:center;gap:0.5rem;margin:0.5rem 0;">
          <div style="width:20px;height:20px;background:rgb(${
            backgroundAnalysis.avgColor.r
          },${backgroundAnalysis.avgColor.g},${
            backgroundAnalysis.avgColor.b
          });border:1px solid #ccc;border-radius:3px;"></div>
          <span>Ana renk: RGB(${backgroundAnalysis.avgColor.r}, ${
            backgroundAnalysis.avgColor.g
          }, ${backgroundAnalysis.avgColor.b})</span>
        </div>
        Parlaklık: ${backgroundAnalysis.brightness}/255 ${
            backgroundAnalysis.brightness > 200 ? "✅" : "❌"
          }<br>
        Renk varyansı: ${backgroundAnalysis.variance} ${
            backgroundAnalysis.variance < 800 ? "✅" : "❌"
          }<br>
        Güven skoru: ${backgroundAnalysis.confidence}%<br>
        Arka plan piksel: ${
          backgroundAnalysis.backgroundPixelCount
        }, Kişi piksel: ${backgroundAnalysis.personPixelCount}
      </div>
    `
        : ""
    }
  `;
}

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
    displayResults(
      predictions,
      inputImage.naturalWidth,
      inputImage.naturalHeight,
      backgroundAnalysis,
      faceCanvas,
      anomalyCanvas
    );
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
      loadingDiv.innerHTML = `
        <div class="loader"></div>
        <div>AI modelleri yükleniyor...</div>
        <div class="progress-container">
          <div class="progress-bar" id="progressBar"></div>
        </div>
      `;
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
        loadingDiv.innerHTML = `
          <svg viewBox="0 0 24 24" width="48" height="48" fill="#38b000">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          <div style="font-weight:600;color:#38b000">🤖 AI modelleri hazır!</div>
          <div style="font-size:0.9em;opacity:0.8">Yüz tanıma + Arka plan analizi aktif</div>
        `;
        setTimeout(() => {
          loadingDiv.style.display = "none";
        }, 2000);
      }
    }, 500);

    if (fileInput) fileInput.disabled = false;
  } catch (e) {
    console.error("Model loading error:", e);
    modelLoaded = false;
    if (loadingDiv) {
      loadingDiv.innerHTML = `
        <svg viewBox="0 0 24 24" width="48" height="48" fill="#e5383b">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <div style="color:#e5383b;font-weight:500;">AI modelleri yüklenemedi</div>
        <div style="font-size:0.9em;opacity:0.8">Lütfen sayfayı yenileyip tekrar deneyin</div>
        <div style="font-size:0.8em;opacity:0.6;margin-top:0.5rem;">Hata: ${e.message}</div>
      `;
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

// Event listeners
if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    resetUI();
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (lastLoadedPhotoURL) URL.revokeObjectURL(lastLoadedPhotoURL);
    lastLoadedPhotoURL = URL.createObjectURL(file);

    if (inputImage) {
      inputImage.src = lastLoadedPhotoURL;
      inputImage.onload = () => {
        if (!analyzing && modelLoaded) {
          processImage();
        }
      };
      inputImage.style.display = "block";
    }
  });
}

if (choosePhotoBtn) {
  choosePhotoBtn.onclick = () => {
    if (fileInput) fileInput.click();
  };
}

// Export function for HTML to call
window.initializeApp = function () {
  console.log("Initializing app...");
  loadModel();
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
