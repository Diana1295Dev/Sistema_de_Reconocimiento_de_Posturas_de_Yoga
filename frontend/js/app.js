// Sistema de Reconocimiento de Posturas de Yoga — frontend estático (sin build tooling).
// Habla con la API de backend/ vía fetch + FormData.

// Ajustar esta URL si el backend se despliega en un host distinto al de desarrollo local.
const API_BASE_URL = "http://localhost:8000";

const POSE_LABELS_ES = {
  downdog: "Perro boca abajo",
  goddess: "Diosa",
  plank: "Plancha",
  tree: "Árbol",
  warrior2: "Guerrero II",
};

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);

const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const errorMessage = document.getElementById("error-message");

const resultEmpty = document.getElementById("result-empty");
const resultLoading = document.getElementById("result-loading");
const resultContent = document.getElementById("result-content");
const predictedPoseName = document.getElementById("predicted-pose-name");
const confidenceValue = document.getElementById("confidence-value");
const probabilityBars = document.getElementById("probability-bars");

const apiBaseUrlLabel = document.getElementById("api-base-url-label");
const healthLink = document.getElementById("health-link");

let selectedFile = null;

apiBaseUrlLabel.textContent = API_BASE_URL;
healthLink.href = `${API_BASE_URL}/health`;

function poseLabel(poseKey) {
  return POSE_LABELS_ES[poseKey] || poseKey;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function validateFile(file) {
  if (!ALLOWED_TYPES.has(file.type)) {
    return "Formato no soportado. Usa una imagen JPG, PNG, WEBP o BMP.";
  }
  if (file.size > MAX_FILE_BYTES) {
    return "El archivo supera el tamaño máximo permitido (10 MB).";
  }
  return null;
}

function setSelectedFile(file) {
  clearError();

  const validationError = validateFile(file);
  if (validationError) {
    showError(validationError);
    selectedFile = null;
    analyzeBtn.disabled = true;
    return;
  }

  selectedFile = file;
  analyzeBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (event) => {
    preview.src = event.target.result;
    preview.hidden = false;
    dropZoneText.hidden = true;
  };
  reader.readAsDataURL(file);

  resultEmpty.hidden = false;
  resultContent.hidden = true;
}

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    setSelectedFile(fileInput.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    setSelectedFile(file);
  }
});

function renderProbabilityBars(probabilities, topPose) {
  probabilityBars.innerHTML = "";

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  for (const [pose, probability] of sorted) {
    const li = document.createElement("li");
    li.className = "probability-bar-row" + (pose === topPose ? " top" : "");

    const label = document.createElement("span");
    label.textContent = poseLabel(pose);

    const track = document.createElement("div");
    track.className = "probability-bar-track";
    const fill = document.createElement("div");
    fill.className = "probability-bar-fill";
    fill.style.width = `${Math.round(probability * 100)}%`;
    track.appendChild(fill);

    const percentLabel = document.createElement("span");
    percentLabel.textContent = `${Math.round(probability * 100)}%`;

    li.append(label, track, percentLabel);
    probabilityBars.appendChild(li);
  }
}

async function analyzeImage() {
  if (!selectedFile) {
    return;
  }

  clearError();
  resultEmpty.hidden = true;
  resultContent.hidden = true;
  resultLoading.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      const detail = errorBody && errorBody.detail ? errorBody.detail : `Error ${response.status}`;
      throw new Error(detail);
    }

    const data = await response.json();

    predictedPoseName.textContent = poseLabel(data.pose);
    confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
    renderProbabilityBars(data.probabilities, data.pose);

    resultContent.hidden = false;
  } catch (err) {
    showError(
      err instanceof TypeError
        ? "No se pudo conectar con la API. ¿Está corriendo en " + API_BASE_URL + "?"
        : err.message
    );
    resultEmpty.hidden = false;
  } finally {
    resultLoading.hidden = true;
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyzeImage);
