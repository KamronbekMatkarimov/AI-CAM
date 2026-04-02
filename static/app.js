function normalizeImagePath(path) {
  if (!path) return "";
  const normalized = path.replace(/\\/g, "/");
  const idx = normalized.indexOf("/outputs/");
  if (idx >= 0) return normalized.slice(idx);
  if (normalized.startsWith("outputs/")) return "/" + normalized;
  return "";
}

let nextRunEpochMs = null;
let uploadedPeopleCount = 0;

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds) || seconds < 0) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function updateCountdown() {
  const el = document.getElementById("nextRunCountdown");
  if (!el) return;
  if (!nextRunEpochMs) {
    el.textContent = "-";
    return;
  }
  const secs = Math.max(0, Math.floor((nextRunEpochMs - Date.now()) / 1000));
  el.textContent = formatDuration(secs);
}

function updateCameraCard(cameraId, data) {
  const countEl = document.getElementById(`${cameraId}Count`);
  const tsEl = document.getElementById(`${cameraId}Ts`);
  const imgEl = document.getElementById(`${cameraId}Img`);
  const emptyEl = document.getElementById(`${cameraId}Empty`);

  const count = data?.people_count ?? 0;
  countEl.textContent = `${count} people`;
  tsEl.textContent = data?.timestamp || "-";

  const imgPath = normalizeImagePath(data?.processed_image_path);
  if (imgPath) {
    imgEl.src = `${imgPath}?t=${Date.now()}`;
    imgEl.style.display = "block";
    emptyEl.style.display = "none";
  } else {
    imgEl.style.display = "none";
    emptyEl.style.display = "grid";
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const payload = await res.json();
    const baseTotal = payload.total_people ?? 0;
    document.getElementById("totalPeople").textContent = baseTotal + uploadedPeopleCount;
    if (payload.scheduler?.next_run_ts) {
      nextRunEpochMs = new Date(payload.scheduler.next_run_ts).getTime();
    }
    updateCountdown();

    Object.entries(payload.cameras || {}).forEach(([cameraId, data]) => {
      updateCameraCard(cameraId, data);
    });
  } catch (err) {
    console.error("Failed to refresh dashboard status", err);
  }
}

async function uploadAndDetectImage(evt) {
  evt.preventDefault();
  const form = document.getElementById("uploadForm");
  const resultEl = document.getElementById("uploadResult");
  const previewEl = document.getElementById("uploadPreview");
  const fileInput = document.getElementById("uploadImageInput");

  if (!fileInput.files || fileInput.files.length === 0) {
    resultEl.textContent = "Please choose an image first.";
    return;
  }

  resultEl.textContent = "Processing image...";

  const formData = new FormData(form);
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const payload = await res.json();
    if (!res.ok) {
      resultEl.textContent = payload.error || "Upload failed.";
      return;
    }

    resultEl.textContent = `Detected: ${payload.people_count} people (${payload.camera_id.toUpperCase()} profile)`;
    previewEl.src = `${payload.processed_image_url}?t=${Date.now()}`;
    previewEl.style.display = "block";

    uploadedPeopleCount = payload.people_count ?? 0;
    // Update total immediately to include upload.
    if (payload.total_people_all != null) {
      document.getElementById("totalPeople").textContent = payload.total_people_all;
    }
  } catch (err) {
    console.error(err);
    resultEl.textContent = "Upload request failed.";
  }
}

refreshStatus();
// Refresh UI continuously so images update as soon as new frames are saved.
setInterval(refreshStatus, 3000);
setInterval(updateCountdown, 1000);

const uploadForm = document.getElementById("uploadForm");
if (uploadForm) {
  uploadForm.addEventListener("submit", uploadAndDetectImage);
}
