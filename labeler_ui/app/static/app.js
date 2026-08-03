const $ = (id) => document.getElementById(id);

let uploadId = null;
let sessionId = null;
let chunkIndex = 0;
let sampleFrame = 0;
let clickMode = 1;
let pendingPoints = [];
let maskPreview = null;
let chunksMeta = [];
let videoMeta = null;
let isMockMode = true;
let currentStep = 1;
let trackStartTime = 0;
let trackFrameTimes = [];

const FLOW = [
  {
    id: 1,
    key: "upload",
    title: "Upload video",
    desc: "Send your video (MP4, AVI, etc.) to the server so we can read frame count and split into chunks.",
    next: "Choose a file and click Upload.",
    eta: () => "~1–5 min for large files over SSH",
  },
  {
    id: 2,
    key: "prepare",
    title: "Prepare chunk",
    desc: "Extract JPEG frames for one chunk (~1000 frames). Repeat for each chunk.",
    next: "Select a chunk and click Prepare.",
    eta: (ctx) => {
      const frames = ctx.chunkFrames || 1000;
      const sec = Math.max(20, Math.round(frames * 0.03));
      return `~${formatDuration(sec)}`;
    },
  },
  {
    id: 3,
    key: "label",
    title: "Click on object",
    desc: "Place green points on what you want to track. Red points exclude areas.",
    next: "Click the image, then go to Preview & Track.",
    eta: () => "~1 min (your clicks)",
  },
  {
    id: 4,
    key: "track",
    title: "Preview & track",
    desc: "Preview the mask on one frame, then SAM3 propagates forward and backward across the chunk. Click near the middle of the chunk for best coverage.",
    next: "Preview mask, then Track entire chunk.",
    eta: (ctx) => {
      const n = ctx.chunkFrames || 1000;
      if (ctx.mock) return n > 500 ? "~30 sec" : "~10 sec";
      const sec = Math.round(n * 0.4);
      return sec > 120 ? `~${formatDuration(sec)} (GPU)` : `~${sec} sec (GPU)`;
    },
  },
  {
    id: 5,
    key: "export",
    title: "Export video",
    desc: "Merge all masks into one annotated MP4 you can download.",
    next: "Click Generate when all chunks are done.",
    eta: (ctx) => {
      const n = ctx.totalFrames || 1000;
      const sec = Math.max(30, Math.round(n * 0.05));
      return `~${formatDuration(sec)}`;
    },
  },
];

function formatDuration(sec) {
  if (sec < 60) return `${sec} sec`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s ? `${m} min ${s} sec` : `${m} min`;
}

function formatElapsed(ms) {
  return formatDuration(Math.max(1, Math.round(ms / 1000)));
}

function ctx() {
  const chunk = chunksMeta.find((c) => c.chunk_index === chunkIndex);
  const chunkFrames = chunk ? chunk.save_end - chunk.save_start + 1 : 1000;
  return {
    mock: isMockMode,
    totalFrames: videoMeta?.frame_count || 1000,
    chunkFrames,
    chunkCount: chunksMeta.length || 1,
    chunksDone: chunksMeta.filter((c) => c.status === "done").length,
  };
}

function renderFlowList() {
  const ol = $("flow-steps");
  ol.innerHTML = "";
  const c = ctx();
  FLOW.forEach((step) => {
    const li = document.createElement("li");
    li.className = "flow-item";
    if (step.id === currentStep) li.classList.add("active");
    if (step.id < currentStep) li.classList.add("done");
    if (step.id === 5 && c.chunksDone === c.chunkCount && c.chunkCount > 0) li.classList.add("done");
    li.innerHTML = `
      <div class="flow-item-head">
        <span class="flow-num">${step.id}</span>
        <span class="flow-name">${step.title}</span>
        <span class="flow-eta">${step.eta(c)}</span>
      </div>
      <div class="flow-bar"><div class="flow-bar-fill" style="width:${stepProgress(step.id)}%"></div></div>`;
    li.addEventListener("click", () => goToStep(step.id, false));
    ol.appendChild(li);
  });
}

function stepProgress(stepId) {
  if (stepId < currentStep) return 100;
  if (stepId > currentStep) return 0;
  return 50;
}

function updateOverallProgress() {
  const c = ctx();
  let pct = 0;
  if (!uploadId) pct = 0;
  else if (c.chunksDone === c.chunkCount && c.chunkCount > 0) pct = 100;
  else {
    const perChunk = 100 / Math.max(1, c.chunkCount);
    pct = Math.min(99, c.chunksDone * perChunk + (currentStep > 1 ? perChunk * 0.3 : 0));
  }
  $("overall-fill").style.width = pct + "%";
  $("overall-pct").textContent = Math.round(pct) + "%";
}

function goToStep(n, updatePanels = true) {
  currentStep = n;
  const step = FLOW[n - 1];
  $("hero-step-num").textContent = `Step ${n} of 5`;
  $("hero-title").textContent = step.title;
  $("hero-desc").textContent = step.desc;
  $("hero-eta").textContent = step.eta(ctx());
  $("hero-next").textContent = step.next ? "👉 " + step.next : "";
  renderFlowList();
  updateOverallProgress();
  if (!updatePanels) return;
  document.querySelectorAll(".panel[data-step]").forEach((p) => {
    const s = Number(p.dataset.step);
    let show = s === n;
    if (n === 4 && (s === 3 || s === 4)) show = true;
    if (n === 3 && s === 3) show = true;
    p.classList.toggle("hidden", !show);
  });
}

function showHeroProgress(show, pct, text) {
  $("hero-progress-wrap").classList.toggle("hidden", !show);
  if (show) {
    $("hero-fill").style.width = (pct || 0) + "%";
    $("hero-progress-text").textContent = text || "";
  }
}

async function checkHealth() {
  const el = $("status");
  try {
    const r = await fetch("/health");
    const j = await r.json();
    if (j.status === "ok") {
      const sam3 = j.sam3_video?.sam3 || j.sam3_video || {};
      isMockMode = !!sam3.mock;
      if (sam3.mock) {
        el.textContent = "Mock mode · fast test";
        el.className = "status-pill mock";
      } else if (sam3.ready) {
        el.textContent = "Real SAM3 · GPU";
        el.className = "status-pill ok";
      } else {
        el.textContent = "SAM3 will load on first track";
        el.className = "status-pill mock";
      }
    } else {
      el.textContent = "Backend degraded";
      el.className = "status-pill err";
    }
  } catch {
    el.textContent = "Offline";
    el.className = "status-pill err";
  }
  renderFlowList();
}

function log(msg) {
  $("log").textContent += msg + "\n";
  $("log").scrollTop = $("log").scrollHeight;
}

function renderPointsTable() {
  const tbody = $("points-body");
  tbody.innerHTML = "";
  pendingPoints.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td><td>${p.x}</td><td>${p.y}</td>
      <td class="${p.label === 1 ? "type-pos" : "type-neg"}">${p.label === 1 ? "+" : "−"}</td>
      <td><button type="button" class="small del-point" data-idx="${i}">✕</button></td>`;
    tbody.appendChild(tr);
  });
  $("points-empty").classList.toggle("hidden", pendingPoints.length > 0);
  $("points-table").classList.toggle("hidden", pendingPoints.length === 0);
}

function deletePoint(idx) {
  pendingPoints.splice(idx, 1);
  maskPreview = null;
  renderPointsTable();
  drawOverlay();
  drawMaskPreview();
}

function clearPoints() {
  pendingPoints = [];
  maskPreview = null;
  renderPointsTable();
  drawOverlay();
  drawMaskPreview();
}

function undoPoint() {
  if (!pendingPoints.length) return;
  pendingPoints.pop();
  maskPreview = null;
  renderPointsTable();
  drawOverlay();
  drawMaskPreview();
}

async function refreshChunkList() {
  if (!uploadId) return;
  const status = await (await fetch(`/api/uploads/${uploadId}/status`)).json();
  chunksMeta = status.chunks || [];
  const ul = $("chunk-list");
  ul.innerHTML = "";
  const sel = $("chunk-select");
  sel.innerHTML = "";

  chunksMeta.forEach((c) => {
    const nFrames = c.save_end - c.save_start + 1;
    const opt = document.createElement("option");
    opt.value = c.chunk_index;
    opt.textContent = `Chunk ${c.chunk_index} · frames ${c.save_start}–${c.save_end} (${nFrames})`;
    if (c.chunk_index === chunkIndex) opt.selected = true;
    sel.appendChild(opt);

    const li = document.createElement("li");
    li.dataset.chunk = c.chunk_index;
    if (c.chunk_index === chunkIndex) li.classList.add("selected");
    const badge =
      c.status === "done"
        ? `<span class="badge done">✓ ${c.mask_count} masks</span>`
        : c.status === "partial"
          ? `<span class="badge pending">partial ${c.mask_count} masks</span>`
          : `<span class="badge pending">waiting</span>`;
    li.innerHTML = `<strong>Chunk ${c.chunk_index}</strong> ${nFrames} frames ${badge}`;
    li.addEventListener("click", () => {
      chunkIndex = c.chunk_index;
      sel.value = String(chunkIndex);
      refreshChunkList();
    });
    ul.appendChild(li);
  });

  const c = ctx();
  if (c.chunksDone === c.chunkCount && c.chunkCount > 0) {
    goToStep(5);
    $("panel-export").classList.remove("hidden");
  }
  renderFlowList();
  updateOverallProgress();
}

// Upload
$("browse-btn").addEventListener("click", () => $("video-file").click());
$("video-file").addEventListener("change", () => {
  const f = $("video-file").files[0];
  $("upload-btn").disabled = !f;
  if (f) {
    $("upload-info").textContent = `Selected: ${f.name} (${(f.size / 1e6).toFixed(1)} MB)`;
    $("drop-zone").classList.add("has-file");
  }
});

$("drop-zone").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop-zone").classList.add("drag");
});
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("drag"));
$("drop-zone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop-zone").classList.remove("drag");
  if (e.dataTransfer.files[0]) {
    $("video-file").files = e.dataTransfer.files;
    $("video-file").dispatchEvent(new Event("change"));
  }
});

$("upload-btn").addEventListener("click", async () => {
  const file = $("video-file").files[0];
  if (!file) return;
  $("upload-btn").disabled = true;
  const t0 = Date.now();
  showHeroProgress(true, 5, "Starting upload…");
  try {
    videoMeta = await uploadWithProgress(file);
  } catch (err) {
    $("upload-info").textContent = String(err.message || err);
    showHeroProgress(false);
    $("upload-btn").disabled = false;
    log(`Upload failed: ${err.message || err}`);
    return;
  }
  uploadId = videoMeta.upload_id;
  const elapsed = formatElapsed(Date.now() - t0);
  const nChunks = Math.ceil(videoMeta.frame_count / (videoMeta.chunk_size || 1000));
  $("upload-info").textContent =
    `✓ ${videoMeta.original_filename} · ${videoMeta.frame_count} frames · ${nChunks} chunk(s) · took ${elapsed}`;
  showHeroProgress(true, 100, `Done in ${elapsed}`);
  log(`Uploaded ${videoMeta.original_filename}`);
  await refreshChunkList();
  goToStep(2);
  $("panel-prepare").classList.remove("hidden");
});

function uploadWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.max(5, Math.min(92, Math.round((e.loaded / e.total) * 92)));
        const mb = (e.loaded / 1e6).toFixed(0);
        const totalMb = (e.total / 1e6).toFixed(0);
        showHeroProgress(true, pct, `Uploading… ${mb} / ${totalMb} MB`);
      } else {
        showHeroProgress(true, 40, `Uploading… ${(e.loaded / 1e6).toFixed(0)} MB sent`);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        showHeroProgress(true, 96, "Probing video…");
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error("Invalid server response"));
        }
        return;
      }
      let detail = xhr.responseText || xhr.statusText;
      try {
        const body = JSON.parse(xhr.responseText);
        detail = body.detail || detail;
      } catch (_) {
        /* plain text error */
      }
      reject(new Error(detail));
    };
    xhr.onerror = () => reject(new Error("Network error — check SSH port forward to :8080"));
    xhr.ontimeout = () => reject(new Error("Upload timed out — try a smaller file or copy to the cluster first"));
    xhr.timeout = 0;
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

$("chunk-select").addEventListener("change", () => {
  chunkIndex = Number($("chunk-select").value || 0);
  refreshChunkList();
});

$("prepare-btn").addEventListener("click", async () => {
  if (!uploadId) return;
  chunkIndex = Number($("chunk-select").value || 0);
  $("prepare-btn").disabled = true;
  const t0 = Date.now();
  showHeroProgress(true, 5, "Extracting frames with ffmpeg…");
  pendingPoints = [];
  maskPreview = null;
  renderPointsTable();
  const r = await fetch(`/api/uploads/${uploadId}/chunks/${chunkIndex}/prepare`, { method: "POST" });
  $("prepare-btn").disabled = false;
  if (!r.ok) {
    $("chunk-info").textContent = await r.text();
    showHeroProgress(false);
    return;
  }
  const data = await r.json();
  sessionId = data.session_id;
  sampleFrame = data.chunk.sample_frame;
  const elapsed = formatElapsed(Date.now() - t0);
  const nFrames = data.chunk.save_end - data.chunk.save_start + 1;
  $("chunk-info").textContent = `✓ Chunk ${chunkIndex} ready in ${elapsed}`;
  $("chunk-summary").classList.remove("hidden");
  $("chunk-summary").innerHTML =
    `<strong>Chunk ${chunkIndex}</strong> · ${nFrames} frames to track · sample frame #${sampleFrame}`;
  showHeroProgress(true, 100, `Extracted ${nFrames} frames in ${elapsed}`);
  log(`Prepared chunk ${chunkIndex}`);
  goToStep(3);
  $("panel-label").classList.remove("hidden");
  $("panel-track").classList.remove("hidden");
  loadFrame();
});

$("resample-btn").addEventListener("click", async () => {
  if (!uploadId) return;
  pendingPoints = [];
  maskPreview = null;
  renderPointsTable();
  const status = await (await fetch(`/api/uploads/${uploadId}/status`)).json();
  const c = (status.chunks || []).find((x) => x.chunk_index === chunkIndex);
  if (!c) return;
  const span = Math.max(1, c.save_end - c.save_start + 1);
  sampleFrame = c.save_start + Math.floor(Math.random() * span);
  loadFrame();
});

function loadFrame() {
  maskPreview = null;
  drawMaskPreview();
  $("frame-label").textContent = `Frame #${sampleFrame} · chunk ${chunkIndex}`;
  const img = $("frame-img");
  img.onload = resizeCanvas;
  img.src = `/api/frame/${uploadId}/${chunkIndex}/${sampleFrame}?t=${Date.now()}`;
}

function resizeCanvas() {
  const img = $("frame-img");
  for (const id of ["overlay", "mask-layer"]) {
    const c = $(id);
    c.width = img.clientWidth;
    c.height = img.clientHeight;
  }
  drawOverlay();
  drawMaskPreview();
}

function drawOverlay() {
  const canvas = $("overlay");
  const ctx2 = canvas.getContext("2d");
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  const img = $("frame-img");
  const sx = canvas.width / (img.naturalWidth || 1);
  const sy = canvas.height / (img.naturalHeight || 1);
  for (const p of pendingPoints) {
    ctx2.beginPath();
    ctx2.arc(p.x * sx, p.y * sy, 7, 0, Math.PI * 2);
    ctx2.fillStyle = p.label === 1 ? "#22c55e" : "#ef4444";
    ctx2.fill();
    ctx2.strokeStyle = "#fff";
    ctx2.lineWidth = 1.5;
    ctx2.stroke();
  }
}

function drawMaskPreview() {
  const canvas = $("mask-layer");
  const ctx2 = canvas.getContext("2d");
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  if (!maskPreview) return;
  const img = new Image();
  img.onload = () => {
    ctx2.globalAlpha = 0.45;
    ctx2.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx2.globalAlpha = 1;
  };
  img.src = "data:image/png;base64," + maskPreview;
}

$("pos-mode").addEventListener("click", () => {
  clickMode = 1;
  $("pos-mode").classList.add("active");
  $("neg-mode").classList.remove("active");
});
$("neg-mode").addEventListener("click", () => {
  clickMode = 0;
  $("neg-mode").classList.add("active");
  $("pos-mode").classList.remove("active");
});

$("overlay").addEventListener("click", (e) => {
  const canvas = $("overlay");
  const img = $("frame-img");
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * (img.naturalWidth / canvas.width));
  const y = Math.round((e.clientY - rect.top) * (img.naturalHeight / canvas.height));
  pendingPoints.push({ x, y, label: clickMode });
  maskPreview = null;
  renderPointsTable();
  drawOverlay();
  drawMaskPreview();
  if (pendingPoints.length >= 1 && currentStep === 3) {
    goToStep(4);
  }
});

$("points-body").addEventListener("click", (e) => {
  if (e.target.classList.contains("del-point")) deletePoint(Number(e.target.dataset.idx));
});
$("undo-btn").addEventListener("click", undoPoint);
$("clear-btn").addEventListener("click", clearPoints);

$("refine-btn").addEventListener("click", async () => {
  if (!sessionId || !pendingPoints.length) return alert("Add at least one click first");
  $("refine-btn").disabled = true;
  goToStep(4);
  const t0 = Date.now();
  showHeroProgress(true, 20, "Running SAM3 on this frame…");
  await syncPoints();
  const r = await fetch("/api/refine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, frame_idx: sampleFrame }),
  });
  $("refine-btn").disabled = false;
  if (!r.ok) {
    let detail = await r.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch (_) {
      /* plain text */
    }
    log("Preview failed: " + detail);
    showHeroProgress(false);
    return;
  }
  const j = await r.json();
  maskPreview = j.mask_preview_b64 || null;
  drawMaskPreview();
  const elapsed = formatElapsed(Date.now() - t0);
  showHeroProgress(true, 100, `Preview ready in ${elapsed} · confidence ${(j.confidence || 0).toFixed(2)}`);
  log(`Preview frame ${sampleFrame} (${elapsed})`);
});

async function syncPoints() {
  await fetch("/api/points", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      frame_idx: sampleFrame,
      points: pendingPoints,
      replace: true,
    }),
  });
}

function estimateTrackRemaining(done, total, elapsedMs) {
  if (done < 2 || !elapsedMs) return null;
  const perFrame = elapsedMs / done;
  const left = (total - done) * perFrame;
  return formatDuration(Math.max(1, Math.round(left / 1000)));
}

$("track-btn").addEventListener("click", async () => {
  if (!sessionId || !pendingPoints.length) return alert("Add at least one click first");
  $("track-btn").disabled = true;
  $("track-progress").classList.remove("hidden");
  $("progress-fill").style.width = "0%";
  trackStartTime = Date.now();
  trackFrameTimes = [];
  goToStep(4);
  showHeroProgress(true, 0, "Starting track job…");
  await syncPoints();
  const r = await fetch("/api/propagate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!r.ok) {
    log("Track failed: " + (await r.text()));
    $("track-btn").disabled = false;
    showHeroProgress(false);
    return;
  }
  const { job_id } = await r.json();
  log(`Tracking chunk ${chunkIndex} (${FLOW[3].eta(ctx())} estimated)`);
  const res = await fetch(`/api/jobs/${job_id}/stream`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let total = 0;
  let progress = 0;
  let trackSucceeded = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const block of parts) {
      for (const line of block.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "start") total = ev.total || 0;
        if (ev.type === "frame") {
          progress = ev.progress || progress + 1;
          const pct = total ? Math.round((progress / total) * 100) : 0;
          $("progress-fill").style.width = pct + "%";
          $("hero-fill").style.width = pct + "%";
          const elapsed = Date.now() - trackStartTime;
          const remaining = estimateTrackRemaining(progress, total, elapsed);
          const txt = `Frame ${ev.frame_idx} · ${progress}/${total} (${pct}%)`;
          $("progress-text").textContent = txt;
          $("hero-progress-text").textContent = txt;
          $("progress-eta").textContent = remaining
            ? `~${remaining} remaining · elapsed ${formatElapsed(elapsed)}`
            : `Elapsed ${formatElapsed(elapsed)}`;
        }
        if (ev.type === "done") {
          trackSucceeded = ev.frames_saved >= (ev.expected || total);
          const elapsed = formatElapsed(Date.now() - trackStartTime);
          const msg = trackSucceeded
            ? `Chunk ${chunkIndex} done · ${ev.frames_saved}/${ev.expected || total} masks in ${elapsed}`
            : `Chunk ${chunkIndex} partial · only ${ev.frames_saved}/${ev.expected || total} masks — click earlier in chunk and re-track`;
          log(msg);
          $("progress-text").textContent = trackSucceeded
            ? `✓ Done · ${ev.frames_saved} masks in ${elapsed}`
            : `⚠ Partial · ${ev.frames_saved}/${ev.expected || total} masks`;
          $("progress-eta").textContent = "";
          showHeroProgress(true, trackSucceeded ? 100 : 60, msg);
          if (trackSucceeded) sessionId = null;
        }
        if (ev.type === "heartbeat") {
          $("progress-text").textContent = ev.message || "Propagating…";
        }
        if (ev.type === "error") {
          log("Track error: " + ev.message);
          $("progress-text").textContent = "Track failed: " + (ev.message || "see Log tab");
          showHeroProgress(false);
        }
      }
    }
  }
  $("track-btn").disabled = false;
  await refreshChunkList();
  if (!trackSucceeded) {
    goToStep(4);
    $("panel-prepare").classList.add("hidden");
    $("panel-label").classList.remove("hidden");
    $("panel-track").classList.remove("hidden");
    return;
  }
  const next = chunksMeta.find((c) => c.status !== "done");
  if (next) {
    chunkIndex = next.chunk_index;
    $("chunk-select").value = String(chunkIndex);
    $("chunk-info").textContent = `Chunk ${chunkIndex - 1} done — prepare chunk ${chunkIndex} next`;
    goToStep(2);
    $("panel-prepare").classList.remove("hidden");
    $("panel-label").classList.add("hidden");
    $("panel-track").classList.add("hidden");
  } else {
    goToStep(5);
    $("panel-export").classList.remove("hidden");
  }
});

$("export-btn").addEventListener("click", async () => {
  if (!uploadId) return;
  $("export-btn").disabled = true;
  $("export-progress").classList.remove("hidden");
  $("export-done").classList.add("hidden");
  const t0 = Date.now();
  const est = FLOW[4].eta(ctx());
  showHeroProgress(true, 30, `Encoding video (${est} estimated)…`);
  $("export-info").textContent = "Extracting frames and applying masks…";
  const r = await fetch(`/api/uploads/${uploadId}/export`, { method: "POST" });
  $("export-btn").disabled = false;
  $("export-progress").classList.add("hidden");
  if (!r.ok) {
    $("export-done").classList.remove("hidden");
    $("export-done").textContent = await r.text();
    showHeroProgress(false);
    return;
  }
  const elapsed = formatElapsed(Date.now() - t0);
  $("export-done").classList.remove("hidden");
  $("export-done").textContent = `✓ Video ready in ${elapsed}`;
  showHeroProgress(true, 100, `Export complete in ${elapsed}`);
  const link = $("download-link");
  link.href = `/api/uploads/${uploadId}/export/download`;
  link.classList.remove("hidden");
  log("Export complete");
  updateOverallProgress();
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

checkHealth().then(() => {
  renderFlowList();
  goToStep(1);
});
window.addEventListener("resize", resizeCanvas);
