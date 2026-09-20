(() => {
  const API_BASE = window.MAGICUT_API_BASE || "";

  const els = {
    videoInput: document.getElementById("video-input"),
    uploadStatus: document.getElementById("upload-status"),
    stepSelect: document.getElementById("step-select"),
    stepProgress: document.getElementById("step-progress"),
    stepResult: document.getElementById("step-result"),
    frameIdx: document.getElementById("frame-idx"),
    loadFrame: document.getElementById("load-frame"),
    keyframe: document.getElementById("keyframe"),
    overlay: document.getElementById("overlay"),
    pointStatus: document.getElementById("point-status"),
    clearPoints: document.getElementById("clear-points"),
    runJob: document.getElementById("run-job"),
    stageText: document.getElementById("stage-text"),
    barFill: document.getElementById("bar-fill"),
    progressStatus: document.getElementById("progress-status"),
    resultVideo: document.getElementById("result-video"),
    downloadLink: document.getElementById("download-link"),
    resultMeta: document.getElementById("result-meta"),
    reset: document.getElementById("reset"),
  };

  const state = {
    jobId: null,
    points: [],
    naturalWidth: 0,
    naturalHeight: 0,
    pollTimer: null,
    objectUrl: null,
  };

  function show(el) {
    el.classList.remove("hidden");
  }

  function hide(el) {
    el.classList.add("hidden");
  }

  async function api(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || JSON.stringify(data);
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    return res;
  }

  function syncCanvasSize() {
    const rect = els.keyframe.getBoundingClientRect();
    els.overlay.width = Math.max(1, Math.floor(rect.width));
    els.overlay.height = Math.max(1, Math.floor(rect.height));
    drawPoints();
  }

  function drawPoints() {
    const ctx = els.overlay.getContext("2d");
    ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    if (!state.naturalWidth || !state.naturalHeight) return;

    const scaleX = els.overlay.width / state.naturalWidth;
    const scaleY = els.overlay.height / state.naturalHeight;

    state.points.forEach((p, i) => {
      const x = p.x * scaleX;
      const y = p.y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = "#e23d2b";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.fillText(String(i + 1), x + 10, y - 8);
    });
  }

  function updateRunEnabled() {
    els.runJob.disabled = !(state.jobId && state.points.length > 0);
    els.pointStatus.textContent = state.points.length
      ? `已选 ${state.points.length} 点：${state.points.map((p) => `(${Math.round(p.x)}, ${Math.round(p.y)})`).join(", ")}`
      : "点击画面选择主角位置";
  }

  function imagePointFromEvent(evt) {
    const rect = els.overlay.getBoundingClientRect();
    const nx = (evt.clientX - rect.left) / rect.width;
    const ny = (evt.clientY - rect.top) / rect.height;
    // object-fit: contain mapping
    const imgRatio = state.naturalWidth / state.naturalHeight;
    const boxRatio = rect.width / rect.height;
    let contentW = rect.width;
    let contentH = rect.height;
    let offsetX = 0;
    let offsetY = 0;
    if (imgRatio > boxRatio) {
      contentH = rect.width / imgRatio;
      offsetY = (rect.height - contentH) / 2;
    } else {
      contentW = rect.height * imgRatio;
      offsetX = (rect.width - contentW) / 2;
    }
    const lx = evt.clientX - rect.left - offsetX;
    const ly = evt.clientY - rect.top - offsetY;
    if (lx < 0 || ly < 0 || lx > contentW || ly > contentH) return null;
    return {
      x: (lx / contentW) * state.naturalWidth,
      y: (ly / contentH) * state.naturalHeight,
    };
  }

  async function loadKeyframe() {
    if (!state.jobId) return;
    const frame = Number(els.frameIdx.value || 0);
    els.pointStatus.textContent = "加载关键帧…";
    const res = await api(`/api/v1/jobs/${state.jobId}/keyframe?frame=${frame}`);
    const blob = await res.blob();
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(blob);
    els.keyframe.onload = () => {
      state.naturalWidth = els.keyframe.naturalWidth;
      state.naturalHeight = els.keyframe.naturalHeight;
      syncCanvasSize();
      updateRunEnabled();
    };
    els.keyframe.src = state.objectUrl;
    const total = res.headers.get("X-Frame-Total");
    els.pointStatus.textContent = `帧 ${res.headers.get("X-Frame-Index")}/${total} · 点击选择主角`;
  }

  async function startFromJob(data, label) {
    state.jobId = data.job_id;
    state.points = [];
    els.uploadStatus.textContent = `${label} · 任务 ${data.job_id}`;
    show(els.stepSelect);
    hide(els.stepProgress);
    hide(els.stepResult);
    await loadKeyframe();
  }

  async function uploadVideo(file) {
    els.uploadStatus.textContent = "上传中…";
    const form = new FormData();
    form.append("video", file, file.name);
    const res = await api("/api/v1/jobs", { method: "POST", body: form });
    const data = await res.json();
    await startFromJob(data, "已上传");
  }

  async function loadDemo() {
    els.uploadStatus.textContent = "准备演示片段…";
    const res = await api("/api/v1/jobs/demo", { method: "POST" });
    const data = await res.json();
    await startFromJob(data, "演示片段就绪");
  }

  async function startProcess() {
    hide(els.stepSelect);
    show(els.stepProgress);
    els.barFill.style.width = "2%";
    els.stageText.textContent = "提交任务…";
    await api(`/api/v1/jobs/${state.jobId}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        keyframe_idx: Number(els.frameIdx.value || 0),
        points: state.points,
        shadow_dilation: 25,
        max_frames: 45,
      }),
    });
    pollStatus();
  }

  function pollStatus() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const res = await api(`/api/v1/jobs/${state.jobId}`);
        const job = await res.json();
        els.barFill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
        els.stageText.textContent = job.stage || job.status;
        els.progressStatus.textContent = `${job.status} · ${Math.round((job.progress || 0) * 100)}%`;
        if (job.status === "completed" && job.result_ready) {
          clearInterval(state.pollTimer);
          await showResult(job);
        } else if (job.status === "failed") {
          clearInterval(state.pollTimer);
          els.stageText.textContent = "失败";
          els.progressStatus.textContent = job.error || "unknown error";
        }
      } catch (err) {
        els.progressStatus.textContent = err.message;
      }
    }, 700);
  }

  async function showResult(job) {
    const res = await api(`/api/v1/jobs/${state.jobId}/result`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    els.resultVideo.src = url;
    els.downloadLink.href = url;
    els.resultMeta.textContent = `mode=${job.mode || "?"} · frames=${job.total_frames ?? "?"} · ${job.elapsed_sec?.toFixed?.(1) ?? "?"}s`;
    hide(els.stepProgress);
    show(els.stepResult);
  }

  els.videoInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadVideo(file);
    } catch (err) {
      els.uploadStatus.textContent = err.message;
    }
  });

  els.demoBtn = document.getElementById("demo-btn");
  els.demoBtn.addEventListener("click", () => {
    loadDemo().catch((err) => {
      els.uploadStatus.textContent = err.message;
    });
  });

  els.loadFrame.addEventListener("click", () => {
    loadKeyframe().catch((err) => {
      els.pointStatus.textContent = err.message;
    });
  });

  els.overlay.addEventListener("click", (evt) => {
    const pt = imagePointFromEvent(evt);
    if (!pt) return;
    state.points.push(pt);
    drawPoints();
    updateRunEnabled();
  });

  els.clearPoints.addEventListener("click", () => {
    state.points = [];
    drawPoints();
    updateRunEnabled();
  });

  els.runJob.addEventListener("click", () => {
    startProcess().catch((err) => {
      els.progressStatus.textContent = err.message;
      show(els.stepProgress);
    });
  });

  els.reset.addEventListener("click", () => {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.jobId = null;
    state.points = [];
    els.videoInput.value = "";
    els.uploadStatus.textContent = "";
    hide(els.stepSelect);
    hide(els.stepProgress);
    hide(els.stepResult);
  });

  window.addEventListener("resize", syncCanvasSize);
})();
