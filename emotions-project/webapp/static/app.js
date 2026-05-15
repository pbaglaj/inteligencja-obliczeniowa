// ---------- Stan globalny ----------
const state = {
  detectors: [],
  emotions: [],
  emotionColors: {
    'Angry':    '#ff3b30',
    'Disgust':  '#1e8a3c',
    'Fear':     '#8b3aff',
    'Happy':    '#ffd400',
    'Neutral':  '#9aa1ad',
    'Sad':      '#3b82f6',
    'Surprise': '#ff8a00',
  },
};

// ---------- Helpers ----------
function $(sel) { return document.querySelector(sel); }
function $all(sel) { return Array.from(document.querySelectorAll(sel)); }

function setStatus(el, msg, kind = '') {
  el.className = 'status' + (kind ? ' ' + kind : '');
  el.textContent = msg;
}

function makeBarsHTML(probabilities) {
  if (!probabilities) return '';
  const html = ['<div class="bars">'];
  for (const name of state.emotions) {
    const p = (probabilities[name] || 0) * 100;
    const color = state.emotionColors[name] || '#cccccc';
    html.push(`<div class="bar-row">
      <span class="name">${name}</span>
      <span class="bar-bg"><span class="bar-fill" style="width:${p.toFixed(1)}%;background:${color}"></span></span>
      <span class="pct">${p.toFixed(1)}%</span>
    </div>`);
  }
  html.push('</div>');
  return html.join('');
}

function buildResultCard({ title, meta, imageSrc, probabilities }) {
  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <h3>${title}</h3>
    <div class="meta">${meta || ''}</div>
    <img src="${imageSrc}">
    ${probabilities ? makeBarsHTML(probabilities) : ''}
  `;
  return card;
}

// ---------- Inicjalizacja: lista detektorow ----------
async function loadDetectors() {
  const res = await fetch('/api/detectors');
  const data = await res.json();
  state.detectors = data.detectors;
  state.emotions = data.emotions;
  const sel = $('#detectorSelect');
  sel.innerHTML = '';
  for (const d of data.detectors) {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = d.label;
    opt.dataset.description = d.description;
    sel.appendChild(opt);
  }
  updateDetectorDesc();
  sel.addEventListener('change', updateDetectorDesc);
}

function updateDetectorDesc() {
  const sel = $('#detectorSelect');
  const opt = sel.options[sel.selectedIndex];
  $('#detectorDesc').textContent = opt ? opt.dataset.description : '';
}

function currentDetector() { return $('#detectorSelect').value; }
function isCompareMode() { return $('#compareMode').checked; }

// ---------- Zakladki ----------
$all('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $all('.tab-btn').forEach(b => b.classList.remove('active'));
    $all('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ---------- ZDJECIE ----------
$('#imageGo').addEventListener('click', async () => {
  const fileInput = $('#imageFile');
  const statusEl = $('#imageStatus');
  const resultsEl = $('#imageResults');
  if (!fileInput.files[0]) {
    setStatus(statusEl, 'Wybierz plik obrazu.', 'error');
    return;
  }
  resultsEl.innerHTML = '';
  setStatus(statusEl, 'Przetwarzanie...');
  const form = new FormData();
  form.append('file', fileInput.files[0]);

  try {
    if (isCompareMode()) {
      const res = await fetch('/api/detect/image_compare', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Blad serwera');
      for (const r of data.results) {
        const card = buildResultCard({
          title: r.label,
          meta: `Twarze: ${r.faces} &middot; ${r.elapsed_ms} ms`,
          imageSrc: r.image,
        });
        resultsEl.appendChild(card);
      }
      setStatus(statusEl, `Por&oacute;wnano 3 detektory.`, 'ok');
    } else {
      form.append('detector', currentDetector());
      const res = await fetch('/api/detect/image', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Blad serwera');
      const largest = data.detections.length
        ? data.detections.reduce((a, b) =>
            (a.bbox[2]*a.bbox[3]) > (b.bbox[2]*b.bbox[3]) ? a : b)
        : null;
      const card = buildResultCard({
        title: state.detectors.find(d => d.id === data.detector)?.label || data.detector,
        meta: `Twarze: ${data.faces} &middot; ${data.elapsed_ms} ms`,
        imageSrc: data.image,
        probabilities: largest ? largest.probabilities : null,
      });
      resultsEl.appendChild(card);
      setStatus(statusEl, `Wykryto ${data.faces} twarz(y).`, 'ok');
    }
  } catch (e) {
    setStatus(statusEl, 'Blad: ' + e.message, 'error');
  }
});

// ---------- FILM ----------
let videoPollTimer = null;
$('#videoGo').addEventListener('click', async () => {
  const fileInput = $('#videoFile');
  const statusEl = $('#videoStatus');
  const progressEl = $('#videoProgress');
  const resultEl = $('#videoResult');

  if (!fileInput.files[0]) {
    setStatus(statusEl, 'Wybierz plik wideo.', 'error');
    return;
  }
  resultEl.innerHTML = '';
  progressEl.style.display = 'block';
  progressEl.value = 0;
  setStatus(statusEl, 'Wysylanie pliku...');

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('detector', currentDetector());

  try {
    const res = await fetch('/api/detect/video', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Blad serwera');
    const jobId = data.job_id;
    setStatus(statusEl, 'Przetwarzanie filmu...');

    if (videoPollTimer) clearInterval(videoPollTimer);
    videoPollTimer = setInterval(async () => {
      try {
        const sres = await fetch(`/api/video/${jobId}/status`);
        const sdata = await sres.json();
        if (sdata.status === 'processing') {
          progressEl.value = sdata.progress || 0;
          let msg = `Przetwarzanie: ${sdata.progress || 0}%`;
          if (sdata.frames_total) msg += ` (${sdata.frames_done}/${sdata.frames_total} klatek)`;
          setStatus(statusEl, msg);
        } else if (sdata.status === 'done') {
          clearInterval(videoPollTimer);
          videoPollTimer = null;
          progressEl.value = 100;
          setStatus(statusEl, 'Gotowe.', 'ok');
          const video = document.createElement('video');
          video.controls = true;
          video.src = `/api/video/${jobId}/result`;
          resultEl.appendChild(video);
          const a = document.createElement('a');
          a.href = `/api/video/${jobId}/result`;
          a.download = `emocje_${jobId}.mp4`;
          a.textContent = 'Pobierz przetworzony film';
          a.style.cssText = 'display:inline-block;margin-top:8px;color:#3b82f6';
          resultEl.appendChild(a);
        } else if (sdata.status === 'error') {
          clearInterval(videoPollTimer);
          videoPollTimer = null;
          setStatus(statusEl, 'Blad: ' + (sdata.error || 'nieznany'), 'error');
        }
      } catch (e) {
        clearInterval(videoPollTimer);
        videoPollTimer = null;
        setStatus(statusEl, 'Blad: ' + e.message, 'error');
      }
    }, 700);
  } catch (e) {
    setStatus(statusEl, 'Blad: ' + e.message, 'error');
  }
});

// ---------- KAMERA ----------
const cam = {
  stream: null,
  active: false,
  busy: false,
  lastFrameTs: 0,
  fpsBuffer: [],
};

$('#camStart').addEventListener('click', async () => {
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video = $('#camVideo');
    video.srcObject = cam.stream;
    video.style.display = 'block';
    cam.active = true;
    $('#camStart').disabled = true;
    $('#camStop').disabled = false;
    requestAnimationFrame(camLoop);
  } catch (e) {
    alert('Nie udalo sie uzyskac dostepu do kamery: ' + e.message);
  }
});

$('#camStop').addEventListener('click', () => {
  cam.active = false;
  if (cam.stream) {
    cam.stream.getTracks().forEach(t => t.stop());
    cam.stream = null;
  }
  $('#camVideo').style.display = 'none';
  $('#camStart').disabled = false;
  $('#camStop').disabled = true;
});

async function camLoop() {
  if (!cam.active) return;
  if (!cam.busy) {
    cam.busy = true;
    try {
      const video = $('#camVideo');
      if (video.videoWidth > 0) {
        const canvas = $('#camCapture');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if ($('#camMirror').checked) {
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -canvas.width, 0);
          ctx.restore();
        } else {
          ctx.drawImage(video, 0, 0);
        }
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        const payload = {
          image: dataUrl,
          mode: isCompareMode() ? 'compare' : 'single',
          detector: currentDetector(),
        };
        const t0 = performance.now();
        const res = await fetch('/api/detect/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        const rtt = performance.now() - t0;

        // Update FPS
        const now = performance.now();
        if (cam.lastFrameTs > 0) {
          const dt = now - cam.lastFrameTs;
          cam.fpsBuffer.push(1000 / dt);
          if (cam.fpsBuffer.length > 10) cam.fpsBuffer.shift();
          const avg = cam.fpsBuffer.reduce((a, b) => a + b, 0) / cam.fpsBuffer.length;
          $('#camFps').textContent = `FPS: ${avg.toFixed(1)} | RTT: ${rtt.toFixed(0)} ms`;
        }
        cam.lastFrameTs = now;

        renderCamResults(data);
      }
    } catch (e) {
      console.error('Blad kamery:', e);
    } finally {
      cam.busy = false;
    }
  }
  requestAnimationFrame(camLoop);
}

function renderCamResults(data) {
  const root = $('#camResults');
  if (data.mode === 'compare') {
    // Update lub create dla 3 kart - aktualizuj in-place, zeby nie migac
    const existing = $all('#camResults .result-card');
    if (existing.length !== data.results.length) {
      root.innerHTML = '';
      for (const r of data.results) {
        const card = buildResultCard({
          title: r.label,
          meta: `Twarze: ${r.faces} &middot; ${r.elapsed_ms} ms`,
          imageSrc: r.image,
        });
        card.dataset.detector = r.detector;
        root.appendChild(card);
      }
    } else {
      data.results.forEach((r, i) => {
        const card = existing[i];
        card.querySelector('h3').textContent = r.label;
        card.querySelector('.meta').innerHTML = `Twarze: ${r.faces} &middot; ${r.elapsed_ms} ms`;
        card.querySelector('img').src = r.image;
      });
    }
  } else {
    // single
    const existing = $all('#camResults .result-card');
    const label = state.detectors.find(d => d.id === data.detector)?.label || data.detector;
    const largest = data.detections && data.detections.length
      ? data.detections.reduce((a, b) =>
          (a.bbox[2]*a.bbox[3]) > (b.bbox[2]*b.bbox[3]) ? a : b)
      : null;
    if (existing.length !== 1) {
      root.innerHTML = '';
      const card = buildResultCard({
        title: label,
        meta: `Twarze: ${data.faces} &middot; ${data.elapsed_ms} ms`,
        imageSrc: data.image,
        probabilities: largest ? largest.probabilities : null,
      });
      root.appendChild(card);
    } else {
      const card = existing[0];
      card.querySelector('h3').textContent = label;
      card.querySelector('.meta').innerHTML = `Twarze: ${data.faces} &middot; ${data.elapsed_ms} ms`;
      card.querySelector('img').src = data.image;
      const oldBars = card.querySelector('.bars');
      if (oldBars) oldBars.remove();
      if (largest) {
        card.insertAdjacentHTML('beforeend', makeBarsHTML(largest.probabilities));
      }
    }
  }
}

// ---------- Start ----------
loadDetectors();
