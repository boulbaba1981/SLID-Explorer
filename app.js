/* SLID Explorer — in-browser inference with ONNX Runtime Web.
   Model outputs: probs (1,14), cams (1,14,9,12), embedding (1,1280). */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const IN_W = 384, IN_H = 288, GH = 9, GW = 12;
  const REGION = {
    'Abnormal': 'Screen', 'Cataract': 'Lens', 'Intraocular lens': 'Lens', 'Lens dislocation': 'Lens',
    'Keratitis': 'Cornea', 'Corneal scarring': 'Cornea', 'Corneal dystrophy': 'Cornea',
    'Corneal / Conjunctival tumor': 'Conjunctiva', 'Pinguecula': 'Conjunctiva', 'Pterygium': 'Conjunctiva',
    'Subconjunctival hemorrhage': 'Conjunctiva', 'Conjunctival injection': 'Conjunctiva',
    'Conjunctival cyst': 'Conjunctiva', 'Pigmented nevus': 'Conjunctiva'
  };
  const DISPLAY = {
    'Abnormal': 'Any abnormality', 'Corneal / Conjunctival tumor': 'Corneal / conjunctival tumour',
    'Subconjunctival hemorrhage': 'Subconjunctival haemorrhage'
  };
  const disp = (n) => DISPLAY[n] || n;

  const state = {
    meta: null, session: null, ready: null,
    bitmap: null, crop: null, result: null,
    camIdx: 0, mode: 'overlay', opacity: 0.6, sort: 'score'
  };

  /* ---------- theme ---------- */
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } }
  };
  const savedTheme = store.get('slid-theme');
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $('#themeBtn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next; store.set('slid-theme', next);
  });

  /* ---------- math ---------- */
  const logit = (p) => { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(p / (1 - p)); };
  const LMIN = -7, LMAX = 7;
  const pos = (p) => (Math.min(Math.max(logit(p), LMIN), LMAX) - LMIN) / (LMAX - LMIN);
  const fmt = (p) => p >= 0.995 ? '>0.99' : p < 0.005 ? '<0.01' : p.toFixed(2);
  const fmtThr = (t) => t < 0.1 ? t.toFixed(3) : t.toFixed(2);

  /* ---------- model loading ---------- */
  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = +res.headers.get('content-length') || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader(); const chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); got += value.length; onProgress(got / total);
    }
    const out = new Uint8Array(got); let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  async function loadModel() {
    const status = $('#modelStatus'), txt = $('#modelTxt'), bar = $('#modelBar');
    try {
      const metaP = fetch('model/model_meta.json').then((r) => r.json());
      ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      // Large binaries are stored in parts (GitHub web-upload friendly) and joined here.
      const parts = ['model/slid_effb0.onnx.part0', 'model/slid_effb0.onnx.part1', 'model/slid_effb0.onnx.part2',
        'vendor/ort-wasm-simd-threaded.wasm.part0', 'vendor/ort-wasm-simd-threaded.wasm.part1'];
      const frac = new Array(parts.length).fill(0);
      const bufs = await Promise.all(parts.map((u, i) => fetchWithProgress(u, (f) => {
        frac[i] = f; bar.style.width = (frac.reduce((a, b) => a + b, 0) / parts.length * 100).toFixed(0) + '%';
      })));
      const join = (arr) => { const n = arr.reduce((a, b) => a + b.length, 0), o = new Uint8Array(n); let k = 0; for (const a of arr) { o.set(a, k); k += a.length; } return o; };
      const bytes = join(bufs.slice(0, 3));
      ort.env.wasm.wasmBinary = join(bufs.slice(3)).buffer;
      txt.textContent = 'Preparing the model…';
      state.session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      state.meta = await metaP;
      buildModelCard(); buildCamSelect();
      status.classList.add('ready'); txt.textContent = 'Model ready. Runs on this device.';
    } catch (err) {
      console.error(err);
      status.classList.add('err');
      txt.textContent = 'The model could not load. Reload the page or try another browser.';
      throw err;
    }
  }
  state.ready = loadModel();
  window.slidExplorer = state;

  /* ---------- input ---------- */
  const drop = $('#drop');
  const pick = (input) => input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); input.value = ''; });
  pick($('#fileIn')); pick($('#camIn')); pick($('#fileIn2'));
  drop.addEventListener('click', (e) => { if (e.target === drop || e.target.closest('.drop-ico,.drop-title,.drop-sub')) $('#fileIn').click(); });
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#fileIn').click(); } });
  ['dragenter', 'dragover'].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); if (t === 'drop' || !e.relatedTarget) drop.classList.remove('over'); }));
  document.addEventListener('drop', (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) handleFile(f); });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith('image/'));
    if (item) handleFile(item.getAsFile());
  });

  async function handleFile(file, label) {
    if (!file.type.startsWith('image/') && !/\.(jpe?g|png|webp|bmp|gif)$/i.test(file.name || '')) {
      alertMsg('That file is not an image. Use a JPEG or PNG photo.'); return;
    }
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
    catch (e) { alertMsg('This image format cannot be read by your browser. Convert it to JPEG or PNG (HEIC photos often need converting).'); return; }
    await analyse(bmp, label);
  }

  function alertMsg(msg) {
    $('#resEmpty').hidden = false; $('#res').hidden = true;
    const t = $('#modelTxt'); const old = t.textContent; t.textContent = msg;
    setTimeout(() => { if (t.textContent === msg) t.textContent = old; }, 6000);
  }

  /* ---------- analysis ---------- */
  // SLID images are 4:3 or about 1.24:1 and were resized to 384 x 288 without cropping.
  // Inside that range we do the same; other shapes are centre-cropped to 4:3.
  function cropBox(w, h) {
    const r = 4 / 3, ar = w / h;
    if (ar >= 1.2 && ar <= 1.36) return { x: 0, y: 0, w, h };
    if (w / h > r + 1e-3) { const cw = Math.round(h * r); return { x: Math.round((w - cw) / 2), y: 0, w: cw, h }; }
    if (w / h < r - 1e-3) { const ch = Math.round(w / r); return { x: 0, y: Math.round((h - ch) / 2), w, h: ch }; }
    return { x: 0, y: 0, w, h };
  }

  /* Area (box) resampling, equivalent to OpenCV INTER_AREA used in training.
     src: Float32Array planar RGB [3][h][w]. */
  function areaWeights(n, m) {
    const s = n / m, out = [];
    for (let j = 0; j < m; j++) {
      const a = j * s, b = (j + 1) * s, idx = [], w = [];
      for (let i = Math.floor(a); i < Math.min(Math.ceil(b), n); i++) {
        const ov = Math.min(b, i + 1) - Math.max(a, i);
        if (ov > 1e-9) { idx.push(i); w.push(ov / s); }
      }
      out.push([idx, w]);
    }
    return out;
  }
  function areaResize(src, w, h, W, H) {
    const wx = areaWeights(w, W), wy = areaWeights(h, H);
    const tmp = new Float32Array(3 * h * W), dst = new Float32Array(3 * H * W);
    for (let c = 0; c < 3; c++) {
      const so = c * w * h, to = c * W * h;
      for (let y = 0; y < h; y++) for (let x = 0; x < W; x++) {
        const [idx, wt] = wx[x]; let v = 0;
        for (let k = 0; k < idx.length; k++) v += src[so + y * w + idx[k]] * wt[k];
        tmp[to + y * W + x] = v;
      }
      const dof = c * W * H;
      for (let y = 0; y < H; y++) {
        const [idx, wt] = wy[y];
        for (let x = 0; x < W; x++) {
          let v = 0;
          for (let k = 0; k < idx.length; k++) v += tmp[to + idx[k] * W + x] * wt[k];
          dst[dof + y * W + x] = v;
        }
      }
    }
    return dst;
  }

  function toTensor(bmp, c) {
    // Read the crop at native resolution (capped at 4096 px on the long side for memory).
    const cap = Math.min(1, 4096 / Math.max(c.w, c.h));
    const w = Math.round(c.w * cap), h = Math.round(c.h * cap);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, c.x, c.y, c.w, c.h, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data, n = w * h;
    let f = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { f[i] = d[4 * i]; f[n + i] = d[4 * i + 1]; f[2 * n + i] = d[4 * i + 2]; }
    let cw = w, ch = h;
    // Training path: long side to 512, then INTER_AREA to 384 x 288.
    if (cw > 512) { const nh = Math.round(ch * 512 / cw); f = areaResize(f, cw, ch, 512, nh); cw = 512; ch = nh; }
    if (cw >= IN_W && ch >= IN_H) f = areaResize(f, cw, ch, IN_W, IN_H);
    else { // small image: let the browser upscale
      const up = document.createElement('canvas'); up.width = IN_W; up.height = IN_H;
      const u = up.getContext('2d', { willReadFrequently: true }); u.drawImage(bmp, c.x, c.y, c.w, c.h, 0, 0, IN_W, IN_H);
      const dd = u.getImageData(0, 0, IN_W, IN_H).data, m = IN_W * IN_H; f = new Float32Array(3 * m);
      for (let i = 0; i < m; i++) { f[i] = dd[4 * i]; f[m + i] = dd[4 * i + 1]; f[2 * m + i] = dd[4 * i + 2]; }
    }
    for (let i = 0; i < f.length; i++) f[i] /= 255;
    return new ort.Tensor('float32', f, [1, 3, IN_H, IN_W]);
  }

  async function analyse(bmp, label) {
    $('#drop').hidden = true; $('#stage').hidden = false;
    state.bitmap = bmp; state.crop = cropBox(bmp.width, bmp.height); state.result = null;
    view.width = 768; view.height = Math.round(768 * state.crop.h / state.crop.w);
    $('.canvas-wrap').style.aspectRatio = `${state.crop.w} / ${state.crop.h}`;
    const note = $('#cropNote');
    const c = state.crop;
    if (c.w !== bmp.width || c.h !== bmp.height) {
      note.hidden = false; note.textContent = `Centre-cropped from ${bmp.width} × ${bmp.height} to 4:3, the shape of the training images.` + (label ? ` SLID reference label: ${label}.` : '');
    } else if (label) {
      note.hidden = false; note.textContent = `SLID reference label: ${label}.`;
    } else note.hidden = true;
    draw(); setBusy(true, state.session ? 'Analysing…' : 'Loading model…');
    try {
      await state.ready;
      setBusy(true, 'Analysing…');
      await new Promise((r) => setTimeout(r, 20));
      const t0 = performance.now();
      const out = await state.session.run({ image: toTensor(bmp, state.crop) });
      const ms = performance.now() - t0;
      state.result = {
        probs: Array.from(out.probs.data), cams: out.cams.data, emb: out.embedding.data, ms
      };
      state.result.ood = oodScore(state.result.emb);
      state.camIdx = defaultCam();
      $('#camSel').value = String(state.camIdx);
      render();
    } catch (e) {
      console.error(e); alertMsg('Analysis failed. Try another image.');
    } finally { setBusy(false); }
  }

  function oodScore(emb) {
    let nrm = 0; for (let i = 0; i < emb.length; i++) nrm += emb[i] * emb[i];
    nrm = Math.sqrt(nrm) || 1;
    let best = -1;
    for (const c of state.meta.ood.centroids) {
      let s = 0; for (let i = 0; i < c.length; i++) s += c[i] * emb[i];
      best = Math.max(best, s / nrm);
    }
    return { sim: best, flagged: best < state.meta.ood.threshold };
  }

  function defaultCam() {
    const T = state.meta.targets, p = state.result.probs;
    let best = -1, bm = -Infinity;
    for (let k = 1; k < T.length; k++) {
      const m = logit(p[k]) - logit(T[k].threshold);
      if (m > 0 && m > bm) { bm = m; best = k; }
    }
    return best > 0 ? best : 0;
  }

  function setBusy(on, msg) { $('#busy').hidden = !on; if (msg) $('#busyTxt').textContent = msg; }

  /* ---------- drawing ---------- */
  const view = $('#view'), vctx = view.getContext('2d');
  const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  function cmap(v) {
    const x = v * (INFERNO.length - 1), i = Math.min(Math.floor(x), INFERNO.length - 2), t = x - i;
    const a = INFERNO[i], b = INFERNO[i + 1];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }

  function camGrid(k) {
    const base = k * GH * GW, g = new Float32Array(GH * GW);
    let mx = 0;
    for (let i = 0; i < GH * GW; i++) { const v = state.result.cams[base + i]; g[i] = v; if (v > mx) mx = v; }
    for (let i = 0; i < g.length; i++) g[i] = mx > 0 ? Math.max(g[i], 0) / mx : 0;
    return { g, mx };
  }

  function heatCanvas(k, W, H) {
    const { g, mx } = camGrid(k);
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    if (mx <= 0) return { cv, empty: true };
    const ctx = cv.getContext('2d'), img = ctx.createImageData(W, H), d = img.data;
    for (let y = 0; y < H; y++) {
      const gy = Math.min(Math.max((y + 0.5) * GH / H - 0.5, 0), GH - 1), y0 = Math.floor(gy), y1 = Math.min(y0 + 1, GH - 1), ty = gy - y0;
      for (let x = 0; x < W; x++) {
        const gx = Math.min(Math.max((x + 0.5) * GW / W - 0.5, 0), GW - 1), x0 = Math.floor(gx), x1 = Math.min(x0 + 1, GW - 1), tx = gx - x0;
        const v = (g[y0 * GW + x0] * (1 - tx) + g[y0 * GW + x1] * tx) * (1 - ty) + (g[y1 * GW + x0] * (1 - tx) + g[y1 * GW + x1] * tx) * ty;
        const [r, gg, b] = cmap(v), o = 4 * (y * W + x);
        d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = 255 * Math.min(1, Math.pow(v, 1.3) * 1.25);
      }
    }
    ctx.putImageData(img, 0, 0);
    return { cv, empty: false };
  }

  function draw() {
    const c = state.crop, W = view.width, H = view.height;
    vctx.clearRect(0, 0, W, H);
    if (!state.bitmap) return;
    vctx.imageSmoothingQuality = 'high';
    vctx.drawImage(state.bitmap, c.x, c.y, c.w, c.h, 0, 0, W, H);
    const legend = $('#camLegend');
    if (state.result && state.mode === 'overlay') {
      const h = heatCanvas(state.camIdx, 192, 144);
      if (!h.empty) {
        vctx.globalAlpha = state.opacity; vctx.imageSmoothingEnabled = true;
        vctx.drawImage(h.cv, 0, 0, W, H); vctx.globalAlpha = 1;
      }
      legend.hidden = false;
      $('#camFor').textContent = h.empty ? `${disp(state.meta.targets[state.camIdx].name)}: no positive evidence`
        : `Evidence for ${disp(state.meta.targets[state.camIdx].name).toLowerCase()}`;
    } else legend.hidden = true;
  }

  /* ---------- results ---------- */
  function render() {
    const { meta, result } = state, T = meta.targets, p = result.probs;
    $('#resEmpty').hidden = true; $('#res').hidden = false;

    const ood = $('#oodBanner');
    ood.hidden = !result.ood.flagged;
    $('#oodSim').textContent = result.ood.sim.toFixed(2);
    $('#oodThr').textContent = meta.ood.threshold.toFixed(2);

    // screening
    const s = p[0], thr = T[0].threshold, flag = s >= thr;
    const box = $('#screen'); box.className = 'screen ' + (flag ? 'flag' : 'clear');
    const pill = $('#screenPill'); pill.className = 'pill ' + (flag ? 'flag' : 'clear');
    pill.textContent = flag ? 'Refer' : 'Not flagged';
    $('#screenTitle').textContent = flag ? 'Abnormal findings likely' : 'No abnormality flagged';
    const nFlag = T.slice(1).filter((t, i) => p[i + 1] >= t.threshold).length;
    $('#screenSub').textContent = `Score ${fmt(s)}, threshold ${fmtThr(thr)} (90% specificity). ` +
      (nFlag ? `${nFlag} lesion${nFlag > 1 ? 's' : ''} above threshold.` : 'No lesion above threshold.') +
      ` Computed in ${Math.round(result.ms)} ms.`;
    const L = Math.PI * 50, gv = $('#gVal');
    gv.style.strokeDasharray = `${L}`; gv.style.strokeDashoffset = `${L * (1 - pos(s))}`;
    const a = Math.PI * (1 - pos(thr));
    const tl = $('#gThr');
    tl.setAttribute('x1', 60 + 38 * Math.cos(a)); tl.setAttribute('y1', 62 - 38 * Math.sin(a));
    tl.setAttribute('x2', 60 + 62 * Math.cos(a)); tl.setAttribute('y2', 62 - 62 * Math.sin(a));
    $('#gNum').textContent = fmt(s);

    renderFindings(); draw();
  }

  function renderFindings() {
    const { meta, result } = state, T = meta.targets, p = result.probs, list = $('#findings');
    const rows = T.slice(1).map((t, i) => ({ k: i + 1, t, p: p[i + 1], m: logit(p[i + 1]) - logit(t.threshold) }));
    list.innerHTML = '';
    const mk = (r) => {
      const li = document.createElement('li');
      const on = r.p >= r.t.threshold;
      li.className = 'frow' + (on ? ' on' : '') + (r.k === state.camIdx ? ' sel' : '');
      li.tabIndex = 0; li.setAttribute('role', 'button');
      li.setAttribute('aria-pressed', r.k === state.camIdx ? 'true' : 'false');
      li.setAttribute('aria-label', `${disp(r.t.name)}: score ${fmt(r.p)}, threshold ${fmtThr(r.t.threshold)}, ${on ? 'above' : 'below'} threshold. Show evidence map.`);
      li.title = `Score ${r.p.toFixed(3)} · threshold ${r.t.threshold.toFixed(3)} · AUROC ${r.t.auroc.toFixed(3)} · sensitivity ${Math.round(r.t.sens_at_spec90 * 100)}% at 90% specificity`;
      li.innerHTML = `<span class="fname"><span class="flagdot"></span><span class="nm"></span>${state.sort === 'score' ? `<span class="region">${REGION[r.t.name]}</span>` : ''}</span>
        <span class="fbar"><i style="width:${(pos(r.p) * 100).toFixed(1)}%"></i><b style="left:calc(${(pos(r.t.threshold) * 100).toFixed(1)}% - 1px)"></b></span>
        <span class="fval">${fmt(r.p)}</span>`;
      li.querySelector('.nm').textContent = disp(r.t.name);
      const act = () => selectCam(r.k);
      li.addEventListener('click', act);
      li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
      return li;
    };
    if (state.sort === 'score') {
      rows.sort((a, b) => b.m - a.m).forEach((r) => list.appendChild(mk(r)));
    } else {
      for (const reg of ['Lens', 'Cornea', 'Conjunctiva']) {
        const h = document.createElement('li'); h.className = 'group-lbl'; h.textContent = reg; list.appendChild(h);
        rows.filter((r) => REGION[r.t.name] === reg).forEach((r) => list.appendChild(mk(r)));
      }
    }
  }

  function selectCam(k) {
    state.camIdx = k; $('#camSel').value = String(k);
    if (state.mode !== 'overlay') setMode('overlay');
    renderFindings(); draw();
  }

  function buildCamSelect() {
    const sel = $('#camSel'); sel.innerHTML = '';
    state.meta.targets.forEach((t, k) => {
      const o = document.createElement('option'); o.value = String(k); o.textContent = disp(t.name); sel.appendChild(o);
    });
    sel.addEventListener('change', () => selectCam(+sel.value));
  }

  function setMode(m) {
    state.mode = m;
    document.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === m)));
    draw();
  }
  document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.querySelectorAll('[data-sort]').forEach((b) => b.addEventListener('click', () => {
    state.sort = b.dataset.sort;
    document.querySelectorAll('[data-sort]').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    if (state.result) renderFindings();
  }));
  $('#opacity').addEventListener('input', (e) => { state.opacity = e.target.value / 100; draw(); });
  $('#dlBtn').addEventListener('click', () => {
    view.toBlob((b) => {
      const a = document.createElement('a'); a.href = URL.createObjectURL(b);
      const nm = state.meta ? state.meta.targets[state.camIdx].name.replace(/[^a-z]+/gi, '_').toLowerCase() : 'view';
      a.download = `slid_explorer_${nm}.png`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, 'image/png');
  });

  /* ---------- examples ---------- */
  fetch('examples/examples.json').then((r) => r.json()).then((ex) => {
    const g = $('#gallery');
    ex.forEach((e) => {
      const b = document.createElement('button'); b.className = 'ex'; b.type = 'button';
      b.setAttribute('aria-label', `Analyse SLID example ${e.slid_file}, labelled ${e.label}`);
      b.innerHTML = `<img loading="lazy" alt="" width="192" height="144"><span></span>`;
      b.querySelector('img').src = e.thumb;
      const s = b.querySelector('span'); s.textContent = e.short;
      const sm = document.createElement('small'); sm.textContent = `SLID ${e.slid_file}`; s.appendChild(sm);
      b.addEventListener('click', async () => {
        document.getElementById('workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
        const blob = await fetch(e.src).then((r) => r.blob());
        const bmp = await createImageBitmap(blob);
        analyse(bmp, `${e.label} (image seen during training)`);
      });
      g.appendChild(b);
    });
  });

  /* ---------- model card ---------- */
  function buildModelCard() {
    const m = state.meta, T = m.targets;
    const macro = T.slice(1).reduce((s, t) => s + t.auroc, 0) / (T.length - 1);
    const facts = [
      ['Backbone', 'EfficientNet-B0, ImageNet weights, frozen'],
      ['Heads', '14 L2-regularised logistic regressions on pooled features'],
      ['Input', '384 × 288 RGB, whole frame'],
      ['Training data', 'SLID, 2,617 images, 13 lesion types'],
      ['Validation', `5-fold grouped CV, ${m.n_groups.toLocaleString()} rebuilt eye groups`],
      ['Screening AUROC', T[0].auroc.toFixed(3)],
      ['Macro lesion AUROC', macro.toFixed(3)],
      ['Explanations', 'Exact class activation maps (12 × 9 grid)']
    ];
    $('#facts').innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    $('#mtable tbody').innerHTML = T.map((t, k) =>
      `<tr class="${k === 0 ? 'lead' : ''}"><td>${disp(t.name)}</td><td>${REGION[t.name]}</td><td class="num">${t.n_pos.toLocaleString()}</td>` +
      `<td class="num">${t.auroc.toFixed(3)}</td><td class="num ${t.sens_at_spec90 < 0.6 ? 'low' : ''}">${Math.round(t.sens_at_spec90 * 100)}%</td>` +
      `<td class="num">${fmtThr(t.threshold)}</td></tr>`).join('');
  }
})();
