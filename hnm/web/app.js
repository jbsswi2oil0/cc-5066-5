'use strict';

const $ = (id) => document.getElementById(id);

const CONVENTIONS = [
  ['opengl', 'OpenGL / Blender / Unity（+Y 绿朝上）'],
  ['directx', 'DirectX / Unreal（−Y 绿朝下）'],
  ['opengl_invert_x', 'OpenGL 系 + 翻转 X'],
  ['directx_invert_x', 'DirectX 系 + 翻转 X/Y'],
];

const FIELD_ALIASES = {
  width_world: 's-widthWorld', widthWorld: 's-widthWorld',
  height_world: 's-heightWorld', heightWorld: 's-heightWorld',
  z_per_height_unit: 's-zScale', zScale: 's-zScale',
  stretch_x: 's-stretchX', stretchX: 's-stretchX',
  stretch_y: 's-stretchY', stretchY: 's-stretchY',
  gamma: 's-gamma', channel: 's-channel', transfer: 's-transfer',
  boundary: 's-boundary', method: 's-method', image: 's-file',
};

async function api(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch (e) { /* fall through */ }
  if (!data) throw { message: `服务异常（HTTP ${res.status}），请确认 python -m hnm.server 正在运行` };
  if (!res.ok || data.ok === false) throw data.error || { message: '未知内核错误' };
  return data;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('图像解码失败'));
    img.src = src;
  });
}

function setFeedback(box, kind, msg) {
  box.hidden = false;
  box.className = `feedback ${kind}`;
  box.textContent = msg;
}
function clearFeedback(box) {
  box.hidden = true;
  box.textContent = '';
  box.className = 'feedback';
}

function markInvalid(field) {
  if (!field) return;
  let id = FIELD_ALIASES[field];
  if (field.includes('islandB') || field.includes('island_B')) id = 'i-b-z';
  const el = id ? $(id) : null;
  if (el) {
    el.classList.add('invalid');
    el.focus?.();
    setTimeout(() => el.classList.remove('invalid'), 4000);
  }
}

function fillConventionSelect(sel) {
  sel.innerHTML = '';
  for (const [id, label] of CONVENTIONS) {
    const o = document.createElement('option');
    o.value = id; o.textContent = label;
    sel.appendChild(o);
  }
}

/* ---------------- 法线 PNG 解码（真实走一遍编码贴图） ---------------- */

async function decodeNormalDataUrl(dataUrl, signs) {
  const img = await loadImage(dataUrl);
  const w = img.naturalWidth, h = img.naturalHeight;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const ctx = off.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const nx = new Float32Array(w * h), ny = new Float32Array(w * h), nz = new Float32Array(w * h);
  const enc = new Uint8ClampedArray(d);
  const [sx, sy, sz] = signs;
  for (let p = 0, i = 0; p < w * h; p++, i += 4) {
    let x = (d[i] / 255) * 2 - 1;
    let y = (d[i + 1] / 255) * 2 - 1;
    let z = (d[i + 2] / 255) * 2 - 1;
    x *= sx; y *= sy; z *= sz;
    const len = Math.hypot(x, y, z) || 1;
    nx[p] = x / len; ny[p] = y / len; nz[p] = z / len;
  }
  return { w, h, nx, ny, nz, enc };
}

/* ---------------- 交互式光照 Canvas ---------------- */

class LightCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dec = null;
  }
  setDecoded(dec) {
    this.dec = dec;
    this.canvas.width = dec.w;
    this.canvas.height = dec.h;
    this.out = this.ctx.createImageData(dec.w, dec.h);
  }
  render(opts) {
    if (!this.dec) return;
    const { w, h, nx, ny, nz } = this.dec;
    const az = opts.az * Math.PI / 180;
    const el = opts.el * Math.PI / 180;
    const lx = Math.cos(el) * Math.cos(az), ly = Math.cos(el) * Math.sin(az), lz = Math.sin(el);
    const v = [0, 0, 1];
    let hx = lx + v[0], hy = ly + v[1], hz = lz + v[2];
    const hl = Math.hypot(hx, hy, hz) || 1;
    hx /= hl; hy /= hl; hz /= hl;
    const amb = opts.ambient, specK = opts.spec, shin = 24;
    const br = 0.72, bg = 0.74, bb = 0.78;
    const out = this.out.data;
    for (let p = 0, i = 0; p < w * h; p++, i += 4) {
      const X = nx[p], Y = ny[p], Z = nz[p];
      let ndl = X * lx + Y * ly + Z * lz;
      if (ndl < 0) ndl = 0;
      const diff = amb + (1 - amb) * ndl;
      let ndh = X * hx + Y * hy + Z * hz;
      if (ndh < 0) ndh = 0;
      const spec = specK * Math.pow(ndh, shin);
      out[i] = Math.min(255, (br * diff + spec) * 255 + 0.5);
      out[i + 1] = Math.min(255, (bg * diff + spec) * 255 + 0.5);
      out[i + 2] = Math.min(255, (bb * diff + spec) * 255 + 0.5);
      out[i + 3] = 255;
    }
    this.ctx.putImageData(this.out, 0, 0);
  }
  pixelEvent(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - rect.left) / rect.width * this.dec.w);
    const y = Math.floor((ev.clientY - rect.top) / rect.height * this.dec.h);
    if (x < 0 || y < 0 || x >= this.dec.w || y >= this.dec.h) return null;
    return { x, y, p: y * this.dec.w + x };
  }
  downloadUrl() {
    return this.canvas.toDataURL('image/png');
  }
}

/* ---------------- 单图模式 ---------------- */

const single = {
  catalog: null,
  uploadDataUrl: null,
  result: null,
  dec: null,
  light: new LightCanvas($('s-lightcanvas')),
};

function collectSingleParams() {
  return {
    sampleId: single.uploadDataUrl ? null : $('s-sample').value,
    image: single.uploadDataUrl,
    channel: $('s-channel').value,
    transfer: $('s-transfer').value,
    gamma: parseFloat($('s-gamma').value) || 2.2,
    invert: $('s-invert').checked,
    widthWorld: parseFloat($('s-widthWorld').value),
    heightWorld: parseFloat($('s-heightWorld').value),
    zScale: parseFloat($('s-zScale').value),
    unit: $('s-unit').value || 'm',
    stretchX: parseFloat($('s-stretchX').value),
    stretchY: parseFloat($('s-stretchY').value),
    boundary: $('s-boundary').value,
    method: $('s-method').value,
    convention: $('s-convention').value,
    flipX: $('s-flipX').checked,
    flipY: $('s-flipY').checked,
    flipZ: $('s-flipZ').checked,
  };
}

function updatePitchReadout() {
  const w = parseFloat($('s-widthWorld').value);
  const h = parseFloat($('s-heightWorld').value);
  const sx = parseFloat($('s-stretchX').value);
  const sy = parseFloat($('s-stretchY').value);
  $('s-stretchX-val').textContent = `${sx.toFixed(2)}×`;
  $('s-stretchY-val').textContent = `${sy.toFixed(2)}×`;
  const dims = $('s-dims').dataset;
  if (dims.w && w > 0 && h > 0) {
    const dx = w * sx / dims.w, dy = h * sy / dims.h;
    $('s-pitch').textContent =
      `像素间距 dx=${dx.toExponential(3)} ${$('s-unit').value}/px，dy=${dy.toExponential(3)} ${$('s-unit').value}/px，宽高比=${(dx / dy).toFixed(3)}`;
  }
}

function applySampleDefaults(id) {
  const e = single.catalog.samples.find((x) => x.id === id);
  if (!e) return;
  $('s-channel').value = e.channel || 'r';
  $('s-transfer').value = e.transfer || 'linear';
  $('s-widthWorld').value = e.widthWorld;
  $('s-heightWorld').value = e.heightWorld;
  $('s-zScale').value = e.zScale;
  $('s-unit').value = e.unit || 'm';
  $('s-boundary').value = e.boundary || 'clamp';
  $('s-method').value = 'central';
  $('s-stretchX').value = 1; $('s-stretchY').value = 1;
  $('s-thumb').src = `/assets/source/${e.file}`;
  $('s-sample-desc').textContent = e.description || '';
  const probe = new Image();
  probe.onload = () => {
    $('s-dims').textContent = `${probe.naturalWidth} × ${probe.naturalHeight} px（实际纹理尺寸）`;
    $('s-dims').dataset.w = probe.naturalWidth;
    $('s-dims').dataset.h = probe.naturalHeight;
    updatePitchReadout();
  };
  probe.src = `/assets/source/${e.file}`;
  updateTransferVisibility();
  updatePitchReadout();
}

function updateTransferVisibility() {
  $('s-gamma-wrap').hidden = $('s-transfer').value !== 'gamma';
}

function renderSingleMetrics(d) {
  const si = d.scale, st = d.stats, en = d.encoding, cv = d.convention;
  const rows = [
    ['纹理像素', `${d.width} × ${d.height}`],
    ['像素间距 dx / dy', `${si.pixelPitchX.toExponential(4)} / ${si.pixelPitchY.toExponential(4)} ${si.unit}`],
    ['纹素宽高比 dx/dy', si.texelAspect.toFixed(4)],
    ['高度标尺 zScale', `${si.zPerHeightUnit} ${si.unit}/单位高度`],
    ['模型拉伸', `${si.stretchX.toFixed(2)}× / ${si.stretchY.toFixed(2)}×`],
    ['平均坡度 |∇H|', st.meanAbsSlope.toFixed(5)],
    ['最大坡度', st.maxSlope.toFixed(5)],
    ['坡度 X 范围', `[${st.slopeXMin.toFixed(4)}, ${st.slopeXMax.toFixed(4)}]`],
    ['坡度 Y 范围', `[${st.slopeYMin.toFixed(4)}, ${st.slopeYMax.toFixed(4)}]`],
    ['nz 最小 / 平均', `${st.normalZMin.toFixed(4)} / ${st.normalZMean.toFixed(4)}`],
    ['编码最大角误差', `${en.maxAngularErrorDeg.toFixed(3)}°`],
    ['编码平均角误差', `${en.meanAngularErrorDeg.toFixed(4)}°`],
    ['编码 RMSE', `${en.rmseAngularErrorDeg.toFixed(4)}°`],
    ['约定预设', cv.presetLabel],
    ['编码符号 (X,Y,Z)', `${cv.signs.x}, ${cv.signs.y}, ${cv.signs.z}`],
    ['绿通道含义', cv.greenChannel],
  ];
  $('s-metrics').innerHTML = rows.map(([k, v]) =>
    `<div class="metric"><span>${k}</span><span>${v}</span></div>`).join('')
    + `<div class="metric full"><span>规范系</span><span>${cv.canonicalFrame}</span></div>`
    + `<div class="metric full"><span>编码</span><span>${cv.encoding}</span></div>`;
  $('s-conv-note').textContent = cv.presetNote;
  $('s-frame-note').textContent = cv.canonicalFrame;
  $('s-conv-badge').textContent = `${cv.preset} · 符号(${cv.signs.x},${cv.signs.y},${cv.signs.z})`;
}

function currentLightOpts() {
  return {
    az: parseFloat($('l-az').value),
    el: parseFloat($('l-el').value),
    ambient: parseFloat($('l-amb').value),
    spec: parseFloat($('l-spec').value),
  };
}

async function runSingleConvert() {
  const btn = $('s-convert');
  btn.disabled = true;
  $('s-spinner').hidden = false;
  clearFeedback($('s-feedback'));
  document.querySelectorAll('.invalid').forEach((e) => e.classList.remove('invalid'));
  try {
    const params = collectSingleParams();
    const d = await api('/api/convert', params);
    single.result = d;
    $('s-normal').src = d.normalPng;
    $('s-download-normal').href = d.normalPng;
    $('s-download-normal').download = `normal_${d.convention.preset}.png`;
    $('s-download-height').href = d.heightPreviewPng;
    renderSingleMetrics(d);

    single.dec = await decodeNormalDataUrl(d.normalPng, [
      d.convention.signs.x, d.convention.signs.y, d.convention.signs.z,
    ]);
    single.light.setDecoded(single.dec);
    single.light.render(currentLightOpts());
    drawSingleVectors();
  } catch (err) {
    const msg = err.message || String(err);
    setFeedback($('s-feedback'), 'error', `转换失败：${msg}`);
    markInvalid(err.field);
  } finally {
    btn.disabled = false;
    $('s-spinner').hidden = true;
  }
}

async function drawSingleVectors() {
  const dec = single.dec;
  const canvas = $('s-vectorcanvas');
  if (!dec) return;
  canvas.width = dec.w; canvas.height = dec.h;
  const ctx = canvas.getContext('2d');
  const img = await loadImage(single.result.normalPng);
  ctx.drawImage(img, 0, 0);
  const step = Math.max(16, Math.round(dec.w / 26));
  const len = step * 0.46;
  ctx.lineWidth = 1.4;
  for (let y = step >> 1; y < dec.h; y += step) {
    for (let x = step >> 1; x < dec.w; x += step) {
      const p = y * dec.w + x;
      const dx = dec.nx[p] * len;
      const dy = -dec.ny[p] * len; // 规范 +Y 朝上，canvas 行向下
      ctx.strokeStyle = 'rgba(8,14,24,.85)';
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx + 1, y + dy + 1); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,.92)';
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke();
      ctx.fillStyle = dec.nx[p] >= 0 ? 'rgba(255,120,120,.95)' : 'rgba(120,200,255,.95)';
      ctx.beginPath(); ctx.arc(x + dx, y + dy, 1.8, 0, Math.PI * 2); ctx.fill();
    }
  }
}

/* ---------------- 尺度一致性验收（图形入口） ---------------- */

const verify = {
  result: null,
  dec: null,
  lit: new LightCanvas($('s-verify-litcanvas')),
};

async function runVerify() {
  const btn = $('s-verify-run');
  const fb = $('s-verify-feedback');
  clearFeedback(fb);
  btn.disabled = true;
  try {
    const params = collectSingleParams();
    // 用真实材质侧光方向：方位=跨坡度方向，低仰角
    params.light = { azimuth: parseFloat($('l-az').value), elevation: 15.0 };
    const d = await api('/api/verify', params);
    verify.result = d;

    const badge = $('s-verify-badge');
    const v = d.verdict || {};
    badge.textContent = v.passed ? '验收 PASS' : '验收 FAIL';
    badge.className = `badge ${v.passed ? 'ok' : 'fail'}`;
    const ref = $('s-verify-ref');
    if (d.referenceAvailable) {
      ref.textContent = '独立解析参照 ✓';
      ref.className = 'badge ok';
    } else {
      ref.textContent = '无解析真值 · 仅可证范围';
      ref.className = 'badge warn';
    }

    renderVerifyMetrics(d);

    // 光照必须用“测得法线”：从内核交付的法线 PNG 逐像素读回后着色
    verify.dec = await decodeNormalDataUrl(d.normalPng, [
      d.convention.signs.x, d.convention.signs.y, d.convention.signs.z,
    ]);
    verify.lit.setDecoded(verify.dec);
    verify.lit.render({
      az: params.light.azimuth, el: params.light.elevation, ambient: 0.22, spec: 0.0,
    });
    $('s-verify-lit-cap').textContent =
      `测得法线 · 侧光 az=${params.light.azimuth}° el=15°（逐像素解码法线参与着色）`;

    if (!v.passed) {
      setFeedback(fb, 'error', v.note || '法线方向或渲染响应与独立参照不符，见下方角差与响应差。');
    } else if (!d.referenceAvailable) {
      setFeedback(fb, 'info', v.note);
    }
  } catch (err) {
    const msg = err.message || String(err);
    setFeedback(fb, 'error', `验收失败：${msg}`);
    $('s-verify-badge').textContent = '验收错误';
    $('s-verify-badge').className = 'badge fail';
    markInvalid(err.field);
  } finally {
    btn.disabled = false;
  }
}

function renderVerifyMetrics(d) {
  const v = d.verdict || {}, sp = d.slopePaths || {};
  const rows = [];
  rows.push(['裁决', v.passed ? 'PASS' : 'FAIL']);
  if (d.referenceAvailable && v.angle) {
    rows.push(['法线角差 max / mean', `${v.angle.max.toFixed(3)}° / ${v.angle.mean.toFixed(3)}°`]);
    rows.push(['角差容差（8-bit 读回）', `${v.angleToleranceDeg}°`]);
    rows.push(['斜光响应差 mean / max', `${v.grazingMeanAbsDelta.toFixed(4)} / ${v.grazingMaxAbsDelta.toFixed(4)}`]);
    rows.push(['响应差容差 mean / max', `${v.grazingMeanTolerance} / ${v.grazingMaxTolerance}`]);
    rows.push(['斜光平均响应（测得 vs 参照）', `${v.grazingResponse.toFixed(4)} / ${v.grazingReference.toFixed(4)}`]);
  } else {
    rows.push(['参照', '无解析真值：仅证明编码往返与两写法一致']);
    rows.push(['编码往返最大角误差', `${(d.encoding?.maxAngularErrorDeg ?? 0).toFixed(3)}°`]);
  }
  rows.push(['切线基内 vs 真实长度', `渲染响应平均差 ${sp.tangentVsTrueLengthRenderMeanAbsDelta?.toFixed(6)}（${sp.agree ? '一致' : '不一致'}）`]);
  if (v.unitMismatch) {
    rows.push(['样例声明尺度自洽', v.unitMismatch.consistent ? '是' : '否（仅提示，非硬失败）']);
  }
  $('s-verify-metrics').innerHTML = rows.map(([k, val]) =>
    `<div class="metric"><span>${k}</span><span>${val}</span></div>`).join('');
}

/* ---------------- 双岛模式 ---------------- */

const islands = {
  result: null,
  dec: null,
  light: new LightCanvas($('i-lightcanvas')),
};

function collectIslandParams(mode) {
  const bz = $('i-b-z').value.trim();
  return {
    mode,
    rowFraction: parseFloat($('i-row').value),
    slopeTol: parseFloat($('i-tol').value),
    expectedPolarity: $('i-expect').value,
    boundary: $('i-boundary').value,
    convention: $('i-convention').value,
    previewZ: parseFloat($('i-previewz').value),
    islandA: {
      sampleId: 'island_A_calibrated',
      widthWorld: parseFloat($('i-a-width').value),
      heightWorld: parseFloat($('i-a-height').value),
      zScale: parseFloat($('i-a-z').value),
      transfer: $('i-a-transfer').value,
      flipCols: $('i-a-flipcols').checked,
      flipRows: $('i-a-fliprows').checked,
    },
    islandB: {
      sampleId: 'island_B_relative',
      widthWorld: parseFloat($('i-b-width').value),
      heightWorld: parseFloat($('i-b-height').value),
      zScale: bz === '' ? null : parseFloat(bz),
      transfer: $('i-b-transfer').value,
      flipCols: $('i-b-flipcols').checked,
      flipRows: $('i-b-fliprows').checked,
    },
  };
}

function activateIslandRTab(name) {
  document.querySelectorAll('[data-irtab]').forEach((b) =>
    b.classList.toggle('active', b.dataset.irtab === name));
  document.querySelectorAll('[data-irpanel]').forEach((p) =>
    p.classList.toggle('active', p.dataset.irpanel === name));
}

function renderProfileCanvas(canvas, prof, checks) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#12161e';
  ctx.fillRect(0, 0, W, H);
  const ml = 60, mr = 250, mt = 46, mb = 44;
  const pw = W - ml - mr, ph = H - mt - mb;
  const xs = prof.x, hs = prof.height;
  const xmin = xs[0], xmax = xs[xs.length - 1];
  let hmin = Math.min(...hs), hmax = Math.max(...hs);
  const pad = Math.max((hmax - hmin) * 0.18, 1e-6);
  hmin -= pad; hmax += pad;
  const X = (x) => ml + (x - xmin) / (xmax - xmin || 1e-12) * pw;
  const Y = (h) => mt + (hmax - h) / (hmax - hmin || 1e-12) * ph;

  ctx.strokeStyle = '#223047'; ctx.fillStyle = '#8295b5'; ctx.font = '12px monospace';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = mt + ph * i / 4;
    ctx.beginPath(); ctx.moveTo(ml, gy); ctx.lineTo(ml + pw, gy); ctx.stroke();
    const val = hmax - (hmax - hmin) * i / 4;
    ctx.fillText(val.toFixed(4), 8, gy + 4);
  }
  // 填充
  ctx.beginPath();
  ctx.moveTo(X(xs[0]), mt + ph);
  xs.forEach((x, idx) => ctx.lineTo(X(x), Y(hs[idx])));
  ctx.lineTo(X(xs[xs.length - 1]), mt + ph);
  ctx.closePath();
  ctx.fillStyle = 'rgba(52,130,190,.35)';
  ctx.fill();
  // 折线
  ctx.beginPath();
  xs.forEach((x, idx) => idx ? ctx.lineTo(X(x), Y(hs[idx])) : ctx.moveTo(X(x), Y(hs[idx])));
  ctx.strokeStyle = '#5ac8ff'; ctx.lineWidth = 2; ctx.stroke();

  // 接缝
  const sx = X(prof.seamX);
  ctx.strokeStyle = '#e6b446'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sx, mt); ctx.lineTo(sx, mt + ph); ctx.stroke();
  ctx.fillStyle = '#e6be5a'; ctx.font = 'bold 14px sans-serif';
  ctx.fillText('接缝', sx + 8, mt - 18);

  // L0 R0
  const sc = prof.seamCol;
  [[sc - 1, '#ff7878', 'L0'], [sc, '#78ffa0', 'R0']].forEach(([idx, color, name]) => {
    if (idx >= 0 && idx < xs.length) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(X(xs[idx]), Y(hs[idx]), 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillText(name, X(xs[idx]) + 8, Y(hs[idx]) - 8);
    }
  });

  ctx.fillStyle = '#e6edf7'; ctx.font = 'bold 18px sans-serif';
  ctx.fillText('跨缝高度剖线 H(X)（物理高度）', ml, 26);
  ctx.fillStyle = '#aabbd4'; ctx.font = '12px monospace';
  ctx.fillText('材料 X（世界单位）', ml, H - 16);

  const lines = [
    ['左坡度', prof.slopeLeft.toFixed(4)],
    ['跨缝坡度', prof.slopeCross.toFixed(4)],
    ['右坡度', prof.slopeRight.toFixed(4)],
    ['坡度残差', prof.slopeResidual.toFixed(4)],
    ['镜像相关', prof.mirrorCorrelation.toFixed(3)],
    ['极性', prof.polarity],
    ['峰/谷', `${prof.peakRelief.toFixed(5)} / ${prof.troughRelief.toFixed(5)}`],
  ];
  let y = mt + 6;
  lines.forEach(([k, v]) => {
    ctx.fillStyle = '#8295b5'; ctx.fillText(k, ml + pw + 20, y);
    ctx.fillStyle = '#dceaff'; ctx.fillText(v, ml + pw + 110, y);
    y += 23;
  });

  if (checks) {
    const ok = checks.passed;
    ctx.fillStyle = ok ? '#46d282' : '#eb5a5a';
    roundRect(ctx, ml + pw + 20, mt + ph - 40, 150, 32, 8); ctx.fill();
    ctx.fillStyle = '#0a0e14'; ctx.font = 'bold 18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(ok ? 'PASS' : 'FAIL', ml + pw + 95, mt + ph - 18);
    ctx.textAlign = 'left';
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function renderIslandProfileNums(d) {
  const p = d.profile, si = d.scaleInfo, f = d.fit;
  const rows = [
    ['合并尺寸', `${d.width} × ${d.height} px`],
    ['统一像素间距 px/py', `${si.px.toExponential(4)} / ${si.py.toExponential(4)} m`],
    ['A/B 目标列数', `${si.colsA} / ${si.colsB}`],
    ['实际使用标尺 zA/zB', `${si.zAUsed.toFixed(5)} / ${si.zBUsed.toFixed(5)} m`],
  ];
  if (f) {
    rows.push(['镜像拟合 B/A 标尺比 k', `${f.scaleRatioBA.toFixed(4)}（理想 1）`]);
    rows.push(['拟合跨缝坡度 β', `${f.crossSlopeBetaFit.toFixed(4)}`]);
    rows.push(['拟合高度残差 RMS', f.fitRmsHeight.toExponential(3)]);
  }
  rows.push(
    ['左/缝/右割线坡度', `${p.slopeLeft.toFixed(4)} / ${p.slopeCross.toFixed(4)} / ${p.slopeRight.toFixed(4)}`],
    ['原始坡度残差 / 容差', `${p.slopeResidual.toFixed(4)} / ${d.checks ? d.checks.slopeTolerance : '—（相对）'}`],
    ['镜像相关系数', p.mirrorCorrelation.toFixed(4)],
    ['极性（凸起/凹槽）', `${p.polarity}，峰 ${p.peakRelief.toFixed(5)}，谷 ${p.troughRelief.toFixed(5)}`],
  );
  $('i-profile-nums').innerHTML = rows.map(([k, v]) =>
    `<div class="metric"><span>${k}</span><span>${v}</span></div>`).join('');
}

async function runIsland(mode) {
  const fb = $('i-feedback');
  clearFeedback(fb);
  document.querySelectorAll('.invalid').forEach((e) => e.classList.remove('invalid'));
  const btn = mode === 'physical' ? $('i-btn-physical') : $('i-btn-relative');
  btn.disabled = true;
  try {
    const d = await api('/api/join', collectIslandParams(mode));
    islands.result = d;
    $('i-normal').src = d.normalPng;
    $('i-download-normal').href = d.normalPng;
    $('i-download-normal').hidden = false;
    $('i-download-height').href = d.heightPng;
    $('i-download-height').hidden = false;

    const mb = $('i-mode-badge');
    if (d.physical) {
      mb.textContent = '物理合并 · 两岛标尺齐全';
      mb.className = 'badge ok';
    } else {
      mb.textContent = '相对预览 · B 强度为临时假设，非物理';
      mb.className = 'badge warn';
    }
    const pb = $('i-pass-badge');
    if (d.checks) {
      pb.hidden = false;
      pb.textContent = d.checks.passed ? '跨缝核对 PASS' : '跨缝核对 FAIL';
      pb.className = `badge ${d.checks.passed ? 'ok' : 'fail'}`;
    } else {
      pb.hidden = true;
    }

    const ul = $('i-messages');
    ul.innerHTML = '';
    const msgs = d.checks ? d.checks.messages : [d.notice];
    msgs.forEach((m, idx) => {
      const li = document.createElement('li');
      let cls = 'info';
      if (d.checks) cls = d.checks.passed ? 'good' : (idx === 0 ? 'bad' : 'info');
      if (/极性错误|不一致|很可能选反/.test(m)) cls = 'bad';
      if (/一致|相符/.test(m)) cls = 'good';
      li.className = cls;
      li.textContent = m;
      ul.appendChild(li);
    });

    renderProfileCanvas($('i-profilecanvas'), d.profile, d.checks || null);
    renderIslandProfileNums(d);

    islands.dec = await decodeNormalDataUrl(d.normalPng, [
      d.convention.signs.x, d.convention.signs.y, d.convention.signs.z,
    ]);
    islands.light.setDecoded(islands.dec);
    islands.light.render({
      az: parseFloat($('il-az').value), el: parseFloat($('il-el').value),
      ambient: 0.22, spec: 0.25,
    });

    activateIslandRTab(d.physical ? 'profile' : 'normal');
    if (d.physical && d.checks && !d.checks.passed) {
      setFeedback(fb, 'error', '物理合并完成，但跨缝核对未通过：请按上面的信息调整 B 标尺或镜像方向。');
    }
  } catch (err) {
    const msg = err.message || String(err);
    setFeedback(fb, 'error', `${err.code === 'scale_required' ? '缺少高度标尺：' : '合并失败：'}${msg}`);
    (err.missing || []).forEach((m) => markInvalid(m));
    markInvalid(err.field);
    $('i-pass-badge').hidden = true;
  } finally {
    btn.disabled = false;
  }
}

/* ---------------- 标签页与事件绑定 ---------------- */

function bindTabs() {
  document.querySelectorAll('.tabs .tab').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.tabs .tab').forEach((x) => x.classList.toggle('active', x === b));
      document.querySelectorAll('.panel').forEach((p) =>
        p.classList.toggle('active', p.id === `panel-${b.dataset.tab}`));
    });
  });
  document.querySelectorAll('[data-rtab]').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('[data-rtab]').forEach((x) =>
        x.classList.toggle('active', x === b));
      document.querySelectorAll('[data-rpanel]').forEach((p) =>
        p.classList.toggle('active', p.dataset.rpanel === b.dataset.rtab));
    });
  });
  document.querySelectorAll('[data-irtab]').forEach((b) => {
    b.addEventListener('click', () => activateIslandRTab(b.dataset.irtab));
  });
  $('btn-tests-info').addEventListener('click', () => {
    document.querySelector('.tab[data-tab="learn"]').click();
  });
}

function bindSingle() {
  fillConventionSelect($('s-convention'));
  let t = null;
  const schedule = () => {
    clearTimeout(t);
    updatePitchReadout();
    t = setTimeout(runSingleConvert, 280);
  };
  const controls = document.querySelectorAll(
    '#panel-single select, #panel-single input[type=number], #panel-single input[type=checkbox], #panel-single input[type=range]'
  );
  controls.forEach((el) => {
    el.addEventListener('input', () => {
      if (el.id === 's-transfer') updateTransferVisibility();
      schedule();
    });
    el.addEventListener('change', schedule);
  });
  $('s-convert').addEventListener('click', runSingleConvert);

  $('s-sample').addEventListener('change', () => {
    single.uploadDataUrl = null;
    applySampleDefaults($('s-sample').value);
    runSingleConvert();
  });

  $('s-file').addEventListener('change', () => {
    const file = $('s-file').files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      single.uploadDataUrl = reader.result;
      $('s-thumb').src = reader.result;
      const probe = new Image();
      probe.onload = () => {
        $('s-dims').textContent = `${probe.naturalWidth} × ${probe.naturalHeight} px（上传图像）`;
        $('s-dims').dataset.w = probe.naturalWidth;
        $('s-dims').dataset.h = probe.naturalHeight;
        updatePitchReadout();
      };
      probe.src = reader.result;
      runSingleConvert();
    };
    reader.readAsDataURL(file);
  });

  ['l-az', 'l-el', 'l-amb', 'l-spec'].forEach((id) => {
    $(id).addEventListener('input', () => {
      $('l-az-val').textContent = `${$('l-az').value}°`;
      $('l-el-val').textContent = `${$('l-el').value}°`;
      $('l-amb-val').textContent = parseFloat($('l-amb').value).toFixed(2);
      $('l-spec-val').textContent = parseFloat($('l-spec').value).toFixed(2);
      single.light.render(currentLightOpts());
      if (verify.dec) {
        verify.lit.render({
          az: parseFloat($('l-az').value), el: 15.0, ambient: 0.22, spec: 0.0,
        });
      }
    });
  });

  $('s-verify-run').addEventListener('click', runVerify);

  $('s-lightcanvas').addEventListener('mousemove', (ev) => {
    if (!single.dec) return;
    const hit = single.light.pixelEvent(ev);
    if (!hit) return;
    const i = hit.p * 4;
    const e = single.dec.enc;
    $('s-hover').textContent =
      `px(${hit.x},${hit.y})  RGB=(${e[i]},${e[i + 1]},${e[i + 2]})  ` +
      `n_canonical=(${single.dec.nx[hit.p].toFixed(3)}, ${single.dec.ny[hit.p].toFixed(3)}, ${single.dec.nz[hit.p].toFixed(3)})`;
  });

  $('s-download-lit').addEventListener('click', (ev) => {
    if (!single.dec) { ev.preventDefault(); return; }
    $('s-download-lit').href = single.light.downloadUrl();
  });
}

function bindIslands() {
  fillConventionSelect($('i-convention'));
  $('i-convention').value = 'opengl';
  $('i-btn-relative').addEventListener('click', () => runIsland('relative'));
  $('i-btn-physical').addEventListener('click', () => runIsland('physical'));
  $('i-apply-a-z').addEventListener('click', () => {
    $('i-b-z').value = $('i-a-z').value;
    $('i-previewz').value = $('i-a-z').value;
    $('i-b-z').classList.remove('invalid');
  });
  $('i-row').addEventListener('input', () => {
    $('i-row-val').textContent = `${Math.round(parseFloat($('i-row').value) * 100)}%`;
  });
  ['il-az', 'il-el'].forEach((id) =>
    $(id).addEventListener('input', () => {
      if (!islands.dec) return;
      islands.light.render({
        az: parseFloat($('il-az').value), el: parseFloat($('il-el').value),
        ambient: 0.22, spec: 0.25,
      });
    }));
}

async function init() {
  bindTabs();
  bindSingle();
  bindIslands();
  try {
    const cat = await (await fetch('/api/samples')).json();
    single.catalog = cat;
    const sel = $('s-sample');
    cat.samples.forEach((e) => {
      const o = document.createElement('option');
      o.value = e.id; o.textContent = e.label;
      sel.appendChild(o);
    });
    sel.value = 'fbm_rock';
    applySampleDefaults('fbm_rock');
    if (cat.islandTruth) {
      const z = Number(cat.islandTruth.zScale.toFixed(5));
      $('i-a-z').value = z;
      $('i-previewz').value = z;
    }
    updatePitchReadout();
    runSingleConvert();
  } catch (err) {
    setFeedback($('s-feedback'), 'error',
      `无法加载样例目录：${err.message}。请确认通过 ./start.sh 或 python -m hnm.server 启动，而不是直接打开 html 文件。`);
  }
}

document.addEventListener('DOMContentLoaded', init);
