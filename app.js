// PeelPixel: original code. Model: BiRefNet_lite (MIT), runtime: transformers.js (Apache-2.0).
import { AutoModel, AutoProcessor, RawImage, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
export const PRO_URL = 'https://jhodges6.gumroad.com/l/ijsmuq?utm_source=peelpixel';
const UNLOCK_CODE = 'PEEL-PRO-2026'; // honor-system unlock shown on the Gumroad content page
const GPU_MODEL = 'onnx-community/BiRefNet_lite-ONNX'; // MIT, best quality, needs WebGPU
const CPU_MODEL = 'onnx-community/ormbg-ONNX';        // Apache-2.0, runs on any CPU via WASM
env.allowLocalModels = false;
const $ = s => document.querySelector(s);
const status = t => { $('#status').textContent = t; };
$('#proLink').href = PRO_URL;
const isPro = () => localStorage.getItem('peelpixel_pro') === '1';
function applyPro() {
  const on = isPro();
  $('#proLocked').hidden = on; $('#proOn').hidden = !on;
  $('#bg').querySelector('option[value=image]').disabled = !on;
}
$('#unlockBtn').onclick = () => {
  const ok = $('#unlockCode').value.trim().toUpperCase() === UNLOCK_CODE;
  if (ok) localStorage.setItem('peelpixel_pro', '1');
  $('#unlockMsg').textContent = ok ? 'Pro unlocked. Thank you!' : 'That code did not match. Check your Gumroad receipt.';
  applyPro();
};
let bgImg = null;
$('#bgfile').onchange = async e => { const f = e.target.files[0]; if (!f) return; bgImg = new Image(); bgImg.src = URL.createObjectURL(f); await bgImg.decode(); $('#bg').value = 'image'; render(); };

let modelP = null;
async function pickDevice() {
  try { const a = navigator.gpu && await navigator.gpu.requestAdapter(); if (a && a.features.has('shader-f16')) return { device: 'webgpu', dtype: 'fp16' }; if (a) return { device: 'webgpu', dtype: 'fp32' }; } catch {}
  return { device: 'wasm', dtype: 'fp32' };
}
function loadModel() {
  if (!modelP) modelP = (async () => {
    status('Downloading AI model (first time only)…');
    const progress_callback = p => { if (p.status === 'progress' && p.file.endsWith('.onnx')) status(`Downloading AI model… ${p.progress.toFixed(0)}%`); };
    let opts = await pickDevice(), model, id = GPU_MODEL, processor = null;
    if (opts.device === 'webgpu') {
      try { model = await AutoModel.from_pretrained(GPU_MODEL, { ...opts, progress_callback }); model.__size = 1024; }
      catch (e) { console.warn('GPU path failed, using CPU', e); model = null; }
    }
    if (!model) {
      opts = { device: 'wasm', dtype: 'uint8' }; id = CPU_MODEL;
      model = await AutoModel.from_pretrained(CPU_MODEL, { ...opts, progress_callback });
      processor = await AutoProcessor.from_pretrained(CPU_MODEL);
    }
    console.log('device', JSON.stringify(opts), id);
    status(`Model ready (${opts.device}). Processing…`);
    return { model, processor };
  })().catch(e => { modelP = null; throw e; });
  return modelP;
}

// ImageNet-normalized NCHW float tensor at S x S (S=1024 on GPU, smaller on CPU to fit WASM memory)
function toInput(image, S) {
  const r = image.rgb ? image.rgb() : image;
  const c = document.createElement('canvas'); c.width = S; c.height = S;
  const src = document.createElement('canvas'); src.width = image.width; src.height = image.height;
  src.getContext('2d').putImageData(new ImageData(r.rgba().data, image.width, image.height), 0, 0);
  const ctx = c.getContext('2d'); ctx.drawImage(src, 0, 0, S, S);
  const d = ctx.getImageData(0, 0, S, S).data, m = [0.485, 0.456, 0.406], sd = [0.229, 0.224, 0.225];
  const f = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) for (let k = 0; k < 3; k++) f[k * S * S + i] = (d[i * 4 + k] / 255 - m[k]) / sd[k];
  return new Tensor('float32', f, [1, 3, S, S]);
}

const items = []; let cur = null;

// returns {img: HTMLCanvasElement original, mask: Uint8ClampedArray alpha (full res)}
async function removeBg(file) {
  const { model, processor } = await loadModel();
  const url = URL.createObjectURL(file);
  const image = await RawImage.fromURL(url);
  let out;
  if (processor) { const { pixel_values } = await processor(image); out = await model({ [model.sessions.model.inputNames[0]]: pixel_values }); }
  else out = await model({ input_image: toInput(image, model.__size) });
  const logits = out[Object.keys(out)[0]];
  const t = (processor ? logits[0] : logits[0].sigmoid()).mul(255).to('uint8');
  const maskImg = await RawImage.fromTensor(t).resize(image.width, image.height);
  const orig = document.createElement('canvas'); orig.width = image.width; orig.height = image.height;
  const im = new Image(); im.src = url; await im.decode(); orig.getContext('2d').drawImage(im, 0, 0);
  const alpha = new Uint8ClampedArray(image.width * image.height);
  const md = maskImg.data, ch = maskImg.channels;
  for (let i = 0; i < alpha.length; i++) alpha[i] = md[i * ch];
  return { orig, alpha, w: image.width, h: image.height };
}

function composite(it, bgMode = $('#bg').value, color = $('#bgcolor').value) {
  const c = document.createElement('canvas'); c.width = it.w; c.height = it.h;
  const ctx = c.getContext('2d');
  const src = it.orig.getContext('2d').getImageData(0, 0, it.w, it.h);
  for (let i = 0; i < it.alpha.length; i++) src.data[i * 4 + 3] = it.alpha[i];
  const tmp = document.createElement('canvas'); tmp.width = it.w; tmp.height = it.h; tmp.getContext('2d').putImageData(src, 0, 0);
  if (bgMode === 'color') { ctx.fillStyle = color; ctx.fillRect(0, 0, it.w, it.h); }
  if (bgMode === 'image' && bgImg && isPro()) { const sc = Math.max(it.w / bgImg.width, it.h / bgImg.height); ctx.drawImage(bgImg, (it.w - bgImg.width * sc) / 2, (it.h - bgImg.height * sc) / 2, bgImg.width * sc, bgImg.height * sc); }
  ctx.drawImage(tmp, 0, 0); return c;
}

function render() {
  if (!cur) return;
  const v = $('#view'); v.width = cur.w; v.height = cur.h;
  v.getContext('2d').drawImage(composite(cur), 0, 0);
  const th = cur.el.querySelector('img'); const s = composite(cur, 'transparent'); th.src = s.toDataURL('image/png');
}

function select(it) { cur = it; items.forEach(x => x.el.classList.toggle('sel', x === it)); $('#editor').hidden = false; render(); }

async function handle(files) {
  files = [...files].filter(f => f.type.startsWith('image/'));
  if (!files.length) return;
  $('#batch h2').hidden = false;
  for (const [n, f] of files.entries()) {
    const el = document.createElement('div'); el.className = 'item'; el.innerHTML = `<img alt=""><div>${f.name.replace(/[<>&]/g, '')}: working…</div>`;
    $('#list').append(el);
    try {
      status(`Processing ${n + 1} of ${files.length}…`);
      const r = await removeBg(f); const it = { ...r, name: f.name.replace(/\.[^.]+$/, ''), el };
      items.push(it); el.querySelector('div').textContent = f.name + ': done'; el.onclick = () => select(it);
      if (!cur || files.length === 1) select(it); else render();
    } catch (e) { console.error(e); el.querySelector('div').textContent = f.name + ': failed (' + e.message + ')'; }
  }
  status(`Done: ${items.length} image(s). Click one to edit, then download.`);
  $('#dlall').hidden = items.length < 2;
  window.__peelDone = true;
}

// drag & drop
const drop = $('#drop');
drop.onclick = () => $('#file').click();
$('#file').onchange = e => handle(e.target.files);
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('hover'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', e => handle(e.dataTransfer.files));

// brush: erase (alpha->0) / restore (alpha->255) in full-res coordinates
let painting = false;
function paint(e) {
  if (!painting || !cur) return;
  const v = $('#view'), r = v.getBoundingClientRect();
  const x = (e.clientX - r.left) * cur.w / r.width, y = (e.clientY - r.top) * cur.h / r.height;
  const rad = +$('#brush').value * cur.w / r.width / 2, val = document.querySelector('input[name=mode]:checked').value === 'erase' ? 0 : 255;
  const x0 = Math.max(0, x - rad | 0), x1 = Math.min(cur.w - 1, x + rad | 0), y0 = Math.max(0, y - rad | 0), y1 = Math.min(cur.h - 1, y + rad | 0);
  for (let j = y0; j <= y1; j++) for (let i = x0; i <= x1; i++) if ((i - x) ** 2 + (j - y) ** 2 <= rad * rad) cur.alpha[j * cur.w + i] = val;
  render();
}
const view = $('#view');
view.addEventListener('pointerdown', e => { painting = true; paint(e); });
view.addEventListener('pointermove', paint);
window.addEventListener('pointerup', () => painting = false);
$('#bg').onchange = $('#bgcolor').oninput = render;

function download(it) {
  composite(it).toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = it.name + '-peelpixel.png'; a.click(); }, 'image/png');
}
$('#dl').onclick = () => cur && download(cur);
$('#dlall').onclick = () => items.forEach((it, i) => setTimeout(() => download(it), i * 400));
applyPro();
window.__peel = { handle, items, composite };
