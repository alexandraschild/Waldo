'use strict';
const models = [
  {name:'Qwen2-VL-2B',acc:.448,ausc:[.752,.716,.845,.826,.827],pvs:[.368,.368,.290,.307,.300]},
  {name:'Qwen2-VL-7B',acc:.562,ausc:[.786,.733,.850,.830,.851],pvs:[.385,.397,.353,.348,.334]},
  {name:'Qwen2.5-VL-3B',acc:.536,ausc:[.830,.814,.832,.823,.846],pvs:[.368,.352,.361,.319,.352]},
  {name:'Qwen2.5-VL-7B',acc:.604,ausc:[.858,.853,.870,.864,.871],pvs:[.382,.397,.359,.351,.349]},
  {name:'Qwen3-VL-2B',acc:.466,ausc:[.803,.799,.827,.817,.836],pvs:[.414,.376,.416,.394,.445]},
  {name:'Qwen3-VL-4B',acc:.600,ausc:[.836,.820,.855,.851,.775],pvs:[.454,.475,.475,.454,.429]},
  {name:'Qwen3-VL-8B',acc:.627,ausc:[.842,.832,.866,.858,.683],pvs:[.428,.414,.414,.384,.398]},
  {name:'Llama-3.2-11B-V',acc:.499,ausc:[.569,.578,.719,.704,.777],pvs:[.493,.483,.477,.456,.475]},
  {name:'LLaVA-1.5-7B',acc:.352,ausc:[.723,.589,.673,.593,.751],pvs:[.307,.317,.365,.334,.306]},
  {name:'Pixtral-12B',acc:.499,ausc:[.521,.507,.707,.694,.890],pvs:[.432,.483,.422,.450,.349]}
];
let mode = 'padding', position = 'center', metric = 'ausc';
const scale = document.querySelector('#scale');
const scene = document.querySelector('#scene');
const canvas = document.querySelector('#scene-canvas');
function updateScene() {
  const value = Number(scale.value);
  document.querySelector('#scale-value').value = value.toFixed(1) + '×';
  document.querySelector('#canvas-size').value = value.toFixed(1) + '× ' + (mode === 'padding' ? 'canvas' : 'pixel dimensions');
  const percentage = mode === 'padding' ? 100 / value : 100;
  scene.style.width = percentage + '%';
  scene.style.height = percentage + '%';
  canvas.classList.toggle('top-left', mode === 'padding' && position === 'top-left');
  document.querySelector('.position-control').disabled = mode === 'pixel';
  document.querySelector('#patch-grid').style.backgroundSize = `${34/value}px ${34/value}px`;
  document.querySelector('#mode-description').textContent = mode === 'padding'
    ? 'Padding adds a blank canvas around the unchanged scene. The image occupies less of the model’s input.'
    : 'Pixel scaling spreads the same information across more pixels. The scene keeps its relative size; resizing this preview to fit the display makes its appearance stay similar.';
}
scale.addEventListener('input', updateScene);
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  mode = button.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  updateScene();
}));
document.querySelectorAll('[data-position]').forEach(button => button.addEventListener('click', () => {
  position = button.dataset.position;
  document.querySelectorAll('[data-position]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  updateScene();
}));
document.querySelector('#show-grid').addEventListener('change', event => document.querySelector('#patch-grid').classList.toggle('visible', event.target.checked));

const conditions = ['1D padding · center','1D padding · top-left','2D padding · center','2D padding · top-left','pixel interpolation'];
function renderResults() {
  const condition = Number(document.querySelector('#condition').value);
  const isAusc = metric === 'ausc';
  document.querySelector('#chart-title').textContent = `${isAusc ? 'Robustness' : 'Prediction instability'} under ${conditions[condition]}`;
  document.querySelector('#chart-units').textContent = isAusc ? 'AUSC · 0–1 · higher is more robust' : 'PVS · 0–1 · fraction of baseline errors corrected';
  const highest = Math.max(...models.map(m => m[metric][condition]));
  document.querySelector('#results-chart').replaceChildren(...models.map(model => {
    const value = model[metric][condition];
    const row = document.createElement('div');
    row.className = 'chart-row' + (value === highest ? ' featured' : '');
    row.innerHTML = `<span class="chart-model">${model.name}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill${model.name.startsWith('Qwen') && value !== highest ? ' qwen' : ''}" style="width:${value*100}%;${!isAusc ? 'background:#d18379' : ''}"></div></div><span class="chart-value">${value.toFixed(3)}</span><span class="chart-baseline" title="Original accuracy">${(model.acc*100).toFixed(1)}% acc.</span>`;
    row.setAttribute('aria-label', `${model.name}: ${metric.toUpperCase()} ${value.toFixed(3)}, original accuracy ${(model.acc*100).toFixed(1)} percent`);
    return row;
  }));
}
document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
  metric = button.dataset.metric;
  document.querySelectorAll('[data-metric]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  renderResults();
}));
document.querySelector('#condition').addEventListener('change', renderResults);
document.querySelector('#copy-citation').addEventListener('click', async event => {
  const button = event.currentTarget;
  const text = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied!';
    document.querySelector('#copy-status').textContent = 'BibTeX citation copied to clipboard.';
    setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 2200);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    button.textContent = 'Select & copy';
    document.querySelector('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
  }
});
updateScene();
renderResults();
