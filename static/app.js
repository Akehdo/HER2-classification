"use strict";

const $ = (id) => document.getElementById(id);
const state = {files: [], selected: 0, busy: false, health: null, response: null, nextId: 1};
const DEFAULT_CLASSES = ["class_0", "class_1+", "class_2+", "class_3+"];
const MAX_FILES = 16, MAX_FILE_SIZE = 10 * 1024 * 1024, MAX_TOTAL_SIZE = 32 * 1024 * 1024 - 64 * 1024;
const modelNames = {resnet18: "ResNet18", resnet50: "ResNet50", efficientnet_b0: "EfficientNet-B0"};
const shortClass = (name) => name.startsWith("class_") ? name.slice(6) : name;
const modelName = (name) => modelNames[name] || name;
const formatSize = (size) => size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} МБ` : `${Math.ceil(size / 1024)} КБ`;
const formatTime = (ms) => ms >= 1000 ? `${(ms / 1000).toFixed(2)} с` : `${Math.round(ms)} мс`;

function icon(name, className = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  if (className) svg.setAttribute("class", className);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function notify(message) {
  $("notice-text").textContent = message;
  $("notice").hidden = false;
  $("announcer").textContent = message;
}

function renderProbabilities(result) {
  const names = state.health?.model?.classes || DEFAULT_CLASSES;
  $("probability-list").replaceChildren();
  for (const name of names) {
    const row = document.createElement("div");
    row.className = "probability-row" + (result?.predicted_class === name ? " winner" : "");
    const label = document.createElement("span");
    label.className = "class-label";
    label.textContent = shortClass(name);
    const track = document.createElement("div");
    track.className = "probability-track";
    const fill = document.createElement("div");
    fill.className = "probability-fill";
    const probability = result?.probabilities[name];
    fill.style.width = probability === undefined ? "0%" : `${probability * 100}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = probability === undefined ? "—" : `${(probability * 100).toFixed(1)}%`;
    row.append(label, track, value);
    $("probability-list").append(row);
  }
}

function renderResults() {
  const result = state.files[state.selected]?.result;
  $("prediction-class").textContent = result ? shortClass(result.predicted_class) : "—";
  $("prediction-check").hidden = !result;
  $("result-indicator").textContent = state.busy ? "Обработка…" : result ? "Анализ завершён" : "Нет данных";
  $("result-indicator").classList.toggle("complete", !!result && !state.busy);
  $("result-description").hidden = !result;
  $("result-description").textContent = result ? `Вероятность ${(result.probability * 100).toFixed(1)}%` : "";
  $("result-timing").hidden = !state.response;
  $("inference-time").textContent = state.response ? formatTime(state.response.inference_ms) : "—";
  $("export-button").disabled = !state.response || state.busy;
  $("export-count").textContent = state.response ? `· ${state.response.results.length}` : "";
  renderProbabilities(result);
}

function updateActions() {
  const ready = state.health?.status === "ready";
  $("analyze-button").disabled = !state.files.length || !ready || state.busy;
  $("analyze-label").textContent = state.busy ? "Анализируем…" : state.files.length > 1 ? `Анализировать · ${state.files.length}` : "Анализировать";
  $("analyze-button").title = !ready ? "Сначала подключите обученную модель" : !state.files.length ? "Выберите изображение патча" : "";
  $("add-files").hidden = !state.files.length;
  $("add-files").disabled = state.busy || state.files.length >= MAX_FILES;
  $("choose-files").disabled = state.busy;
  $("file-input").disabled = state.busy;
  $("remove-file").disabled = state.busy;
  $("processing-overlay").hidden = !state.busy;
  $("result-panel").setAttribute("aria-busy", String(state.busy));
  document.querySelector(".input-panel").classList.toggle("processing", state.busy);
}

function renderQueue() {
  $("queue").replaceChildren();
  $("queue").hidden = state.files.length < 2;
  state.files.forEach((entry, index) => {
    const button = document.createElement("button");
    button.className = "queue-item";
    button.type = "button";
    button.setAttribute("aria-label", `Патч ${index + 1}: ${entry.file.name}`);
    button.setAttribute("aria-pressed", String(index === state.selected));
    button.title = entry.file.name;
    button.disabled = state.busy;
    const img = new Image();
    img.alt = "";
    img.src = entry.url;
    img.onerror = () => img.replaceWith(icon("image"));
    button.append(img);
    if (entry.result) button.append(icon("check", "queue-check"));
    button.addEventListener("click", () => { state.selected = index; renderFiles(); });
    $("queue").append(button);
  });
}

function renderFiles() {
  const entry = state.files[state.selected];
  $("upload-empty").hidden = !!entry;
  $("patch-preview").hidden = true;
  $("unsupported-preview").hidden = true;
  $("expand-preview").hidden = true;
  $("remove-file").hidden = !entry;
  $("file-strip").hidden = !entry;
  $("input-tag").hidden = !entry;
  $("input-tag").textContent = entry ? `${state.selected + 1} / ${state.files.length}` : "";
  $("image-dimensions").hidden = true;
  if (entry) {
    $("file-name").textContent = entry.file.name;
    $("file-meta").textContent = formatSize(entry.file.size);
    const preview = $("patch-preview");
    preview.onload = () => {
      if (state.files[state.selected]?.id !== entry.id) return;
      preview.hidden = false;
      $("unsupported-preview").hidden = true;
      $("expand-preview").hidden = false;
      $("image-dimensions").hidden = false;
      $("image-dimensions").textContent = `${preview.naturalWidth} × ${preview.naturalHeight} px`;
    };
    preview.onerror = () => {
      if (state.files[state.selected]?.id !== entry.id) return;
      preview.hidden = true;
      $("expand-preview").hidden = true;
      $("unsupported-preview").hidden = false;
    };
    preview.src = entry.url;
  } else {
    $("patch-preview").removeAttribute("src");
    $("file-name").textContent = "";
    $("file-meta").textContent = "";
    $("image-dimensions").textContent = "";
  }
  renderQueue();
  renderResults();
  updateActions();
}

function invalidateResults() {
  state.response = null;
  for (const entry of state.files) entry.result = null;
}

function addFiles(files) {
  if (state.busy) return;
  const incoming = Array.from(files);
  if (!incoming.length) return;
  $("notice").hidden = true;
  if (state.files.length + incoming.length > MAX_FILES) return notify("За один анализ можно загрузить до 16 патчей.");
  const invalid = incoming.find((file) => !/\.(png|jpe?g|tiff?|webp)$/i.test(file.name));
  if (invalid) return notify(`Файл «${invalid.name}»: выберите PNG, JPEG, TIFF или WebP.`);
  const tooBig = incoming.find((file) => file.size > MAX_FILE_SIZE || file.size === 0);
  if (tooBig) return notify(`Файл «${tooBig.name}» пустой или превышает 10 МБ.`);
  const total = [...state.files.map((entry) => entry.file), ...incoming].reduce((sum, file) => sum + file.size, 0);
  if (total > MAX_TOTAL_SIZE) return notify("Общий размер загрузки превышает 32 МБ. Уменьшите количество или размер патчей.");
  const firstNew = state.files.length;
  invalidateResults();
  for (const file of incoming) state.files.push({id: state.nextId++, file, url: URL.createObjectURL(file), result: null});
  state.selected = firstNew;
  renderFiles();
  $("announcer").textContent = `Добавлено патчей: ${incoming.length}. Всего: ${state.files.length}.`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {signal: AbortSignal.timeout(8000), cache: "no-store"});
    if (!response.ok) throw new Error();
    state.health = await response.json();
    if (state.response && state.response.model.version !== state.health.model?.version) invalidateResults();
    const ready = state.health.status === "ready";
    $("model-status").textContent = ready ? `${modelName(state.health.model.name)} · ${state.health.model.device.toUpperCase()}` : "Модель не подключена";
    $("model-status").title = ready ? state.health.model.checkpoint : "";
    $("model-status").classList.toggle("ready", ready);
    $("model-banner").hidden = ready;
    $("banner-message").textContent = state.health.message?.startsWith("Не удалось") ? "Не удалось загрузить модель." : "Модель не подключена.";
  } catch {
    state.health = null;
    $("model-status").textContent = "Нет связи";
    $("model-status").title = "";
    $("model-status").classList.remove("ready");
    $("model-banner").hidden = false;
    $("banner-message").textContent = "Сервер недоступен.";
  }
  updateActions();
  renderResults();
}

async function analyze() {
  if (state.busy || !state.files.length || state.health?.status !== "ready") return;
  state.busy = true;
  invalidateResults();
  $("notice").hidden = true;
  renderFiles();
  const form = new FormData();
  for (const entry of state.files) form.append("files", entry.file, entry.file.name);
  try {
    const response = await fetch("/predict", {method: "POST", body: form, signal: AbortSignal.timeout(180000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Не удалось обработать запрос.");
    if (data.results.length !== state.files.length) throw new Error("Сервер вернул неполный результат. Повторите анализ.");
    state.response = data;
    data.results.forEach((result, index) => { state.files[index].result = result; });
    $("announcer").textContent = `Анализ завершён. Обработано патчей: ${data.results.length}.`;
  } catch (error) {
    notify(error.name === "TimeoutError" ? "Сервер не ответил за 3 минуты. Проверьте его состояние и повторите анализ." : error.message || "Нет связи с сервером.");
  } finally {
    state.busy = false;
    renderFiles();
  }
}

function exportCsv() {
  if (!state.response) return;
  const {model, results, inference_ms} = state.response;
  // Имена файлов остаются текстом при открытии CSV в Excel.
  const cell = (value) => {
    let text = String(value ?? "");
    if (/^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = [["filename", "width", "height", "predicted_class", ...model.classes.map((name) => `prob_${name}`), "model", "checkpoint", "checkpoint_version", "batch_inference_ms"]];
  for (const result of results) rows.push([result.filename, result.width, result.height, result.predicted_class, ...model.classes.map((name) => result.probabilities[name]), model.name, model.checkpoint, model.version, inference_ms]);
  const content = "\uFEFF" + rows.map((row) => row.map(cell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([content], {type: "text/csv;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = `her2-results-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

for (const id of ["choose-files", "add-files"]) $(id).addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => { addFiles(event.target.files); event.target.value = ""; });
$("analyze-button").addEventListener("click", analyze);
$("export-button").addEventListener("click", exportCsv);
$("dismiss-notice").addEventListener("click", () => { $("notice").hidden = true; });
$("remove-file").addEventListener("click", () => {
  if (state.busy) return;
  const removed = state.files.splice(state.selected, 1)[0];
  if (removed) URL.revokeObjectURL(removed.url);
  state.selected = Math.max(0, Math.min(state.selected, state.files.length - 1));
  invalidateResults();
  renderFiles();
});

let dragDepth = 0;
$("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); if (!state.busy) { dragDepth++; $("drop-overlay").hidden = false; } });
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); event.dataTransfer.dropEffect = state.busy ? "none" : "copy"; });
$("drop-zone").addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-overlay").hidden = true; });
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-overlay").hidden = true; addFiles(event.dataTransfer.files); });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

document.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
document.querySelectorAll("dialog").forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog && (event.clientX < dialog.getBoundingClientRect().left || event.clientX > dialog.getBoundingClientRect().right || event.clientY < dialog.getBoundingClientRect().top || event.clientY > dialog.getBoundingClientRect().bottom)) dialog.close(); }));
$("expand-preview").addEventListener("click", () => { $("full-preview").src = state.files[state.selected].url; $("image-dialog").showModal(); });
$("refresh-health").addEventListener("click", checkHealth);
window.addEventListener("beforeunload", () => { for (const entry of state.files) URL.revokeObjectURL(entry.url); });
renderFiles();
checkHealth();
setInterval(() => { if (!document.hidden && !state.busy) checkHealth(); }, 20000);
