"use strict";

// ---- element handles ----------------------------------------------------
const $ = (id) => document.getElementById(id);
const els = {
  prompt: $("prompt"), steps: $("steps"), seed: $("seed"), size: $("size"),
  guidance: $("guidance"), generate: $("generate"),
  modeButtons: Array.from(document.querySelectorAll(".toggle-btn")),
  alpha: $("alpha"), alphaVal: $("alpha-val"), hint: $("hint"),
  canvas: $("canvas"), display: $("display"), marker: $("marker"),
  spinner: $("spinner"), spinnerText: $("spinner-text"),
  tokens: $("tokens"), legend: $("legend"), legendCap: $("legend-cap"),
  layer: $("layer"), layerField: $("layer-field"),
  includeSpecial: $("include-special"), specialField: $("special-field"),
  source: $("source"), sourceField: $("source-field"),
};

// ---- state --------------------------------------------------------------
const state = {
  sessionId: null, width: 512, mode: "image2text", alpha: 0.6, layer: "",
  includeSpecial: false, source: "score",
  activeToken: null, lastSelf: null, lastI2t: null, baseImage: null,
  layersCross: [], layersSelf: [], crossRes: null, selfRes: null,
};

// all three modes now support per-layer inspection
const LAYER_MODES = new Set(["cross", "image2text", "self"]);

const HINTS = {
  cross: "Click a token below to highlight the image regions that attend to it.",
  self: "Click anywhere on the image to see which regions that spot attends to.",
  image2text: "Click anywhere on the image to color each token by how much that region attends to it.",
};

const LEGEND_CAPTIONS = {
  cross: "attention to selected token",
  image2text: "attention from selected region",
  self: "attention from selected region",
};

const CLICK_MODES = new Set(["self", "image2text"]);

// ---- networking ---------------------------------------------------------
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---- generation ---------------------------------------------------------
async function generate() {
  els.generate.disabled = true;
  showSpinner("Generating image… (first run loads the model)");
  try {
    const data = await postJSON("/api/generate", {
      prompt: els.prompt.value,
      steps: Number(els.steps.value),
      seed: Number(els.seed.value),
      size: Number(els.size.value),
      guidance: Number(els.guidance.value),
    });
    onGenerated(data);
  } catch (err) {
    els.hint.textContent = "⚠ " + err.message;
  } finally {
    hideSpinner();
    els.generate.disabled = false;
  }
}

function onGenerated(data) {
  state.sessionId = data.session_id;
  state.width = data.width;
  state.baseImage = data.image;
  state.activeToken = null;
  state.lastSelf = null;
  state.lastI2t = null;
  state.layer = "";
  state.layersCross = data.layers || [];
  state.layersSelf = data.self_layers || [];
  state.crossRes = data.cross_res || null;
  state.selfRes = data.self_res || null;
  els.canvas.classList.add("ready");
  els.display.src = data.image;
  els.marker.classList.add("hidden");
  renderTokens(data.tokens);
  populateLayers(state.mode);
  updateModeChrome();
  clearChipColors();
  els.hint.textContent = HINTS[state.mode];
}

// cross and image-to-text share the cross-attention layers; self has its own
function layersForMode(mode) {
  return mode === "self" ? state.layersSelf : state.layersCross;
}

function populateLayers(mode) {
  const res = mode === "self" ? state.selfRes : state.crossRes;
  const avgLabel = res ? `Average (${res}² layers)` : "Average";
  els.layer.innerHTML = `<option value="">${avgLabel}</option>`;
  layersForMode(mode).forEach((ly) => {
    const opt = document.createElement("option");
    opt.value = ly.name;
    opt.textContent = ly.label;
    els.layer.appendChild(opt);
  });
  els.layer.value = "";
  state.layer = "";
}

// show/hide the layer dropdown and the include-special checkbox per mode
function updateModeChrome() {
  const ready = Boolean(state.sessionId);
  els.layerField.classList.toggle("hidden", !(ready && LAYER_MODES.has(state.mode)));
  els.specialField.classList.toggle("hidden", !(ready && state.mode === "image2text"));
  els.sourceField.classList.toggle("hidden", !(ready && state.mode === "cross"));
  // legend shown in every mode; gradient matches the colormap of that mode
  els.legend.classList.toggle("hidden", !ready);
  els.legend.classList.toggle("jet", state.mode !== "image2text");
  els.legendCap.textContent = LEGEND_CAPTIONS[state.mode];
}

// ---- tokens (cross-attention) ------------------------------------------
function renderTokens(tokens) {
  els.tokens.innerHTML = "";
  tokens.forEach((tok) => {
    const chip = document.createElement("button");
    chip.className = tok.special ? "chip special" : "chip";
    chip.textContent = tok.label;
    chip.dataset.index = tok.index;
    if (tok.special) chip.title = "special marker token (attention sink)";
    chip.addEventListener("click", () => selectToken(tok.index, chip));
    els.tokens.appendChild(chip);
  });
}

async function selectToken(index, chip) {
  if (!state.sessionId || state.mode !== "cross") return;
  state.activeToken = index;
  markActiveChip(chip);
  try {
    const data = await postJSON("/api/cross", {
      session_id: state.sessionId, token_index: index, alpha: state.alpha,
      layer: state.layer, source: state.source,
    });
    els.display.src = data.overlay;
  } catch (err) {
    els.hint.textContent = "⚠ " + err.message;
  }
}

function markActiveChip(chip) {
  els.tokens.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  if (chip) chip.classList.add("active");
}

// ---- click on image (self-attention OR image-to-text) ------------------
async function onCanvasClick(event) {
  if (!CLICK_MODES.has(state.mode) || !state.sessionId) return;
  const rect = els.display.getBoundingClientRect();
  const relX = (event.clientX - rect.left) / rect.width;
  const relY = (event.clientY - rect.top) / rect.height;
  if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;

  const x = Math.round(clamp01(relX) * state.width);
  const y = Math.round(clamp01(relY) * state.width);
  placeMarker(event, rect);
  if (state.mode === "self") {
    state.lastSelf = { x, y };
    await fetchSelf(x, y);
  } else {
    state.lastI2t = { x, y };
    await fetchImage2Text(x, y);
  }
}

async function fetchSelf(x, y) {
  try {
    const data = await postJSON("/api/self", {
      session_id: state.sessionId, x, y, alpha: state.alpha, layer: state.layer,
    });
    els.display.src = data.overlay;
  } catch (err) {
    els.hint.textContent = "⚠ " + err.message;
  }
}

// ---- image-to-text: color the token chips by attention weight ----------
async function fetchImage2Text(x, y) {
  try {
    const data = await postJSON("/api/image2text", {
      session_id: state.sessionId, x, y, layer: state.layer,
      include_special: state.includeSpecial,
    });
    colorChips(data.tokens);
  } catch (err) {
    els.hint.textContent = "⚠ " + err.message;
  }
}

function colorChips(tokens) {
  const byIndex = {};
  tokens.forEach((t) => { byIndex[t.index] = t.weight; });
  els.tokens.querySelectorAll(".chip").forEach((chip) => {
    const w = byIndex[Number(chip.dataset.index)];
    if (w === undefined) {        // token not part of the score -> leave neutral
      chip.classList.remove("weighted");
      chip.style.background = "";
      chip.style.color = "";
      return;
    }
    chip.classList.add("weighted");
    chip.style.background = weightToColor(w);
    chip.style.color = w > 0.55 ? "#1a0030" : "#f2f0ff";
  });
}

function clearChipColors() {
  els.tokens.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.remove("weighted");
    chip.style.background = "";
    chip.style.color = "";
  });
}

// jet-style ramp matching the legend: indigo -> teal -> lime -> yellow -> red
function weightToColor(w) {
  const stops = [
    [0.00, [46, 22, 177]],   // #2E16B1
    [0.30, [0, 157, 146]],   // #009D92
    [0.55, [189, 244, 0]],   // #bdf400
    [0.78, [255, 203, 0]],   // #FFCB00
    [1.00, [255, 65, 0]],    // #FF4100
  ];
  w = clamp01(w);
  for (let i = 1; i < stops.length; i++) {
    if (w <= stops[i][0]) {
      const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
      const t = (w - p0) / (p1 - p0);
      const c = c0.map((v, k) => Math.round(v + t * (c1[k] - v)));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  return "rgb(255, 65, 0)";
}

function placeMarker(event, rect) {
  const canvasRect = els.canvas.getBoundingClientRect();
  els.marker.style.left = (event.clientX - canvasRect.left) + "px";
  els.marker.style.top = (event.clientY - canvasRect.top) + "px";
  els.marker.classList.remove("hidden");
}

// ---- mode + alpha -------------------------------------------------------
function setMode(mode) {
  state.mode = mode;
  els.modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  els.canvas.classList.toggle("self-mode", CLICK_MODES.has(mode));
  els.tokens.classList.toggle("hidden-tokens", mode === "self");
  populateLayers(mode);     // cross/image2text and self have different layer lists
  updateModeChrome();
  els.marker.classList.add("hidden");
  markActiveChip(null);
  clearChipColors();
  state.activeToken = null;
  state.lastSelf = null;
  state.lastI2t = null;
  if (state.baseImage) els.display.src = state.baseImage;
  els.hint.textContent = state.sessionId ? HINTS[mode] : "Generate an image to begin.";
}

function onIncludeSpecialChange() {
  state.includeSpecial = els.includeSpecial.checked;
  if (state.mode === "image2text" && state.lastI2t) {
    fetchImage2Text(state.lastI2t.x, state.lastI2t.y);
  }
}

function onSourceChange() {
  state.source = els.source.value;
  if (state.mode === "cross" && state.activeToken !== null) {
    const chip = els.tokens.querySelector(`.chip[data-index="${state.activeToken}"]`);
    selectToken(state.activeToken, chip);
  }
}

function onAlphaChange() {
  state.alpha = Number(els.alpha.value);
  els.alphaVal.textContent = state.alpha.toFixed(2);
  if (state.mode === "cross" && state.activeToken !== null) {
    const chip = els.tokens.querySelector(`.chip[data-index="${state.activeToken}"]`);
    selectToken(state.activeToken, chip);
  } else if (state.mode === "self" && state.lastSelf) {
    fetchSelf(state.lastSelf.x, state.lastSelf.y);
  }
  // image-to-text mode does not use the overlay-strength slider.
}

function onLayerChange() {
  state.layer = els.layer.value;
  if (state.mode === "cross" && state.activeToken !== null) {
    const chip = els.tokens.querySelector(`.chip[data-index="${state.activeToken}"]`);
    selectToken(state.activeToken, chip);
  } else if (state.mode === "image2text" && state.lastI2t) {
    fetchImage2Text(state.lastI2t.x, state.lastI2t.y);
  }
}

// ---- helpers ------------------------------------------------------------
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function showSpinner(text) {
  els.spinnerText.textContent = text;
  els.spinner.classList.remove("hidden");
}
function hideSpinner() { els.spinner.classList.add("hidden"); }

// ---- wire up ------------------------------------------------------------
els.generate.addEventListener("click", generate);
els.modeButtons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
els.alpha.addEventListener("input", onAlphaChange);
els.layer.addEventListener("change", onLayerChange);
els.includeSpecial.addEventListener("change", onIncludeSpecialChange);
els.source.addEventListener("change", onSourceChange);
els.canvas.addEventListener("click", onCanvasClick);

// sync the UI to the default mode (Image -> Text) on load
setMode(state.mode);
