// UI helpers for the case explorer: guided tour, linked views, structure search,
// uncertainty controls, trust badges and the tumor-location card.
//
// Plain ES module, no build step. Everything here is DOM/maths only; scene changes
// (visibility, opacity, camera presets) are passed in as callbacks by viewer.js.
import * as THREE from "three";

/* ------------------------------------------------------------------ */
/* Small utilities                                                     */
/* ------------------------------------------------------------------ */

const SEQUENCE_LABELS = { flair: "FLAIR", t1: "T1", t1c: "T1c", t2: "T2" };

export const DEFAULT_ROWS = 5;

export async function fetchJson(url) {
  const response = await fetch(url);
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }
  if (!response.ok) throw new Error((payload && payload.error) || response.statusText || "Request failed");
  return payload;
}

/** Location report + uncertainty availability. Each part is null when unavailable. */
export async function loadExplorerData(id) {
  const base = `/api/cases/${encodeURIComponent(id)}`;
  const [location, uncertainty] = await Promise.all([
    fetchJson(`${base}/location.json`).catch((error) => {
      console.warn("location.json unavailable:", error);
      return null;
    }),
    fetchJson(`${base}/uncertainty`).catch((error) => {
      console.warn("uncertainty unavailable:", error);
      return null;
    }),
  ]);
  return { location, uncertainty };
}

function finite(value) {
  const number = Number(value);
  return value !== null && value !== undefined && value !== "" && Number.isFinite(number) ? number : null;
}

/** Trim and end with exactly one full stop (reasons from the API may already end with one). */
function sentence(text) {
  const trimmed = String(text || "").trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function fmtMl(value) {
  const number = finite(value);
  return number === null ? null : `${number.toFixed(1)} mL`;
}

function isTyping(target) {
  if (!target || !target.tagName) return false;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable === true;
}

/* ------------------------------------------------------------------ */
/* Trust badges + report link + volume card                            */
/* ------------------------------------------------------------------ */

export function renderTrustBadges(location) {
  const qc = document.querySelector("#qc-badge");
  const plausibility = document.querySelector("#plausibility-badge");
  const model = document.querySelector("#model-badge");
  const setChip = (el, text, status, title) => {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("good", "caution", "unreliable");
    if (status) el.classList.add(status);
    el.title = title || "";
  };
  if (!location) {
    setChip(qc, "QC unavailable", null, "No location report for this case.");
    setChip(plausibility, "Plausibility unavailable", null, "");
    return;
  }
  const status = location.qc && location.qc.status;
  const reasons = (location.qc && location.qc.reasons) || [];
  setChip(qc, `QC: ${status || "unknown"}`, ["good", "caution", "unreliable"].includes(status) ? status : null, reasons.join("\n"));

  const plaus = location.plausibility;
  if (plaus && typeof plaus.passed === "boolean") {
    const wt = fmtMl(plaus.wt_ml);
    const floor = fmtMl(plaus.floor_ml);
    const detail = [wt && `whole tumor ${wt}`, floor && `size floor ${floor}`].filter(Boolean).join(", ");
    setChip(plausibility, `Plausibility: ${plaus.passed ? "passed" : "failed"}`, plaus.passed ? "good" : "unreliable", detail);
  } else {
    setChip(plausibility, "Plausibility: unknown", null, "");
  }
  if (model && location.model_note) model.textContent = location.model_note;
}

export function wireReportLink(id) {
  const link = document.querySelector("#download-report");
  if (!link) return;
  link.href = `/api/cases/${encodeURIComponent(id)}/report.html`;
  link.setAttribute("download", `${id}_report.html`);
  link.target = "_blank";
  link.rel = "noopener";
}

/* ------------------------------------------------------------------ */
/* Tumor location card                                                 */
/* ------------------------------------------------------------------ */

function sortedRows(rows) {
  return [...(rows || [])].sort((a, b) => {
    const aOver = a.relation === "overlaps" ? 0 : 1;
    const bOver = b.relation === "overlaps" ? 0 : 1;
    if (aOver !== bOver) return aOver - bOver;
    return (finite(a.distance_mm) ?? 0) - (finite(b.distance_mm) ?? 0);
  });
}

export function locationRowText(row) {
  const where = row.relation === "overlaps" ? "overlaps" : `${(finite(row.distance_mm) ?? 0).toFixed(1)} mm`;
  return `${row.name} — ${where}`;
}

/**
 * One-line summary + top-N nearest structures (with a Show all toggle) + legend.
 * `data` = { withheld, reason, headline, regions: {WT: rows, ET: rows}, marginMm }.
 */
export function renderTumorLocation(data, limit = DEFAULT_ROWS) {
  const summary = document.querySelector("#tumor-summary-line");
  const host = document.querySelector("#tumor-location");
  const toggle = document.querySelector("#location-toggle");
  const legend = document.querySelector("#location-legend");
  if (!host) return;
  let expanded = false;

  const summaryText = data.withheld
    ? `Location withheld: ${sentence(data.reason || "reason not recorded")}`
    : data.headline || "No location summary available.";
  if (summary) summary.textContent = summaryText;

  const render = () => {
    host.replaceChildren();
    if (data.withheld || data.hasSummary === false) {
      if (toggle) toggle.hidden = true;
      if (legend) legend.hidden = true;
      return;
    }
    let overflow = false;
    let any = false;
    for (const [region, title] of [["WT", "Whole tumor"], ["ET", "Enhancing tumor"]]) {
      const rows = sortedRows(data.regions[region]);
      if (rows.length > limit) overflow = true;
      if (rows.length) any = true;
      const shown = expanded ? rows : rows.slice(0, limit);
      const block = document.createElement("div");
      block.className = "location-region";
      const heading = document.createElement("h3");
      heading.textContent = title;
      const list = document.createElement("ul");
      if (!shown.length) {
        const item = document.createElement("li");
        item.textContent = "None within range.";
        list.appendChild(item);
      }
      for (const row of shown) {
        const item = document.createElement("li");
        item.textContent = locationRowText(row);
        list.appendChild(item);
      }
      block.append(heading, list);
      host.appendChild(block);
    }
    if (toggle) {
      toggle.hidden = !overflow;
      toggle.textContent = expanded ? "Show fewer" : "Show all";
    }
    if (legend) {
      legend.hidden = !any;
      legend.replaceChildren();
      const over = document.createElement("span");
      over.className = "overlaps";
      over.textContent = "Overlaps tumor";
      const near = document.createElement("span");
      near.className = "near";
      near.textContent = `Within ${(finite(data.marginMm) ?? 10).toFixed(0)} mm`;
      legend.append(over, near);
    }
  };
  if (toggle) {
    toggle.onclick = () => {
      expanded = !expanded;
      render();
    };
  }
  render();
}

/* ------------------------------------------------------------------ */
/* Camera animation                                                    */
/* ------------------------------------------------------------------ */

let activeAnimation = null;
const cancelHooked = new WeakSet();

export function capturePose(camera, controls) {
  return {
    position: camera.position.clone(),
    target: controls.target.clone(),
    near: camera.near,
    far: camera.far,
  };
}

export function applyPose(camera, controls, pose) {
  camera.position.copy(pose.position);
  controls.target.copy(pose.target);
  camera.near = pose.near;
  camera.far = pose.far;
  camera.updateProjectionMatrix();
  controls.update();
}

export function cancelCameraAnimation() {
  if (activeAnimation) cancelAnimationFrame(activeAnimation.frame);
  activeAnimation = null;
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Glide the camera to `pose`: target lerps, the orbit direction slerps (no cutting through the head). */
export function animatePose(camera, controls, pose, { ms = 1000 } = {}) {
  cancelCameraAnimation();
  if (!cancelHooked.has(controls)) {
    cancelHooked.add(controls);
    controls.addEventListener("start", cancelCameraAnimation);
  }
  const from = capturePose(camera, controls);
  const offsetFrom = from.position.clone().sub(from.target);
  const offsetTo = pose.position.clone().sub(pose.target);
  const radiusFrom = offsetFrom.length() || 1;
  const radiusTo = offsetTo.length() || 1;
  const dirFrom = offsetFrom.clone().normalize();
  const dirTo = offsetTo.clone().normalize();
  const rotation = new THREE.Quaternion().setFromUnitVectors(dirFrom, dirTo);
  const identity = new THREE.Quaternion();
  const q = new THREE.Quaternion();
  const started = performance.now();
  const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const duration = reduce ? 0 : ms;
  const state = { frame: 0 };
  activeAnimation = state;
  const step = (now) => {
    if (activeAnimation !== state) return;
    const raw = duration > 0 ? Math.min((now - started) / duration, 1) : 1;
    const t = easeInOut(raw);
    q.copy(identity).slerp(rotation, t);
    const radius = radiusFrom + (radiusTo - radiusFrom) * t;
    const target = from.target.clone().lerp(pose.target, t);
    camera.position.copy(target).add(dirFrom.clone().applyQuaternion(q).multiplyScalar(radius));
    controls.target.copy(target);
    camera.near = from.near + (pose.near - from.near) * t;
    camera.far = from.far + (pose.far - from.far) * t;
    camera.updateProjectionMatrix();
    controls.update();
    if (raw < 1) {
      state.frame = requestAnimationFrame(step);
    } else {
      activeAnimation = null;
      applyPose(camera, controls, pose);
    }
  };
  state.frame = requestAnimationFrame(step);
}

/**
 * Run `mutate` (which may move the camera instantly, e.g. applyPreset), then
 * rewind the camera and glide to wherever `mutate` left it.
 */
export function transitionWithCamera(camera, controls, mutate, options = {}) {
  cancelCameraAnimation();
  const before = capturePose(camera, controls);
  mutate();
  const after = capturePose(camera, controls);
  applyPose(camera, controls, before);
  animatePose(camera, controls, after, options);
}

/* ------------------------------------------------------------------ */
/* Linked views: geometry                                              */
/* ------------------------------------------------------------------ */

const inverseCache = new WeakMap();

/** World (mm) → voxel index via the inverted 4×4 affine. Returns null if singular. */
export function worldToVoxel(affine, world, shape) {
  let inverse = inverseCache.get(affine);
  if (inverse === undefined) {
    const matrix = new THREE.Matrix4();
    matrix.set(...affine.flat());
    inverse = Math.abs(matrix.determinant()) < 1e-12 ? null : matrix.clone().invert();
    inverseCache.set(affine, inverse);
  }
  if (!inverse) return null;
  const v = new THREE.Vector3(world.x, world.y, world.z).applyMatrix4(inverse);
  const ijk = [v.x, v.y, v.z].map((value, axis) => {
    const rounded = Math.round(value);
    return Math.min(Math.max(rounded, 0), shape[axis] - 1);
  });
  return ijk.every(Number.isFinite) ? ijk : null;
}

/** Same orientation convention as the slice PNGs (see recon/slices.orient_ras_slice). */
export function orientedPixelToIjk(axis, index, row, col, shape) {
  const [nx, ny, nz] = shape;
  if (axis === 2) return [nx - 1 - col, ny - 1 - row, index];
  if (axis === 1) return [nx - 1 - col, index, nz - 1 - row];
  return [index, ny - 1 - col, nz - 1 - row];
}

export function ijkToPixel(axis, ijk, shape) {
  const [nx, ny, nz] = shape;
  if (axis === 2) return { col: nx - 1 - ijk[0], row: ny - 1 - ijk[1] };
  if (axis === 1) return { col: nx - 1 - ijk[0], row: nz - 1 - ijk[2] };
  return { col: ny - 1 - ijk[1], row: nz - 1 - ijk[2] };
}

/* ------------------------------------------------------------------ */
/* Tumor contour from overlay alpha                                    */
/* ------------------------------------------------------------------ */

const contourCache = new WeakMap();

/**
 * Edge of the overlay's alpha mask, drawn as a crisp 1-px outline.
 * mode "fill": the overlay is itself a ring of contours (label masks), so holes are
 *   filled first and the outer edge of the filled region is traced (whole-tumor outline).
 * mode "threshold": soft maps (probability); trace the edge where alpha >= threshold.
 */
export function contourCanvas(img, { mode = "fill", threshold = 8, color = [94, 234, 212] } = {}) {
  const key = `${mode}:${threshold}:${color.join(",")}`;
  let perImage = contourCache.get(img);
  if (perImage && perImage.key === key) return perImage.canvas;
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  const out = document.createElement("canvas");
  out.width = Math.max(width, 1);
  out.height = Math.max(height, 1);
  if (!width || !height) return out;
  const work = document.createElement("canvas");
  work.width = width;
  work.height = height;
  const wctx = work.getContext("2d", { willReadFrequently: true });
  wctx.drawImage(img, 0, 0);
  const pixels = wctx.getImageData(0, 0, width, height).data;
  const inside = new Uint8Array(width * height);
  for (let i = 0; i < inside.length; i += 1) inside[i] = pixels[i * 4 + 3] >= threshold ? 1 : 0;

  let region = inside;
  if (mode === "fill") {
    // Flood the exterior from the border; everything not reached is "inside".
    const outside = new Uint8Array(width * height);
    const stack = [];
    const push = (x, y) => {
      if (x < 0 || y < 0 || x >= width || y >= height) return;
      const at = y * width + x;
      if (outside[at] || inside[at]) return;
      outside[at] = 1;
      stack.push(at);
    };
    for (let x = 0; x < width; x += 1) {
      push(x, 0);
      push(x, height - 1);
    }
    for (let y = 0; y < height; y += 1) {
      push(0, y);
      push(width - 1, y);
    }
    while (stack.length) {
      const at = stack.pop();
      const x = at % width;
      const y = (at - x) / width;
      push(x + 1, y);
      push(x - 1, y);
      push(x, y + 1);
      push(x, y - 1);
    }
    region = new Uint8Array(width * height);
    for (let i = 0; i < region.length; i += 1) region[i] = outside[i] ? 0 : 1;
  }

  const octx = out.getContext("2d");
  const image = octx.createImageData(width, height);
  const has = (x, y) => x >= 0 && y >= 0 && x < width && y < height && region[y * width + x] === 1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!region[y * width + x]) continue;
      if (has(x - 1, y) && has(x + 1, y) && has(x, y - 1) && has(x, y + 1)) continue;
      const at = (y * width + x) * 4;
      image.data[at] = color[0];
      image.data[at + 1] = color[1];
      image.data[at + 2] = color[2];
      image.data[at + 3] = 255;
    }
  }
  octx.putImageData(image, 0, 0);
  perImage = { key, canvas: out };
  contourCache.set(img, perImage);
  return out;
}

/** Contour settings for each overlay source. */
export function contourOptionsFor(source) {
  if (source === "prob") return { mode: "threshold", threshold: 108 }; // alpha = 255 * p * 0.85 → p ≈ 0.5
  return { mode: "fill", threshold: 8 };
}

/* ------------------------------------------------------------------ */
/* ACS strip (axial / coronal / sagittal at the crosshair)             */
/* ------------------------------------------------------------------ */

const ACS_SIZE = 288;
const imageCache = new Map();
const acsTokens = [0, 0, 0];
const acsLayout = new WeakMap();

function loadImage(url) {
  if (imageCache.has(url)) return imageCache.get(url);
  const promise = new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`image failed: ${url}`));
    img.src = url;
  });
  imageCache.set(url, promise);
  promise.catch(() => imageCache.delete(url));
  if (imageCache.size > 90) imageCache.delete(imageCache.keys().next().value);
  return promise;
}

export function acsIndices(volume, crosshair) {
  const mid = volume.shape.map((length) => Math.floor((length - 1) / 2));
  return crosshair ? crosshair.map((value, axis) => Math.min(Math.max(value, 0), volume.shape[axis] - 1)) : mid;
}

/**
 * Paint the three orthogonal views (FLAIR + overlay + contour + crosshair).
 * opts: { volume, crosshair, sliceUrl(axis, index), overlayUrl(axis, index), overlayOpacity, source }
 */
export function paintAcsStrip(opts) {
  const root = document.querySelector("#acs-strip");
  if (!root || !opts.volume || !opts.volume.shape) return;
  const { volume } = opts;
  const indices = acsIndices(volume, opts.crosshair);
  root.querySelectorAll("figure[data-acs]").forEach((figure) => {
    const axis = Number(figure.dataset.acs);
    const canvas = figure.querySelector("canvas");
    if (!canvas || ![0, 1, 2].includes(axis)) return;
    const index = indices[axis];
    const caption = figure.querySelector("figcaption");
    if (caption) {
      if (!caption.dataset.base) caption.dataset.base = caption.textContent;
      caption.textContent = `${caption.dataset.base} · ${index}`;
    }
    const token = ++acsTokens[axis];
    figure.querySelectorAll(".acs-error").forEach((el) => el.remove());
    Promise.all([
      loadImage(opts.sliceUrl(axis, index)),
      loadImage(opts.overlayUrl(axis, index)).catch(() => null),
    ])
      .then(([base, overlay]) => {
        if (acsTokens[axis] !== token) return;
        drawAcs(canvas, base, overlay, axis, indices, volume, opts);
      })
      .catch((error) => {
        console.warn("ACS slice failed:", error);
        if (acsTokens[axis] !== token) return;
        const msg = document.createElement("div");
        msg.className = "acs-error";
        msg.textContent = `Could not load slice: ${error && error.message ? error.message : error}`;
        figure.appendChild(msg);
      });
  });
}

function drawAcs(canvas, base, overlay, axis, indices, volume, opts) {
  const size = ACS_SIZE;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  const extent = (volume.extent_mm && volume.extent_mm[String(axis)]) || [base.width, base.height];
  const aspect = Number(extent[0]) / Math.max(Number(extent[1]), 1e-6);
  const w = aspect >= 1 ? size : size * aspect;
  const h = aspect >= 1 ? size / aspect : size;
  const ox = (size - w) / 2;
  const oy = (size - h) / 2;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(base, ox, oy, w, h);
  if (overlay) {
    if ((opts.overlayOpacity ?? 0.85) > 0) {
      ctx.save();
      ctx.globalAlpha = opts.overlayOpacity ?? 0.85;
      ctx.drawImage(overlay, ox, oy, w, h);
      ctx.restore();
    }
    ctx.drawImage(contourCanvas(overlay, contourOptionsFor(opts.source)), ox, oy, w, h);
  }
  if (opts.crosshair) {
    const { col, row } = ijkToPixel(axis, indices, volume.shape);
    const x = ox + ((col + 0.5) / base.width) * w;
    const y = oy + ((row + 0.5) / base.height) * h;
    ctx.strokeStyle = "#ffee58";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, oy);
    ctx.lineTo(x, oy + h);
    ctx.moveTo(ox, y);
    ctx.lineTo(ox + w, y);
    ctx.stroke();
  }
  acsLayout.set(canvas, { ox, oy, w, h, cols: base.width, rows: base.height, size, axis, indices: indices.slice() });
}

/** Click an ACS view to move the crosshair. `getVolume()` is read lazily (volume may be swapped). */
export function wireAcsStrip(getVolume, onPick) {
  const root = document.querySelector("#acs-strip");
  if (!root) return;
  root.addEventListener("click", (event) => {
    const canvas = event.target.closest("canvas");
    const layout = canvas && acsLayout.get(canvas);
    const volume = getVolume();
    if (!layout || !volume || !volume.shape) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const x = ((event.clientX - rect.left) / rect.width) * layout.size;
    const y = ((event.clientY - rect.top) / rect.height) * layout.size;
    if (x < layout.ox || x > layout.ox + layout.w || y < layout.oy || y > layout.oy + layout.h) return;
    const col = Math.round(((x - layout.ox) / layout.w) * layout.cols - 0.5);
    const row = Math.round(((y - layout.oy) / layout.h) * layout.rows - 0.5);
    const clampedCol = Math.min(Math.max(col, 0), layout.cols - 1);
    const clampedRow = Math.min(Math.max(row, 0), layout.rows - 1);
    const ijk = orientedPixelToIjk(layout.axis, layout.indices[layout.axis], clampedRow, clampedCol, volume.shape)
      .map((value, axis) => Math.min(Math.max(value, 0), volume.shape[axis] - 1));
    onPick(ijk);
  });
}

/* ------------------------------------------------------------------ */
/* Click vs drag on the 3D canvas                                      */
/* ------------------------------------------------------------------ */

/** Calls `onClick(event)` for a primary-button press that did not orbit/drag. */
export function onCanvasClick(canvas, onClick, { maxMove = 5, maxMs = 700 } = {}) {
  let down = null;
  canvas.addEventListener("pointerdown", (event) => {
    down = { x: event.clientX, y: event.clientY, t: performance.now(), button: event.button };
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!down) return;
    const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
    const elapsed = performance.now() - down.t;
    const ok = down.button === 0 && event.button === 0 && moved <= maxMove && elapsed <= maxMs;
    down = null;
    if (ok) onClick(event);
  });
  canvas.addEventListener("pointercancel", () => {
    down = null;
  });
}

/* ------------------------------------------------------------------ */
/* Uncertainty                                                         */
/* ------------------------------------------------------------------ */

/**
 * Wires #uncertainty-modes [data-unc]. `onChange(source)` runs after the active source changes.
 * Returns { get(): source, set(source) }.
 */
export function setupUncertainty(uncertainty, { initial = "model", onChange } = {}) {
  const host = document.querySelector("#uncertainty-modes");
  const diceEl = document.querySelector("#uncertainty-dice");
  const reasons = (uncertainty && uncertainty.reasons) || {};
  const available = {
    model: uncertainty ? uncertainty.has_model !== false : true,
    second: Boolean(uncertainty && uncertainty.has_second),
    prob: Boolean(uncertainty && uncertainty.has_prob),
  };
  let current = available[initial] === false ? "model" : initial;
  const buttons = host ? [...host.querySelectorAll("[data-unc]")] : [];
  const reasonFor = (source) => {
    if (reasons[source]) return reasons[source];
    if (source === "prob") return "Needs the model, not available in this demo";
    if (source === "second") return "No expert ground-truth mask for this case.";
    return "Not available for this case";
  };
  const sync = () => {
    buttons.forEach((button) => {
      const source = button.dataset.unc;
      const ok = available[source] !== false;
      button.disabled = !ok;
      button.title = ok ? "" : reasonFor(source);
      if (source === "prob") {
        button.textContent = ok ? "Probability" : "Needs the model, not available in this demo";
      }
      button.classList.toggle("is-active", source === current);
      button.setAttribute("aria-pressed", String(source === current));
    });
    if (diceEl) {
      const dice = uncertainty && uncertainty.dice;
      if (dice && Object.keys(dice).length) {
        const part = (label, value) => (finite(value) === null ? null : `${label} ${Number(value).toFixed(3)}`);
        diceEl.textContent = `Dice, model vs second mask: ${[part("WT", dice.wt), part("TC", dice.tc), part("ET", dice.et)].filter(Boolean).join(" · ")}`;
      } else if (!available.second) {
        diceEl.textContent = reasonFor("second");
      } else {
        diceEl.textContent = "Dice unavailable until two masks are present.";
      }
    }
  };
  const set = (source) => {
    if (available[source] === false || source === current) return;
    current = source;
    sync();
    if (onChange) onChange(current);
  };
  buttons.forEach((button) => button.addEventListener("click", () => set(button.dataset.unc)));
  sync();
  return { get: () => current, set };
}

/* ------------------------------------------------------------------ */
/* Structure explorer                                                  */
/* ------------------------------------------------------------------ */

export function buildStructureIndex(manifest, names, labelOf) {
  return [...names].map((name) => {
    const entry = (manifest.structures || {})[name] || {};
    const label = labelOf(name, entry);
    const haystack = [name, label, entry.full_name, entry.short_name, entry.legend, entry.group]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return { name, label, group: entry.group || "", haystack, lower: String(label).toLowerCase() };
  });
}

export function searchStructures(index, query, limit = 8) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const scored = [];
  for (const item of index) {
    let score = -1;
    if (item.lower.startsWith(q)) score = 0;
    else if (item.lower.split(/[\s(),/_-]+/).some((word) => word.startsWith(q))) score = 1;
    else if (item.haystack.includes(q)) score = 2;
    if (score >= 0) scored.push({ item, score });
  }
  scored.sort((a, b) => a.score - b.score || a.item.label.localeCompare(b.item.label));
  return scored.slice(0, limit).map((entry) => entry.item);
}

/** Volume (cm³) and distance to tumor for the structure card. */
export function structureCardMeta(name, manifest, location) {
  const entry = (manifest.structures || {})[name] || {};
  const parts = [];
  const volume = finite(entry.volume_cm3);
  if (volume !== null) parts.push(`Volume ${volume.toFixed(2)} cm³`);
  if (entry.group === "Tumor" || entry.source === "model") return parts.join(" · ");

  if (location && location.withheld) {
    parts.push("Distance to tumor withheld");
    return parts.join(" · ");
  }
  const summary = manifest.tumor_location || (location && location.tumor_location) || null;
  const label = entry.full_name;
  const find = (rows) => (rows || []).find((row) => row.name === name || (label && row.name === label));
  const described = [];
  const describe = (title, row) => {
    if (!row) return;
    described.push(row.relation === "overlaps" ? `overlaps ${title}` : `${(finite(row.distance_mm) ?? 0).toFixed(1)} mm from ${title}`);
  };
  if (summary) {
    describe("whole tumor", find(summary.WT));
    describe("enhancing tumor", find(summary.ET));
  }
  const tract = (manifest.tract_proximity || {})[name];
  if (tract) {
    if (tract.wt_overlaps) described.push("overlaps whole tumor");
    else if (finite(tract.wt_distance_mm) !== null) described.push(`${Number(tract.wt_distance_mm).toFixed(1)} mm from whole tumor`);
    if (tract.et_overlaps) described.push("overlaps enhancing tumor");
    else if (finite(tract.et_distance_mm) !== null) described.push(`${Number(tract.et_distance_mm).toFixed(1)} mm from enhancing tumor`);
  }
  if (described.length) parts.push([...new Set(described)].join("; "));
  else if (summary) parts.push(`No tumor proximity recorded (margin ${(finite(summary.margin_mm) ?? 10).toFixed(0)} mm)`);
  if (entry.source === "atlas-estimated") parts.push("atlas-estimated");
  return parts.join(" · ");
}

/**
 * Autocomplete + structure card + display-mode buttons.
 * Scene effects are delegated: onMode(name, mode), onClear(), onShowAll().
 */
export function createStructureExplorer({ index, manifest, getLocation, labelOf, onMode, onClear, onShowAll }) {
  const input = document.querySelector("#structure-search");
  const suggest = document.querySelector("#structure-suggest");
  const card = document.querySelector("#structure-card");
  const nameEl = document.querySelector("#structure-card-name");
  const metaEl = document.querySelector("#structure-card-meta");
  const modeButtons = [...document.querySelectorAll("#structure-card [data-mode]")];
  const state = { selected: null, mode: null, matches: [], active: -1 };

  const closeSuggest = () => {
    state.matches = [];
    state.active = -1;
    if (suggest) {
      suggest.hidden = true;
      suggest.replaceChildren();
    }
    if (input) input.setAttribute("aria-expanded", "false");
  };
  const paintActive = () => {
    if (!suggest) return;
    [...suggest.querySelectorAll("button")].forEach((button, position) => {
      button.setAttribute("aria-selected", String(position === state.active));
      if (position === state.active) button.scrollIntoView({ block: "nearest" });
    });
  };
  const renderSuggest = () => {
    if (!suggest || !input) return;
    state.matches = searchStructures(index, input.value);
    state.active = state.matches.length ? 0 : -1;
    suggest.replaceChildren();
    if (!state.matches.length) {
      suggest.hidden = true;
      return;
    }
    state.matches.forEach((item) => {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = item.group ? `${item.label} · ${item.group}` : item.label;
      button.addEventListener("click", () => select(item.name, { apply: true }));
      li.appendChild(button);
      suggest.appendChild(li);
    });
    suggest.hidden = false;
    input.setAttribute("aria-expanded", "true");
    paintActive();
  };
  const paintModes = () => {
    modeButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.mode === state.mode);
      button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
    });
  };

  function select(name, { apply = true, mode = "context" } = {}) {
    const entry = (manifest.structures || {})[name];
    if (!entry) return;
    state.selected = name;
    const label = labelOf(name, entry);
    if (nameEl) nameEl.textContent = label;
    if (metaEl) metaEl.textContent = structureCardMeta(name, manifest, getLocation());
    if (card) card.hidden = false;
    if (input && apply) input.value = label;
    closeSuggest();
    state.mode = null;
    if (apply) setMode(mode);
    else paintModes();
  }

  function setMode(mode) {
    if (!state.selected) return;
    state.mode = mode;
    paintModes();
    onMode(state.selected, mode);
  }

  function clear() {
    state.selected = null;
    state.mode = null;
    if (card) card.hidden = true;
    if (input) input.value = "";
    closeSuggest();
    paintModes();
    onClear();
  }

  function showAll() {
    state.mode = null;
    paintModes();
    onShowAll();
  }

  if (input) {
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-controls", "structure-suggest");
    input.setAttribute("aria-expanded", "false");
    input.addEventListener("input", renderSuggest);
    input.addEventListener("focus", () => {
      if (input.value.trim()) renderSuggest();
    });
    input.addEventListener("blur", () => setTimeout(closeSuggest, 120));
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        if (!state.matches.length) return;
        event.preventDefault();
        const step = event.key === "ArrowDown" ? 1 : -1;
        state.active = (state.active + step + state.matches.length) % state.matches.length;
        paintActive();
      } else if (event.key === "Enter") {
        const item = state.matches[Math.max(state.active, 0)];
        if (item) {
          event.preventDefault();
          select(item.name, { apply: true });
          input.blur();
        }
      } else if (event.key === "Escape") {
        event.stopPropagation();
        if (state.matches.length) closeSuggest();
        else {
          clear();
          input.blur();
        }
      }
    });
  }
  if (suggest) suggest.addEventListener("pointerdown", (event) => event.preventDefault());
  modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));

  return {
    select,
    setMode,
    clear,
    showAll,
    get selected() {
      return state.selected;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Shortcuts + help overlay                                            */
/* ------------------------------------------------------------------ */

export function setupShortcuts({ tour, explorer }) {
  const help = document.querySelector("#shortcuts-help");
  const openBtn = document.querySelector("#shortcuts-help-btn");
  const closeBtn = document.querySelector("#shortcuts-close");
  const setHelp = (open) => {
    if (!help) return;
    help.hidden = !open;
    if (open && closeBtn) closeBtn.focus();
    else if (!open && openBtn && document.activeElement === closeBtn) openBtn.focus();
  };
  if (openBtn) openBtn.addEventListener("click", () => setHelp(help ? help.hidden : false));
  if (closeBtn) closeBtn.addEventListener("click", () => setHelp(false));
  if (help) {
    help.addEventListener("click", (event) => {
      if (event.target === help) setHelp(false);
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const typing = isTyping(event.target);
    if (event.key === "Escape") {
      if (help && !help.hidden) setHelp(false);
      else if (tour && tour.active) tour.stop();
      else if (explorer && explorer.selected) explorer.clear();
      return;
    }
    if (typing) return;
    if (tour && tour.active) {
      if (event.key === "ArrowRight") {
        event.preventDefault();
        tour.next();
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        tour.back();
        return;
      }
    }
    const key = event.key.toLowerCase();
    if (key === "i" && explorer && explorer.selected) explorer.setMode("isolate");
    else if (key === "g" && explorer && explorer.selected) explorer.setMode("ghost");
    else if (key === "a" && explorer) explorer.showAll();
    else if (event.key === "?") setHelp(help ? help.hidden : false);
  });
}

/* ------------------------------------------------------------------ */
/* Guided tour                                                         */
/* ------------------------------------------------------------------ */

function nameOf(label, manifest) {
  const entry = (manifest.structures || {})[label];
  return (entry && (entry.full_name || entry.legend)) || label;
}

function labelVolumeMl(label, location, manifest) {
  const volumes = (location && location.volumes_ml) || {};
  const key = label === "enhancing" ? "et_ml" : `${label}_ml`;
  const fromLocation = finite(volumes[key]);
  if (fromLocation !== null) return fromLocation;
  const fromManifest = finite(((manifest.structures || {})[label] || {}).volume_cm3);
  return fromManifest;
}

/**
 * Six steps; every sentence is assembled from location.json / manifest / volume metadata.
 * Step.nearestNames lists structure names the viewer may reveal in step 6.
 */
export function buildTourSteps({ location, manifest, volume }) {
  const loc = location || {};
  const fallbackSeqs = Object.keys((volume && volume.windows) || {}).map((key) => SEQUENCE_LABELS[key] || key.toUpperCase());
  const sequences = (loc.sequences && loc.sequences.length ? loc.sequences : fallbackSeqs).filter(Boolean);

  const brain = (manifest.structures || {}).brain;
  const brainMl = brain ? finite(brain.volume_cm3) : null;

  const labels = (loc.segmentation_labels && loc.segmentation_labels.length
    ? loc.segmentation_labels
    : Object.entries(manifest.structures || {}).filter(([, entry]) => entry.group === "Tumor").map(([name]) => name));
  const labelParts = labels.map((label) => {
    const ml = fmtMl(labelVolumeMl(label, loc, manifest));
    return ml ? `${nameOf(label, manifest)} (${ml})` : nameOf(label, manifest);
  });

  const volumes = loc.volumes_ml || {};
  const wt = fmtMl(volumes.wt_ml);
  const tc = fmtMl(volumes.tc_ml);
  const et = fmtMl(volumes.et_ml);
  const volumeParts = [wt && `whole tumor ${wt}`, tc && `tumor core ${tc}`, et && `enhancing ${et}`].filter(Boolean);
  const plaus = loc.plausibility;
  const floorNote = plaus && plaus.passed === false && fmtMl(plaus.floor_ml)
    ? ` Below the ${fmtMl(plaus.floor_ml)} size floor.`
    : "";

  const nearest = (loc.nearest && loc.nearest.length ? loc.nearest : (loc.tumor_location && loc.tumor_location.WT) || []).slice();
  const top = sortedRows(nearest).slice(0, DEFAULT_ROWS);
  const modelNote = loc.model_note ? ` (${loc.model_note})` : "";

  const locationCaption = loc.withheld
    ? `Location withheld: ${sentence(loc.withheld_reason || "reason not recorded")}`
    : loc.headline || (top[0] ? `Nearest structure: ${locationRowText(top[0])}.` : "No location summary available.");
  const nearestCaption = loc.withheld
    ? `Nearest structures withheld: ${sentence(loc.withheld_reason || "reason not recorded")}`
    : top.length
      ? `Nearest structures: ${top.map(locationRowText).join("; ")}.`
      : "No nearby structures were recorded for this case.";

  return [
    {
      id: "sequences",
      title: "MRI sequences",
      preset: "glass-tumor",
      highlight: "#acs-strip",
      caption: sequences.length
        ? `${sequences.length} MRI sequence${sequences.length === 1 ? "" : "s"} loaded: ${sequences.join(", ")}. Compare them in the slice panel.`
        : "MRI sequences for this case are shown in the slice panel.",
    },
    {
      id: "brain",
      title: "Brain extracted",
      preset: "glass-tumor",
      highlight: "#groups",
      caption: brainMl !== null
        ? `Brain extracted from the scan and shown as a glass shell (${brainMl.toFixed(0)} cm³).`
        : "Brain extracted from the scan and shown as a glass shell.",
    },
    {
      id: "segmentation",
      title: "Tumor segmentation",
      preset: "tumor-focus",
      highlight: '[data-group="Tumor"]',
      caption: labelParts.length
        ? `Segmented labels: ${labelParts.join(", ")}${modelNote}.`
        : `No segmentation labels were recorded${modelNote}.`,
    },
    {
      id: "volume",
      title: "Tumor volume",
      preset: "tumor-focus",
      highlight: "#volume-card",
      caption: volumeParts.length ? `Volumes: ${volumeParts.join(", ")}.${floorNote}` : "Tumor volumes were not recorded for this case.",
    },
    {
      id: "location",
      title: "Tumor location",
      preset: "glass-tumor",
      highlight: "#tumor-card",
      caption: locationCaption,
    },
    {
      id: "nearest",
      title: "Nearest structures",
      preset: "glass-tumor",
      highlight: "#tumor-location",
      caption: nearestCaption,
      nearestNames: loc.withheld ? [] : top.map((row) => row.name),
    },
  ];
}

const highlighted = new Set();
function setHighlight(selector) {
  highlighted.forEach((el) => {
    el.style.boxShadow = "";
    el.style.borderRadius = "";
  });
  highlighted.clear();
  if (!selector) return;
  document.querySelectorAll(selector).forEach((el) => {
    el.style.boxShadow = "0 0 0 2px rgba(94, 234, 212, 0.85)";
    el.style.borderRadius = "6px";
    highlighted.add(el);
  });
  const first = document.querySelector(selector);
  if (first && first.scrollIntoView) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

/**
 * Guided tour controller bound to #tour-overlay & friends.
 * getSteps() is evaluated on each start; onStep(step, index, steps) applies scene changes.
 */
export function createTour({ getSteps, onStep, onExit }) {
  const overlay = document.querySelector("#tour-overlay");
  const caption = document.querySelector("#tour-caption");
  const fill = document.querySelector("#tour-progress-fill");
  const backBtn = document.querySelector("#tour-back");
  const nextBtn = document.querySelector("#tour-next");
  const exitBtn = document.querySelector("#tour-exit");
  const startBtn = document.querySelector("#start-tour");
  const state = { steps: [], index: 0, active: false };

  const render = () => {
    const step = state.steps[state.index];
    if (!step || !overlay) return;
    overlay.hidden = false;
    if (caption) {
      caption.replaceChildren();
      const title = document.createElement("strong");
      title.textContent = `${state.index + 1}/${state.steps.length} · ${step.title}. `;
      const text = document.createElement("span");
      text.textContent = step.caption;
      caption.append(title, text);
    }
    if (fill) fill.style.width = `${((state.index + 1) / state.steps.length) * 100}%`;
    if (backBtn) backBtn.disabled = state.index === 0;
    if (nextBtn) nextBtn.textContent = state.index === state.steps.length - 1 ? "Finish" : "Next";
    setHighlight(step.highlight);
    if (onStep) onStep(step, state.index, state.steps);
  };

  const api = {
    get active() {
      return state.active;
    },
    start() {
      state.steps = getSteps();
      state.index = 0;
      state.active = true;
      render();
    },
    goTo(index) {
      if (!state.active) return;
      state.index = Math.min(Math.max(index, 0), state.steps.length - 1);
      render();
    },
    next() {
      if (!state.active) return;
      if (state.index >= state.steps.length - 1) api.stop();
      else api.goTo(state.index + 1);
    },
    back() {
      if (state.active) api.goTo(state.index - 1);
    },
    stop() {
      if (!state.active) return;
      state.active = false;
      if (overlay) overlay.hidden = true;
      setHighlight(null);
      if (onExit) onExit();
    },
  };
  if (startBtn) startBtn.addEventListener("click", () => api.start());
  if (nextBtn) nextBtn.addEventListener("click", () => api.next());
  if (backBtn) backBtn.addEventListener("click", () => api.back());
  if (exitBtn) exitBtn.addEventListener("click", () => api.stop());
  return api;
}
