import * as THREE from "three";

const TUMOR_COMPARTMENTS = ["edema", "necrosis", "enhancing"];
const TOUCH_RELATIONS = new Set(["overlaps", "within_10mm"]);

function structureEntries(manifest) {
  return Object.entries(manifest?.structures || {}).filter(([, entry]) => entry && !entry.volume_only);
}

function namesInGroup(manifest, group) {
  return structureEntries(manifest)
    .filter(([, entry]) => entry.group === group)
    .map(([name]) => name);
}

function resolveShowNames(show, manifest) {
  const structures = manifest?.structures || {};
  const found = [];
  for (const item of show || []) {
    const entry = structures[item];
    if (entry && !entry.volume_only) {
      found.push(item);
      continue;
    }
    for (const [name, row] of Object.entries(structures)) {
      if (row && !row.volume_only && row.group === item) found.push(name);
    }
  }
  return [...new Set(found)];
}

function pushStep(steps, manifest, step) {
  if (!resolveShowNames(step.show, manifest).length) return;
  steps.push(step);
}

export function buildTour(manifest) {
  const steps = [];
  const structures = manifest?.structures || {};

  const wholeShow = ["Tissue", "Brainstem & Cerebellum", "Brain"].filter(
    (group) => namesInGroup(manifest, group).length > 0,
  );
  pushStep(steps, manifest, {
    title: "Whole brain",
    text: "Brain tissue meshes present in this case.",
    show: wholeShow,
    focus: null,
  });

  const compartments = TUMOR_COMPARTMENTS.filter((name) => structures[name] && !structures[name].volume_only);
  const tumorNames = namesInGroup(manifest, "Tumor");
  if (compartments.length || tumorNames.length) {
    const show = compartments.length ? compartments : tumorNames;
    const volumeNames = compartments.length ? compartments : tumorNames;
    const total = volumeNames.reduce((sum, name) => sum + (Number(structures[name].volume_cm3) || 0), 0);
    const parts = volumeNames.map((name) => {
      const label = structures[name].legend || structures[name].full_name || name;
      const volume = Number(structures[name].volume_cm3);
      return Number.isFinite(volume) ? `${label}: ${volume.toFixed(1)} cm³` : label;
    });
    pushStep(steps, manifest, {
      title: "Tumor",
      text: [`Total volume ${total.toFixed(1)} cm³.`, ...parts].join(" "),
      show,
      focus: show[0] || null,
    });
  }

  const touchRows = [];
  const seen = new Set();
  const summary = manifest?.tumor_location || {};
  for (const region of ["WT", "ET"]) {
    for (const row of summary[region] || []) {
      if (!TOUCH_RELATIONS.has(row.relation)) continue;
      if (!row.name || seen.has(row.name)) continue;
      if (!structures[row.name] || structures[row.name].volume_only) continue;
      seen.add(row.name);
      touchRows.push(row);
      if (touchRows.length >= 5) break;
    }
    if (touchRows.length >= 5) break;
  }
  if (touchRows.length) {
    const touchNames = touchRows.map((row) => row.name);
    const show = [...new Set([...touchNames, ...compartments])];
    const lines = touchRows.map((row) => {
      if (row.relation === "overlaps") return `${row.name}: overlaps`;
      return `${row.name}: ${Number(row.distance_mm).toFixed(1)} mm`;
    });
    pushStep(steps, manifest, {
      title: "What the tumor touches",
      text: `${lines.join(". ")}.`,
      show,
      focus: touchNames[0] || null,
    });
  }

  const deep = namesInGroup(manifest, "Subcortical");
  pushStep(steps, manifest, {
    title: "Deep structures",
    text: "Subcortical structures in this case.",
    show: ["Subcortical"],
    focus: deep[0] || null,
  });

  const vents = namesInGroup(manifest, "Ventricles");
  pushStep(steps, manifest, {
    title: "Ventricles",
    text: "Ventricles in this case.",
    show: ["Ventricles"],
    focus: vents[0] || null,
  });

  const tracts = namesInGroup(manifest, "Tracts");
  pushStep(steps, manifest, {
    title: "Fibre pathways",
    text: "Pathways are atlas estimates, not measured from this scan.",
    show: ["Tracts"],
    focus: tracts[0] || null,
  });

  return steps;
}

function framingPose(objects) {
  const present = objects.filter(Boolean);
  const box = new THREE.Box3();
  present.forEach((mesh) => box.expandByObject(mesh));
  if (box.isEmpty()) {
    return {
      position: new THREE.Vector3(120, -180, 80),
      target: new THREE.Vector3(0, 0, 0),
      near: 0.1,
      far: 10000,
    };
  }
  const target = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 100;
  return {
    position: target.clone().add(new THREE.Vector3(size * 0.9, -size * 1.3, size * 0.55)),
    target,
    near: Math.max(size / 1000, 0.01),
    far: size * 20,
  };
}

export class TourPlayer {
  constructor({
    steps,
    byName,
    camera,
    controls,
    setVisible,
    syncLegend,
    onChange,
    onExit,
  }) {
    this.steps = steps || [];
    this.byName = byName;
    this.camera = camera;
    this.controls = controls;
    this.setVisible = setVisible;
    this.syncLegend = syncLegend;
    this.onChange = onChange;
    this.onExit = onExit;
    this.index = 0;
    this.active = false;
    this.savedVisibility = null;
    this._motionRaf = 0;
    this._motionCancelled = false;
    this._onControlsStart = () => {
      this._motionCancelled = true;
    };
  }

  get step() {
    return this.steps[this.index] || null;
  }

  get count() {
    return this.steps.length;
  }

  start() {
    if (!this.steps.length) return false;
    this.savedVisibility = new Map();
    this.byName.forEach((mesh, name) => this.savedVisibility.set(name, Boolean(mesh.visible)));
    this.active = true;
    this.index = 0;
    this.applyStep(0);
    return true;
  }

  next() {
    if (!this.active) return;
    if (this.index >= this.steps.length - 1) {
      this.exit();
      return;
    }
    this.applyStep(this.index + 1);
  }

  back() {
    if (!this.active || this.index <= 0) return;
    this.applyStep(this.index - 1);
  }

  exit() {
    if (!this.active && !this.savedVisibility) return;
    this._cancelMotion();
    this._restoreGlassOpacity();
    if (this.savedVisibility) {
      this.savedVisibility.forEach((visible, name) => this.setVisible(name, visible));
      this.savedVisibility = null;
    }
    this.active = false;
    if (this.syncLegend) this.syncLegend();
    if (this.onExit) this.onExit();
  }

  applyStep(index) {
    if (index < 0 || index >= this.steps.length) return;
    this.index = index;
    const step = this.steps[index];
    const showNames = this._resolveShow(step.show);
    for (const mesh of this._glassMeshes()) showNames.add(mesh.name);
    this.byName.forEach((_mesh, name) => this.setVisible(name, showNames.has(name)));
    this._setGlassOpacity(index === 0 ? null : 0.22);
    if (this.syncLegend) this.syncLegend();
    const focusMesh = step.focus ? this.byName.get(step.focus) : null;
    const frameMeshes = focusMesh
      ? [focusMesh]
      : [...showNames].map((name) => this.byName.get(name)).filter(Boolean);
    this._animateTo(framingPose(frameMeshes));
    if (this.onChange) this.onChange(step, index, this.steps.length);
  }

  _glassMeshes() {
    return [...this.byName.values()].filter(
      (mesh) => mesh.userData.glass || mesh.name === "brain",
    );
  }

  _setGlassOpacity(opacity) {
    for (const mesh of this._glassMeshes()) {
      const base = Number(mesh.userData.baseOpacity);
      const value = opacity == null
        ? (Number.isFinite(base) ? base : 0.7)
        : opacity;
      if (mesh.material?.uniforms?.opacity) {
        mesh.material.uniforms.opacity.value = value;
      } else if (mesh.material) {
        mesh.material.opacity = value;
        mesh.material.transparent = value < 0.999;
      }
    }
  }

  _restoreGlassOpacity() {
    this._setGlassOpacity(null);
  }

  _resolveShow(show) {
    const names = new Set();
    for (const item of show || []) {
      if (this.byName.has(item)) {
        names.add(item);
        continue;
      }
      this.byName.forEach((mesh, name) => {
        if (mesh.userData.group === item) names.add(name);
      });
    }
    return names;
  }

  _cancelMotion() {
    this._motionCancelled = true;
    if (this._motionRaf) {
      cancelAnimationFrame(this._motionRaf);
      this._motionRaf = 0;
    }
    if (this._listeningControls) {
      this.controls.removeEventListener("start", this._onControlsStart);
      this._listeningControls = false;
    }
  }

  _animateTo(pose, durationMs = 800) {
    this._cancelMotion();
    this._motionCancelled = false;
    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    const endPos = pose.position;
    const endTarget = pose.target;
    this.camera.near = pose.near;
    this.camera.far = pose.far;
    this.camera.updateProjectionMatrix();
    this.controls.addEventListener("start", this._onControlsStart);
    this._listeningControls = true;
    const started = performance.now();
    const tick = (now) => {
      if (this._motionCancelled) {
        this._cancelMotion();
        return;
      }
      const u = Math.min(1, (now - started) / durationMs);
      const eased = u * u * (3 - 2 * u);
      this.camera.position.lerpVectors(startPos, endPos, eased);
      this.controls.target.lerpVectors(startTarget, endTarget, eased);
      this.controls.update();
      if (u < 1) {
        this._motionRaf = requestAnimationFrame(tick);
      } else {
        this._cancelMotion();
        this._motionCancelled = false;
      }
    };
    this._motionRaf = requestAnimationFrame(tick);
  }
}
