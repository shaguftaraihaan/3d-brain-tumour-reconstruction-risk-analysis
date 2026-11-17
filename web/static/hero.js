// Landing-page hero: a fully procedural, artificial brain (no patient data, no dataset).
// Two folded hemispheres, cerebellum, brainstem, deep nuclei and an illustrative tumor are
// generated in code with ridged noise. Falls back to the static screenshot if WebGL fails.
import * as THREE from "three";

const host = document.getElementById("brain");

// ---- small seeded 3D value noise -------------------------------------------------------
function makeNoise(seed = 7) {
  const P = new Uint8Array(512), perm = Array.from({ length: 256 }, (_, i) => i);
  let s = seed;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 255; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  for (let i = 0; i < 512; i++) P[i] = perm[i & 255];
  const val = (x, y, z) => P[P[P[x & 255] + (y & 255)] + (z & 255)] / 255;
  const f = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  return (x, y, z) => {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const xf = x - xi, yf = y - yi, zf = z - zi, u = f(xf), v = f(yf), w = f(zf);
    const l = (a, b, t) => a + (b - a) * t;
    return l(
      l(l(val(xi, yi, zi), val(xi + 1, yi, zi), u), l(val(xi, yi + 1, zi), val(xi + 1, yi + 1, zi), u), v),
      l(l(val(xi, yi, zi + 1), val(xi + 1, yi, zi + 1), u), l(val(xi, yi + 1, zi + 1), val(xi + 1, yi + 1, zi + 1), u), v),
      w
    );
  };
}
const noise = makeNoise(11);
const ridged = (x, y, z) => 1 - Math.abs(noise(x, y, z) * 2 - 1);

// ---- geometry builders ---------------------------------------------------------------
function hemisphere(side) {
  const g = new THREE.SphereGeometry(1, 260, 190);
  const p = g.attributes.position, v = new THREE.Vector3();
  const R = { x: 47, y: 52, z: 72 };
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    // base ellipsoid; flatten the lower surface a little, bulge the temporal region
    let x = v.x * R.x, y = v.y * R.y, z = v.z * R.z;
    if (y < 0) y *= 0.72;
    x *= 1 - 0.16 * Math.max(0, z / 72);   // narrower frontal pole
    const dir = new THREE.Vector3(v.x, v.y, v.z).normalize();
    // folds: domain-warped ridged noise, two octaves
    const wx = x + 14 * noise(x * 0.05 + 3, y * 0.05, z * 0.05), wy = y + 14 * noise(x * 0.05, y * 0.05 + 9, z * 0.05), wz = z + 14 * noise(x * 0.05, y * 0.05, z * 0.05 + 5);
    const gyri = ridged(wx * 0.075, wy * 0.075, wz * 0.075) * 0.55 + ridged(wx * 0.17, wy * 0.17, wz * 0.17) * 0.45;
    const amp = 9.5;
    const d = Math.pow(gyri, 2.2) * amp;
    x += dir.x * d * R.x / 52; y += dir.y * d; z += dir.z * d * R.z / 52;
    // flat medial wall, with the longitudinal fissure between the hemispheres
    const sx = side < 0 ? -1 : 1;
    let lx = x * sx;                      // positive = lateral
    lx = Math.max(lx, 1.5 - Math.max(0, (1 - Math.abs(y) / 40)) * 0.0);
    p.setXYZ(i, sx * (lx + 3), y + 4, z);
  }
  g.computeVertexNormals();
  return g;
}

function cerebellum() {
  const g = new THREE.SphereGeometry(1, 110, 80), p = g.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    let x = v.x * 40, y = v.y * 16, z = v.z * 26;
    const fold = Math.abs(Math.sin(y * 1.6 + noise(x * 0.1, y * 0.1, z * 0.1) * 3)) * 1.4;
    const dir = v.clone().normalize();
    x += dir.x * fold; y += dir.y * fold; z += dir.z * fold;
    p.setXYZ(i, x, y - 40, z - 50);
  }
  g.computeVertexNormals();
  return g;
}

function bumpyBlob(r, k, seed, detail = 64) {
  const g = new THREE.SphereGeometry(1, detail, Math.round(detail * 0.75)), p = g.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const b = 1 + k * (noise(v.x * 1.7 + seed, v.y * 1.7, v.z * 1.7) - 0.5) * 2;
    p.setXYZ(i, v.x * r * b, v.y * r * b, v.z * r * b);
  }
  g.computeVertexNormals();
  return g;
}

// ---- scene ---------------------------------------------------------------------------
async function start(el) {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 1, 2000);
  camera.position.set(0, 14, 340);
  scene.add(new THREE.AmbientLight(0x8890ff, 0.35));

  // Glowing surface with a rim light and a height gradient; uReveal sweeps it into view.
  const surface = (low, high, strength, opacity) =>
    new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.FrontSide,
      uniforms: {
        uLow: { value: new THREE.Color(low) }, uHigh: { value: new THREE.Color(high) },
        uStrength: { value: strength }, uOpacity: { value: opacity },
        uMin: { value: -62 }, uMax: { value: 70 }, uReveal: { value: 0 },
        uBand: { value: new THREE.Color(0.4, 1.0, 0.9) },
      },
      vertexShader: `
        varying vec3 vN; varying vec3 vV; varying float vH;
        uniform float uMin; uniform float uMax;
        void main(){
          vec4 mv = modelViewMatrix * vec4(position,1.0);
          vN = normalize(normalMatrix * normal);
          vV = normalize(-mv.xyz);
          vH = clamp((position.y - uMin) / (uMax - uMin), 0.0, 1.0);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vN; varying vec3 vV; varying float vH;
        uniform float uReveal; uniform vec3 uBand;
        uniform vec3 uLow; uniform vec3 uHigh; uniform float uStrength; uniform float uOpacity;
        void main(){
          if (vH > uReveal) discard;
          float band = smoothstep(uReveal - 0.05, uReveal, vH) * step(uReveal, 0.999);
          float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
          float lit = 0.18 + 0.32 * max(dot(normalize(vN), normalize(vec3(-0.4,0.8,0.6))), 0.0);
          vec3 base = mix(uLow, uHigh, vH);
          vec3 col = base * (lit + fres * uStrength) + uBand * band * 1.6;
          gl_FragColor = vec4(col, uOpacity * (0.35 + 0.65 * fres + lit * 0.4));
        }`,
    });
  const glow = (hex, opacity) => new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
  const solid = (hex, emissive, intensity) => new THREE.MeshStandardMaterial({ color: hex, emissive, emissiveIntensity: intensity, roughness: 0.35, metalness: 0.1, transparent: true });

  const brain = new THREE.Group();
  const surfaces = [];
  const add = (geo, mat, pos = [0, 0, 0], order = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(...pos); m.renderOrder = order; brain.add(m); return m; };
  const cortex = () => { const m = surface(0x6f4bff, 0x35e0c4, 1.9, 0.92); surfaces.push(m); return m; };
  const cereb = () => { const m = surface(0x3a2fd0, 0x4aa8ff, 1.5, 0.78); surfaces.push(m); return m; };

  add(hemisphere(-1), cortex());
  add(hemisphere(1), cortex());
  add(cerebellum(), cereb(), [0, 0, 0]);

  // brainstem
  const stem = new THREE.CylinderGeometry(9, 6.5, 52, 40, 1, true);
  stem.translate(0, -38, -14); stem.rotateX(0.0);
  const sm = surface(0x3a2fd0, 0x4aa8ff, 1.5, 0.7); surfaces.push(sm);
  add(stem, sm);

  // illustrative tumor, three layers, in the right hemisphere
  const T = [27, -2, 22];
  add(bumpyBlob(21, 0.16, 2, 56), glow(0xffa53a, 0.32), T, 3);        // swelling
  add(bumpyBlob(12.5, 0.14, 6, 56), solid(0xff3d8a, 0xff2f7a, 1.1), T, 5); // enhancing shell
  add(bumpyBlob(6.5, 0.12, 9, 40), solid(0x2a0a3a, 0x7a1fa8, 0.9), T, 4); // necrotic core

  const group = new THREE.Group();
  group.add(brain);
  const warm = new THREE.PointLight(0xff7a45, 5200, 400, 1.6);
  warm.position.set(...T);
  brain.add(warm);
  scene.add(group);

  // drifting particles
  const n = 700, pts = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const cs = [new THREE.Color(0x35e0c4), new THREE.Color(0x8b7bff), new THREE.Color(0xff5fa8)];
  for (let i = 0; i < n; i++) {
    const r = 130 + Math.random() * 260, a = Math.random() * Math.PI * 2, b = Math.acos(2 * Math.random() - 1);
    pts.set([r * Math.sin(b) * Math.cos(a), r * Math.cos(b) * 0.8, r * Math.sin(b) * Math.sin(a)], i * 3);
    const c = cs[Math.random() < 0.15 ? 2 : Math.random() < 0.5 ? 0 : 1];
    col.set([c.r, c.g, c.b], i * 3);
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pts, 3));
  pg.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const dust = new THREE.Points(pg, new THREE.PointsMaterial({ size: 2.2, vertexColors: true, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(dust);

  const resize = () => { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
  resize();
  new ResizeObserver(resize).observe(el);

  let mx = 0, my = 0;
  window.addEventListener("pointermove", (e) => { mx = (e.clientX / innerWidth - 0.5) * 2; my = (e.clientY / innerHeight - 0.5) * 2; }, { passive: true });
  let visible = true;
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(el);

  // Opening sequence: scanned into view bottom-up while the camera pulls back, then the tumor ignites.
  const tumorMats = [];
  brain.traverse((o) => { if (o.isMesh && o.renderOrder >= 3) { o.material.transparent = true; tumorMats.push([o.material, o.material.opacity]); } });
  const INTRO_DELAY = reduce ? 0 : 2.1, REVEAL = 2.4, IGNITE = 1.0;
  const ease = (x) => 1 - Math.pow(1 - Math.min(Math.max(x, 0), 1), 3);
  const clock = new THREE.Clock();
  let spin = 1.25;
  const frame = () => {
    requestAnimationFrame(frame);
    if (!visible) return;
    const dt = clock.getDelta(), t = clock.elapsedTime;
    const k = reduce ? 1 : ease((t - INTRO_DELAY) / REVEAL);
    for (const m of surfaces) m.uniforms.uReveal.value = k >= 1 ? 2 : k;
    camera.position.z = 340 + (1 - k) * 170;
    const ig = reduce ? 1 : ease((t - INTRO_DELAY - REVEAL + 0.3) / IGNITE);
    for (const [m, o0] of tumorMats) m.opacity = o0 * ig;
    if (!reduce) spin += dt * 0.22;
    group.rotation.y = spin + mx * 0.25;
    group.rotation.x = my * 0.08;
    dust.rotation.y -= reduce ? 0 : dt * 0.03;
    warm.intensity = 5200 * (0.85 + 0.15 * Math.sin(t * 1.6)) * ig;
    renderer.render(scene, camera);
  };
  frame();
  el.classList.add("ready");
}

if (host) start(host).catch((err) => { console.warn("hero scene unavailable:", err); host.classList.add("no-webgl"); });
