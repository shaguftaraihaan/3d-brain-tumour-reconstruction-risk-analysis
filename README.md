# BraTS case explorer

Local Flask viewer for BraTS reconstructions, plus a **backend-free static demo** that can be hosted on Vercel / Cloudflare Pages / Netlify.

Research prototype, not for diagnosis. Demo cases are derived from the **public BraTS dataset** — check the [BraTS data terms](https://www.med.upenn.edu/cbica/brats2020/data.html) before publishing a hosted copy.

## Deploy the static demo on Vercel

1. Build the static site locally (needs the three demo cases already reconstructed under `outputs/` and the matching `data/cases/` NIfTIs for slice export):

   ```bash
   python scripts/export_static_demo.py \
     --cases BraTS20_Training_002,BraTS20_Training_004,BraTS20_Training_022 \
     --stride 2
   ```

   This writes `dist/static_demo/` (viewer assets + `scene.glb` + pre-rendered FLAIR / overlay PNGs + JSON / report HTML). Target size is under 100 MB; the exporter raises stride / reduces PNG colors if needed.

2. Preview locally without Flask:

   ```bash
   python -m http.server 4173 --directory dist/static_demo
   # open http://127.0.0.1:4173/index.html?case=BraTS20_Training_002
   ```

3. Deploy the folder to Vercel (no build step — `vercel.json` points `outputDirectory` at `dist/static_demo`):

   ```bash
   npx vercel link          # once per machine / project
   npx vercel --prod        # from the repo root, after export
   ```

   Or in the Vercel dashboard: import the repo, set **Output Directory** to `dist/static_demo`, leave **Build Command** empty, and either commit a pre-built `dist/static_demo` or run the exporter in CI before deploy.

4. Confirm `.glb` responses use `Content-Type: model/gltf-binary` (configured in `vercel.json`) and that HTML/JS/CSS are `no-cache` while `.glb` / `.png` / `vendor/` are immutable long-cache.

Do **not** upload private patient scans into the static export. Upload / modality switching / window-level / probability maps stay disabled in static mode; run the Flask app locally for full analysis.
