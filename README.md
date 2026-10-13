# PeelPixel: browser-only background remover (Money Lab, built Oct 12, 2026)
Static site: `index.html`, `style.css`, `app.js`, `LICENSES.md`. Deploy those 4 files to GitHub Pages. **Do not deploy `test/`** (GFDL test photo).
- GPU (WebGPU + browser) → BiRefNet_lite (MIT, 1024px). Otherwise → ormbg uint8 on WASM (Apache-2.0).
- Before publishing: set `PRO_URL` in app.js to the Gumroad $19 lifetime link.
- Test: `python3 -m http.server 8765` then `node test/run.mjs` (playwright-core from /workspace/tools/rea).
