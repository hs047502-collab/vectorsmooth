# VectorSmooth Pro — marketplace-safe vector smoothing

**Smooth the path, never change the subject.** Every candidate is measured
against the original and rolled back automatically if it moves too far.

`index.html` is a **fully self-contained single-file app** (Python engine +
samples embedded, ~150 KB). Double-click it, or host it anywhere — it runs
100% in the browser; user files never leave the device. Only the one-time
Python-engine download (Pyodide CDN) needs internet.

## Features

- 🔒 **Subject-lock guarantee** — 4 enforced layers: cleanup budget,
  neighbour-clearance caps (paths can never merge), a separation backstop,
  and hard render-vs-render QA gates (breach → original returned untouched)
- 📦 **Unlimited batch** — drag & drop files *and* folders (recursive),
  Smooth-all with progress + stop, per-file inspector, download-all `.zip`
- ✅ **Marketplace checklists** — Adobe Stock, Shutterstock, Vecteezy,
  Freepik, EPS 8/10, icon sets + diagnostic readiness scores
- 🔍 **Proof views** — before/after slider, anchor + L/C overlays,
  pixel-QA diff, per-path dev/budget table, EPS export with re-open check
- 🐍 **Python CLI too** — `tools/smooth.py` for automation, same engine

## Use it (3 ways)

**A. Double-click** — download `index.html`, open it. Done.

**B. GitHub Pages (public link for your users)**
1. Push this folder to a GitHub repo (root must contain `index.html`)
2. Repo → *Settings → Pages → Deploy from a branch* → `main` / root
3. Share the `https://<you>.github.io/<repo>/` link

**C. CLI**
```bash
pip install numpy scipy pillow
python3 tools/smooth.py input.svg --preset auto --mode safe --convert-shapes --out clean.svg
python3 tools/smooth.py input.svg --eps out.eps --eps-level 10
```

## For developers

```bash
python3 -m pytest tests/ -q        # 22 tests: solver, fitter, guard, subject-lock, regression
python3 tools/build_standalone.py  # re-embed engine into index.html after ANY core/ change
```

Layout: `core/` engine · `tools/smooth.py` CLI · `tools/build_standalone.py`
embedder · `tests/` suite · `samples/` demo files · `index.html` shipped app.

## Selling / licensing

You own this code and the page. The app has no accounts, no tracking, no
backend — price it however you like (one-time file, paid link, membership…).
If you ever want key-based activation, that needs a tiny license server;
everything else already works as a static file.
