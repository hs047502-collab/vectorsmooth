# VectorSmooth Pro — Changelog

## v1.2.0 (2026-09-21) — Marketplace-ready EPS

EPS downloads now upload error-free to Shutterstock, Adobe Stock, Vecteezy
and Freepik — and every file carries a preflight report proving it.

- ✅ **Spec-compliant EPS structure** — full DSC header (Creator, CreationDate,
  Clean7Bit, LanguageLevel 2, HiResBoundingBox), RGB-only paints,
  Illustrator 8/10-compatible operators; the forbidden `showpage` operator
  is gone and every export self-validates before download.
- 🛂 **Marketplace preflight on every EPS** — 21 checks: file structure,
  open paths, strokes, live text, raster, filters, transparency, gradients,
  `<use>`, clipping, artwork size (MP), file size, artboard. Blockers list
  the exact Illustrator fix; clean files get a `marketplace-ready` badge.
- 📦 **EPS batch ZIP** — batch download now honors the SVG/EPS toggle and
  includes a `preflight.txt` report (READY/BLOCKED per file + fixes).
- 📋 Checklist/locks catch up: transparency, gradient, `<use>` and clipping
  checks added to all marketplace profiles; EPS profiles measure the real
  EPS bytes and bbox megapixels.

## v1.1.0 (2026-09-21) — Detail-lock release

Quality release: holes, cutouts and small details are now protected by a
dedicated lock, and the QA gates stop discarding legitimate smoothing.

- 🔒 **Detail lock (new, 5th lock)** — every hole/cutout pair is verified
  twice: geometrically (child still inside parent, area change within the
  measured boundary-movement budget) and at the render stage (hole-interior
  pixels must paint pixel-identical). Violations roll back to the original.
- 🧱 **Hole-wall budget caps** — smoothing near a hole wall is capped so a
  thin wall can never collapse or merge, even in Pro mode.
- ⚖️ **Marginal-IoU policy** — the strict 0.985 safe-mode floor is unchanged,
  but IoU alone can no longer discard good work: if IoU is the sole failing
  gate (≥ 0.97 absolute floor) and winding, chamfer, changed-pixel and the
  detail lock all verify clean, the smoothing is kept — with an honest
  `accepted*` mark and a note explaining why.
- 📊 **Paths & holes accounting** — results header shows `paths before → after`
  and `holes before → after`; the Pixel QA tab shows the detail-lock line
  (holes, interior pixels verified, changed, rollbacks).
- 🐛 **Compound export fix** — multi-subpath elements stay ONE `<path>`
  (even-odd holes no longer fill solid on export); EPS emits a single
  `eofill` per compound.
- 🐛 **SAFE keeps tiny details** — tiny-but-real artwork is no longer deleted
  as fragments in Safe mode (Pro still cleans).
- 📝 Report notes cleaned up (no more `inf` units in messages).

## v1.0.0 — Initial release

- Single-file app (`index.html`, engine + samples embedded, runs in-browser)
- Subject-lock guarantee: cleanup budget, neighbour-clearance caps,
  separation backstop, render-vs-render QA gates (IoU / changed / chamfer)
- SVG + EPS (level 8/10) export with re-open validation, batch ZIP
