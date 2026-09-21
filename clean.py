"""Topology-safe, shape-preserving smoothing pipeline.

Pipeline per subpath:
  cleanup (dup / near-dup / collinear / zigzag / spike)
  -> uniform resample -> corner classify -> denoise (budget = measured displacement)
  -> corner rebuild (outward intersection) -> error-bounded cubic fit
  -> guard (deviation + self-intersection + winding) -> accept or ROLLBACK.

Decision hierarchy: topology > silhouette fidelity > corners > curvature >
details > node count > file size. Node count is never #1.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    bbox_diag, detect_corners, estimate_noise, fit_span, guard_candidate,
    poly_len, polyline_deviation, rebuild_corners, resample, sample_segments,
    seg_lengths, signed_area, smooth, turn_angles, jitter_stats,
    detect_regular_features, points_in_polygon,
)

PRESETS = {
    'clean': {'smoothness': 15, 'fidelity': 92},
    'icon': {'smoothness': 25, 'fidelity': 90},
    'illustration': {'smoothness': 45, 'fidelity': 75},
    'traced': {'smoothness': 65, 'fidelity': 70},
    'heavy': {'smoothness': 85, 'fidelity': 55},
}

AUTO_FIDELITY = 80


def auto_smoothness(subpaths, D):
    stats = [jitter_stats(sp['pts'], sp['closed']) for sp in subpaths
             if len(sp['pts']) >= 16]
    if not stats:
        return 45, 0.0
    med_rel = float(np.median([s[0] for s in stats]))
    med_abs = float(np.median([s[1] for s in stats]))
    s_rel = 15.0 + 40.0 * med_rel
    # absolute term: jitter visible against a ~0.1%-of-artwork just-noticeable scale
    s_abs = 15.0 + 25.0 * (med_abs / max(D * 0.001, 1e-9))
    s = min(90.0, max(10.0, max(s_rel, s_abs)))
    return s, med_rel


def smoothness_label(s):
    if s < 20:
        return 'clean'
    if s < 35:
        return 'icon'
    if s < 55:
        return 'illustration'
    if s < 75:
        return 'traced'
    return 'heavy'


# ------------------------------------------------------------- cleanup stages

def _remove_duplicates(pts, closed):
    """Exact duplicate points (same coordinate twice)."""
    p = np.asarray(pts, float)
    if len(p) < 2:
        return p, 0
    keep = np.ones(len(p), bool)
    keep[1:] = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])) > 1e-12
    if closed and len(p) > 2 and keep.sum() > 2:
        first = np.where(keep)[0]
        if np.hypot(*(p[first[-1]] - p[first[0]])) <= 1e-12:
            keep[first[-1]] = False
    return p[keep], int(len(p) - keep.sum())


def _merge_near_duplicates(pts, closed, thr):
    """Near-duplicate merge; threshold scales with artwork size, never fixed px."""
    p = np.asarray(pts, float)
    if len(p) < 2 or thr <= 0:
        return p, 0
    kept = [p[0]]
    for q in p[1:]:
        if float(np.hypot(*(q - kept[-1]))) >= thr:
            kept.append(q)
    out = np.array(kept)
    if closed and len(out) > 2 and float(np.hypot(*(out[-1] - out[0]))) < thr:
        out = out[:-1]
    return out, int(len(p) - len(out))


def _remove_collinear(pts, closed, max_deg=1.5, passes=2):
    """Drop middle point when three consecutive points are ~one straight line."""
    p = np.asarray(pts, float)
    removed = 0
    for _ in range(passes):
        n = len(p)
        if n < (4 if closed else 3):
            break
        t = np.abs(turn_angles(p, 1, closed))
        keep = t >= max_deg
        if not closed:
            keep[0] = True
            keep[-1] = True
        if keep.all() or keep.sum() < 3:
            break
        removed += int(n - keep.sum())
        p = p[keep]
    return p, removed


def _remove_zigzag(pts, closed, D, amp_frac=0.0022, len_frac=0.0035):
    """Micro-zigzag runs (small amplitude + high frequency) -> straight chord.

    Only runs whose perpendicular amplitude is visually insignificant collapse;
    anything bigger (feather/fur/serration scale) is left alone.
    """
    p = np.asarray(pts, float)
    n = len(p)
    if n < 8:
        return p, 0
    t = turn_angles(p, 1, closed)
    if closed:
        nxt = np.roll(p, -1, axis=0)
        prv = np.roll(p, 1, axis=0)
    else:
        nxt = np.vstack([p[1:], p[-1]])
        prv = np.vstack([p[0], p[:-1]])
    sp = np.minimum(np.hypot((nxt - p)[:, 0], (nxt - p)[:, 1]),
                    np.hypot((p - prv)[:, 0], (p - prv)[:, 1]))
    if not closed:
        sp[0] = sp[-1] = np.inf
    qual = (np.abs(t) >= 4.0) & (np.abs(t) <= 70.0) & (sp <= D * len_frac)
    drop = np.zeros(n, bool)
    removed = 0
    i = 0
    while i < n:
        if drop[i] or not qual[i]:
            i += 1
            continue
        run = [i]
        j = i
        while len(run) < n:
            k = (j + 1) % n if closed else j + 1
            if (not closed and k >= n) or k == i or drop[k] or not qual[k]:
                break
            if np.sign(t[k]) == np.sign(t[j]) and len(run) >= 1 and k != i:
                # allow one same-sign step, then the run must alternate
                pass
            # strict alternation required from the 3rd point on
            if len(run) >= 2 and np.sign(t[k]) == np.sign(t[run[-1]]):
                # check it is not just noise: break the run
                if np.sign(t[k]) == np.sign(t[run[-2]]):
                    break
            run.append(k)
            j = k
        if len(run) >= 4:
            a, b = p[run[0]], p[run[-1]]
            chord = float(np.hypot(*(b - a)))
            if chord > 1e-12 and chord >= 2 * D * len_frac:
                q = p[np.array(run)]
                amp = float(np.abs((q[:, 0] - a[0]) * (b[1] - a[1])
                                     - (q[:, 1] - a[1]) * (b[0] - a[0])).max() / chord)
                if amp < D * amp_frac:
                    for r in run[1:-1]:
                        drop[r] = True
                    removed += len(run) - 2
                    i = run[-1]
        i += 1
    if removed and (~drop).sum() >= 3:
        return p[~drop], removed
    return p, 0


def _remove_spikes(pts, closed, D, base_frac=0.008, h_frac=0.030, passes=3):
    """Accidental spikes removed; raster-reference-checked spikes preserved.

    A protrusion is only removed when it is narrow + tall AND a lightly
    smoothed reference of the same polyline does not protrude there
    (intentional beaks/horns/thorns survive).
    """
    w = np.asarray(pts, float).copy()
    removed = 0
    for _ in range(passes):
        n = len(w)
        if n < 8:
            break
        ref = smooth(w, 1.0, closed)
        t = turn_angles(w, 1, closed)
        cand = np.where(np.abs(t) > 55.0)[0]
        drop = []
        for i in cand:
            if not closed and (i == 0 or i == n - 1):
                continue
            ip = (i - 1) % n
            iq = (i + 1) % n
            a, b, c = w[ip], w[i], w[iq]
            base = float(np.hypot(*(c - a)))
            if base < 1e-9 or base > D * base_frac:
                continue
            h = abs((b[0] - a[0]) * (c[1] - a[1])
                   - (b[1] - a[1]) * (c[0] - a[0])) / base
            if h < base * 2.0 or h > D * h_frac:
                continue
            ar, br, cr = ref[ip], ref[i], ref[iq]
            baser = float(np.hypot(*(cr - ar)))
            hr = (abs((br[0] - ar[0]) * (cr[1] - ar[1])
                     - (br[1] - ar[1]) * (cr[0] - ar[0])) / baser
                  if baser > 1e-9 else 0.0)
            if hr > 0.5 * h:
                continue  # intentional: the reference protrudes too
            drop.append(i)
        if not drop:
            break
        w = np.delete(w, drop, axis=0)
        removed += len(drop)
    return w, removed


def _span_pts(a, i, j, closed):
    n = len(a)
    if not closed:
        return a[i:j + 1]
    if i == j:
        return np.vstack([a[i:], a[:i + 1]])
    if j > i:
        return a[i:j + 1]
    return np.vstack([a[i:], a[:j + 1]])


# ------------------------------------------------------------------ pipeline

def _resolve_params(subpaths, o, D):
    preset = (o.get('preset') or 'auto').lower()
    if preset not in PRESETS and preset != 'auto':
        preset = 'auto'
    s_opt = o.get('smoothness')
    f_opt = o.get('fidelity')
    if preset == 'auto':
        s_auto, noise_med = auto_smoothness(subpaths, D)
        s = float(s_opt) if s_opt is not None else s_auto
        f = float(f_opt) if f_opt is not None else AUTO_FIDELITY
        preset_used = 'auto->%s' % smoothness_label(s)
    else:
        s = float(s_opt) if s_opt is not None else PRESETS[preset]['smoothness']
        f = float(f_opt) if f_opt is not None else PRESETS[preset]['fidelity']
        preset_used = preset
    s = min(100.0, max(0.0, s))
    f = min(100.0, max(0.0, f))
    mode = (o.get('mode') or 'safe').lower()
    if mode not in ('safe', 'pro'):
        mode = 'safe'
    remove_tiny = bool(o.get('remove_tiny', True))
    if mode == 'safe':
        f = max(f, 80.0)  # SAFE MARKETPLACE CLEAN never drops fidelity
        remove_tiny = False  # ... and never deletes artwork, however small
    return {
        'preset_used': preset_used, 'mode': mode,
        'smoothness': s, 'fidelity': f,
        'movement_cap': o.get('movement_cap'),
        'noise_target': o.get('noise_target'),
        'round_corners': bool(o.get('round_corners')) and mode == 'pro',
        'remove_tiny': remove_tiny,
        'auto_close': bool(o.get('auto_close', True)),
        'convert_shapes': bool(o.get('convert_shapes', False)),
    }


def _smooth_one(sp, D, P, notes, clear_cap=None):
    pts = np.asarray(sp['pts'], float)
    closed = bool(sp['closed'])
    res = {'closed': closed, 'anchors_before': int(sp['anchors']),
           'status': 'smoothed', 'reason': 'ok', 'dev': 0.0, 'budget': 0.0,
           'segments': [], 'start': pts[0].copy() if len(pts) else np.zeros(2),
           'dense_orig': pts.copy(), 'dense_new': pts.copy(),
           'removed': {}, 'corners': 0, 'rounded': 0, 'cusps': 0,
           'jitter_before': 0.0, 'jitter_after': 0.0, 'auto_closed': False}
    if len(pts) < 2:
        res['status'] = 'degenerate'
        return res
    if len(pts) == 2 and not closed:
        res['status'] = 'passthrough'  # a plain line: nothing to smooth
        return res
    if sp.get('is_shape') and not P.get('convert_shapes', False):
        res['status'] = 'passthrough'
        res['reason'] = 'shape kept (convert_shapes off)'
        res['dense_orig'] = pts
        res['dense_new'] = pts
        return res
    try:
        np.linalg.inv(sp['M_exp'])
    except np.linalg.LinAlgError:
        res['status'] = 'passthrough'
        res['reason'] = 'singular transform, kept verbatim'
        return res

    # start closed loops at the sharpest turn so no feature straddles the seam
    if closed and len(pts) > 16:
        t0 = np.abs(turn_angles(pts, 2, True))
        pts = np.roll(pts, -int(np.argmax(t0)), axis=0)

    jb_rel, jb_abs = jitter_stats(pts, closed)
    res['jitter_before'] = jb_rel

    # ---- stage A: cleanup (skipped entirely at 0% = Original) ----
    if P['smoothness'] <= 0:
        res['status'] = 'passthrough'
        res['reason'] = 'smoothness 0% (Original)'
        res['dense_orig'] = pts
        res['dense_new'] = pts
        return res
    fid = P['fidelity']
    # ---- fidelity cap: the solemn promise (never exceed, even at 100%) ----
    # Fixed BEFORE cleanup so structural ops (spike/zigzag removal) obey the
    # same budget as smoothing: subject change is measured against the RAW
    # input, never against an already-altered curve.
    if P['movement_cap'] is not None:
        B_cap = float(P['movement_cap'])
    else:
        B_cap = D * (0.0005 + (1 - fid / 100.0) * 0.02)
        if P['mode'] == 'safe':
            B_cap = min(B_cap, D * 0.004)
    if clear_cap is not None and clear_cap < B_cap:
        B_cap = float(clear_cap)
        # recorded, then aggregated into ONE note (dozens of neighbours
        # would otherwise spam the notes panel with near-identical lines)
        res['clear_cap'] = (round(B_cap, 3), round(clear_cap / 0.4, 3))
    B_cap = max(B_cap, D * 1e-4)
    merge_tol = D * (1e-4 + (1 - fid / 100.0) * 9e-4)
    raw = pts.copy()
    pts, n_dup = _remove_duplicates(pts, closed)
    pts, n_merge = _merge_near_duplicates(pts, closed, merge_tol)
    pts, n_col = _remove_collinear(pts, closed)
    pts_s, n_zig = _remove_zigzag(pts, closed, D)
    pts_s, n_spk = _remove_spikes(pts_s, closed, D)
    step_c = max(poly_len(raw, closed) / 6000.0, D * 0.0008, 1e-6)
    need = 3 if closed else 2
    if len(pts_s) >= need and len(raw) >= need:
        cd = polyline_deviation(resample(raw, step_c, closed),
                                resample(pts_s, step_c, closed))
    else:
        cd = float('inf')
    if cd <= B_cap:
        pts, cleanup_dev = pts_s, cd
    else:
        # structural cleanup would move the subject past the budget: keep the
        # harmless part (dups/collinear are ~zero-dev), drop the rest.
        pts, n_zig, n_spk = pts, 0, 0
        if len(pts) >= need:
            cd2 = polyline_deviation(resample(raw, step_c, closed),
                                     resample(pts, step_c, closed))
        else:
            cd2 = float('inf')
        if np.isfinite(cd):
            why = 'would move %.2f u' % cd
        else:
            why = 'would erase the path'
        if cd2 <= B_cap:
            cleanup_dev = cd2
            notes.append('structural cleanup skipped (%s, over the %.2f u '
                         'budget) - subject kept' % (why, B_cap))
        else:
            pts, cleanup_dev = raw.copy(), 0.0
            n_dup = n_merge = n_col = 0
            notes.append('cleanup skipped entirely (%s) - subject kept'
                         % why)
    res['removed'] = {'duplicates': n_dup + n_merge, 'collinear': n_col,
                      'zigzag': n_zig, 'spikes': n_spk}
    res['cleanup_dev'] = round(float(cleanup_dev), 4)
    if len(pts) < (3 if closed else 2):
        res['status'] = 'reverted'
        res['reason'] = 'cleanup would erase the path'
        res['dense_orig'] = np.asarray(sp['pts'], float)
        res['dense_new'] = res['dense_orig']
        return res

    # ---- open path handling ----
    if not closed and P['auto_close'] and len(pts) >= 3:
        gap = float(np.hypot(*(pts[-1] - pts[0])))
        if gap < D * 0.002:
            closed = True
            res['auto_closed'] = True
            notes.append('auto-closed a %.3f-unit gap' % gap)
    res['closed'] = closed

    # ---- tiny/stray objects ----
    diag = bbox_diag(pts)
    if diag < max(D * 0.004, 0.75):
        if P['remove_tiny']:
            res['status'] = 'removed'
            res['reason'] = 'tiny fragment (%.2f u)' % diag
            res['dense_new'] = np.zeros((0, 2))
            return res
        res['status'] = 'passthrough'
        res['reason'] = 'tiny object kept (remove_tiny off)'
        res['dense_new'] = pts
        res['dense_orig'] = pts
        return res

    # ---- clean-input passthrough: never touch what is already clean ----
    removed_total = sum(res['removed'].values())
    if removed_total == 0 and jb_abs < max(D * 2e-4, 0.02) and jb_rel < 0.08:
        res['status'] = 'passthrough'
        res['reason'] = 'already clean'
        res['dense_orig'] = pts
        res['dense_new'] = pts
        return res

    # ---- uniform resample (physical units from here on) ----
    total = poly_len(pts, closed)
    step0 = max(total / 6000.0, D * 0.0008, 1e-6)
    dense = resample(pts, step0, closed)
    step = total / len(dense)
    # the guard compares against the RAW input: cleanup + smoothing + fit
    # share one budget, so the subject can never drift in uncounted stages.
    res['dense_orig'] = resample(raw, step0, closed)
    _sg = seg_lengths(pts, closed)
    _sg = _sg[_sg > 1e-12]
    med_spacing = float(np.median(_sg)) if len(_sg) else step

    # ---- corner classification (after pre-smooth, physical window) ----
    window_len = D * 0.010
    sharp_deg = 55.0 - fid * 0.35
    mask, turns, mag, k = detect_corners(dense, step, window_len, sharp_deg,
                                         closed, med_spacing=med_spacing)
    while mask.sum() > max(8, len(dense) // 50) and sharp_deg < 120:
        sharp_deg *= 1.25  # adaptive: never drown in "corners"
        mask, turns, mag, k = detect_corners(dense, step, window_len, sharp_deg,
                                             closed, med_spacing=med_spacing)
    corners = sorted(int(i) for i in np.where(mask)[0])
    feats = detect_regular_features(dense, step, closed)
    if feats:
        corners = sorted(set(corners) | set(feats))
        notes.append('%d repeated feature(s) pinned (teeth/serration guard)'
                     % len(feats))
    res['corners'] = len([c for c in corners if mag[c] < 140])
    res['cusps'] = len([c for c in corners if mag[c] >= 140])
    res['rounded'] = int(((mag >= 12.0) & (mag < sharp_deg)).sum())

    # B_cap was fixed before cleanup: every stage shares one budget.

    # ---- corner-protective denoise with sigma search (strong -> gentle) ----
    # sigma must span MULTIPLE vertices to average white noise: a kernel
    # narrower than the vertex spacing only rounds interpolation kinks.
    nt = P['noise_target']
    if nt is not None:
        sigma_px = D * 0.005 * float(nt)
    else:
        sigma_px = max(D * (0.0004 + 0.006 * (P['smoothness'] / 100.0)),
                       med_spacing * (2.0 + 3.0 * (P['smoothness'] / 100.0)))
    sigma_px = min(sigma_px, D * 0.05)
    sigma_s = sigma_px / step
    # ---- split points (sigma-independent) ----
    if corners:
        bounds = corners
    elif closed:
        bounds = [0, len(dense) // 2]
    else:
        bounds = [0, len(dense) - 1]
    if not closed and bounds[0] != 0:
        bounds = [0] + bounds
    if not closed and bounds[-1] != len(dense) - 1:
        bounds = bounds + [len(dense) - 1]

    # ---- nested search: sigma (strong -> gentle) x fit-tol (loose -> tight).
    # Every passing combination is collected; fewest segments wins, smallest
    # deviation breaks ties. Nothing may exceed B_cap, ever.
    candidates = []
    last_reason, last_dev = 'no candidate', float('inf')
    kb = max(2, k // 2)
    for factor in (1.0, 0.5, 0.25, 0.125):
        sm_try = smooth(dense, sigma_s * factor, closed)
        sm_r = rebuild_corners(sm_try, corners, kb, step, closed, dense,
                               max_move=B_cap * 0.5)
        if P['round_corners']:
            for i in corners:
                if mag[i] < 140:
                    a = sm_r[(i - kb) % len(sm_r)] if closed else sm_r[max(0, i - kb)]
                    b = sm_r[(i + kb) % len(sm_r)] if closed else sm_r[min(len(sm_r) - 1, i + kb)]
                    new = 0.55 * sm_r[i] + 0.45 * (a + b) / 2
                    mv = float(np.hypot(*(new - dense[i])))
                    lim = B_cap * 0.25
                    sm_r[i] = (dense[i] + (new - dense[i]) * (lim / mv)
                               if mv > lim else new)
        # displacement of the rebuilt candidate: pinning/snapping already
        # fixed the corner round-off, so every point counts, no exclusions.
        disp = np.hypot((sm_r - dense)[:, 0], (sm_r - dense)[:, 1])
        dmax_try = float(disp.max()) if len(disp) else 0.0
        # total displacement vs RAW = cleanup + smoothing (+ fit later)
        if cleanup_dev + dmax_try > B_cap:
            last_reason = ('smoothing moves %.2f u, over the %.2f u cap'
                           % (cleanup_dev + dmax_try, B_cap))
            last_dev = cleanup_dev + dmax_try
            continue
        budget_try = min(B_cap, cleanup_dev + dmax_try * 1.5
                         + max(D * 2e-4, B_cap * 0.02))
        headroom_try = max(budget_try - cleanup_dev - dmax_try, D * 1e-6)
        for ci in corners:  # keep corner moves inside the budget accounting
            if float(np.hypot(*(sm_r[ci] - dense[ci]))) > budget_try * 0.5:
                sm_r[ci] = dense[ci]
        if closed:
            spans = [_span_pts(sm_r, a, b, True)
                     for a, b in zip(bounds, bounds[1:] + bounds[:1])]
        else:
            spans = [sm_r[a:b + 1] for a, b in zip(bounds[:-1], bounds[1:])]
        fine_step_try = min(step, max(headroom_try * 0.15, total / 30000.0, 1e-9))
        fine_orig = resample(raw, fine_step_try, closed)
        area_orig = signed_area(fine_orig)
        for tol in (headroom_try * 1.0, headroom_try * 0.5,
                    headroom_try * 0.25, headroom_try * 0.125,
                    headroom_try * 0.0625):
            segs = []
            for span in spans:
                segs.extend(fit_span(span, max(tol, 1e-9)))
            if not segs:
                continue
            cand = sample_segments(sm_r[bounds[0]], segs, step=fine_step_try)
            ok, reason, dev = guard_candidate(fine_orig, cand, budget_try,
                                              closed, area_orig)
            if ok:
                candidates.append((segs, cand, dev, tol, sm_r, budget_try,
                                   factor, dmax_try))
                break
            last_reason, last_dev = reason, dev
    accepted = (min(candidates, key=lambda c: (len(c[0]), c[2]))
                if candidates else None)
    if P['round_corners']:
        notes.append('corners rounded (explicit option, pro mode)')
    if accepted is None:
        res['status'] = 'reverted'
        res['reason'] = last_reason
        res['budget'] = B_cap
        # revert outputs the budget-legal cleaned curve (NOT raw, NOT the
        # rejected candidate): reported dev is the cleanup displacement.
        res['dev'] = cleanup_dev
        res['dense_new'] = dense
        idx = np.linspace(0, len(pts) - 1, min(len(pts), 400)).astype(int)
        res['segments'] = [('L', pts[j].copy()) for j in idx[1:]]
        res['start'] = pts[idx[0]].copy()
        return res
    segs, cand, dev, tol_used, sm, budget, factor, dmax = accepted
    res['budget'] = budget
    if factor < 1.0:
        notes.append('gentler smoothing used to respect fidelity (x%.3g)'
                     % factor)
    n_after = len(segs) if closed else 1 + len(segs)
    if n_after > max(4, 1.25 * sp['anchors']):
        res['status'] = 'reverted'  # more nodes is never "better"
        res['reason'] = 'smoothing would add nodes (%d -> %d)' % (
            sp['anchors'], n_after)
        res['dev'] = dev
        res['dense_new'] = dense
        idx = np.linspace(0, len(pts) - 1, min(len(pts), 400)).astype(int)
        res['segments'] = [('L', pts[j].copy()) for j in idx[1:]]
        res['start'] = pts[idx[0]].copy()
        return res
    res['segments'] = segs
    res['start'] = sm[bounds[0]].copy()
    res['dense_new'] = cand
    res['dev'] = dev
    res['tol_used'] = tol_used
    res['jitter_after'] = estimate_noise(
        cand[::max(1, len(cand) // 2000)], closed)

    # ---- clean-input passthrough: byte-identical when nothing was needed ----
    removed_total = sum(res['removed'].values())
    if (removed_total == 0 and len(segs) >= 0.97 * max(1, sp['anchors'])
            and dev < D * 5e-4):
        res['status'] = 'passthrough'
        res['reason'] = 'already clean'
        res['dense_new'] = dense
        return res
    return res


def _pairwise_min_dist(a, b, cap=1200):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    if len(a) > cap:
        a = a[::len(a) // cap + 1]
    if len(b) > cap:
        b = b[::len(b) // cap + 1]
    d, _ = cKDTree(a).query(b)
    return float(d.min())


def _raw_dense(subpaths, D):
    """Capped resample of every raw input path (for clearance/holes)."""
    raws = []
    for sp in subpaths:
        pts = np.asarray(sp['pts'], float)
        if len(pts) >= 2:
            st = max(poly_len(pts, bool(sp['closed'])) / 1500.0, D * 0.001,
                     1e-9)
            raws.append(resample(pts, st, bool(sp['closed'])))
        else:
            raws.append(pts)
    return raws


def _densify_poly(a, closed, D):
    a = np.asarray(a, float)
    if len(a) >= 8 or len(a) < 2:
        return a
    st = max(poly_len(a, bool(closed)) / 1500.0, D * 0.001, 1e-9)
    return np.asarray(resample(a, st, bool(closed)), float)


def _find_holes(dense_list, closed_list):
    """Containment pairs (child, parent): strictly inside + positive gap.

    A child that touches its parent (shared icon edges) is NOT a hole.
    """
    out = []
    n = len(dense_list)
    for i in range(n):
        if not closed_list[i]:
            continue
        a = np.asarray(dense_list[i], float)
        if len(a) < 8:
            continue
        aa = abs(signed_area(a))
        if aa <= 0:
            continue
        for j in range(n):
            if i == j or not closed_list[j]:
                continue
            b = np.asarray(dense_list[j], float)
            if len(b) < 8 or abs(signed_area(b)) <= aa:
                continue
            blo, bhi = b.min(axis=0) - 1e-9, b.max(axis=0) + 1e-9
            if (a.min(axis=0) < blo).any() or (a.max(axis=0) > bhi).any():
                continue
            ac = a if len(a) <= 800 else a[::len(a) // 800 + 1]
            if not points_in_polygon(ac, b).all():
                continue
            gap = _pairwise_min_dist(a, b)
            step = poly_len(b, True) / max(len(b), 1)
            if gap < max(step, 1e-9):  # touching/shared edge: not a hole
                continue
            out.append({'child': i, 'parent': j, 'gap': gap,
                        'area': aa, 'perim': poly_len(a, True)})
    return out


def _clearance_caps(subpaths, D):
    """Per-path budget cap from neighbour clearance: paths can never merge.

    Each path may move at most 40% of its gap to the nearest neighbour, so
    even two neighbours moving straight at each other keep 20% of the gap.
    """
    raws = _raw_dense(subpaths, D)
    caps = []
    for i, a in enumerate(raws):
        gap = float('inf')
        for j, b in enumerate(raws):
            if i == j:
                continue
            gap = min(gap, _pairwise_min_dist(a, b))
        caps.append(gap * 0.4 if np.isfinite(gap) else None)
    return caps


def _revert_to_original(subpaths, results, k, reason):
    r = results[k]
    if r['status'] in ('smoothed', 'passthrough'):
        r['status'] = 'reverted'
        r['reason'] = reason
        r['dev'] = 0.0
        r['dense_new'] = r['dense_orig']
        pts = np.asarray(subpaths[k]['pts'], float)
        m = min(len(pts), 400)
        ix = np.linspace(0, len(pts) - 1, m).astype(int)
        r['segments'] = [('L', pts[t].copy()) for t in ix[1:]]
        r['start'] = pts[ix[0]].copy()


def _detail_lock_check(subpaths, results, holes, notes):
    """Detail lock: holes stay holes (containment + area consistency).

    For every input containment pair the child must still sit inside its
    parent afterwards, and its area may only change by what the measured
    boundary movement allows (area <= perimeter * deviation + slack).
    Violating pairs are reverted to the original.
    """
    fixed = 0
    for h in holes:
        i, j = h['child'], h['parent']
        ri, rj = results[i], results[j]
        if ri['status'] == 'removed' or rj['status'] == 'removed':
            continue
        ni = np.asarray(ri['dense_new'], float)
        nj = np.asarray(rj['dense_new'], float)
        oi = np.asarray(ri['dense_orig'], float)
        if len(ni) < 3 or len(nj) < 3 or len(oi) < 3:
            continue
        iv = ni if len(ni) <= 800 else ni[::len(ni) // 800 + 1]
        ok_in = bool(points_in_polygon(iv, nj).all())
        dev = max(float(ri['dev']), 0.0)
        step = poly_len(oi, True) / max(len(oi), 1)
        allowed = h['perim'] * (dev + step) + h['area'] * 0.02
        d_area = abs(abs(signed_area(ni)) - h['area'])
        if not ok_in or d_area > allowed:
            fixed += 1
            for k in (i, j):
                _revert_to_original(subpaths, results, k,
                                    'hole/detail failed the detail-lock - '
                                    'kept original')
    if fixed:
        notes.append('%d hole/detail(s) failed the detail-lock '
                     '(containment/area) - kept original' % fixed)
    return fixed


def _separation_check(subpaths, results, notes):
    """Revert pairs whose open gap closed to a touch (backstop for caps).

    Pairs already touching in the INPUT (shared edges in icons) are exempt:
    only a gap that was clearly open and is now closed counts as a violation.
    """
    touched = 0
    gmin_before, gmin_after = float('inf'), float('inf')
    idx = [i for i, r in enumerate(results)
           if r['status'] != 'removed' and len(r['dense_new']) > 0
           and len(r['dense_orig']) > 0]
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            before = _pairwise_min_dist(results[i]['dense_orig'],
                                        results[j]['dense_orig'])
            after = _pairwise_min_dist(results[i]['dense_new'],
                                       results[j]['dense_new'])
            gmin_before = min(gmin_before, before)
            gmin_after = min(gmin_after, after)
            ni = np.asarray(results[i]['dense_new'], float)
            nj = np.asarray(results[j]['dense_new'], float)
            si = poly_len(ni, results[i]['closed']) / max(len(ni), 1)
            sj = poly_len(nj, results[j]['closed']) / max(len(nj), 1)
            eps = max(si, sj, 1e-9)
            if after < eps and before > 3 * eps:
                touched += 1
                for k in (i, j):
                    _revert_to_original(subpaths, results, k,
                                        'neighbour paths would touch/merge - '
                                        'kept original')
    if touched:
        notes.append('%d neighbour pair(s) would touch after smoothing - '
                     'kept original' % touched)
    return touched, gmin_before, gmin_after


def smooth_artwork(art, options):
    subpaths = art['subpaths']
    all_pts = [sp['pts'] for sp in subpaths if len(sp['pts'])]
    if all_pts:
        big = np.vstack(all_pts)
        D = float(np.hypot(*(big.max(axis=0) - big.min(axis=0))))
    else:
        D = float(max(art['vb'][2], art['vb'][3]))
    D = max(D, 1e-6)
    P = _resolve_params(subpaths, options or {}, D)
    notes = []
    caps = _clearance_caps(subpaths, D)
    raws = _raw_dense(subpaths, D)
    holes_in = _find_holes(raws, [bool(sp['closed']) for sp in subpaths])
    for h in holes_in:
        # hole walls also respect their own mean width (area/perimeter):
        # a hole may not move more than half of it, so it can never close.
        wcap = max(h['area'] / max(h['perim'], 1e-9) * 0.5, D * 1e-4)
        i = h['child']
        caps[i] = wcap if caps[i] is None else min(caps[i], wcap)
    results = [_smooth_one(sp, D, P, notes, clear_cap=c)
               for sp, c in zip(subpaths, caps)]
    touched, gap_before, gap_after = _separation_check(subpaths, results,
                                                       notes)
    detail_fixed = _detail_lock_check(subpaths, results, holes_in, notes)
    out_dense = [np.asarray(r['dense_new'], float)
                 if r['status'] != 'removed' else np.zeros((0, 2))
                 for r in results]
    # Densify coarse outputs (passthrough rect = 4 corners) exactly like the
    # input side, so the after-count uses the same ruler as the before-count.
    out_dense = [_densify_poly(a, bool(c), D) for a, c in
                 zip(out_dense, [r['closed'] for r in results])]
    holes_after = _find_holes(out_dense, [r['closed'] for r in results])
    capped = [r['clear_cap'] for r in results if r.get('clear_cap')]
    if capped:
        notes.append('%d path(s): budget capped by neighbour gaps '
                     '(tightest %.3f u cap on a %.3f u gap - '
                     'paths can never touch)'
                     % (len(capped), min(c[0] for c in capped),
                        min(c[1] for c in capped)))
    for r in results:
        if r['status'] == 'removed':
            r['anchors_after'] = 0
        elif r['status'] in ('passthrough', 'reverted', 'degenerate'):
            r['anchors_after'] = r['anchors_before']
        elif r['closed']:
            r['anchors_after'] = len(r['segments'])
        else:
            r['anchors_after'] = 1 + len(r['segments'])
    n_rev = sum(1 for r in results if r['status'] == 'reverted')
    if subpaths and all(r['status'] in ('passthrough', 'degenerate') for r in results):
        notes.append('file already clean - returned untouched')
    elif subpaths and all(r['status'] in ('reverted', 'passthrough', 'degenerate', 'removed')
                          for r in results) and n_rev:
        notes.append('all smoothing candidates failed the guard - original kept where possible')
    if P['round_corners']:
        pass  # already noted per path
    summary = {
        'anchors_before': int(sum(r['anchors_before'] for r in results)),
        'anchors_after': int(sum(r['anchors_after'] for r in results)),
        'deviation_max': round(max([r['dev'] for r in results] + [0.0]), 4),
        'budget': round(max([r['budget'] for r in results] + [0.0]), 4),
        'self_intersections': 0,  # filled by QA/render stage verify
        'reverted': n_rev,
        'removed_tiny': sum(1 for r in results if r['status'] == 'removed'),
        'smoothed': sum(1 for r in results if r['status'] == 'smoothed'),
        'passthrough': sum(1 for r in results if r['status'] in ('passthrough', 'degenerate')),
        'spikes_removed': sum(r['removed'].get('spikes', 0) for r in results),
        'zigzag_removed': sum(r['removed'].get('zigzag', 0) for r in results),
        'dups_removed': sum(r['removed'].get('duplicates', 0) for r in results),
        'collinear_removed': sum(r['removed'].get('collinear', 0) for r in results),
        'corners_kept': sum(r['corners'] + r['cusps'] for r in results),
        'art_diag': round(D, 3),
        'pairs_touched': touched,
        'min_gap_before': (round(gap_before, 4)
                           if np.isfinite(gap_before) else None),
        'min_gap_after': (round(gap_after, 4)
                          if np.isfinite(gap_after) else None),
        'paths_before': len(subpaths),
        'paths_after': sum(1 for r in results
                           if r['status'] != 'removed'),
        'holes_before': len(holes_in),
        'holes_after': len(holes_after),
        'detail_fixed': detail_fixed,
    }
    return {'params': P, 'results': results, 'summary': summary,
            'notes': sorted(set(notes)), 'diag': D, 'holes': holes_in}
