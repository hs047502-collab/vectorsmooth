"""Numeric geometry helpers: resample, denoise, corners, Bezier fit, gates."""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def seg_lengths(pts, closed):
    d = np.diff(np.asarray(pts, float), axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    if closed and len(pts) > 1:
        p = np.asarray(pts, float)
        L = np.append(L, float(np.hypot(*(p[0] - p[-1]))))
    return L


def poly_len(pts, closed):
    return float(seg_lengths(pts, closed).sum())


def bbox_diag(pts):
    p = np.asarray(pts, float)
    if len(p) == 0:
        return 0.0
    return float(np.hypot(*(p.max(axis=0) - p.min(axis=0))))


def signed_area(pts):
    p = np.asarray(pts, float)
    if len(p) < 3:
        return 0.0
    return float(0.5 * (np.dot(p[:, 0], np.roll(p[:, 1], -1))
                        - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def points_in_polygon(pts, poly):
    """Vectorized ray-cast point-in-polygon. Returns a bool array."""
    p = np.asarray(pts, float)
    q = np.asarray(poly, float)
    n = len(q)
    if n < 3 or len(p) == 0:
        return np.zeros(len(p), bool)
    x = p[:, 0][:, None]
    y = p[:, 1][:, None]
    x1, y1 = q[:, 0][None, :], q[:, 1][None, :]
    x2, y2 = np.roll(q[:, 0], -1)[None, :], np.roll(q[:, 1], -1)[None, :]
    cond = ((y1 <= y) & (y2 > y)) | ((y2 <= y) & (y1 > y))
    with np.errstate(divide='ignore', invalid='ignore'):
        xin = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
    cross = cond & (xin > x)
    return (cross.sum(axis=1) % 2) == 1


def resample(pts, step, closed=True):
    """Uniform arc-length resample. Closed: wrap included, no dup endpoint."""
    pts = np.asarray(pts, float)
    if len(pts) < 2 or step <= 0:
        return pts.copy()
    ext = np.vstack([pts, pts[:1]]) if closed else pts
    d = np.diff(ext, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    total = float(seg.sum())
    if total < 1e-12:
        return pts.copy()
    n = max(8, int(round(total / step)))
    s_target = np.linspace(0, total, n, endpoint=not closed)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.searchsorted(cum, s_target, side='right') - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (s_target - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return ext[idx] + (ext[idx + 1] - ext[idx]) * t[:, None]


def smooth(pts, sigma, closed=True):
    if sigma is None or sigma < 0.35:
        return np.asarray(pts, float).copy()
    return ndimage.gaussian_filter1d(np.asarray(pts, float), sigma, axis=0,
                                     mode='wrap' if closed else 'nearest')


def turn_angles(pts, k=1, closed=True):
    """Signed turn angle (degrees) at each point using +-k neighbours."""
    p = np.asarray(pts, float)
    n = len(p)
    if n < 3:
        return np.zeros(n)
    if closed:
        ii = np.arange(n)
        a = p[(ii - k) % n]
        c = p[(ii + k) % n]
    else:
        ii = np.arange(n)
        a = p[np.clip(ii - k, 0, n - 1)]
        c = p[np.clip(ii + k, 0, n - 1)]
    v1 = p - a
    v2 = c - p
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def detect_corners(pts, step, window_len, sharp_deg, closed, presmooth=True,
                   med_spacing=None):
    """Corner mask via physical-length window on a pre-smoothed curve.

    Returns (mask, turns, magnitudes, k). Classification happens only after
    the pre-smooth, otherwise noise looks like corners. The pre-smooth
    scales with the noise wavelength (median vertex spacing) so dense
    jitter can never fake dozens of corners.
    """
    p = np.asarray(pts, float)
    n = len(p)
    if n < 8:
        return np.zeros(n, bool), np.zeros(n), np.zeros(n), 1
    k = max(2, int(round(window_len / max(step, 1e-9))))
    k = min(k, max(2, n // 8 if closed else n // 4))
    pre_px = min(window_len * 0.75,
                 max(window_len / 4.0, (med_spacing or 0.0) * 0.75))
    work = (smooth(p, pre_px / max(step, 1e-9), closed) if presmooth else p)
    turns = turn_angles(work, k, closed)
    mag = np.abs(turns)
    mask = np.zeros(n, bool)
    cand = np.where(mag >= sharp_deg)[0]
    if closed:
        for idx in cand:
            window = np.take(mag, range(idx - k, idx + k + 1), mode='wrap')
            if mag[idx] >= window.max() - 1e-9:
                mask[idx] = True
    else:
        for idx in cand:
            lo, hi = max(0, idx - k), min(n, idx + k + 1)
            if mag[idx] >= mag[lo:hi].max() - 1e-9:
                mask[idx] = True
    return mask, turns, mag, k


def detect_regular_features(pts, step, closed):
    """Pin repeated motifs (gear teeth, serrations, stitches, star tips).

    High-turn points with PERIODIC spacing are features by definition, even
    when a fixed window would miss them. Random noise never forms regular
    chains, so this only ever fires on real repeated structure.
    Returns sorted indices (refined to local turn maxima).
    """
    p = np.asarray(pts, float)
    n = len(p)
    if n < 32:
        return []
    t = np.abs(turn_angles(p, 2, closed))
    cand = [int(i) for i in np.where(t > 20.0)[0]]
    if len(cand) < 6:
        return []
    gaps = np.diff(np.array(cand, dtype=float))
    if closed:
        gaps = np.append(gaps, cand[0] + n - cand[-1])
    med = float(np.median(gaps)) if len(gaps) else 0.0
    if med < 3:
        return []
    feats = []
    run = [cand[0]]
    order = list(range(len(cand))) if not closed else list(range(len(cand)))
    for gi in range(len(gaps)):
        nxt = cand[(gi + 1) % len(cand)]
        if abs(float(gaps[gi]) - med) <= 0.35 * med:
            if closed or gi + 1 < len(cand):
                run.append(nxt)
            else:
                break
        else:
            if len(run) >= 6:
                feats.extend(run)
            run = [nxt]
            if not closed and gi + 1 >= len(cand):
                break
    if len(run) >= 6:
        feats.extend(run)
    if not feats:
        return []
    r = max(1, int(med // 4))
    out = []
    for f in feats:
        if closed:
            window = np.take(t, range(f - r, f + r + 1), mode='wrap')
            out.append((f - r + int(np.argmax(window))) % n)
        else:
            lo, hi = max(0, f - r), min(n, f + r + 1)
            out.append(lo + int(np.argmax(t[lo:hi])))
    return sorted(set(out))


def _fit_line(P):
    c = P.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        return c, Vt[0]
    except np.linalg.LinAlgError:
        d = P[-1] - P[0]
        n = float(np.hypot(*d))
        return c, (d / n if n > 1e-12 else np.array([1.0, 0.0]))


def _intersect(l1, l2):
    p, d = l1
    q, e = l2
    A = np.column_stack([d, -e])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    try:
        t = np.linalg.solve(A, q - p)
    except np.linalg.LinAlgError:
        return None
    return p + d * t[0]


def rebuild_corners(smooth_pts, corner_idx, w, step, closed, orig_pts,
                    max_move=None):
    """Rebuild corners as the outward intersection of the two smoothed edges.

    Pinning raw corner samples would freeze a jitter band around every tip;
    intersecting the neighbouring edges keeps tips sharp *and* clean.
    max_move budget-caps every rebuild so corners can never blow the budget.
    """
    out = np.asarray(smooth_pts, float).copy()
    orig = np.asarray(orig_pts, float)
    n = len(out)
    if n == 0:
        return out
    cap = w * step * 4
    if max_move is not None:
        cap = min(cap, max_move)
    snap_tol = w * step * 0.5
    snapped = np.zeros(n, bool)
    pins = {}
    for i in corner_idx:
        if closed:
            i1 = [(i - 2 * w + j) % n for j in range(w)]
            i2 = [(i + w + j) % n for j in range(w)]
        else:
            if i - 2 * w < 0 or i + 2 * w >= n:
                out[i] = orig[i]
                snapped[i] = True
                pins[i] = orig[i]
                continue
            i1 = list(range(i - 2 * w, i - w))
            i2 = list(range(i + w, i + 2 * w))
        # lines fit on the ORIGINAL flanks (noise-averaged), never on the
        # smoothed curve: smoothing would have already dragged the corner.
        A = orig[np.array(i1)]
        B = orig[np.array(i2)]
        l1, l2 = _fit_line(A), _fit_line(B)
        P = _intersect(l1, l2)
        if P is None or float(np.hypot(*(P - orig[i]))) > cap:
            out[i] = orig[i]  # no trustworthy rebuild: pin original exactly
            snapped[i] = True
            pins[i] = orig[i]
            continue
        out[i] = P
        snapped[i] = True
        pins[i] = P
        # snap flanks onto the rebuilt edges, extending along the feature
        # while the original stays on the line (wide spikes need wide snaps;
        # curving flanks stop the extension by themselves). First-wins keeps
        # neighbouring zones from overlapping.
        for side, line in ((-1, l1), (1, l2)):
            pt, dr = line
            misses = 0
            for j in range(1, 8 * w + 1):
                idx = (i + side * j) % n if closed else i + side * j
                if not closed and (idx < 0 or idx >= n):
                    break
                if snapped[idx]:
                    break
                o = orig[idx]
                if float(np.hypot(*(o - (pt + dr * float(np.dot(o - pt, dr)))))) > snap_tol:
                    misses += 1  # isolated jitter crossings don't end the zone
                    if misses >= 3:
                        break
                    continue
                misses = 0
                v = out[idx] - pt
                out[idx] = pt + dr * float(np.dot(v, dr))
                snapped[idx] = True
    # dissolve 1-sample steps left by skipped violators (collinear snapped
    # runs are untouched by this); corners are re-pinned exactly after.
    out = smooth(out, 1.0, closed)
    for i, P in pins.items():
        out[i] = P
    return out


def _poly_tangent(q, at_start):
    """Unit tangent from a least-squares quadratic fit (osculating parabola).

    A plain chord gives the MIDPOINT tangent (~2 deg off at the endpoint on
    gentle arcs); the parabola's derivative AT the endpoint is ~10x more
    accurate, which is what long cubic windows need.
    """
    q = np.asarray(q, float)
    m = len(q)
    if m < 3:
        d = q[-1] - q[0]
        n = float(np.hypot(*d))
        return d / n if n > 1e-12 else np.array([1.0, 0.0])
    d = q[-1] - q[0]
    L = float(np.hypot(*d))
    if L < 1e-12:
        return np.array([1.0, 0.0])
    ux, uy = d / L
    xr = (q[:, 0] - q[0, 0]) * ux + (q[:, 1] - q[0, 1]) * uy
    yr = -(q[:, 0] - q[0, 0]) * uy + (q[:, 1] - q[0, 1]) * ux
    A = np.column_stack([xr * xr, xr, np.ones(m)])
    try:
        (a, b, _), *_ = np.linalg.lstsq(A, yr, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return np.array([ux, uy])
    x0 = 0.0 if at_start else float(xr[-1])
    slope = 2 * a * x0 + b
    nrm = math.hypot(1.0, slope)
    return np.array([(ux - slope * uy) / nrm, (uy + slope * ux) / nrm])


def fit_single_cubic(pts):
    """Least-squares cubic with endpoint tangents. Returns ((P0..P3), err, imax)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    P0, P3 = pts[0].copy(), pts[-1].copy()
    chord = float(np.hypot(*(P3 - P0)))
    if n >= 6:
        m = max(4, min(12, n // 4))
        t0 = _poly_tangent(pts[:m], True)
        t1 = -_poly_tangent(pts[-m:], False)  # backward-pointing
    else:
        m = max(2, n // 4)
        t0 = pts[m] - pts[0]
        n0 = float(np.hypot(*t0))
        t0 = t0 / n0 if n0 > 1e-12 else np.array([1.0, 0.0])
        t1 = pts[n - 1 - m] - pts[-1]
        n1 = float(np.hypot(*t1))
        t1 = t1 / n1 if n1 > 1e-12 else np.array([-1.0, 0.0])
    if len(pts) >= 2:
        d = np.diff(pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        tot = float(seg.sum())
        u = (np.concatenate([[0.0], np.cumsum(seg)]) / tot if tot > 1e-12
             else np.linspace(0, 1, len(pts)))
    else:
        u = np.array([0.0])
    B0 = (1 - u) ** 3
    B1 = 3 * u * (1 - u) ** 2
    B2 = 3 * u * u * (1 - u)
    B3 = u ** 3
    A1 = t0[None, :] * B1[:, None]
    A2 = t1[None, :] * B2[:, None]
    # residual vs the endpoints' full Bernstein weight (Schneider):
    # fitted = P0*(B0+B1) + P3*(B2+B3) + a1*A1 + a2*A2
    tmp = pts - (P0[None, :] * (B0 + B1)[:, None]
                 + P3[None, :] * (B2 + B3)[:, None])
    C00 = float((A1 * A1).sum())
    C01 = float((A1 * A2).sum())
    C11 = float((A2 * A2).sum())
    X0 = float((A1 * tmp).sum())
    X1 = float((A2 * tmp).sum())
    det = C00 * C11 - C01 * C01
    if abs(det) > 1e-9 * (abs(C00 * C11) + 1e-18):
        a1 = (X0 * C11 - X1 * C01) / det
        a2 = (C00 * X1 - C01 * X0) / det
    else:
        a1 = a2 = chord / 3.0
    # handle-length clamp: no giant handles, no overshoot loops
    cap = max(chord * 1.2, 1e-9)
    a1 = float(min(max(a1, 0.0), cap))
    a2 = float(min(max(a2, 0.0), cap))
    P1 = P0 + t0 * a1
    P2 = P3 + t1 * a2
    F = (B0[:, None] * P0 + B1[:, None] * P1
         + B2[:, None] * P2 + B3[:, None] * P3)
    err = np.hypot((F - pts)[:, 0], (F - pts)[:, 1])
    imax = int(np.argmax(err))
    return (P0, P1, P2, P3), float(err[imax]), imax


def _cubic_dense_err(P0, P1, P2, P3, pts):
    """Max distance from a finely-sampled cubic to the polyline.

    Least-squares error is only measured AT data points; between samples a
    cubic can overshoot. Dense verification closes that hole.
    """
    pts = np.asarray(pts, float)
    tree = cKDTree(pts)
    n = max(16, len(pts) * 3)
    t = np.linspace(0, 1, n)
    mt = 1 - t
    Q = ((mt ** 3)[:, None] * P0 + (3 * mt * mt * t)[:, None] * P1
         + (3 * mt * t * t)[:, None] * P2 + (t ** 3)[:, None] * P3)
    d, _ = tree.query(Q)
    return float(d.max())


def _line_dev(pts):
    pts = np.asarray(pts, float)
    if len(pts) <= 2:
        return 0.0
    chord = pts[-1] - pts[0]
    L = float(np.hypot(*chord))
    if L < 1e-12:
        return 0.0
    dev = np.abs((pts[:, 0] - pts[0, 0]) * chord[1]
                 - (pts[:, 1] - pts[0, 1]) * chord[0]) / L
    return float(dev.max())


def _cubic_fit_err(pts, tol):
    """One cubic fit -> (seg, err), dense-verified when data-err passes."""
    pts = np.asarray(pts, float)
    if len(pts) <= 2:
        return ('L', pts[-1].copy()), 0.0
    (_, P1, P2, _), derr, _ = fit_single_cubic(pts)
    seg = ('C', P1, P2, pts[-1].copy())
    if derr > tol:
        return seg, derr
    cerr = _cubic_dense_err(pts[0], P1, P2, pts[-1], pts)
    return seg, max(derr, cerr)


def fit_span(pts, tol):
    """Greedy error-bounded fit of an open point span -> segment list.

    At each step the maximal line window AND the maximal cubic window are
    found (binary search) and whichever reaches farther wins: straights
    collapse to one L, arcs to a few C's, anchors land at natural
    transitions. Minimal segments, no micro-kink confetti.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    if n <= 2:
        return [('L', pts[-1].copy())] if n == 2 else []
    out = []
    i = 0
    while i < n - 1:
        # full-span check first: window error is NOT monotonic in length
        # (the chord refits every window), so binary search alone can miss
        # a passing full span; clean runs then collapse in one step.
        if _line_dev(pts[i:]) <= tol:
            bestL = n - 1
        else:
            bestL = i + 1
            lo, hi = i + 2, n - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if _line_dev(pts[i:mid + 1]) <= tol:
                    bestL = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
        seg_full, err_full = _cubic_fit_err(pts[i:], tol)
        if err_full <= tol:
            bestC, bestCseg = n - 1, seg_full
        else:
            bestC, bestCseg = i + 1, ('L', pts[i + 1].copy())
            lo, hi = i + 2, n - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                seg, err = _cubic_fit_err(pts[i:mid + 1], tol)
                if err <= tol:
                    bestC, bestCseg = mid, seg
                    lo = mid + 1
                else:
                    hi = mid - 1
        if bestL >= bestC:
            out.append(('L', pts[bestL].copy()))
            i = bestL
        else:
            out.append(bestCseg)
            i = bestC
        if len(out) > n + 4:  # paranoia: never loop forever
            break
    return out


def sample_segments(start, segments, step=None, per=14):
    """Dense-sample a segment chain (for deviation gates + QA render).

    With step set, samples scale with segment length so long straight runs
    are NOT under-sampled (under-sampling here caused false guard reverts).
    """
    pts = [np.asarray(start, float).copy()]
    cur = pts[0]
    for s in segments:
        if s[0] == 'L':
            p = np.asarray(s[1], float)
            seglen = float(np.hypot(*(p - cur)))
            n = max(2, int(np.ceil(seglen / step))) if step else per
            for t in np.linspace(0, 1, min(n, 20000) + 1)[1:]:
                pts.append(cur + (p - cur) * t)
            cur = p
        else:
            _, c1, c2, p = s
            c1 = np.asarray(c1, float)
            c2 = np.asarray(c2, float)
            p = np.asarray(p, float)
            if step:
                approx = (float(np.hypot(*(c1 - cur)))
                          + float(np.hypot(*(c2 - c1)))
                          + float(np.hypot(*(p - c2))))
                n = max(4, int(np.ceil(approx / step)))
            else:
                n = per
            for t in np.linspace(0, 1, min(n, 20000) + 1)[1:]:
                mt = 1 - t
                pts.append((mt ** 3) * cur + 3 * mt * mt * t * c1
                           + 3 * mt * t * t * c2 + (t ** 3) * p)
            cur = p
    return np.array(pts)


def polyline_deviation(a, b):
    """Symmetric max distance between two dense polylines."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    d1, _ = cKDTree(a).query(b)
    d2, _ = cKDTree(b).query(a)
    return float(max(d1.max(), d2.max()))


def has_self_intersection(poly, closed=True):
    """Proper (crossing) self-intersection test, adjacency excluded."""
    p = np.asarray(poly, float)
    if len(p) < 4:
        return False
    a = p if closed else p[:-1]
    b = np.roll(p, -1, axis=0) if closed else p[1:]
    n = len(a)
    if n < 4:
        return False
    for i in range(n):
        js = np.arange(i + 2, n)
        if closed and i == 0:
            js = js[js != (n - 1)]
        if len(js) == 0:
            continue
        p1, p2 = a[i], b[i]
        q1, q2 = a[js], b[js]
        o1 = (p2[0] - p1[0]) * (q1[:, 1] - p1[1]) - (p2[1] - p1[1]) * (q1[:, 0] - p1[0])
        o2 = (p2[0] - p1[0]) * (q2[:, 1] - p1[1]) - (p2[1] - p1[1]) * (q2[:, 0] - p1[0])
        o3 = (q2[:, 0] - q1[:, 0]) * (p1[1] - q1[:, 1]) - (q2[:, 1] - q1[:, 1]) * (p1[0] - q1[:, 0])
        o4 = (q2[:, 0] - q1[:, 0]) * (p2[1] - q1[:, 1]) - (q2[:, 1] - q1[:, 1]) * (p2[0] - q1[:, 0])
        if bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0))):
            return True
    return False


def guard_candidate(orig_dense, cand_dense, budget, closed=True, area_orig=None):
    """Accept-or-rollback gate. Returns (ok, reason, deviation)."""
    dev = polyline_deviation(orig_dense, cand_dense)
    if not np.isfinite(dev):
        return False, 'empty candidate geometry', dev
    if dev > budget:
        return False, 'deviation %.3f over budget %.3f' % (dev, budget), dev
    cand = np.asarray(cand_dense, float)
    if len(cand) > 1500:
        step = poly_len(cand, closed) / 1500.0
        cand = resample(cand, step, closed)
    if has_self_intersection(cand, closed):
        return False, 'would introduce a self-intersection', dev
    if area_orig is not None and abs(area_orig) > 0:
        ac = signed_area(cand_dense if len(cand_dense) >= 3 else cand)
        if abs(ac) < 1e-9 and abs(area_orig) > 0:
            return False, 'candidate collapsed (zero area)', dev
        if (ac > 0) != (area_orig > 0):
            return False, 'winding direction flipped', dev
        ratio = abs(ac / area_orig)
        if ratio < 0.5 or ratio > 2.0:
            return False, 'area changed too much (x%.2f)' % ratio, dev
    return True, 'ok', dev


def jitter_stats(pts, closed):
    """(spacing-relative jitter, absolute jitter in px)."""
    p = np.asarray(pts, float)
    if len(p) < 8:
        return 0.0, 0.0
    s = smooth(p, 1.0, closed)
    resid = np.hypot((p - s)[:, 0], (p - s)[:, 1])
    seg = seg_lengths(p, closed)
    seg = seg[seg > 1e-12]
    sp = float(np.median(seg)) if len(seg) else 1.0
    med = float(np.median(resid))
    return med / (sp + 1e-12), med


def estimate_noise(pts, closed):
    """Spacing-independent jitter estimate (median residual / median spacing)."""
    return jitter_stats(pts, closed)[0]
