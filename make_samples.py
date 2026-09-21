"""Generate deterministic test artwork (noisy star / blob / dense square)."""
import numpy as np
from pathlib import Path

SAMPLES = {'star': 'star_jitter.svg',
           'blob': 'blob_noise.svg',
           'square': 'rounded_square_dense.svg'}


def _d_closed(pts):
    p = np.asarray(pts, float)
    s = 'M%.2f %.2f' % tuple(p[0])
    for q in p[1:]:
        s += 'L%.2f %.2f' % tuple(q)
    return s + 'Z'


def _wrap(path_id, fill, d, extra=''):
    return ("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">\n"""
            """<rect width="600" height="600" fill="#ffffff"/>\n"""
            """<path id="%s" fill="%s" d="%s"/>\n%s</svg>\n"""
            % (path_id, fill, d, extra))


def make_star(rng):
    cx, cy, R, r, n = 300.0, 300.0, 220.0, 95.0, 5
    tips = []
    for i in range(2 * n):
        ang = -np.pi / 2 + i * np.pi / n
        rad = R if i % 2 == 0 else r
        tips.append([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
    tips = np.array(tips)
    out = []
    for i in range(len(tips)):
        a, b = tips[i], tips[(i + 1) % len(tips)]
        for k in range(9):
            t = k / 9.0
            p = a + (b - a) * t
            if k > 0:
                p = p + rng.normal(0, 1.1, 2)
            out.append(p)
    out = np.array(out)
    out[13] += np.array([10.0, -8.0])  # one broken spike
    return _wrap('star', '#f5a623', _d_closed(out))


def make_blob(rng):
    cx, cy = 300.0, 300.0
    th = np.linspace(0, 2 * np.pi, 151)[:-1]
    rad = (200.0 + 6.0 * np.sin(3 * th) + 4.0 * np.sin(7 * th + 1.0)
           + rng.normal(0, 2.5, len(th)))
    pts = np.column_stack([cx + rad * np.cos(th), cy + rad * np.sin(th)])
    return _wrap('blob', '#2f6fed', _d_closed(pts))


def make_square(rng):
    x, y, w, h, r = 130.0, 130.0, 340.0, 340.0, 70.0
    pts = []
    # dense collinear points along the four straight edges
    edges = [((x + r, y), (x + w - r, y)),
             ((x + w, y + r), (x + w, y + h - r)),
             ((x + w - r, y + h), (x + r, y + h)),
             ((x, y + h - r), (x, y + r))]
    corners = [((x + w - r, y + r), -90, 0),
               ((x + w - r, y + h - r), 0, 90),
               ((x + r, y + h - r), 90, 180),
               ((x + r, y + r), 180, 270)]
    for (a, b), ((ccx, ccy), a0, a1) in zip(edges, corners):
        a, b = np.array(a), np.array(b)
        L = float(np.hypot(*(b - a)))
        for k in range(int(L / 5)):
            t = k / max(1, int(L / 5))
            p = a + (b - a) * t + rng.normal(0, 0.35, 2)
            pts.append(p)
        for k in range(12):
            th = np.radians(a0 + (a1 - a0) * k / 11)
            pts.append([ccx + r * np.cos(th) + rng.normal(0, 0.3),
                        ccy + r * np.sin(th) + rng.normal(0, 0.3)])
    return _wrap('tile', '#7b61ff', _d_closed(np.array(pts)))


def ensure_samples(force=False):
    d = Path(__file__).resolve().parent.parent / 'samples'
    d.mkdir(exist_ok=True)
    makers = {'star': make_star, 'blob': make_blob, 'square': make_square}
    rng = np.random.default_rng(7)
    for key, fname in SAMPLES.items():
        p = d / fname
        if force or not p.exists():
            p.write_text(makers[key](rng))
    return d


if __name__ == '__main__':
    ensure_samples(force=True)
    print('samples written')
