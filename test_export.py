"""Export integrity: compounds keep their holes, tiny art survives SAFE.

Regression cover for the two subject-changing export bugs: multi-subpath
elements were split into one <path> per subpath (filling evenodd holes
solid), and SAFE mode deleted small details as 'tiny fragments'.
"""
import re

import numpy as np

from core.svg_parse import parse_svg
from core.clean import smooth_artwork
from core.export import build_svg, to_eps

OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _noisy_sq(x0, y0, x1, y1, n=40, amp=1.5, seed=9):
    rng = np.random.default_rng(seed)
    per = []
    for xa, ya, xb, yb in [(x0, y0, x1, y0), (x1, y0, x1, y1),
                           (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        edge = np.stack([np.linspace(xa, xb, n), np.linspace(ya, yb, n)],
                         axis=1)[:-1]
        per.append(edge)
    p = np.concatenate(per) + rng.normal(0, amp, (4 * (n - 1), 2))
    return 'M' + 'L'.join('%.1f,%.1f' % tuple(q) for q in p) + 'Z'


def _compound_art():
    d = (_noisy_sq(100, 100, 500, 500) + ' '
         + _noisy_sq(200, 200, 400, 400, n=30))
    return parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
        '<path d="%s" fill="black" fill-rule="evenodd"/></svg>' % d)


def test_compound_stays_one_path():
    art = _compound_art()
    out = smooth_artwork(art, dict(OPTS))
    assert [r['status'] for r in out['results']] == ['smoothed', 'smoothed']
    for r in out['results']:
        assert r['dev'] <= r['budget'], r
    svg = build_svg(art, out['results'])
    paths = re.findall(r'<path[^>]*>', svg)
    assert len(paths) == 1, paths
    assert paths[0].count('M') == 2, paths[0][:100]
    assert 'evenodd' in paths[0]
    # and it re-opens as the same compound
    art2 = parse_svg(svg)
    assert len(art2['subpaths']) == 2


def test_compound_revert_stays_one_path():
    art = _compound_art()
    results = []
    for sp in art['subpaths']:
        pts = np.asarray(sp['pts'], float)
        results.append({'status': 'reverted', 'reason': 'forced',
                        'dev': 0.0, 'budget': 1.0,
                        'segments': [('L', q.copy()) for q in pts[1:]],
                        'start': pts[0].copy(), 'closed': sp['closed']})
    svg = build_svg(art, results)
    paths = re.findall(r'<path[^>]*>', svg)
    assert len(paths) == 1, paths
    assert paths[0].count('M') == 2, paths[0][:100]
    assert 'evenodd' in paths[0]


def test_eps_single_eofill_for_compound():
    art = _compound_art()
    out = smooth_artwork(art, dict(OPTS))
    eps, _ = to_eps(art, out['results'], level=10)
    assert eps.count('newpath') == 1, eps.count('newpath')
    assert eps.count('eofill') == 1


def test_tiny_kept_in_safe_removed_in_pro():
    tiny = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
            '<path d="M50,50L550,50L550,550L50,550Z"/>'
            '<path d="M300,300L301.5,300L301.5,301.5L300,301.5Z" '
            'fill="white"/></svg>')
    art = parse_svg(tiny)
    safe = smooth_artwork(art, dict(OPTS))
    assert safe['summary']['removed_tiny'] == 0, safe['summary']
    assert len(re.findall(r'<path', build_svg(art, safe['results']))) == 2
    art = parse_svg(tiny)
    pro = smooth_artwork(art, dict(OPTS, mode='pro'))
    assert pro['summary']['removed_tiny'] == 1, pro['summary']
    assert len(re.findall(r'<path', build_svg(art, pro['results']))) == 1
