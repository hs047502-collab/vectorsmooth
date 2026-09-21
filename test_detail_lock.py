"""Detail lock: holes and cutouts survive smoothing.

A hole that gets filled in (or a detail that escapes its parent) is a
subject change, so containment pairs are verified geometrically
(containment + area consistency) and at the render stage (hole-interior
pixels must paint identically). Violations revert to the original.
"""
import numpy as np

from core.svg_parse import parse_svg
from core.clean import smooth_artwork, _detail_lock_check, _find_holes
from core.pipeline import run, hole_interior_samples, verify_hole_render
from core.raster import render_mask

OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _thin_wall_art():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 2 * np.pi, 160)
    ro = 200 + rng.normal(0, 2.5, len(t))
    outer = ('M' + 'L'.join('%.1f,%.1f' % (300 + r * np.cos(a), 300 + r * np.sin(a))
                            for a, r in zip(t, ro)) + 'Z')
    t2 = np.linspace(0, 2 * np.pi, 60)
    ri = 30 + rng.normal(0, 2.0, len(t2))
    hole = ('M' + 'L'.join('%.1f,%.1f' % (465 + r * np.cos(a), 300 + r * np.sin(a))
                           for a, r in zip(t2, ri)) + 'Z')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
            '<path d="%s %s" fill="black" fill-rule="evenodd"/></svg>'
            % (outer, hole))


def test_thin_wall_hole_survives_both_modes():
    for mode in ('safe', 'pro'):
        opts = dict(OPTS, mode=mode)
        out = run(_thin_wall_art(), opts)
        s = out['summary']
        assert s['holes_before'] == 1, '%s: %s' % (mode, s)
        assert s['holes_after'] == 1, '%s: %s' % (mode, s)
        assert out['qa']['detail']['failed'] == 0, '%s: %s' % (mode, out['qa']['detail'])
        assert out['qa']['detail']['samples'] > 0, '%s: no samples' % mode


def _sq(x0, y0, x1, y1, n=50):
    """Dense closed square (n points per edge)."""
    per = []
    for xa, ya, xb, yb in [(x0, y0, x1, y0), (x1, y0, x1, y1),
                           (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        edge = np.stack([np.linspace(xa, xb, n), np.linspace(ya, yb, n)],
                        axis=1)[:-1]
        per.append(edge)
    return np.concatenate(per)

def _fake_pair(child_new):
    parent = _sq(0., 0., 600., 600.)
    child = _sq(200., 200., 400., 400.)
    holes = _find_holes([parent, child], [True, True])
    assert len(holes) == 1 and holes[0]['child'] == 1
    subpaths = [{'pts': parent, 'closed': True},
                {'pts': child, 'closed': True}]
    results = [
        {'status': 'smoothed', 'closed': True, 'dev': 0.5,
         'dense_orig': parent, 'dense_new': parent},
        {'status': 'smoothed', 'closed': True, 'dev': 0.5,
         'dense_orig': child, 'dense_new': np.asarray(child_new, float)},
    ]
    return subpaths, results, holes


def test_detail_check_catches_escape():
    subpaths, results, holes = _fake_pair(
        _sq(700., 700., 800., 800.))  # child moved fully outside parent
    notes = []
    fixed = _detail_lock_check(subpaths, results, holes, notes)
    assert fixed == 1
    assert results[1]['status'] == 'reverted'
    assert results[0]['status'] == 'reverted'  # pair reverts together
    assert any('detail-lock' in n for n in notes)


def test_detail_check_catches_area_blowup():
    child = _sq(200., 200., 400., 400.)
    big = (child - 300) * 1.5 + 300  # x2.25 area, still inside parent
    subpaths, results, holes = _fake_pair(big)
    notes = []
    fixed = _detail_lock_check(subpaths, results, holes, notes)
    assert fixed == 1
    assert results[1]['status'] == 'reverted'


def test_detail_check_keeps_good_pair():
    child = _sq(200., 200., 400., 400.)
    subpaths, results, holes = _fake_pair(child + 0.3)  # sub-dev nudge
    notes = []
    fixed = _detail_lock_check(subpaths, results, holes, notes)
    assert fixed == 0
    assert results[1]['status'] == 'smoothed'
    assert notes == []


def test_hole_render_verify_catches_fill():
    parent = _sq(100., 100., 500., 500.)
    child = _sq(200., 200., 400., 400.)
    holes = [{'child': 1, 'parent': 0, 'area': 40000.0, 'perim': 800.0,
              'gap': 100.0}]
    mkres = lambda dn: {'status': 'smoothed', 'closed': True, 'dev': 0.0,
                        'dense_orig': dn, 'dense_new': dn}
    results = [mkres(parent), mkres(child)]
    vb = (0.0, 0.0, 600.0, 600.0)
    ds = hole_interior_samples(holes, results, vb, 1.0)
    assert len(ds) > 0
    mo = render_mask([(parent, True, True), (child, True, True)], vb)
    mn_same = render_mask([(parent, True, True), (child, True, True)], vb)
    ok, fail = verify_hole_render(mo, mn_same, ds)
    assert fail == 0 and ok == len(ds)
    mn_filled = render_mask([(parent, True, True)], vb)  # hole filled in
    ok2, fail2 = verify_hole_render(mo, mn_filled, ds)
    assert fail2 > 0, 'filled-in hole went undetected'


def test_summary_counts_paths_and_holes():
    art = parse_svg(_thin_wall_art())
    out = smooth_artwork(art, dict(OPTS))
    s = out['summary']
    assert (s['paths_before'], s['paths_after']) == (2, 2), s
    assert (s['holes_before'], s['holes_after']) == (1, 1), s
    assert s['detail_fixed'] == 0
    assert len(out['holes']) == 1
