"""End-to-end regression: the three sample files must smooth well.

Pinned behaviour (safe/auto): real node reduction, deviation inside the
per-path budget, zero reverts. Margins are deliberately loose so tiny
numeric wobble does not fail the suite; the hard promise is dev<=budget.
"""
from pathlib import Path

from core.svg_parse import parse_svg
from core.clean import smooth_artwork

SAMPLES = Path(__file__).resolve().parent.parent / 'samples'
OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _run(name):
    art = parse_svg((SAMPLES / (name + '.svg')).read_text())
    return art, smooth_artwork(art, dict(OPTS))


def test_star_jitter():
    art, out = _run('star_jitter')
    s = out['summary']
    assert s['reverted'] == 0, out['results']
    assert s['anchors_after'] <= 0.7 * s['anchors_before'], s
    for sp, r in zip(art['subpaths'], out['results']):
        if sp['tag'] == 'rect':
            continue
        assert r['status'] == 'smoothed', r
        assert r['dev'] <= r['budget'], r


def test_blob_noise():
    art, out = _run('blob_noise')
    s = out['summary']
    assert s['reverted'] == 0, out['results']
    assert s['anchors_after'] <= 0.7 * s['anchors_before'], s
    for sp, r in zip(art['subpaths'], out['results']):
        if sp['tag'] == 'rect':
            continue
        assert r['status'] == 'smoothed', r
        assert r['dev'] <= r['budget'], r


def test_rounded_square_dense():
    art, out = _run('rounded_square_dense')
    s = out['summary']
    assert s['reverted'] == 0, out['results']
    # dense polyline -> a handful of segments
    assert s['anchors_after'] <= 0.25 * s['anchors_before'], s
    for sp, r in zip(art['subpaths'], out['results']):
        if sp['tag'] == 'rect':
            continue
        assert r['status'] == 'smoothed', r
        assert r['dev'] <= r['budget'], r


def test_promise_holds_in_pro_mode():
    for name in ('star_jitter', 'blob_noise', 'rounded_square_dense'):
        art = parse_svg((SAMPLES / (name + '.svg')).read_text())
        opts = dict(OPTS, mode='pro')
        out = smooth_artwork(art, opts)
        for r in out['results']:
            if r['status'] == 'smoothed':
                assert r['dev'] <= r['budget'], (name, r)
