"""Gate policy: marginal IoU is exonerated only when every other lock
verifies clean - otherwise the original is still returned.

The 0.985 safe-mode floor stays strict; exoneration just stops IoU from
single-handedly discarding good smoothing when the detail lock, chamfer,
changed-pixel and winding checks all agree the artwork is intact.
"""
import numpy as np

from core import __version__
from core.pipeline import run, _iou_exonerated

OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _wheel_svg():
    t = np.linspace(0, 2 * np.pi, 120)
    tire = ('M' + 'L'.join('%.1f,%.1f' % (300 + 150 * np.cos(a),
                                          300 + 150 * np.sin(a))
                           for a in t) + 'Z')
    rim = ('M' + 'L'.join('%.1f,%.1f' % (300 + 90 * np.cos(a),
                                         300 + 90 * np.sin(a))
                          for a in t) + 'Z')
    rng = np.random.default_rng(3)
    spokes = []
    for k in range(4):
        a = k * np.pi / 2 + 0.3
        cx, cy = 300 + 60 * np.cos(a), 300 + 60 * np.sin(a)
        sq = (np.array([(cx - 14, cy - 14), (cx + 14, cy - 14),
                        (cx + 14, cy + 14), (cx - 14, cy + 14)])
              + rng.normal(0, 1.2, (4, 2)))
        spokes.append('M' + 'L'.join('%.1f,%.1f' % tuple(q) for q in sq)
                      + 'Z')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
            '<path d="%s" fill="black" fill-rule="evenodd"/></svg>'
            % (tire + ' ' + rim + ' ' + ' '.join(spokes)))


def test_marginal_iou_exonerated_when_all_else_clean():
    svg = _wheel_svg()
    out = run(svg, dict(OPTS))
    g = out['qa']['gates']['iou']
    assert not g['pass'] and g['value'] < g['floor'], g  # genuinely marginal
    assert g.get('exonerated') is True, g
    assert out['global_revert'] is False
    assert out['svg_out'] != svg  # smoothing actually delivered
    s = out['summary']
    assert (s['holes_before'], s['holes_after']) == (9, 9), s
    assert any('smoothing kept' in n for n in out['notes']), out['notes']


def _m(iou):
    return {'silhouette_iou': iou}


def test_exoneration_policy_unit():
    ds = [(1, 1), (2, 2)]  # real verified hole samples
    assert _iou_exonerated(_m(0.978), ['iou'], True, ds, 0) is True
    assert _iou_exonerated(_m(0.9699), ['iou'], True, ds, 0) is False  # floor
    assert _iou_exonerated(_m(0.978), ['iou', 'changed'], True, ds, 0) is False
    assert _iou_exonerated(_m(0.978), ['changed'], True, ds, 0) is False
    assert _iou_exonerated(_m(0.978), ['iou'], True, ds, 1) is False  # dfail
    assert _iou_exonerated(_m(0.978), ['iou'], True, [], 0) is False  # no proof
    assert _iou_exonerated(_m(0.978), ['iou'], False, ds, 0) is False  # wind
    assert _iou_exonerated(_m(0.990), ['iou'], True, ds, 0) is True  # moot-but-true


def test_clean_art_untouched_by_policy():
    out = run(open('samples/blob_noise.svg').read(), dict(OPTS))
    g = out['qa']['gates']['iou']
    assert g['pass'] and 'exonerated' not in g, g
    assert out['global_revert'] is False


def test_engine_version_reported():
    out = run(open('samples/blob_noise.svg').read(), dict(OPTS))
    assert out['engine_version'] == __version__ == '1.2.0', out.get('engine_version')
