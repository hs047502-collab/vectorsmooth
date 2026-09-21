"""Subject-lock tests: the main design must NEVER change.

Layers (all enforced, not advisory):
1. cleanup budget - spike/zigzag removal shares the movement budget
2. clearance caps - neighbour paths can never merge (40% of gap each)
3. separation backstop - pairs that would touch are reverted
4. QA gates - render-vs-render floors trip a global revert to the original
"""
import numpy as np

from core.svg_parse import parse_svg
from core.clean import smooth_artwork, _separation_check
from core.pipeline import qa_gates_pass

OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _svg(paths):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
            + ''.join('<path d="%s"/>' % d for d in paths) + '</svg>')


def _d(pts, closed=True):
    s = 'M%.2f,%.2f' % tuple(pts[0]) + ''.join('L%.2f,%.2f' % tuple(p)
                                                for p in pts[1:])
    return s + 'Z' if closed else s


def test_big_spike_is_kept_as_subject():
    # 15 u spike: _remove_spikes would take it (under its own 3%-of-D cap),
    # but the cleanup budget (safe B_cap ~2.9 u) must veto the removal.
    # midpoints offset 2 u so collinear cleanup keeps them (n >= 8 gate)
    sq = [(50, 550), (52, 300), (50, 50), (200, 48), (298, 50),
          (300, 35), (302, 50), (400, 48), (550, 50), (548, 300),
          (550, 550), (300, 552)]
    art = parse_svg(_svg([_d(sq)]))
    out = smooth_artwork(art, dict(OPTS))
    s = out['summary']
    assert s['spikes_removed'] == 0, s
    r = out['results'][0]
    assert r['dev'] <= r['budget'], r
    tip = np.asarray(r['dense_orig'])[:, 1].min()
    assert tip <= 36.0, 'raw spike tip missing from dense_orig: %s' % tip
    assert any('subject kept' in n for n in out['notes']), out['notes']


def test_small_spike_is_still_cleaned():
    # 2.2 u spike on a 1 u base of a dense uniform line: genuine artifact
    # scale, inside the budget -> removed; the far rect only sets D.
    x = np.linspace(100, 500, 801)
    # gentle wave (turn ~2.8 deg/pt) so collinear cleanup keeps the points
    y = 300.0 + 0.5 * np.sin(2 * np.pi * (x - 100) / 10.0)
    y[400] += 2.2
    line = list(zip(x, y))
    rect = ('M0,0L600,0L600,600L0,600Z')
    art = parse_svg(_svg([_d(line, False), rect]))
    opts = dict(OPTS, convert_shapes=False)
    out = smooth_artwork(art, opts)
    assert out['summary']['spikes_removed'] == 1, out['summary']
    r = out['results'][0]
    assert r['dev'] <= r['budget'], r


def test_neighbour_paths_can_never_merge():
    rng = np.random.default_rng(5)
    x = np.linspace(100, 500, 81)
    wob = 3 * np.sin(2 * np.pi * (x - 100) / 80)
    n1 = rng.normal(0, 1.0, 81)
    p1 = list(zip(x, 300 + wob + n1))
    p2 = list(zip(x, 302 + wob + n1))  # same noise: gap exactly 2 u
    art = parse_svg(_svg([_d(p1, False), _d(p2, False)]))
    out = smooth_artwork(art, dict(OPTS))
    for r in out['results']:
        assert r['budget'] <= 0.85, r  # 40% of the 2 u gap
        if r['status'] == 'smoothed':
            assert r['dev'] <= r['budget'], r
    from scipy.spatial import cKDTree
    a = np.asarray(out['results'][0]['dense_new'], float)
    b = np.asarray(out['results'][1]['dense_new'], float)
    gap = float(cKDTree(a).query(b)[0].min())
    assert gap > 0.2, 'neighbour gap closed: %s' % gap
    assert any('neighbour gap' in n for n in out['notes']), out['notes']


def test_separation_backstop_reverts_touching_pairs():
    x = np.linspace(0, 100, 60)
    before_a = np.stack([x, np.zeros(60)], axis=1)
    before_b = np.stack([x, np.full(60, 8.0)], axis=1)
    after_b = np.stack([x, np.full(60, 0.05)], axis=1)  # moved onto `a`
    mk = lambda do, dn: {'status': 'smoothed', 'closed': False,
                         'dense_orig': do, 'dense_new': dn}
    results = [mk(before_a, before_a), mk(before_b, after_b)]
    subpaths = [{'pts': before_a}, {'pts': before_b}]
    notes = []
    touched, gb, ga = _separation_check(subpaths, results, notes)
    assert touched == 1, (touched, gb, ga)
    assert all(r['status'] == 'reverted' for r in results), results
    assert all(r['dev'] == 0.0 for r in results), results


def test_separation_backstop_ignores_shared_edges():
    # icon-style adjacent shapes already touching in the input: not a crime.
    x = np.linspace(0, 100, 60)
    a = np.stack([x, np.zeros(60)], axis=1)
    b = np.stack([x, np.full(60, 0.1)], axis=1)
    mk = lambda do, dn: {'status': 'smoothed', 'closed': False,
                         'dense_orig': do, 'dense_new': dn}
    results = [mk(a, a), mk(b, b)]
    notes = []
    touched, _, _ = _separation_check(results and [{'pts': a}, {'pts': b}],
                                       results, notes)
    assert touched == 0, notes


def test_qa_gates():
    good = {'silhouette_iou': 0.998, 'pixels_changed_pct': 0.2,
            'chamfer_px': 0.15}
    ok, g = qa_gates_pass(good, 'safe')
    assert ok and g['pass'], g
    bad = dict(good, silhouette_iou=0.95)
    ok, g = qa_gates_pass(bad, 'safe')
    assert not ok and not g['iou']['pass'] and g['changed']['pass'], g
    bad = dict(good, pixels_changed_pct=9.0)
    ok, g = qa_gates_pass(bad, 'pro')
    assert not ok and not g['changed']['pass'], g
    # pro floors are looser than safe floors
    mid = dict(good, silhouette_iou=0.98)
    assert not qa_gates_pass(mid, 'safe')[0]
    assert qa_gates_pass(mid, 'pro')[0]
