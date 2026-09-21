"""Marketplace-ready EPS: spec-compliant file structure + honest preflight.

The EPS must open error-free anywhere (Illustrator 8/10, Inkscape,
marketplace validators): DSC header, no showpage, RGB-only paints,
Level-1 operators, %%EOF trailer. And the preflight must flag every
known marketplace blocker (open paths, strokes, text, raster, filters,
transparency, gradients, <use>, clips, size) so files pass review.
"""
from core.export import to_eps, validate_eps_text
from core.pipeline import export_eps
from core.svg_parse import parse_svg

OPTS = {'preset': 'auto', 'mode': 'safe', 'convert_shapes': True}


def _svg(body, vb='0 0 2000 2000'):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="%s">%s</svg>'
            % (vb, body))


def _statuses(preflight):
    return {c['id']: c['status'] for c in preflight}


def test_eps_structure_compliant():
    art = parse_svg(_svg('<path d="M100,100L1900,100L1900,1900L100,1900Z" '
                         'fill="black"/>'))
    from core.clean import smooth_artwork
    out = smooth_artwork(art, dict(OPTS))
    for level in (10, 8):
        eps, _ = to_eps(art, out['results'], level=level)
        assert validate_eps_text(eps) == [], validate_eps_text(eps)
        assert eps.split('\n')[0] == '%!PS-Adobe-3.0 EPSF-3.0'
        assert '%%BoundingBox: 0 0 2000 2000' in eps
        assert '%%HiResBoundingBox: 0 0 2000.000 2000.000' in eps
        assert 'EPS %d compatible' % level in eps
        assert 'showpage' not in eps, 'showpage forbidden in EPS'
        assert eps.split('\n')[-2] == '%%EOF'
        assert eps.isascii()
        assert 'setcmykcolor' not in eps and 'setgray' not in eps


def test_eps_validator_catches_corruption():
    art = parse_svg(_svg('<path d="M10,10L50,10L50,50L10,50Z" fill="red"/>'))
    from core.clean import smooth_artwork
    out = smooth_artwork(art, dict(OPTS))
    eps, _ = to_eps(art, out['results'], level=10)
    assert validate_eps_text(eps) == []
    assert validate_eps_text(eps.replace('%%EOF', '')) != []
    assert validate_eps_text(eps.replace('newpath', 'showpage\nnewpath', 1)) != []
    assert validate_eps_text('garbage\n') != []
    assert validate_eps_text(eps.replace('%%BoundingBox', '%%BBox')) != []


def test_preflight_clean_art_passes():
    eps, notes, out = export_eps(
        _svg('<path d="M100,100L1900,100L1900,1900L100,1900Z" fill="black"/>'),
        dict(OPTS), level=10)
    pf = out['preflight']
    assert pf[0]['id'] == 'format' and pf[0]['status'] == 'pass', pf[0]
    st = _statuses(pf)
    for cid in ('closed', 'stroke', 'fonts', 'raster', 'effects',
                'transparency', 'gradients', 'use', 'clipping'):
        assert st[cid] == 'pass', (cid, pf)
    assert not [c for c in pf if c['status'] == 'fail']
    assert any('all %d marketplace checks passed' % len(pf) in n
               for n in notes), notes


def test_preflight_flags_every_blocker():
    cases = [
        ('closed', '<path d="M100,100L900,900" fill="none" stroke="black"/>',
         ['closed']),  # open, unstroked? no: stroked too
        ('stroke', '<path d="M100,100L900,100L900,900L100,900Z" fill="none" '
                   'stroke="black" stroke-width="4"/>', ['stroke']),
        ('fonts', '<text x="10" y="20">Hi</text>'
                  '<path d="M10,10L50,10L50,50L10,50Z" fill="black"/>',
         ['fonts']),
        ('raster', '<image href="a.png" x="0" y="0" width="10" height="10"/>'
                   '<path d="M10,10L50,10L50,50L10,50Z" fill="black"/>',
         ['raster']),
        ('transparency', '<path d="M10,10L50,10L50,50L10,50Z" fill="black" '
                         'opacity="0.5"/>', ['transparency']),
        ('gradients', '<defs><linearGradient id="g"><stop offset="0" '
                      'stop-color="black"/></linearGradient></defs>'
                      '<path d="M10,10L50,10L50,50L10,50Z" fill="url(#g)"/>',
         ['gradients']),
    ]
    for cid, body, must_fail in cases:
        _, _, out = export_eps(_svg(body), dict(OPTS), level=10)
        st = _statuses(out['preflight'])
        for m in must_fail:
            assert st[m] == 'fail', (cid, m, out['preflight'])
    # open-path case also trips stroke (it is stroked) - both flagged
    _, _, out = export_eps(
        _svg('<path d="M100,100L900,900" fill="none" stroke="black"/>'),
        dict(OPTS), level=10)
    st = _statuses(out['preflight'])
    assert st['closed'] == 'fail' and st['stroke'] == 'fail'


def test_preflight_uses_eps_bytes_and_bbox_mp():
    _, _, out = export_eps(
        _svg('<path d="M100,100L1900,100L1900,1900L100,1900Z" fill="black"/>'),
        dict(OPTS), level=10)
    size = [c for c in out['preflight'] if c['id'] == 'size'][0]
    assert size['status'] == 'pass' and 'KB' in size['detail'], size
    mp = [c for c in out['preflight'] if c['id'] == 'mp'][0]
    assert 'bbox' in mp['label'] and mp['status'] == 'warn', mp  # 3.24MP<4


def test_preflight_level8_profile():
    _, _, out = export_eps(
        _svg('<path d="M10,10L50,10L50,50L10,50Z" fill="black"/>'),
        dict(OPTS), level=8)
    assert out['preflight'][0]['label'] == 'EPS 8 file structure'
