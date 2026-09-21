"""Marketplace preflight checklist + diagnostic quality scores.

Scores are internal diagnostics, never an acceptance guarantee - every
marketplace also applies its own moderation review.
"""
from .geometry import signed_area

SITES = {
    'generic_svg': dict(label='Generic SVG', formats='SVG', max_mb=None,
                        mp_min=None, mp_max=None, forbid_open=False,
                        forbid_stroke=False, forbid_text=False,
                        forbid_raster=False, forbid_filter=False,
                        require_00=False, transparent_bg=False,
                        forbid_transparency=False,
                        forbid_gradient=False, forbid_use=False),
    'adobe_stock': dict(label='Adobe Stock', formats='AI / EPS / SVG', max_mb=45,
                        mp_min=15, mp_max=65, mp_basis='artboard',
                        forbid_open=False, forbid_stroke=False, forbid_text=True,
                        forbid_raster=True, forbid_filter=False,
                        require_00=True, transparent_bg=True,
                        forbid_transparency=False,
                        forbid_gradient=False, forbid_use=False),
    'shutterstock': dict(label='Shutterstock', formats='EPS 10 / EPS 8',
                         max_mb=100, mp_min=4, mp_max=25, mp_basis='bbox',
                         forbid_open=True, forbid_stroke=True, forbid_text=True,
                         forbid_raster=True, forbid_filter=False,
                         require_00=False, transparent_bg=False,
                        forbid_transparency=True,
                        forbid_gradient=True, forbid_use=True),
    'vecteezy': dict(label='Vecteezy', formats='EPS 10', max_mb=None,
                     mp_min=4, mp_max=25, mp_basis='bbox',
                     forbid_open=True, forbid_stroke=True, forbid_text=True,
                     forbid_raster=True, forbid_filter=True,
                     require_00=False, transparent_bg=False,
                        forbid_transparency=True,
                        forbid_gradient=True, forbid_use=True),
    'freepik': dict(label='Freepik', formats='EPS (CS6 / CC2017+)',
                    max_mb=80, min_kb=500, mp_min=None, mp_max=None,
                    forbid_open=True, forbid_stroke=True, forbid_text=True,
                    forbid_raster=True, forbid_filter=False,
                    require_00=False, transparent_bg=False,
                    forbid_transparency=True,
                    forbid_gradient=True, forbid_use=True),
    'eps10': dict(label='EPS 10', formats='EPS 10', max_mb=100,
                  mp_min=4, mp_max=25, mp_basis='bbox',
                  forbid_open=True, forbid_stroke=True, forbid_text=True,
                  forbid_raster=True, forbid_filter=True,
                  require_00=False, transparent_bg=False,
                        forbid_transparency=True,
                        forbid_gradient=True, forbid_use=True),
    'eps8': dict(label='EPS 8', formats='EPS 8', max_mb=100,
                 mp_min=4, mp_max=25, mp_basis='bbox',
                 forbid_open=True, forbid_stroke=True, forbid_text=True,
                 forbid_raster=True, forbid_filter=True,
                 require_00=False, transparent_bg=False,
                        forbid_transparency=True,
                        forbid_gradient=True, forbid_use=True),
    'icon': dict(label='Icon set', formats='SVG', max_mb=None,
                 mp_min=None, mp_max=None, forbid_open=False,
                 forbid_stroke=False, forbid_text=True,
                 forbid_raster=True, forbid_filter=False,
                 require_00=True, transparent_bg=True,
                        forbid_transparency=False,
                        forbid_gradient=False, forbid_use=False),
    'silhouette': dict(label='Silhouette', formats='SVG / EPS', max_mb=None,
                       mp_min=None, mp_max=None, forbid_open=False,
                       forbid_stroke=True, forbid_text=True,
                       forbid_raster=True, forbid_filter=False,
                       require_00=False, transparent_bg=False,
                        forbid_transparency=False,
                        forbid_gradient=False, forbid_use=False),
}

SITE_ORDER = ['generic_svg', 'adobe_stock', 'shutterstock', 'vecteezy',
              'freepik', 'eps10', 'eps8', 'icon', 'silhouette']


def _ck(cid, label, status, detail=''):
    return {'id': cid, 'label': label, 'status': status, 'detail': detail}


def build_checklist(ctx, site_key):
    """ctx: dict of measured facts (see server.py). Returns (checks, scores)."""
    site = SITES.get(site_key, SITES['generic_svg'])
    C = []
    n_open = ctx.get('open_paths', 0)
    if n_open == 0:
        C.append(_ck('closed', 'Closed paths', 'pass', 'all paths closed'))
    elif site['forbid_open']:
        C.append(_ck('closed', 'Closed paths', 'fail',
                     '%d open path(s) - this profile forbids them' % n_open))
    else:
        C.append(_ck('closed', 'Closed paths', 'warn',
                     '%d open path(s) kept as-is' % n_open))
    C.append(_ck('dups', 'No duplicate points', 'pass',
                 '%d removed' % ctx.get('dups_removed', 0)))
    if ctx.get('spikes_removed', 0):
        C.append(_ck('spikes', 'No accidental spikes', 'pass',
                     '%d spike(s) repaired' % ctx['spikes_removed']))
    else:
        C.append(_ck('spikes', 'No accidental spikes', 'pass', 'none found'))
    tiny_left = ctx.get('tiny_remaining', 0)
    if tiny_left:
        C.append(_ck('fragments', 'No micro fragments', 'warn',
                     '%d tiny object(s) kept (remove_tiny off)' % tiny_left))
    else:
        C.append(_ck('fragments', 'No micro fragments', 'pass',
                     '%d removed' % ctx.get('tiny_removed', 0)))
    si = ctx.get('self_intersections', 0)
    C.append(_ck('selfint', 'No self-intersections',
                 'pass' if si == 0 else 'fail',
                 'none' if si == 0 else '%d found - FIX BEFORE UPLOAD' % si))
    C.append(_ck('winding', 'Winding / compound paths preserved',
                 'pass' if ctx.get('winding_ok', True) else 'fail',
                 'outer/inner orientation intact'
                 if ctx.get('winding_ok', True) else 'winding changed!'))
    if ctx.get('subpaths_before', 0) == ctx.get('subpaths_after', 0):
        C.append(_ck('topocount', 'Topology: subpath count', 'pass',
                     '%d subpaths intact' % ctx.get('subpaths_after', 0)))
    else:
        C.append(_ck('topocount', 'Topology: subpath count', 'warn',
                     '%d -> %d (tiny/degenerate only)' % (
                         ctx.get('subpaths_before', 0),
                         ctx.get('subpaths_after', 0))))
    off = ctx.get('off_artboard', 0)
    C.append(_ck('offboard', 'No objects outside artboard',
                 'pass' if off == 0 else 'warn',
                 'all inside' if off == 0 else '%d flagged' % off))
    size_kb = ctx.get('out_bytes', 0) / 1024.0
    if site.get('max_mb') and ctx.get('out_bytes', 0) > site['max_mb'] * 1024 * 1024:
        C.append(_ck('size', 'File size <= %d MB' % site['max_mb'], 'fail',
                     '%.1f KB' % size_kb))
    elif site.get('min_kb') and size_kb < site['min_kb']:
        C.append(_ck('size', 'File size in range', 'warn',
                     '%.1f KB (profile min %d KB)' % (size_kb, site['min_kb'])))
    else:
        C.append(_ck('size', 'File size valid', 'pass', '%.1f KB' % size_kb))
    mp = ctx.get('mp_value')
    if site.get('mp_min') is not None and mp is not None:
        ok = site['mp_min'] <= mp <= site['mp_max']
        C.append(_ck('mp', 'Artwork size %g-%g MP (%s)' % (
            site['mp_min'], site['mp_max'], site.get('mp_basis', 'artboard')),
            'pass' if ok else 'warn',
            'current %.2f MP%s' % (mp, '' if ok else
                                  ' - rescale artboard, geometry unaffected')))
    else:
        C.append(_ck('mp', 'Artwork size', 'pass',
                     '%.2f MP (no constraint)' % (mp or 0)))
    n_stroke = ctx.get('stroked_paths', 0)
    if n_stroke == 0:
        C.append(_ck('stroke', 'Strokes expanded / none', 'pass', 'no strokes'))
    elif site['forbid_stroke']:
        C.append(_ck('stroke', 'Strokes expanded', 'fail',
                     '%d stroked path(s) - expand to fills' % n_stroke))
    else:
        C.append(_ck('stroke', 'Strokes', 'warn',
                     '%d stroked path(s) preserved' % n_stroke))
    if ctx.get('n_text', 0):
        C.append(_ck('fonts', 'Fonts outlined', 'fail' if site['forbid_text'] else 'warn',
                     '%d <text> element(s) - convert to outlines'
                     % ctx['n_text']))
    else:
        C.append(_ck('fonts', 'Fonts outlined', 'pass', 'no live text'))
    C.append(_ck('raster', 'No embedded raster', 'pass' if not ctx.get('n_image', 0) else 'fail',
                 'none' if not ctx.get('n_image', 0)
                 else '%d <image> element(s)' % ctx['n_image']))
    if ctx.get('n_filter', 0):
        C.append(_ck('effects', 'No unsupported effects',
                     'fail' if site['forbid_filter'] else 'warn',
                     '%d filter(s) present' % ctx['n_filter']))
    else:
        C.append(_ck('effects', 'No unsupported effects', 'pass', 'none'))
    svg_only = site['formats'] == 'SVG'
    n_tr = ctx.get('n_transparent', 0)
    if n_tr == 0:
        C.append(_ck('transparency', 'No transparency', 'pass', 'all opaque'))
    elif site['forbid_transparency']:
        C.append(_ck('transparency', 'No transparency', 'fail',
                     '%d transparent path(s) - flatten in Illustrator '
                     '(no transparency in EPS 8/10)' % n_tr))
    elif svg_only:
        C.append(_ck('transparency', 'Transparency', 'pass',
                     '%d transparent path(s) - preserved in SVG' % n_tr))
    else:
        C.append(_ck('transparency', 'Transparency', 'warn',
                     '%d transparent path(s) - flattened in EPS export'
                     % n_tr))
    n_gr = ctx.get('n_gradient', 0)
    if n_gr == 0:
        C.append(_ck('gradients', 'No gradients', 'pass', 'solid paints only'))
    elif site['forbid_gradient']:
        C.append(_ck('gradients', 'No gradients', 'fail',
                     '%d gradient-filled path(s) - expand to fills in '
                     'Illustrator (EPS export flattens them)' % n_gr))
    elif svg_only:
        C.append(_ck('gradients', 'Gradients', 'pass',
                     '%d gradient fill(s) - preserved in SVG' % n_gr))
    else:
        C.append(_ck('gradients', 'Gradients', 'warn',
                     '%d gradient fill(s) - flattened in EPS export' % n_gr))
    n_use = ctx.get('n_use', 0)
    if n_use == 0:
        C.append(_ck('use', 'No <use> instances', 'pass', 'none'))
    elif site['forbid_use']:
        C.append(_ck('use', 'No <use> instances', 'fail',
                     '%d <use> instance(s) - expand in Illustrator '
                     '(dropped in EPS export)' % n_use))
    elif svg_only:
        C.append(_ck('use', '<use> instances', 'pass',
                     '%d <use> instance(s) - preserved in SVG' % n_use))
    else:
        C.append(_ck('use', '<use> instances', 'warn',
                     '%d <use> instance(s) - dropped in EPS export' % n_use))
    if ctx.get('n_clip', 0) == 0:
        C.append(_ck('clipping', 'No clipping paths', 'pass', 'none'))
    elif svg_only:
        C.append(_ck('clipping', 'Clipping paths', 'pass',
                     'clipPath(s) present - preserved in SVG'))
    else:
        C.append(_ck('clipping', 'Clipping paths', 'warn',
                     'clipPath(s) present - ignored in EPS export, expand '
                     'clipping masks in Illustrator and compare'))
    if ctx.get('artboard_at_origin', True):
        C.append(_ck('origin', 'Artboard at (0,0)', 'pass', 'viewBox origin 0,0'))
    else:
        C.append(_ck('origin', 'Artboard at (0,0)',
                     'warn' if site['require_00'] else 'pass',
                     'viewBox starts at %s' % ctx.get('vb_origin', '?')))
    if ctx.get('fullbleed_bg', False):
        C.append(_ck('bg', 'Transparent background',
                     'warn' if site['transparent_bg'] else 'pass',
                     'full-bleed opaque rect found'
                     if site['transparent_bg'] else 'opaque bg rect kept'))
    else:
        C.append(_ck('bg', 'Transparent background', 'pass', 'no bg rect'))

    fails = sum(1 for c in C if c['status'] == 'fail')
    warns = sum(1 for c in C if c['status'] == 'warn')
    iou = ctx.get('iou', 1.0)
    cham = ctx.get('chamfer_px', 0.0)
    red = ctx.get('reduction', 0.0)
    jb, ja = ctx.get('jitter_before', 0.0), ctx.get('jitter_after', 0.0)

    gq = 100 - (30 if si else 0) - (10 if (n_open and site['forbid_open']) else
                                    min(9, 3 * n_open))
    gq -= min(6, 2 * tiny_left) + (5 if (n_stroke and site['forbid_stroke']) else 0)
    gq -= (5 if ctx.get('n_text') else 0) + (5 if ctx.get('n_image') else 0)
    gq -= (5 if (ctx.get('n_transparent') and site['forbid_transparency']) else 0)
    gq -= (5 if (ctx.get('n_gradient') and site['forbid_gradient']) else 0)
    gq -= (5 if (ctx.get('n_use') and site['forbid_use']) else 0)
    topo = 100 - (40 if si else 0) - (0 if ctx.get('winding_ok', True) else 20)
    if ctx.get('subpaths_before', 0) != ctx.get('subpaths_after', 0):
        topo -= 10
    if jb < 0.05:
        smooth_q = 100.0
    elif ctx.get('n_smoothed', 0) == 0:
        smooth_q = 60.0
    else:
        smooth_q = max(5.0, min(99.0, 100.0 * (1.0 - ja / max(jb, 1e-9))))
    if red >= 0.9:
        node_q = 99
    elif red >= 0.7:
        node_q = 95
    elif red >= 0.5:
        node_q = 90
    elif red >= 0.3:
        node_q = 80
    elif red >= 0.1:
        node_q = 65
    elif red > 0:
        node_q = 50
    else:
        node_q = 100 if jb < 0.05 else 30
    scores = {
        'geometry_quality': int(max(0, min(100, round(gq)))),
        'silhouette_fidelity': round(max(0.0, min(100.0, 100.0 * iou
                                                 - min(5.0, cham))), 2),
        'topology': int(max(0, min(100, topo))),
        'curve_smoothness': round(smooth_q, 1),
        'node_optimization': int(node_q),
        'marketplace_readiness': int(max(0, min(100, 100 - 20 * fails - 4 * warns))),
    }
    return C, scores
