"""One pipeline used by the web server and the CLI.

run(svg_text, opts) -> everything the UI/CLI needs: smoothed SVG, summary,
notes, pixel QA, marketplace checklist + scores, compare PNG.
"""
import base64
import time

import numpy as np
from scipy.spatial import cKDTree

from .svg_parse import parse_svg
from . import __version__ as ENGINE_VERSION
from .clean import smooth_artwork
from .raster import render_mask, compare_masks, compare_png
from .validate import build_checklist, SITES
from .export import build_svg, to_eps, validate_eps_text
from .geometry import (has_self_intersection, resample, poly_len, signed_area,
                       points_in_polygon)


def _cap(pts, n=3000, closed=True):
    p = np.asarray(pts, float)
    if len(p) <= n or len(p) < 2:
        return p
    step = poly_len(p, closed) / n
    return resample(p, max(step, 1e-9), closed)


QA_GATES = {
    # subject-lock floors: the render-vs-render proof that the artwork did
    # not change. Breach -> the ORIGINAL file is returned, no exceptions.
    'safe': {'iou': 0.985, 'changed': 2.0, 'chamfer': 1.5},
    'pro': {'iou': 0.975, 'changed': 4.0, 'chamfer': 2.5},
}


def qa_gates_pass(m, mode):
    """Hard subject-lock verdict on a render-compare dict. (ok, gates)"""
    g = QA_GATES.get(mode, QA_GATES['safe'])
    gates = {
        'iou': {'value': float(m['silhouette_iou']), 'floor': g['iou'],
                'pass': bool(m['silhouette_iou'] >= g['iou'])},
        'changed': {'value': float(m['pixels_changed_pct']),
                    'ceil': g['changed'],
                    'pass': bool(m['pixels_changed_pct'] <= g['changed'])},
        'chamfer': {'value': float(m['chamfer_px']), 'ceil': g['chamfer'],
                    'pass': bool(m['chamfer_px'] <= g['chamfer'])},
    }
    gates['pass'] = all(v['pass'] for v in gates.values())
    return gates['pass'], gates


IOU_EXONERATE_FLOOR = 0.97


def _iou_exonerated(m, failed, wind_ok, samples, dfail):
    """Marginal-IoU exoneration: IoU alone may not discard good work.

    Fires only when ALL of these hold: IoU is the SOLE failing gate, it
    still clears the absolute 0.97 floor, winding is preserved, and the
    detail lock verified hole interiors pixel-identical (samples > 0
    proves real holes existed to verify). Anything else -> no exoneration,
    the original is returned as before.
    """
    if failed != ['iou']:
        return False
    if float(m['silhouette_iou']) < IOU_EXONERATE_FLOOR:
        return False
    if not wind_ok:
        return False
    if dfail or not samples:
        return False
    return True


def verify_hole_render(mo, mn, samples):
    """Hole interiors must render IDENTICALLY before/after.

    samples: list of (row, col) pixels strictly inside holes.
    A filled-in cutout flips background->paint and fails the check.
    Returns (ok_count, fail_count).
    """
    ok = fail = 0
    h, w = mo.shape
    for (r, c) in samples:
        if not (0 <= r < h and 0 <= c < w):
            continue
        if abs(float(mo[r, c]) - float(mn[r, c])) < 0.5:
            ok += 1
        else:
            fail += 1
    return ok, fail


def hole_interior_samples(holes, results, vb, scale, cap=400):
    """Artwork-coords sample points strictly inside surviving holes."""
    vx, vy, _, _ = vb
    px = 1.0 / max(scale, 1e-9)  # artwork units per QA pixel
    samples = []
    for h in holes:
        i, j = h['child'], h['parent']
        ri, rj = results[i], results[j]
        if ri['status'] == 'removed' or rj['status'] == 'removed':
            continue
        if h['area'] < (2 * px) ** 2:
            continue  # subpixel at QA resolution: cannot verify, skip
        ci = np.asarray(ri['dense_orig'], float)
        cj = np.asarray(rj['dense_orig'], float)
        ni = np.asarray(ri['dense_new'], float)
        if len(ci) < 8 or len(cj) < 8 or len(ni) < 8:
            continue
        c = ci.mean(axis=0)
        idx = np.linspace(0, len(ci) - 1, 9).astype(int)
        cands = np.vstack([c, ci[idx] + (c - ci[idx]) * 0.25])
        inside = (points_in_polygon(cands, ci)
                  & points_in_polygon(cands, cj))
        dev_sum = max(float(ri['dev']), 0.0) + max(float(rj['dev']), 0.0)
        step = poly_len(ni, True) / max(len(ni), 1)
        need = dev_sum + step
        for q in cands[inside]:
            # unreliable if a moved wall could have reached the sample
            if cKDTree(ni).query(q)[0] < need:
                continue
            col = int(np.floor((q[0] - vx) * scale))
            row = int(np.floor((q[1] - vy) * scale))
            samples.append((row, col))
            if len(samples) >= cap:
                return samples
    return samples


def run(svg_text, opts, do_qa=True):
    t0 = time.time()
    opts = dict(opts or {})
    art = parse_svg(svg_text)
    pipe = smooth_artwork(art, opts)
    P = pipe['params']
    results = pipe['results']
    summary = dict(pipe['summary'])
    notes = list(pipe['notes'])
    if art['stats'].get('degenerate'):
        notes.append('%d degenerate anchor(s) dropped'
                     % art['stats']['degenerate'])

    # global revert: nothing improved -> original bytes untouched
    global_revert = (summary['smoothed'] == 0 and summary['removed_tiny'] == 0
                     and len(art['subpaths']) > 0)
    if global_revert:
        svg_out = svg_text
        if summary['reverted']:
            notes.append('guard reverted every path - original file returned untouched')
    else:
        svg_out = build_svg(art, results)
        try:
            parse_svg(svg_out)  # re-open exported file: must parse
        except ValueError as e:
            notes.append('export re-parse failed (%s) - original returned' % e)
            svg_out = svg_text
            global_revert = True

    # verify self-intersections on input + output geometry
    n_before = n_after = 0
    for sp, r in zip(art['subpaths'], results):
        do = _cap(r['dense_orig'], 1200, sp['closed'])
        if len(do) >= 4 and has_self_intersection(do, sp['closed']):
            n_before += 1
        if r['status'] == 'removed':
            continue
        dn = _cap(r['dense_new'], 1200, r['closed'])
        if len(dn) >= 4 and has_self_intersection(dn, r['closed']):
            n_after += 1
    summary['self_intersections'] = n_after
    if n_before and n_after:
        notes.append('input already had %d self-intersecting path(s) - kept as-is, fix manually'
                     % n_before)

    # winding verify
    wind_ok = True
    for sp, r in zip(art['subpaths'], results):
        if r['status'] == 'removed':
            continue
        ao = signed_area(r['dense_orig'])
        an = signed_area(r['dense_new'])
        if abs(ao) > 1e-9 and abs(an) > 1e-9 and (ao > 0) != (an > 0):
            wind_ok = False

    # pixel QA
    qa = None
    cmp_b64 = None
    iou, cham, changed, ssim = 1.0, 0.0, 0.0, 1.0
    if do_qa:
        items_o, items_n = [], []
        for sp, r in zip(art['subpaths'], results):
            filled = (sp['fill'] or '').strip().lower() not in (
                '', 'none', 'transparent')
            items_o.append((_cap(r['dense_orig'], 3000, sp['closed']),
                            sp['closed'], filled))
            if r['status'] == 'removed':
                continue
            items_n.append((_cap(r['dense_new'], 3000, r['closed']),
                            r['closed'], filled))
        mo = render_mask(items_o, art['vb'])
        mn = render_mask(items_n, art['vb'])
        m = compare_masks(mo, mn)
        iou, cham = m['silhouette_iou'], m['chamfer_px']
        changed, ssim = m['pixels_changed_pct'], m['ssim_alpha']
        qa = {'geometry': {'hausdorff_max': summary['deviation_max']},
              'render': m,
              'topology': {
                  'winding_preserved': wind_ok,
                  'subpaths_before': len(art['subpaths']),
                  'subpaths_after': len(art['subpaths']) - summary['removed_tiny'],
                  'closed_before': sum(1 for sp in art['subpaths'] if sp['closed']),
                  'closed_after': sum(1 for r in results
                                      if r['closed'] and r['status'] != 'removed')}}
        cmp_b64 = base64.b64encode(compare_png(mo, mn)).decode()
        gates_ok, gates = qa_gates_pass(m, P['mode'])
        qa['gates'] = gates
        # detail lock, render stage: hole/cutout interiors must paint
        # identically (a filled-in hole flips pixels even when IoU is
        # high). Verified BEFORE the gate verdict so a clean detail
        # lock can exonerate a marginal IoU (smoothing footprint, not
        # damage) - but a failed detail lock never gets exonerated.
        sc = 600.0 / max(art['vb'][2], 1e-9)
        ds = hole_interior_samples(pipe.get('holes', []), results,
                                   art['vb'], sc)
        dok, dfail = verify_hole_render(mo, mn, ds)
        qa['detail'] = {'holes': len(pipe.get('holes', [])),
                        'samples': len(ds), 'failed': dfail}
        if not gates_ok and not global_revert:
            failed = [k for k in ('iou', 'changed', 'chamfer')
                      if not gates[k]['pass']]
            if _iou_exonerated(m, failed, wind_ok, ds, dfail):
                gates['iou']['exonerated'] = True
                notes.append('IoU %.4f just under the %.3f bar, but every '
                             'other lock verifies clean (%d hole-interior '
                             'pixels identical) - smoothing kept'
                             % (m['silhouette_iou'], gates['iou']['floor'],
                                len(ds)))
            else:
                notes.append('subject-lock gate tripped (%s) - original '
                             'file returned untouched' % ', '.join(failed))
                svg_out = svg_text
                global_revert = True
        if dfail and not global_revert:
            notes.append('detail-lock tripped (%d hole interior pixel(s) '
                         'changed rendering) - original file returned '
                         'untouched' % dfail)
            svg_out = svg_text
            global_revert = True

    # marketplace context
    vx, vy, vw, vh = art['vb']
    off = 0
    union = None
    for sp, r in zip(art['subpaths'], results):
        if r['status'] == 'removed' or len(r['dense_new']) == 0:
            continue
        q = np.asarray(r['dense_new'])
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        if x0 < vx - 1e-6 or y0 < vy - 1e-6 or x1 > vx + vw + 1e-6 or y1 > vy + vh + 1e-6:
            off += 1
        box = np.array([x0, y0, x1, y1])
        union = box if union is None else np.array(
            [min(union[0], box[0]), min(union[1], box[1]),
             max(union[2], box[2]), max(union[3], box[3])])
    stroked = sum(1 for sp, r in zip(art['subpaths'], results)
                  if r['status'] != 'removed'
                  and (sp['stroke'] or '').strip().lower() not in ('', 'none'))

    def _painted(sp):
        f = (sp['fill'] or '').strip().lower() not in ('', 'none', 'transparent')
        s = (sp['stroke'] or '').strip().lower() not in ('', 'none')
        return f, s

    n_transparent = 0
    n_grad = 0
    for sp, r in zip(art['subpaths'], results):
        if r['status'] == 'removed':
            continue
        f, s = _painted(sp)
        if not (f or s):
            continue
        try:
            o = float(sp.get('opacity', '1'))
            fo = float(sp.get('fill_opacity', '1'))
            so = float(sp.get('stroke_opacity', '1'))
        except (TypeError, ValueError):
            o = fo = so = 1.0
        if o < 1 - 1e-9 or (f and fo < 1 - 1e-9) or (s and so < 1 - 1e-9):
            n_transparent += 1
        if ((sp['fill'] or '').strip().lower().startswith('url(') or
                (sp['stroke'] or '').strip().lower().startswith('url(')):
            n_grad += 1
    fullbleed = False
    for sp, r in zip(art['subpaths'], results):
        if sp['tag'] != 'rect' or r['status'] == 'removed':
            continue
        if (sp['fill'] or '').strip().lower() in ('', 'none', 'transparent'):
            continue
        if sp['opacity'] != '1' or sp['fill_opacity'] != '1':
            continue
        q = np.asarray(r['dense_new'])
        if len(q) == 0:
            continue
        w = q[:, 0].max() - q[:, 0].min()
        h = q[:, 1].max() - q[:, 1].min()
        if w * h >= 0.98 * vw * vh:
            fullbleed = True
    mp_board = art['W'] * art['H'] / 1e6
    mp_bbox = ((union[2] - union[0]) * (union[3] - union[1]) / 1e6
              if union is not None else 0.0)
    mp_key = (opts.get('marketplace') or 'auto').lower()
    site_key = mp_key if mp_key in SITES else 'generic_svg'
    basis = SITES[site_key].get('mp_basis', 'artboard')
    before = summary['anchors_before']
    after = summary['anchors_after']
    red = (1.0 - after / before) if before else 0.0
    jb = float(np.mean([r['jitter_before'] for r in results])) if results else 0.0
    sm_j = [r['jitter_after'] for r in results if r['status'] == 'smoothed']
    ja = float(np.mean(sm_j)) if sm_j else jb
    ctx = {
        'open_paths': sum(1 for r in results
                          if not r['closed'] and r['status'] != 'removed'),
        'dups_removed': summary['dups_removed'],
        'spikes_removed': summary['spikes_removed'],
        'tiny_removed': summary['removed_tiny'],
        'tiny_remaining': sum(1 for r in results if r['status'] == 'passthrough'
                              and 'tiny' in r.get('reason', '')),
        'self_intersections': n_after,
        'winding_ok': wind_ok,
        'subpaths_before': len(art['subpaths']),
        'subpaths_after': len(art['subpaths']) - summary['removed_tiny'],
        'off_artboard': off,
        'out_bytes': len(svg_out.encode('utf-8')),
        'mp_value': mp_bbox if basis == 'bbox' else mp_board,
        'stroked_paths': stroked,
        'n_text': art['stats']['n_text'],
        'n_image': art['stats']['n_image'],
        'n_filter': art['stats']['n_filter'],
        'n_transparent': n_transparent,
        'n_gradient': n_grad,
        'n_use': art['stats']['n_use'],
        'n_clip': art['stats']['n_clip'],
        'artboard_at_origin': abs(vx) < 1e-9 and abs(vy) < 1e-9,
        'vb_origin': '%.1f,%.1f' % (vx, vy),
        'fullbleed_bg': fullbleed,
        'iou': iou, 'chamfer_px': cham, 'reduction': red,
        'jitter_before': jb, 'jitter_after': ja,
        'n_smoothed': summary['smoothed'],
    }
    checklist, scores = build_checklist(ctx, site_key)
    if qa is not None and 'gates' in qa:
        g = qa['gates']
        checklist.append({
            'id': 'subject_lock', 'label': 'Subject lock (silhouette)',
            'status': 'pass' if g['pass'] else 'fail',
            'detail': 'IoU %.4f / %.3f floor, changed %.2f%% / %.1f%% max, '
                      'chamfer %.2f / %.1f px max'
                      % (g['iou']['value'], g['iou']['floor'],
                         g['changed']['value'], g['changed']['ceil'],
                         g['chamfer']['value'], g['chamfer']['ceil'])})
    if summary.get('min_gap_before') is not None:
        nt = summary.get('pairs_touched', 0)
        checklist.append({
            'id': 'separation', 'label': 'Path separation',
            'status': 'pass',
            'detail': ('min neighbour gap %.3f -> %.3f u, nothing merged'
                       % (summary['min_gap_before'], summary['min_gap_after'])
                       if not nt else
                       '%d pair(s) would touch - kept original' % nt)})

    opt_bits = ['mode=%s' % P['mode'],
                'smooth=%.0f' % P['smoothness'],
                'fid=%.0f' % P['fidelity']]
    if P['movement_cap'] is not None:
        opt_bits.append('cap=%s' % P['movement_cap'])
    if P['noise_target'] is not None:
        opt_bits.append('nt=%s' % P['noise_target'])
    if opts.get('convert_shapes'):
        opt_bits.append('shapes->path')
    if P['round_corners']:
        opt_bits.append('round')
    if not P['remove_tiny']:
        opt_bits.append('keep-tiny')

    seconds = round(time.time() - t0, 2)
    return {
        'art': art, 'results': results, 'params': P,
        'svg_out': svg_out, 'global_revert': global_revert,
        'summary': summary, 'notes': sorted(set(notes)),
        'qa': qa, 'compare_png_b64': cmp_b64,
        'checklist': checklist, 'scores': scores,
        'site_key': site_key,
        'site_label': ('Auto (core checks)' if mp_key == 'auto'
                       else SITES[site_key]['label']),
        'mp': {'artboard': round(mp_board, 3), 'bbox': round(mp_bbox, 3)},
        'ctx': ctx, 'seconds': seconds,
        'options': ' '.join(opt_bits),
        'preset_used': P['preset_used'],
        'engine_version': ENGINE_VERSION,
    }


def export_eps(svg_text, opts, level=10):
    out = run(svg_text, opts, do_qa=False)
    eps, eps_notes = to_eps(out['art'], out['results'], level=level)
    probs = validate_eps_text(eps)
    prof = 'eps8' if level == 8 else 'eps10'
    ectx = dict(out['ctx'])
    ectx['out_bytes'] = len(eps.encode('utf-8'))
    ectx['mp_value'] = out['mp']['bbox']  # EPS profiles measure the bbox
    pre, _ = build_checklist(ectx, prof)
    preflight = [{
        'id': 'format', 'label': 'EPS %d file structure' % level,
        'status': 'pass' if not probs else 'fail',
        'detail': ('DSC header, RGB paints, AI-%d-compatible ops only' % level
                   if not probs else '; '.join(probs)),
    }] + pre
    out['preflight'] = preflight
    notes = sorted(set(out['notes']) | set(eps_notes))
    fails = [c['label'] for c in preflight if c['status'] == 'fail']
    if fails:
        notes.append('EPS marketplace blockers (%d): %s - fix in Illustrator '
                     'before upload' % (len(fails), ', '.join(fails)))
    else:
        notes.append('EPS preflight: all %d marketplace checks passed'
                     % len(preflight))
    return eps, notes, out
