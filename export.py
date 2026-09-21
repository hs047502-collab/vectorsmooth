"""Export: rebuilt SVG (structure-preserving) + EPS 8/10 writer."""
import copy
import datetime
import math
import re

import numpy as np
import xml.etree.ElementTree as ET

from . import __version__ as ENGINE_VERSION
from .svg_parse import GEOM_TAGS, apply_mat

DROP_ATTRS = {'d', 'transform', 'x', 'y', 'width', 'height', 'rx', 'ry',
              'cx', 'cy', 'r', 'points', 'x1', 'y1', 'x2', 'y2'}


def _local(tag):
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _fmt(v):
    s = '%.3f' % float(v)
    s = s.rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def segments_to_d(start, segments, closed):
    parts = ['M%s %s' % (_fmt(start[0]), _fmt(start[1]))]
    for s in segments:
        if s[0] == 'L':
            parts.append('L%s %s' % (_fmt(s[1][0]), _fmt(s[1][1])))
        else:
            _, c1, c2, p = s
            parts.append('C%s %s %s %s %s %s' % (
                _fmt(c1[0]), _fmt(c1[1]), _fmt(c2[0]), _fmt(c2[1]),
                _fmt(p[0]), _fmt(p[1])))
    if closed:
        parts.append('Z')
    return ''.join(parts)


def _mk_path(like_el):
    t = like_el.tag
    if '}' in t:
        return ET.Element(t.split('}')[0] + '}path')
    return ET.Element('path')


def _to_local(M_exp, pts):
    try:
        inv = np.linalg.inv(M_exp)
    except np.linalg.LinAlgError:
        return None
    return apply_mat(inv, np.asarray(pts, float))


def _seg_local(M_exp, start, segments):
    pts = [np.asarray(start, float)]
    for s in segments:
        if s[0] == 'L':
            pts.append(np.asarray(s[1], float))
        else:
            pts.extend([np.asarray(s[1], float), np.asarray(s[2], float),
                        np.asarray(s[3], float)])
    loc = _to_local(M_exp, np.array(pts))
    if loc is None:
        return None, None
    out = []
    i = 1
    for s in segments:
        if s[0] == 'L':
            out.append(('L', loc[i]))
            i += 1
        else:
            out.append(('C', loc[i], loc[i + 1], loc[i + 2]))
            i += 3
    return loc[0], out


def build_svg(art, results, global_revert_text=None):
    """Rebuild the SVG: defs/order/groups preserved, only changed geometry new.

    Elements whose subpaths are all verbatim (passthrough/noop) are copied
    byte-for-byte, so clean inputs come out byte-identical.
    """
    if global_revert_text is not None:
        return global_revert_text
    by_el = {}
    for sp, r in zip(art['subpaths'], results):
        by_el.setdefault(id(sp['el']), []).append((sp, r))
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    root = art['root']
    new_root = ET.Element(root.tag, dict(root.attrib))
    new_root.text = root.text
    new_root.tail = None

    def rebuild(src_parent, dst_parent):
        for child in list(src_parent):
            tag = _local(child.tag)
            if tag in GEOM_TAGS and id(child) in by_el:
                outs = by_el[id(child)]
                if all(r['status'] in ('passthrough',) for _, r in outs):
                    dst_parent.append(copy.deepcopy(child))
                    continue
                parts = []
                failed = False
                for sp, r in outs:
                    if r['status'] in ('removed', 'degenerate'):
                        continue
                    if r['status'] == 'passthrough':
                        # mixed element: render this subpath from original pts
                        segs = [('L', q.copy()) for q in sp['pts'][1:]]
                        start = sp['pts'][0].copy()
                        closed = sp['closed']
                    else:
                        segs, start, closed = (r['segments'], r['start'], r['closed'])
                    st, sg = _seg_local(sp['M_exp'], start, segs)
                    if st is None:
                        dst_parent.append(copy.deepcopy(child))
                        failed = True
                        break
                    parts.append(segments_to_d(st, sg, closed))
                if failed or not parts:
                    continue
                nel = _mk_path(child)
                for k, v in child.attrib.items():
                    if _local(k) not in DROP_ATTRS:
                        nel.set(k, v)
                # compound elements stay ONE path: splitting subpaths into
                # separate elements would fill evenodd/nonzero holes and
                # visibly change the subject.
                nel.set('d', ' '.join(parts))
                nel.tail = child.tail
                dst_parent.append(nel)
            elif len(child):
                shell = ET.Element(child.tag, dict(child.attrib))
                shell.text = child.text
                shell.tail = child.tail
                rebuild(child, shell)
                dst_parent.append(shell)
            else:
                dst_parent.append(copy.deepcopy(child))

    rebuild(root, new_root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        new_root, encoding='unicode')


# ------------------------------------------------------------------- EPS

_NAMED = {'black': (0, 0, 0), 'white': (1, 1, 1), 'red': (1, 0, 0),
          'green': (0, 0.5, 0), 'blue': (0, 0, 1), 'yellow': (1, 1, 0),
          'none': None}


def parse_color(s):
    s = (s or '').strip().lower()
    if s in ('', 'none', 'transparent'):
        return None
    if s in _NAMED:
        return _NAMED[s]
    if s.startswith('#'):
        h = s[1:]
        try:
            if len(h) == 3:
                return tuple(int(c * 2, 16) / 255.0 for c in h)
            if len(h) == 6:
                return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            return (0, 0, 0)
        return (0, 0, 0)
    if s.startswith('rgb'):
        import re
        nums = re.findall(r'-?\d*\.?\d+%?', s)
        vals = []
        for x in nums[:3]:
            if x.endswith('%'):
                vals.append(float(x[:-1]) / 100.0)
            else:
                vals.append(float(x) / 255.0)
        if len(vals) == 3:
            return tuple(max(0.0, min(1.0, v)) for v in vals)
        return (0, 0, 0)
    if s.startswith('url('):
        return 'gradient'
    return (0, 0, 0)


def to_eps(art, results, level=10):
    """Minimal, dependency-free EPS 8/10 writer (filled paths + strokes)."""
    W, H = art['W'], art['H']
    L = []
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L.append('%!PS-Adobe-3.0 EPSF-3.0')
    L.append('%%%%BoundingBox: 0 0 %d %d' % (math.ceil(W), math.ceil(H)))
    L.append('%%%%HiResBoundingBox: 0 0 %.3f %.3f' % (W, H))
    L.append('%%%%Creator: VectorSmooth Pro v%s (EPS %d compatible)'
             % (ENGINE_VERSION, level))
    L.append('%%Title: smoothed artwork')
    L.append('%%%%CreationDate: %s' % stamp)
    L.append('%%DocumentData: Clean7Bit')
    L.append('%%LanguageLevel: 2')
    L.append('%%Pages: 1')
    L.append('%%EndComments')
    notes = []
    n_paths = 0
    Y = lambda y: H - y  # noqa: E731 - SVG y-down -> PS y-up
    # group subpaths by source element: compounds share one path so that a
    # single eofill/fill keeps their holes (filling subpaths one by one
    # would paint every hole solid).
    groups, order = {}, []
    for sp, r in zip(art['subpaths'], results):
        k = id(sp['el'])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append((sp, r))
    for k in order:
        outs = [(sp, r) for sp, r in groups[k]
                if r['status'] not in ('removed', 'degenerate')]
        if not outs:
            continue
        sp0 = outs[0][0]
        geoms = []
        for sp, r in outs:
            if r['status'] == 'passthrough':
                segs = [('L', q.copy()) for q in sp['pts'][1:]]
                start = sp['pts'][0].copy()
                closed = sp['closed']
            else:
                segs, start, closed = r['segments'], r['start'], r['closed']
            geoms.append((start, segs, closed))
        fill = parse_color(sp0['fill'])
        stroke = parse_color(sp0['stroke'])
        if fill == 'gradient' or stroke == 'gradient':
            notes.append('gradient fill flattened to gray in EPS')
            if fill == 'gradient':
                fill = (0.5, 0.5, 0.5)
            if stroke == 'gradient':
                stroke = (0.5, 0.5, 0.5)
        if fill is None and stroke is None:
            continue
        try:
            sw = float(sp0['stroke_width'])
        except (TypeError, ValueError):
            sw = 1.0
        def emit_trace():
            for (st0, sg, cl) in geoms:
                L.append('%.3f %.3f moveto' % (st0[0], Y(st0[1])))
                for s in sg:
                    if s[0] == 'L':
                        L.append('%.3f %.3f lineto' % (s[1][0], Y(s[1][1])))
                    else:
                        _, c1, c2, p = s
                        L.append('%.3f %.3f %.3f %.3f %.3f %.3f curveto' % (
                            c1[0], Y(c1[1]), c2[0], Y(c2[1]), p[0], Y(p[1])))
                if cl:
                    L.append('closepath')

        L.append('newpath')
        emit_trace()
        if fill is not None:
            L.append('%.4f %.4f %.4f setrgbcolor' % fill)
            L.append('eofill' if 'evenodd' in (sp0['fill_rule'] or '') else 'fill')
            # re-trace for stroke: simpler to re-emit path
            if stroke is not None:
                L.append('newpath')
                emit_trace()
                L.append('%.4f %.4f %.4f setrgbcolor %.3f setlinewidth stroke'
                         % (stroke + (sw,)))
        else:
            L.append('%.4f %.4f %.4f setrgbcolor %.3f setlinewidth stroke'
                     % (stroke + (sw,)))
        n_paths += 1
    if n_paths == 0:
        notes.append('no paintable geometry found')
    L.append('%%EOF')
    return '\n'.join(L) + '\n', sorted(set(notes))


def validate_eps_text(eps):
    """Structural self-check on our own EPS output.

    Returns a list of problem strings (empty = valid): DSC magic,
    bounding boxes, comment/body order, no showpage (forbidden in EPS),
    %%EOF trailer, 7-bit ASCII only.
    """
    probs = []
    t = eps.replace('\r\n', '\n').replace('\r', '\n')
    lines = t.split('\n')
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty or nonempty[0].strip() != '%!PS-Adobe-3.0 EPSF-3.0':
        probs.append('missing EPS magic header')
    m = re.search(r'^%%BoundingBox:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',
                  t, re.M)
    if not m:
        probs.append('missing/invalid %%BoundingBox')
    elif int(m.group(3)) <= 0 or int(m.group(4)) <= 0:
        probs.append('empty %%BoundingBox')
    if not re.search(r'^%%HiResBoundingBox:', t, re.M):
        probs.append('missing %%HiResBoundingBox')
    if not re.search(r'^%%Creator:', t, re.M):
        probs.append('missing %%Creator')
    if not re.search(r'^%%EndComments\s*$', t, re.M):
        probs.append('missing %%EndComments')
    else:
        head, _, body = t.partition('%%EndComments')
        if re.search(r'^[^%\n]*\bmoveto\b', head, re.M):
            probs.append('drawing ops before %%EndComments')
        _ = body
    if any(ln.strip() == 'showpage' for ln in lines):
        probs.append('forbidden showpage operator in EPS body')
    if not nonempty or nonempty[-1].strip() != '%%EOF':
        probs.append('missing %%EOF trailer')
    if not t.isascii():
        probs.append('non-ASCII bytes in EPS')
    return probs
