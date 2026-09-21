"""SVG parsing: geometry -> subpaths (world coords), structure kept for rebuild.

Every geometric element (<path> + shapes) becomes one or more subpaths with
flattened polylines in *world* (artboard) coordinates. Transforms are baked
into the points; the ancestor matrix is stored so export can convert back to
local coordinates and keep the element in place inside its <g>.
"""
import copy
import math
import re

import numpy as np
import xml.etree.ElementTree as ET

TOKEN = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')
NUM = re.compile(r'-?\d*\.?\d+(?:[eE][-+]?\d+)?')
TRANS = re.compile(r'(\w+)\s*\(([^)]*)\)')

GEOM_TAGS = {'path', 'rect', 'circle', 'ellipse', 'polygon', 'polyline', 'line'}
# containers whose subtree may hold real artwork (clip/mask/pattern skipped: semantics risk)
SKIP_GEOM_IN = {'clipPath', 'mask', 'pattern'}
INHERIT_KEYS = ('fill', 'stroke', 'stroke-width', 'fill-rule', 'fill-opacity',
                'stroke-opacity', 'opacity', 'stroke-linejoin', 'stroke-linecap')


def _local(tag):
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _parse_len(v, default=0.0):
    if v is None:
        return default
    m = NUM.search(str(v))
    try:
        return float(m.group(0)) if m else default
    except ValueError:
        return default


def parse_transform(s):
    """Parse an SVG transform attribute into a 3x3 matrix."""
    M = np.eye(3)
    if not s:
        return M
    for name, argstr in TRANS.findall(s):
        try:
            nums = [float(x) for x in NUM.findall(argstr)]
        except ValueError:
            continue
        if name == 'translate' and nums:
            tx = nums[0]
            ty = nums[1] if len(nums) > 1 else 0.0
            T = np.eye(3)
            T[0, 2] = tx
            T[1, 2] = ty
            M = M @ T
        elif name == 'scale' and nums:
            sx = nums[0]
            sy = nums[1] if len(nums) > 1 else sx
            T = np.eye(3)
            T[0, 0] = sx
            T[1, 1] = sy
            M = M @ T
        elif name == 'rotate' and nums:
            a = math.radians(nums[0])
            c, sn = math.cos(a), math.sin(a)
            T = np.eye(3)
            T[0, 0] = c
            T[0, 1] = -sn
            T[1, 0] = sn
            T[1, 1] = c
            if len(nums) == 3:
                cx, cy = nums[1], nums[2]
                M = (M @ np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                     @ T @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]))
            else:
                M = M @ T
        elif name == 'matrix' and len(nums) == 6:
            a, b, c, d, e, f = nums
            M = M @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif name == 'skewX' and nums:
            t = math.tan(math.radians(nums[0]))
            M = M @ np.array([[1, t, 0], [0, 1, 0], [0, 0, 1]])
        elif name == 'skewY' and nums:
            t = math.tan(math.radians(nums[0]))
            M = M @ np.array([[1, 0, 0], [t, 1, 0], [0, 0, 1]])
    return M


def apply_mat(M, pts):
    p = np.asarray(pts, dtype=float)
    if p.size == 0:
        return p.reshape(0, 2)
    h = np.column_stack([p, np.ones(len(p))])
    return (h @ M.T)[:, :2]


# ---------------------------------------------------------------- flattening

def _cubic_flat(p0, p1, p2, p3, tol):
    """Adaptive-subdivision flatten of one cubic; returns points excluding start."""
    out = []
    stack = [(np.asarray(p0, float), np.asarray(p1, float),
              np.asarray(p2, float), np.asarray(p3, float))]
    guard = 0
    while stack and guard < 20000:
        guard += 1
        a, b, c, d = stack.pop()
        ux, uy = d[0] - a[0], d[1] - a[1]
        L = math.hypot(ux, uy)
        if L < 1e-12:
            dev = max(math.hypot(b[0] - a[0], b[1] - a[1]),
                      math.hypot(c[0] - a[0], c[1] - a[1]))
        else:
            dev = max(abs((b[0] - a[0]) * uy - (b[1] - a[1]) * ux),
                      abs((c[0] - a[0]) * uy - (c[1] - a[1]) * ux)) / L
        if dev <= tol or L < 1e-9:
            out.append(d)
        else:
            ab = (a + b) / 2
            bc = (b + c) / 2
            cd = (c + d) / 2
            abc = (ab + bc) / 2
            bcd = (bc + cd) / 2
            m = (abc + bcd) / 2
            stack.append((m, bcd, cd, d))
            stack.append((a, ab, abc, m))
    return out


def _arc_points(p0, rx, ry, phi_deg, large, sweep, p1, tol=0.2):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    rx, ry = abs(float(rx)), abs(float(ry))
    if rx < 1e-9 or ry < 1e-9 or (abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12):
        return [np.array([x1, y1])]
    phi = math.radians(phi_deg % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x0 - x1) / 2.0, (y0 - y1) / 2.0
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx *= s
        ry *= s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    f = math.sqrt(max(0.0, num / max(den, 1e-18)))
    if bool(large) == bool(sweep):
        f = -f
    cxp, cyp = f * rx * y1p / ry, -f * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x0 + x1) / 2.0
    cy = sp * cxp + cp * cyp + (y0 + y1) / 2.0

    def _ang(ux, uy, vx, vy):
        d = math.hypot(ux, uy) * math.hypot(vx, vy)
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / max(d, 1e-18)))
        a = math.acos(c)
        return -a if ux * vy - uy * vx < 0 else a

    th1 = _ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = _ang((x1p - cxp) / rx, (y1p - cyp) / ry,
               (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    if sweep and dth < 0:
        dth += 2 * math.pi
    approx_r = max(rx, ry)
    n = max(4, min(256, int(abs(dth) * approx_r / max(tol, 1e-6) / 4) + 4))
    pts = []
    for t in np.linspace(th1, th1 + dth, n + 1)[1:]:
        ct, st = math.cos(t), math.sin(t)
        pts.append(np.array([cp * rx * ct - sp * ry * st + cx,
                             sp * rx * ct + cp * ry * st + cy]))
    return pts


def _parse_path_d(d, tol):
    """Parse path data -> (subpaths, n_degenerate). pts exclude duplicated closure."""
    toks = TOKEN.findall(d or '')
    subs = []
    cur = None
    cmd = None
    rel = False
    first_move = True
    i, n = 0, len(toks)
    pos = np.zeros(2)
    pc = None
    pq = None
    ndeg = 0

    def is_cmd(t):
        return len(t) == 1 and t.isalpha()

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def have(k):
        if i + k > n:
            return False
        return all(not is_cmd(toks[j]) for j in range(i, i + k))

    def new_sub():
        nonlocal cur
        cur = {'pts': [], 'closed': False, 'anchors': 0}
        subs.append(cur)
        return cur

    def ensure():
        return cur if cur is not None else new_sub()

    while i < n:
        t = toks[i]
        if is_cmd(t):
            cmd = t.upper()
            rel = t.islower()
            i += 1
            if cmd == 'Z':
                if cur is not None and cur['pts']:
                    cur['closed'] = True
                    pos = cur['pts'][0].copy()
                cur = None
                pc = None
                pq = None
                cmd = None
                continue
            if cmd == 'M':
                first_move = True
            if cmd not in 'MLHVCSQTA':
                cmd = None
                continue
        if cmd is None:
            i += 1  # stray number
            continue
        eff = 'L' if (cmd == 'M' and not first_move) else cmd
        if eff == 'M':
            if not have(2):
                cmd = None
                continue
            x, y = num(), num()
            p = pos + np.array([x, y]) if rel else np.array([x, y], float)
            new_sub()
            cur['pts'].append(p)
            cur['anchors'] += 1
            pos = p.copy()
            first_move = False
            pc = None
            pq = None
        elif eff == 'L':
            if not have(2):
                cmd = None
                continue
            x, y = num(), num()
            p = pos + np.array([x, y]) if rel else np.array([x, y], float)
            c = ensure()
            c['pts'].append(p)
            c['anchors'] += 1
            pos = p.copy()
            pc = None
            pq = None
        elif eff == 'H':
            if not have(1):
                cmd = None
                continue
            x = num()
            p = np.array([pos[0] + x if rel else x, pos[1]])
            c = ensure()
            c['pts'].append(p)
            c['anchors'] += 1
            pos = p.copy()
            pc = None
            pq = None
        elif eff == 'V':
            if not have(1):
                cmd = None
                continue
            y = num()
            p = np.array([pos[0], pos[1] + y if rel else y])
            c = ensure()
            c['pts'].append(p)
            c['anchors'] += 1
            pos = p.copy()
            pc = None
            pq = None
        elif eff in ('C', 'S'):
            need = 6 if eff == 'C' else 4
            if not have(need):
                cmd = None
                continue
            if eff == 'C':
                x1, y1 = num(), num()
                p1 = pos + np.array([x1, y1]) if rel else np.array([x1, y1], float)
            else:
                p1 = 2 * pos - pc if pc is not None else pos.copy()
            x2, y2 = num(), num()
            x, y = num(), num()
            p2 = pos + np.array([x2, y2]) if rel else np.array([x2, y2], float)
            p = pos + np.array([x, y]) if rel else np.array([x, y], float)
            c = ensure()
            if not c['pts']:
                c['pts'].append(pos.copy())
            c['pts'].extend(_cubic_flat(pos, p1, p2, p, tol))
            c['anchors'] += 1
            pc = p2.copy()
            pq = None
            pos = p.copy()
        elif eff in ('Q', 'T'):
            if eff == 'Q':
                if not have(4):
                    cmd = None
                    continue
                x1, y1 = num(), num()
                q1 = pos + np.array([x1, y1]) if rel else np.array([x1, y1], float)
            else:
                if not have(2):
                    cmd = None
                    continue
                q1 = 2 * pos - pq if pq is not None else pos.copy()
            x, y = num(), num()
            p = pos + np.array([x, y]) if rel else np.array([x, y], float)
            c1 = pos + (2.0 / 3.0) * (q1 - pos)
            c2 = p + (2.0 / 3.0) * (q1 - p)
            c = ensure()
            if not c['pts']:
                c['pts'].append(pos.copy())
            c['pts'].extend(_cubic_flat(pos, c1, c2, p, tol))
            c['anchors'] += 1
            pq = q1.copy()
            pc = None
            pos = p.copy()
        elif eff == 'A':
            if not have(7):
                cmd = None
                continue
            rx, ry, rot = num(), num(), num()
            laf, saf = num(), num()
            x, y = num(), num()
            p = pos + np.array([x, y]) if rel else np.array([x, y], float)
            c = ensure()
            if not c['pts']:
                c['pts'].append(pos.copy())
            c['pts'].extend(_arc_points(pos, rx, ry, rot, laf, saf, p, tol))
            c['anchors'] += 1
            pc = None
            pq = None
            pos = p.copy()
        else:
            cmd = None

    out = []
    for s in subs:
        if len(s['pts']) >= 2:
            s['pts'] = np.array(s['pts'], float)
            out.append(s)
        else:
            ndeg += 1
    return out, ndeg


def _shape_subpaths(el):
    t = _local(el.tag)
    a = el.attrib
    if t == 'rect':
        x = _parse_len(a.get('x'))
        y = _parse_len(a.get('y'))
        w = _parse_len(a.get('width'))
        h = _parse_len(a.get('height'))
        if w <= 0 or h <= 0:
            return []
        rx = _parse_len(a.get('rx'))
        ry = _parse_len(a.get('ry'))
        if a.get('rx') is not None and a.get('ry') is None:
            ry = rx
        if a.get('ry') is not None and a.get('rx') is None:
            rx = ry
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx <= 0 and ry <= 0:
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
            return [{'pts': pts, 'closed': True, 'anchors': 4}]
        pts = []
        for cx, cy, a0, a1 in ((x + w - rx, y + ry, -90, 0),
                              (x + w - rx, y + h - ry, 0, 90),
                              (x + rx, y + h - ry, 90, 180),
                              (x + rx, y + ry, 180, 270)):
            for k in range(7):
                th = math.radians(a0 + (a1 - a0) * k / 6)
                pts.append((cx + rx * math.cos(th), cy + ry * math.sin(th)))
        return [{'pts': np.array(pts, float), 'closed': True, 'anchors': 4}]
    if t == 'circle':
        cx = _parse_len(a.get('cx'))
        cy = _parse_len(a.get('cy'))
        r = _parse_len(a.get('r'))
        if r <= 0:
            return []
        th = np.linspace(0, 2 * math.pi, 49)[:-1]
        return [{'pts': np.column_stack([cx + r * np.cos(th),
                                         cy + r * np.sin(th)]),
                 'closed': True, 'anchors': 4}]
    if t == 'ellipse':
        cx = _parse_len(a.get('cx'))
        cy = _parse_len(a.get('cy'))
        rx = _parse_len(a.get('rx'))
        ry = _parse_len(a.get('ry'))
        if rx <= 0 or ry <= 0:
            return []
        th = np.linspace(0, 2 * math.pi, 49)[:-1]
        return [{'pts': np.column_stack([cx + rx * np.cos(th),
                                         cy + ry * np.sin(th)]),
                 'closed': True, 'anchors': 4}]
    if t in ('polygon', 'polyline'):
        nums = [float(x) for x in NUM.findall(a.get('points', ''))]
        if len(nums) < 4:
            return []
        pts = np.array(nums, float).reshape(-1, 2)
        if len(pts) < 2:
            return []
        return [{'pts': pts, 'closed': (t == 'polygon'), 'anchors': len(pts)}]
    if t == 'line':
        x1 = _parse_len(a.get('x1'))
        y1 = _parse_len(a.get('y1'))
        x2 = _parse_len(a.get('x2'))
        y2 = _parse_len(a.get('y2'))
        return [{'pts': np.array([[x1, y1], [x2, y2]], float),
                 'closed': False, 'anchors': 2}]
    return []


def _parse_style(s):
    out = {}
    for part in (s or '').split(';'):
        if ':' in part:
            k, v = part.split(':', 1)
            out[k.strip().lower()] = v.strip()
    return out


def _own_style(attrib):
    st = _parse_style(attrib.get('style'))
    out = {}
    for k in INHERIT_KEYS:
        if k in attrib:
            out[k] = attrib[k]
        elif k in st:
            out[k] = st[k]
    return out


def _walk(el, M_parent, inh, art, tol, skip, el_counter):
    for child in list(el):
        tag = _local(child.tag)
        M_here = M_parent @ parse_transform(child.attrib.get('transform'))
        child_inh = dict(inh)
        child_inh.update(_own_style(child.attrib))
        if tag in GEOM_TAGS and not skip:
            if tag == 'path':
                subs, ndeg = _parse_path_d(child.attrib.get('d', ''), tol)
            else:
                subs, ndeg = _shape_subpaths(child), 0
            art['stats']['degenerate'] = art['stats'].get('degenerate', 0) + ndeg
            if not subs:
                continue
            el_idx = el_counter[0]
            el_counter[0] += 1
            for si, s in enumerate(subs):
                pts_w = apply_mat(M_here, s['pts'])
                art['subpaths'].append({
                    'pts': pts_w,
                    'closed': s['closed'],
                    'anchors': s['anchors'],
                    'fill': child_inh.get('fill', 'black'),
                    'stroke': child_inh.get('stroke', 'none'),
                    'stroke_width': child_inh.get('stroke-width', '1'),
                    'fill_rule': child_inh.get('fill-rule', 'nonzero'),
                    'opacity': child_inh.get('opacity', '1'),
                    'fill_opacity': child_inh.get('fill-opacity', '1'),
                    'stroke_opacity': child_inh.get('stroke-opacity', '1'),
                    'linejoin': child_inh.get('stroke-linejoin', ''),
                    'el': child,
                    'el_idx': el_idx,
                    'sub_idx': si,
                    'n_subs': len(subs),
                    'M_exp': M_parent.copy(),  # ancestors only; own transform baked+ dropped
                    'tag': tag,
                    'is_shape': tag != 'path',
                })
        else:
            if tag == 'text':
                art['stats']['n_text'] += 1
            elif tag == 'image':
                art['stats']['n_image'] += 1
            elif tag in ('linearGradient', 'radialGradient'):
                art['stats']['n_gradient'] += 1
            elif tag == 'filter':
                art['stats']['n_filter'] += 1
            elif tag == 'clipPath':
                art['stats']['n_clip'] += 1
            elif tag == 'style':
                art['stats']['n_style'] += 1
            elif tag == 'use':
                art['stats']['n_use'] += 1
            if len(child):
                _walk(child, M_here, child_inh, art, tol,
                      skip or tag in SKIP_GEOM_IN, el_counter)


def parse_svg(text):
    """Parse SVG text into an artwork dict. Raises ValueError on bad input."""
    if text is None:
        raise ValueError('empty file')
    if '<!ENTITY' in text:
        raise ValueError('SVG with entities is not accepted')
    data = text.encode('utf-8') if isinstance(text, str) else bytes(text)
    if len(data) > 12 * 1024 * 1024:
        raise ValueError('file too large (>12 MB)')
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError('not valid SVG/XML: %s' % e)
    if _local(root.tag) != 'svg':
        raise ValueError('root element is not <svg>')
    a = root.attrib
    vb = None
    if a.get('viewBox'):
        nums = NUM.findall(a['viewBox'])
        if len(nums) == 4:
            try:
                vb = tuple(float(x) for x in nums)
            except ValueError:
                vb = None
    W = _parse_len(a.get('width'), 0) or (vb[2] if vb else 300.0)
    H = _parse_len(a.get('height'), 0) or (vb[3] if vb else 300.0)
    if vb is None:
        vb = (0.0, 0.0, float(W), float(H))
    if vb[2] <= 0 or vb[3] <= 0:
        raise ValueError('invalid viewBox size')
    tol = min(0.5, max(0.02, max(vb[2], vb[3]) / 2000.0))
    art = {'root': root, 'W': float(W), 'H': float(H), 'vb': vb,
           'flat_tol': tol, 'subpaths': [], 'raw_len': len(data),
           'stats': {'n_text': 0, 'n_image': 0, 'n_filter': 0,
                     'n_gradient': 0, 'n_clip': 0, 'n_use': 0,
                     'n_style': 0, 'degenerate': 0}}
    _walk(root, np.eye(3), {}, art, tol, False, [0])
    return art
