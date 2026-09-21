"""Pixel-level QA: scanline-render original vs smoothed, then compare.

Renders filled geometry (even-odd) with numpy only - no external renderer
needed - and reports silhouette IoU, changed pixels, edge chamfer distance
and an SSIM-style alpha similarity, plus a before/after/heatmap compare PNG.
"""
import io

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


def render_mask(items, vb, out_w=600):
    """items: list of (pts, closed, filled). Returns float mask HxW in 0..1."""
    vx, vy, vw, vh = vb
    scale = out_w / max(vw, 1e-9)
    out_h = max(1, int(round(vh * scale)))
    E = []
    for pts, closed, filled in items:
        p = np.asarray(pts, float)
        if not filled or len(p) < 3:
            continue
        q = (p - np.array([vx, vy])) * scale
        qq = np.vstack([q, q[:1]])  # implicit close for fill
        x1, y1 = qq[:-1, 0], qq[:-1, 1]
        x2, y2 = qq[1:, 0], qq[1:, 1]
        keep = np.abs(y2 - y1) > 1e-9
        for a, b, c, d in zip(x1[keep], y1[keep], x2[keep], y2[keep]):
            E.append((a, b, c, d))
    mask = np.zeros((out_h, out_w), np.float32)
    if not E:
        return mask
    E = np.array(E, float)
    for r in range(out_h):
        yc = r + 0.5
        cross = ((E[:, 1] <= yc) & (E[:, 3] > yc)) | ((E[:, 3] <= yc) & (E[:, 1] > yc))
        if not np.any(cross):
            continue
        e = E[cross]
        xs = e[:, 0] + (yc - e[:, 1]) / (e[:, 3] - e[:, 1]) * (e[:, 2] - e[:, 0])
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a = max(0, int(np.floor(xs[k])))
            b = min(out_w, int(np.ceil(xs[k + 1])))
            if b > a:
                mask[r, a:b] = 1.0
    return mask


def ssim_alpha(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb_ = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb_ + c2)
    if den <= 0:
        return 1.0 if float(abs(mu_a - mu_b)) < 1e-12 else 0.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / den)


def compare_masks(a, b):
    A = np.asarray(a) > 0.5
    B = np.asarray(b) > 0.5
    inter = float((A & B).sum())
    union = float((A | B).sum())
    iou = inter / union if union else 1.0
    changed = float((A != B).mean() * 100.0)
    ea = A ^ ndimage.binary_erosion(A)
    eb = B ^ ndimage.binary_erosion(B)
    if ea.sum() == 0 or eb.sum() == 0:
        cham = 0.0
    else:
        dt_b = ndimage.distance_transform_edt(~eb)
        dt_a = ndimage.distance_transform_edt(~ea)
        cham = float((dt_b[ea].mean() + dt_a[eb].mean()) / 2.0)
    return {'silhouette_iou': round(float(iou), 6),
            'pixels_changed_pct': round(changed, 4),
            'chamfer_px': round(cham, 4),
            'ssim_alpha': round(ssim_alpha(a, b), 6)}


def _mask_to_rgb(mask, fg=(20, 20, 25), bg=(255, 255, 255)):
    m = (np.asarray(mask) > 0.5)
    img = np.zeros((m.shape[0], m.shape[1], 3), np.uint8)
    img[:, :] = bg
    img[m] = fg
    return Image.fromarray(img)


def heatmap_image(mask_o, mask_s):
    A = np.asarray(mask_o) > 0.5
    B = np.asarray(mask_s) > 0.5
    h, w = A.shape
    img = np.full((h, w, 3), 247, np.uint8)  # same background
    img[A & B] = (76, 154, 99)    # GREEN  = safe / identical
    img[A & ~B] = (229, 72, 77)   # RED    = removed / moved away
    img[B & ~A] = (245, 166, 35)  # AMBER  = added / moved in
    return Image.fromarray(img)


def compare_png(mask_o, mask_s, panel_w=520):
    """Side-by-side ORIGINAL | SMOOTHED | DIFFERENCE panel as PNG bytes."""
    panels = [_mask_to_rgb(mask_o),
              _mask_to_rgb(mask_s),
              heatmap_image(mask_o, mask_s)]
    labels = ['ORIGINAL (rendered)', 'SMOOTHED (rendered)',
              'DIFFERENCE  green=same  red/orange=changed']
    scaled = []
    for im in panels:
        r = panel_w / max(im.width, 1)
        scaled.append(im.resize((panel_w, max(1, int(im.height * r))), Image.NEAREST))
    H = max(im.height for im in scaled)
    bar, pad = 26, 8
    W = panel_w * 3 + pad * 4
    canvas = Image.new('RGB', (W, H + bar + pad * 2), (13, 17, 23))
    dr = ImageDraw.Draw(canvas)
    for i, im in enumerate(scaled):
        x = pad + i * (panel_w + pad)
        canvas.paste(im, (x, bar + pad))
        dr.text((x + 6, 6), labels[i], fill=(230, 237, 243))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return buf.getvalue()
