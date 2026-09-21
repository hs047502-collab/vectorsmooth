"""CLI: python3 tools/smooth.py artwork.svg --preset auto --site shutterstock."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import export_eps, run as run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='vectorsmooth CLI')
    ap.add_argument('file')
    ap.add_argument('--preset', default='auto')
    ap.add_argument('--mode', default='safe', choices=['safe', 'pro'])
    ap.add_argument('--smoothness', type=float, default=None)
    ap.add_argument('--fidelity', type=float, default=None)
    ap.add_argument('--site', default='auto', dest='marketplace')
    ap.add_argument('--movement-cap', type=float, default=None)
    ap.add_argument('--noise-target', type=float, default=None)
    ap.add_argument('--convert-shapes', action='store_true')
    ap.add_argument('--round-corners', action='store_true')
    ap.add_argument('--keep-tiny', action='store_true')
    ap.add_argument('--no-autoclose', action='store_true')
    ap.add_argument('--no-qa', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--eps', default=None)
    ap.add_argument('--eps-level', type=int, default=10, choices=[8, 10])
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    text = Path(args.file).read_text()
    opts = {
        'preset': args.preset, 'mode': args.mode,
        'smoothness': args.smoothness, 'fidelity': args.fidelity,
        'marketplace': args.marketplace,
        'movement_cap': args.movement_cap,
        'noise_target': args.noise_target,
        'convert_shapes': args.convert_shapes,
        'round_corners': args.round_corners,
        'remove_tiny': not args.keep_tiny,
        'auto_close': not args.no_autoclose,
    }
    out = run_pipeline(text, opts, do_qa=not args.no_qa)
    if args.out:
        Path(args.out).write_text(out['svg_out'])
    if args.eps:
        eps, _, _ = export_eps(text, opts, level=args.eps_level)
        Path(args.eps).write_text(eps)

    if args.json:
        print(json.dumps({
            'preset_used': out['preset_used'], 'seconds': out['seconds'],
            'options': out['options'], 'summary': out['summary'],
            'notes': out['notes'], 'qa': out['qa'],
            'checklist': out['checklist'], 'scores': out['scores'],
            'site': out['site_label'], 'mp': out['mp'],
            'global_revert': out['global_revert']}, indent=1))
        return

    s = out['summary']
    print('preset %s | %s | %ss | %s' % (
        out['preset_used'], out['options'], out['seconds'], out['site_label']))
    print('anchors %d -> %d | max dev %.3f u (budget %.3f) | IoU %s | reverted %d' % (
        s['anchors_before'], s['anchors_after'], s['deviation_max'], s['budget'],
        out['qa']['render']['silhouette_iou'] if out['qa'] else 'n/a', s['reverted']))
    for n in out['notes']:
        print('note: %s' % n)
    print('--- checklist ---')
    for c in out['checklist']:
        mark = {'pass': 'PASS', 'warn': 'WARN', 'fail': 'FAIL'}[c['status']]
        print('[%s] %s %s' % (mark, c['label'],
                              ('- ' + c['detail']) if c['detail'] else ''))
    print('--- scores ---')
    for k, v in out['scores'].items():
        print('%s: %s' % (k, v))


if __name__ == '__main__':
    main()
