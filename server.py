"""vectorsmooth web service: UI + /api/smooth + /api/export.

Run:  pip install -r requirements.txt
      python3 server.py --port 8000   ->  http://localhost:8000
"""
import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, request, send_file  # noqa: E402

from core.pipeline import export_eps, run as run_pipeline  # noqa: E402

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024


def _opts_from_request():
    v = request.values

    def f(key):
        s = v.get(key, '')
        if s == '' or s is None:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def b(key, default=False):
        s = v.get(key, '')
        if s == '' or s is None:
            return default
        return s.strip().lower() in ('1', 'true', 'yes', 'on')

    opts = {
        'preset': v.get('preset', 'auto'),
        'mode': v.get('mode', 'safe'),
        'smoothness': f('smoothness'),
        'fidelity': f('fidelity'),
        'marketplace': v.get('marketplace', 'auto'),
        'movement_cap': f('movement_cap'),
        'noise_target': f('noise_target'),
        'convert_shapes': b('convert_shapes', False),
        'round_corners': b('round_corners', False),
        'remove_tiny': b('remove_tiny', True),
        'auto_close': b('auto_close', True),
    }
    return opts, b('qa', True)


def _read_upload():
    f = request.files.get('file')
    if f is None:
        return None, 'no file uploaded (field "file")'
    raw = f.read()
    if not raw:
        return None, 'uploaded file is empty'
    if len(raw) > 12 * 1024 * 1024:
        return None, 'file too large (>12 MB)'
    try:
        return (f.filename or 'upload.svg', raw.decode('utf-8')), None
    except UnicodeDecodeError:
        return None, 'file is not UTF-8 text (SVG must be XML text)'


@app.get('/')
def index():
    return send_file(ROOT / 'index.html')


@app.get('/favicon.ico')
def favicon():
    return Response(status=204)


@app.get('/api/health')
def health():
    return jsonify({'ok': True})


@app.get('/api/sample/<name>')
def sample(name):
    from tools.make_samples import ensure_samples, SAMPLES
    ensure_samples()
    if name not in SAMPLES:
        return jsonify({'error': 'unknown sample: %s' % name}), 404
    return send_file(ROOT / 'samples' / SAMPLES[name], mimetype='image/svg+xml')


@app.post('/api/smooth')
def api_smooth():
    try:
        up, err = _read_upload()
        if err:
            return jsonify({'error': err}), 400
        name, text = up
        opts, do_qa = _opts_from_request()
        out = run_pipeline(text, opts, do_qa=do_qa)
        return jsonify({
            'name': name,
            'preset_used': out['preset_used'],
            'mode': out['params']['mode'],
            'seconds': out['seconds'],
            'options': out['options'],
            'summary': out['summary'],
            'notes': out['notes'],
            'smoothed_svg': out['svg_out'],
            'qa': out['qa'],
            'compare_png_b64': out['compare_png_b64'],
            'checklist': out['checklist'],
            'scores': out['scores'],
            'site': {'key': out['site_key'], 'label': out['site_label']},
            'mp': out['mp'],
            'global_revert': out['global_revert'],
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'internal engine error'}), 500


@app.post('/api/export')
def api_export():
    try:
        up, err = _read_upload()
        if err:
            return jsonify({'error': err}), 400
        name, text = up
        opts, _ = _opts_from_request()
        fmt = (request.values.get('format') or 'svg').lower()
        base = Path(name).stem or 'artwork'
        if fmt == 'svg':
            out = run_pipeline(text, opts, do_qa=False)
            data = out['svg_out']
            return Response(data, mimetype='image/svg+xml',
                            headers={'Content-Disposition':
                                     'attachment; filename="%s.smooth.svg"' % base})
        if fmt in ('eps10', 'eps8'):
            level = 10 if fmt == 'eps10' else 8
            eps, _, _ = export_eps(text, opts, level=level)
            return Response(eps, mimetype='application/postscript',
                            headers={'Content-Disposition':
                                     'attachment; filename="%s.eps%d.eps"'
                                     % (base, level)})
        return jsonify({'error': 'unknown format: %s (svg/eps10/eps8)' % fmt}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'internal engine error'}), 500


def main():
    ap = argparse.ArgumentParser(description='vectorsmooth web service')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()
    from tools.make_samples import ensure_samples
    ensure_samples()
    print('vectorsmooth at http://%s:%d' % (
        'localhost' if args.host == '0.0.0.0' else args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
