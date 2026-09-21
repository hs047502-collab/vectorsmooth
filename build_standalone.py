#!/usr/bin/env python3
"""Inline core/*.py + samples/*.svg into index.html embedded blocks.

Makes index.html a fully self-contained single file (works from file://,
GitHub Pages, any static host). Only the Pyodide CDN still needs internet.

Idempotent: run after any engine change:
    python3 tools/build_standalone.py
"""
import datetime
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / 'index.html'

CORE = ['__init__.py', 'svg_parse.py', 'geometry.py', 'clean.py',
        'raster.py', 'validate.py', 'export.py', 'pipeline.py']
SAMPLES = ['star_jitter.svg', 'blob_noise.svg', 'rounded_square_dense.svg']


def block(ident, content):
    if '</script' in content.lower():
        raise ValueError('%s contains </script - cannot embed raw' % ident)
    # loader strips exactly one leading + one trailing newline
    return '<script type="%s" id="%s">\n%s\n</script>' % (
        'text/x-python' if ident.startswith('emb-core-') else 'text/plain',
        ident, content)


def main():
    html = HTML.read_text(encoding='utf-8')
    total = 0
    digest = hashlib.sha256()
    for name in CORE:
        ident = 'emb-core-' + name
        content = (ROOT / 'core' / name).read_text(encoding='utf-8')
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        pat = re.compile(r'<script type="[^"]*" id="%s">.*?</script>'
                         % re.escape(ident), re.S)
        if not pat.search(html):
            sys.exit('placeholder missing for %s - index.html out of sync'
                     % ident)
        html = pat.sub(lambda m: block(ident, content), html, count=1)
        total += len(content)
        digest.update(content.encode('utf-8'))
    for name in SAMPLES:
        ident = 'emb-sample-' + name
        content = (ROOT / 'samples' / name).read_text(encoding='utf-8')
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        pat = re.compile(r'<script type="[^"]*" id="%s">.*?</script>'
                         % re.escape(ident), re.S)
        if not pat.search(html):
            sys.exit('placeholder missing for %s - index.html out of sync'
                     % ident)
        html = pat.sub(lambda m: block(ident, content), html, count=1)
        total += len(content)
        digest.update(content.encode('utf-8'))
    stamp = '<!--BUILD %s engine-%s-->' % (
        datetime.date.today().isoformat(), digest.hexdigest()[:12])
    if '<!--BUILD-->' in html:
        html = html.replace('<!--BUILD-->', stamp)
    else:
        html = re.sub(r'<!--BUILD .*?-->', stamp, html)
    opens = len(re.findall(r'<script(?=[\s>])', html))
    closes = html.count('</script')
    if opens != closes:
        sys.exit('unbalanced script tags (%d vs %d) - a literal </script '
                 'may hide in app JS (breaks browsers)' % (opens, closes))
    HTML.write_text(html, encoding='utf-8')
    print('embedded %d bytes into %s (%s)' % (total, HTML, stamp[10:-3]))


if __name__ == '__main__':
    main()
