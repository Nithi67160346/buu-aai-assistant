#!/usr/bin/env python3
"""Collect ONLY curated AAI official BUU pages and optionally AAI curriculum PDFs.

Deliberately no recursive crawling: excludes CS, SE, ITDI, generic faculty news.
Reads robots.txt (fail closed on error), respects host/scope, rate-limits.
The source list can be reviewed/edited in this file before fetching.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE = 'https://www.informatics.buu.ac.th'
HOST = 'www.informatics.buu.ac.th'
UA = 'AAI-RAG-Learning-Prototype/1.0 (public pages, low request rate)'
TIMEOUT = (10, 30)

# Manually selected, official AAI-only sources; no other faculty content.
AAI_PAGE = f'{BASE}/?page_id=4321'
AAI_NEWS = {
    'AAI orientation preparation': f'{BASE}/?p=16028',
    'AAI researchers at BUU RISE 2026': f'{BASE}/?p=15169',
}
AAI_PDFS = {
    'AAI curriculum 2567': f'{BASE}/wp-content/uploads/2024/01/AAI-Program-67.pdf',
    'AAI 2567 course plan': f'{BASE}/wp-content/uploads/2024/07/planAAI67.pdf',
}
AAI_TERMS = ('AAI', 'ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ')
MAX_HTML = 3_000_000
MAX_PDF = 22_000_000


def is_official_url(url):
    u = urlsplit(url)
    return u.scheme == 'https' and u.hostname == HOST and u.username is None and u.password is None


def robots_rules(session):
    r = session.get(BASE + '/robots.txt', timeout=TIMEOUT, allow_redirects=False)
    if r.status_code == 404:
        robot = RobotFileParser()
        robot.parse([])
        return robot
    if r.status_code != 200 or 'html' in r.headers.get('Content-Type', '').lower():
        raise RuntimeError('robots.txt could not be verified; stopped without crawling.')
    robot = RobotFileParser()
    robot.parse(r.text.splitlines())
    return robot


def get_bounded(session, robots, url, max_size):
    if not is_official_url(url) or not robots.can_fetch(UA, url):
        raise RuntimeError('URL is outside the allowed host or blocked by robots.txt: ' + url)
    with session.get(url, timeout=TIMEOUT, allow_redirects=False, stream=True) as r:
        # Never follow a redirect to another site without checking its rules.
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code} for {url}')
        data = bytearray()
        for part in r.iter_content(chunk_size=65536):
            data.extend(part)
            if len(data) > max_size:
                raise RuntimeError(f'File over size cap: {url}')
        return bytes(data), r.headers.get('Content-Type', '').lower()


def extract_html(data):
    soup = BeautifulSoup(data, 'html.parser')
    head = soup.select_one('article h1.entry-title, h1.entry-title, h1')
    title = head.get_text(' ', strip=True) if head else soup.title.get_text(' ', strip=True) if soup.title else ''
    # Keep only article content; remove menus, footers and promotional sidebar.
    root = (soup.select_one('article .entry-content') or soup.select_one('.entry-content')
            or soup.select_one('main article') or soup.select_one('main'))
    if root is None:
        raise RuntimeError('No main article found; no text saved to avoid unrelated navigation content.')
    for elt in root.select('nav, footer, aside, script, style, noscript, form, .widget, .related-posts'):
        elt.decompose()
    lines = [re.sub(r'\s+', ' ', line).strip() for line in root.get_text('\n', strip=True).splitlines()]
    return re.sub(r'\s+', ' ', title), '\n'.join(x for i, x in enumerate(lines) if x and (i == 0 or x != lines[i-1]))


def aai_title(title):
    return ('ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ' in title or
            re.search(r'(?<![A-Za-z])AAI(?![A-Za-z])', title, re.I) is not None)


def main():
    p = argparse.ArgumentParser(description='AAI-only public page collector')
    p.add_argument('--output', type=Path, default=Path('docs_aai'))
    p.add_argument('--with-pdf', action='store_true', help='Download AAI official curriculum 2567 and course-plan PDFs')
    p.add_argument('--no-news', action='store_true', help='Only AAI curriculum page, not curated AAI news')
    p.add_argument('--delay', type=float, default=2.0, help='Seconds between requests (at least 1)')
    args = p.parse_args()
    if args.delay < 0:
        p.error('delay must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [('AAI program overview', AAI_PAGE, 'html')]
    if not args.no_news:
        jobs += [(k, v, 'html') for k, v in AAI_NEWS.items()]
    if args.with_pdf:
        jobs += [(k, v, 'pdf') for k, v in AAI_PDFS.items()]

    session = requests.Session()
    session.headers.update({'User-Agent': UA, 'Accept': 'text/html,application/pdf'})
    try:
        robots = robots_rules(session)
    except (RuntimeError, requests.RequestException) as e:
        sys.exit(f'[stopped] {e}')

    manifest = {}
    succeeded = 0
    for name, url, kind in jobs:
        try:
            time.sleep(max(1, args.delay))
            contents, ctype = get_bounded(session, robots, url, MAX_PDF if kind == 'pdf' else MAX_HTML)
            digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
            if kind == 'html':
                if 'html' not in ctype and not contents.lstrip().lower().startswith((b'<!doctype', b'<html')):
                    raise RuntimeError('not an HTML response')
                title, text = extract_html(contents)
                if not aai_title(title):
                    raise RuntimeError('HTML title does not identify AAI, skipping')
                if len(text) < 80:
                    raise RuntimeError('too little main content')
                filename = f'aai_page_{digest}.txt'
                (args.output / filename).write_text(f'SOURCE_URL: {url}\nTITLE: {title}\n\n{text}\n', encoding='utf-8')
            else:
                if not contents.startswith(b'%PDF-'):
                    raise RuntimeError('not a real PDF')
                filename = f'aai_pdf_{digest}.pdf'
                (args.output / filename).write_bytes(contents)
            manifest[filename] = {'source_url': url, 'title': name, 'type': kind}
            print(f'[saved] {name}: {filename}')
            succeeded += 1
        except (RuntimeError, requests.RequestException, OSError) as e:
            print(f'[skipped] {name}: {e}')

    # Keep the corpus scoped on re-fetch; do not leave stale copies of previously
    # generated AAI pages/PDFs in the output directory.
    if succeeded:
        for old in args.output.iterdir():
            if old.is_file() and (old.name.startswith('aai_page_') or old.name.startswith('aai_pdf_')) and old.name not in manifest:
                old.unlink()
        (args.output / 'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished: {succeeded}/{len(jobs)} AAI-only items, path: {args.output.resolve()}')
    if not succeeded:
        sys.exit('Nothing fetched. Check access, robots.txt, and current site layout.')
    print('Next: python rag_pi4.py index ./docs_aai')


if __name__ == '__main__':
    main()
