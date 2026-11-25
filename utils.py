import csv
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_HERE      = os.path.dirname(os.path.abspath(__file__))
URL_CSV    = os.path.join(_HERE, 'benign_urls.csv')
CACHE_DIR  = os.path.join(_HERE, 'html_cache')
USER_AGENT = 'Mozilla/5.0 (compatible; PhishingResearch/1.0)'

# Tranco top sites list (https://tranco-list.eu/) — a research-oriented ranking
# of popular domains. The permanent URL always resolves to the latest daily list.
TRANCO_URL = 'https://tranco-list.eu/top-1m.csv.zip'

# Domains matching these substrings are skipped when building the benign list.
_EXCLUDE_KEYWORDS = (
    'crypto', 'bitcoin', 'btc', 'ethereum', 'binance', 'coinbase',
    'blockchain', 'wallet', 'nft', 'dogecoin', 'litecoin', 'kraken',
    'metamask', 'bitfinex', 'bittrex', 'poloniex', 'coinmarketcap',
    'coingecko', 'tether', 'usdt', 'ledger', 'trezor', 'solana', 'cardano',
    'polkadot', 'ripple', 'defi', 'web3', 'opensea', 'uniswap',
    'pancakeswap', 'bitpay', 'miner', 'mining',
)


def _excluded(domain: str) -> bool:
    d = domain.lower()
    return any(k in d for k in _EXCLUDE_KEYWORDS)


def build_benign_urls(target: int = 1000, dest: str = URL_CSV) -> str:
    """Download the Tranco top sites list and write benign URLs to ``dest``.

    Fetches the current Tranco top-1M ranking (https://tranco-list.eu/),
    keeps the top ``target`` domains, and writes them to a single-column CSV
    (header: ``url``). The list is generated locally on first run rather than
    shipped in the repository.

    Tranco is a research-oriented ranking aggregated from several upstream
    providers; please review the Tranco site for the terms that apply to the
    underlying data before redistributing any list you generate.
    """
    print(f'Downloading the Tranco top sites list from {TRANCO_URL} ...')
    req = urllib.request.Request(TRANCO_URL, headers={'User-Agent': USER_AGENT})
    raw = urllib.request.urlopen(req, timeout=60).read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            reader = csv.reader(io.TextIOWrapper(fh, encoding='utf-8'))
            urls = []
            for row in reader:
                if len(row) < 2:
                    continue
                domain = row[1].strip()
                if not domain or _excluded(domain):
                    continue
                urls.append(f'https://{domain}')
                if len(urls) >= target:
                    break

    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['url'])
        writer.writerows([u] for u in urls)

    print(f'Wrote {len(urls)} benign URLs to {dest}')
    return dest


def crawl_benign_pages(
    target:  int = 500,
    workers: int = 50,
    timeout: int = 8,
) -> list[dict]:
    """
    Crawl benign pages from benign_urls.csv and return a list of
    {"url": ..., "html": ..., "label": 0} records.

    Pages are cached to html_cache/0_{md5(url)}.json — reruns skip
    already-fetched URLs. If benign_urls.csv is not present it is generated
    automatically from the Tranco top sites list (see build_benign_urls).
    """
    if not os.path.exists(URL_CSV):
        print('benign_urls.csv not found — generating it from the Tranco top sites list ...')
        build_benign_urls(target=max(target * 2, 1000))

    with open(URL_CSV, newline='', encoding='utf-8') as f:
        urls = [row['url'] for row in csv.DictReader(f) if row.get('url')]

    os.makedirs(CACHE_DIR, exist_ok=True)

    existing_files = [p for p in os.listdir(CACHE_DIR) if p.startswith('0_') and p.endswith('.json')]
    existing       = len(existing_files)
    print(f'Cache has {existing} benign pages. Target: {target}.')

    records = []
    for fname in existing_files:
        with open(os.path.join(CACHE_DIR, fname), encoding='utf-8') as fh:
            rec = json.load(fh)
        if rec.get('html'):
            records.append(rec)

    if existing >= target:
        print(f'Target already met — returning {len(records)} cached records.')
        return records[:target]

    needed = target - existing
    print(f'Fetching {needed} more pages ({workers} workers, {timeout}s timeout) ...')

    def _fetch_one(url):
        url_hash   = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f'0_{url_hash}.json')
        if os.path.exists(cache_path):
            return None
        try:
            resp = requests.get(url, timeout=timeout, headers={'User-Agent': USER_AGENT}, allow_redirects=True)
            resp.raise_for_status()
            ct   = resp.headers.get('Content-Type', '')
            if resp.status_code == 200 and 'text/html' in ct:
                record = {'url': url, 'html': resp.text, 'label': 0}
                with open(cache_path, 'w', encoding='utf-8') as fh:
                    json.dump(record, fh)
                return record
        except Exception:
            return None

    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, u): u for u in urls if not os.path.exists(
            os.path.join(CACHE_DIR, f'0_{hashlib.md5(u.encode(), usedforsecurity=False).hexdigest()}.json')
        )}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                records.append(rec)
                saved += 1
                if saved % 50 == 0:
                    print(f'  {existing + saved}/{target} pages cached ...', flush=True)
                if saved >= needed:
                    for f in futures:
                        f.cancel()
                    break

    print(f'Done — {len(records)} benign records ready ({saved} newly fetched).')
    return records[:target]
