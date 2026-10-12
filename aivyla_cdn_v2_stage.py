"""Opt-in V2 full-object authenticated CDN transport for ANY verified R2 model.

Only an R2 catalog model is accepted. The calling stage retains original receipt
SHA256, HEAD/ETag verification, safetensors checks, and disk lifecycle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import urllib.parse

from aivyla_r2_registry import parse_model

CHUNK = 64 * 1024 * 1024
API_URL = 'https://studio.radioarvila.gr/api/internal/aivyla-cdn-v2/token'
CDN_URL = os.getenv('AIVYLA_CDN_V2_URL', '').rstrip('/')
USER_AGENT = 'Aivyla-Global-CDN-V2/1.0'


class CDNIntegrityError(Exception):
    """Never silently bypass content identity errors."""


class _Token:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.token = None
        self.model_id = None
        self.deadline = 0.0

    def get(self, rejected=None):
        with self.lock:
            if (self.token and time.monotonic() < self.deadline - 30 and
                    (rejected is None or rejected != self.token)):
                return self.model_id, self.token
            secret = os.getenv('AIVYLA_CDN_ISSUER_KEY', '')
            if len(secret) < 32:
                raise RuntimeError('AIVYLA_CDN_ISSUER_KEY missing in isolated dev endpoint')
            req = urllib.request.Request(API_URL, method='POST',
                data=json.dumps({'model_key': self.model.key,
                                 'etag': self.model.etag,
                                 'bytes': self.model.bytes}).encode('utf-8'),
                headers={'Authorization': 'Bearer ' + secret,
                         'Content-Type': 'application/json', 'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=25) as response:
                if response.status != 200:
                    raise RuntimeError('Unexpected CDN issuer status')
                data = json.load(response)
            model_id = data.get('model_id')
            token = data.get('access_token')
            ttl = data.get('expires_in')
            if (not isinstance(model_id, str) or len(model_id) != 26 or
                    not model_id.startswith('m-') or not all(x in '0123456789abcdef' for x in model_id[2:]) or
                    not isinstance(token, str) or not token or
                    type(ttl) is not int or not 60 <= ttl <= 300):
                raise RuntimeError('Malformed CDN v2 token issuer response')
            self.token, self.model_id = token, model_id
            self.deadline = time.monotonic() + ttl
            return model_id, token


def _write_at(fd, data, offset):
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        if n <= 0:
            raise OSError('Short CDN file write')
        view = view[n:]
        offset += n


def download_file(model, destination: Path):
    parsed = parse_model(model.key)
    if not parsed or parsed.category != model.category or parsed.filename != model.filename:
        raise ValueError('CDN refuses object not in R2 model catalog namespace')
    if type(model.bytes) is not int or not 8 <= model.bytes <= 256 * 1024**3 or not model.etag:
        raise ValueError('Invalid model size/ETag')
    parsed_url = urllib.parse.urlsplit(CDN_URL)
    if (parsed_url.scheme != 'https' or not parsed_url.hostname or
            not parsed_url.hostname.endswith('.workers.dev') or parsed_url.username or
            parsed_url.password or parsed_url.path or parsed_url.query or parsed_url.fragment):
        raise RuntimeError('AIVYLA_CDN_V2_URL must be an HTTPS Cloudflare Workers hostname')
    tokens = _Token(model)
    tokens.get()  # fail before creating an incomplete file
    max_workers = max(1, min(12, int(os.getenv('AIVYLA_CDN_V2_WORKERS', '8'))))
    pieces = (model.bytes + CHUNK - 1) // CHUNK
    destination = Path(destination)
    fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, model.bytes)

        def part(index):
            offset = index * CHUNK
            wanted = min(CHUNK, model.bytes - offset)
            for attempt in range(3):
                identity, token = tokens.get()
                url = f'{CDN_URL}/v2/{identity}/part-{index:05d}.bin'
                req = urllib.request.Request(url, headers={
                    'Authorization': 'Bearer ' + token,
                    'Accept-Encoding': 'identity', 'User-Agent': USER_AGENT})
                try:
                    with urllib.request.urlopen(req, timeout=120) as response:
                        if response.status != 200:
                            raise RuntimeError('Unexpected CDN v2 response status')
                        etag = (response.headers.get('X-Aivyla-Source-Etag') or '').strip('"')
                        size = response.headers.get('X-Aivyla-Source-Size')
                        if etag != model.etag or size != str(model.bytes):
                            raise CDNIntegrityError('CDN v2 object version differs from catalog')
                        length = response.headers.get('Content-Length')
                        if length is not None and length != str(wanted):
                            raise CDNIntegrityError('CDN v2 part size mismatch')
                        pos = offset
                        while pos < offset + wanted:
                            buf = response.read(min(1024 * 1024, offset + wanted - pos))
                            if not buf:
                                raise CDNIntegrityError('Truncated CDN v2 part')
                            _write_at(fd, buf, pos)
                            pos += len(buf)
                        if response.read(1):
                            raise CDNIntegrityError('Overlong CDN v2 part')
                        return
                except urllib.error.HTTPError as error:
                    if error.code in (401, 403) and attempt < 2:
                        # Rotate the CDN grant once rejected; preserve the full
                        # original R2 integrity checks on any subsequent fallback.
                        tokens.get(rejected=token)
                        continue
                    if error.code == 403:
                        # Access denial is not proof of corrupt model bytes.
                        # Raise an ordinary error to trigger verified direct R2.
                        ray = error.headers.get('CF-Ray', 'unknown') if error.headers else 'unknown'
                        raise RuntimeError(
                            f'CDN v2 HTTP 403 after token refresh: '
                            f'{model.key}, part={index}, cf_ray={ray}') from error
                    if error.code in (404, 409, 412, 416):
                        # Object identity/version/range problems fail closed.
                        raise CDNIntegrityError(f'CDN v2 rejected model identity: {error.code}') from error
                    if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                        raise
                except (TimeoutError, ConnectionError, OSError):
                    if attempt == 2:
                        raise
                time.sleep(0.5 * (attempt + 1))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(part, i) for i in range(pieces)]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        os.fsync(fd)
    finally:
        os.close(fd)
