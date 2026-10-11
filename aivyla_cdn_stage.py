"""Opt-in, fail-closed transport for the isolated Krea2 Aivyla CDN pilot.

This module never supplies model metadata: aivyla_r2_stage.py still validates
catalog size, ETag, complete-file SHA256 receipts, and safetensors headers.
Only the staging downloader in aivyla-global should call download_file().
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

MODEL_KEY = 'models/split_files/diffusion_models/krea2_turbo_fp8_scaled.safetensors'
MODEL_ID = 'pilot-ce1747db6aaf7cf4'
API_URL = 'https://studio.radioarvila.gr/api/internal/aivyla-cdn-pilot/token'
CDN_URL = 'https://aivyla-r2-cdn-pilot.radioarvilaofficial.workers.dev'
CHUNK = 64 * 1024 * 1024
UA = 'Aivyla-CDN-Model-Stage/1.0'


class CDNIntegrityError(Exception):
    """Content/version mismatch: DO NOT hide this with direct R2 fallback."""


class _Tokens:
    def __init__(self):
        self._lock = threading.Lock()
        self._token = None
        self._expiry = 0.0

    def get(self, rejected=None):
        with self._lock:
            if self._token and time.monotonic() < self._expiry - 30 and (
                rejected is None or rejected != self._token
            ):
                return self._token
            issuer_key = os.environ.get('AIVYLA_CDN_ISSUER_KEY', '')
            if len(issuer_key) < 32:
                raise RuntimeError('CDN issuer key not configured for dev worker')
            body = json.dumps({'model_id': MODEL_ID}).encode('utf-8')
            request = urllib.request.Request(
                API_URL, data=body, method='POST',
                headers={'Authorization': 'Bearer ' + issuer_key,
                         'Content-Type': 'application/json', 'User-Agent': UA})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.load(response)
            token = payload.get('access_token')
            ttl = payload.get('expires_in')
            if not isinstance(token, str) or not token or not isinstance(ttl, int) or not 60 <= ttl <= 300:
                raise RuntimeError('Invalid CDN token response')
            self._token = token
            self._expiry = time.monotonic() + ttl
            return token


def _write_at(fd, data, offset):
    view = memoryview(data)
    while view:
        count = os.pwrite(fd, view, offset)
        if count <= 0:
            raise OSError('Short CDN file write')
        offset += count
        view = view[count:]


def download_file(model, destination: Path):
    """Fetch a complete, version-pinned R2 model through authenticated 64 MiB CDN chunks.

    Streaming writes with bounded memory, e.g. eight 1 MiB chunks in flight.
    The caller owns the partial file lifecycle, SHA256 verification and publish.
    Any error leaves destination untrusted; caller must unlink or replace it.
    """
    if model.key != MODEL_KEY:
        raise ValueError('CDN pilot only permits Krea2 source key')
    if not isinstance(model.bytes, int) or model.bytes < 4 * CHUNK or model.bytes > 64 * 1024**3:
        raise ValueError('Unexpected pilot source size')
    if not model.etag:
        raise CDNIntegrityError('R2 catalog ETag missing')
    token_provider = _Tokens()
    token_provider.get()  # fail early before creating the destination
    workers = max(1, min(12, int(os.environ.get('AIVYLA_CDN_PILOT_WORKERS', '8'))))
    count = (model.bytes + CHUNK - 1) // CHUNK
    if count > 99999:
        raise ValueError('Too many CDN parts')
    destination = Path(destination)
    # Caller uses a unique .aivyla-r2-part path. Do not open in append mode.
    fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, model.bytes)

        def one(index):
            offset = index * CHUNK
            wanted = min(CHUNK, model.bytes - offset)
            url = f'{CDN_URL}/v1/{MODEL_ID}/part-{index:05d}.bin'
            for attempt in range(3):
                token = token_provider.get()
                request = urllib.request.Request(url, headers={
                    'Authorization': 'Bearer ' + token,
                    'User-Agent': UA,
                    'Accept-Encoding': 'identity',
                })
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 200:
                            raise RuntimeError(f'Unexpected CDN status {response.status}')
                        # R2 head metadata must match the catalog that stage_workflow selected.
                        etag = (response.headers.get('X-Aivyla-Source-Etag') or '').strip('"')
                        size = response.headers.get('X-Aivyla-Source-Size')
                        if etag != model.etag or size != str(model.bytes):
                            raise CDNIntegrityError('CDN object version does not match R2 catalog')
                        content_length = response.headers.get('Content-Length')
                        if content_length is not None and content_length != str(wanted):
                            raise CDNIntegrityError(f'CDN length mismatch, part {index}')
                        pos = offset
                        while pos < offset + wanted:
                            data = response.read(min(1024 * 1024, offset + wanted - pos))
                            if not data:
                                raise CDNIntegrityError(f'Incomplete CDN body, part {index}')
                            _write_at(fd, data, pos)
                            pos += len(data)
                        if response.read(1):
                            raise CDNIntegrityError(f'CDN overlong body, part {index}')
                        return
                except urllib.error.HTTPError as exc:
                    if exc.code == 401 and attempt < 2:
                        token_provider.get(rejected=token)
                        continue
                    if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                        raise
                except (TimeoutError, ConnectionError, OSError):
                    if attempt == 2:
                        raise
                time.sleep(0.5 * (attempt + 1))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, index) for index in range(count)]
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
