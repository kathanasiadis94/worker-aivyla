"""Verified lazy R2 model fetch + Global Volume fallback for test workers.

Called by the R2-only wrapper before forwarding a workflow to ComfyUI.
No writes to Global Volume or to any S3 object.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

from aivyla_r2_registry import Registry
import aivyla_global_cache as legacy

CACHE = Path(os.getenv('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'))
RESERVE_GIB = float(os.getenv('AIVYLA_GLOBAL_CACHE_MIN_FREE_GIB', '6'))
R2_ENABLED = os.getenv('AIVYLA_R2_ENABLED', '0') == '1'
R2_FALLBACK = os.getenv('AIVYLA_R2_FALLBACK', '1') == '1'
LOCK = threading.RLock()
_registry = None


class R2IntegrityError(Exception):
    """R2 object failed integrity checks; never fall back silently."""



def registry():
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def _manifest(dst):
    return dst.with_name(dst.name + '.aivyla-cache.json')


def _local_ok(dst, fp):
    if not dst.is_file():
        return False
    try:
        receipt = json.loads(_manifest(dst).read_text())
        return (dst.stat().st_size == fp['source_size'] and
                all(receipt.get(k) == value for k, value in fp.items())
                and len(receipt.get('sha256', '')) == 64)
    except (OSError, ValueError):
        return False


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_manifest(dst, fp, digest):
    receipt = _manifest(dst)
    temp = receipt.with_name(receipt.name + '.tmp')
    temp.write_text(json.dumps({**fp, 'sha256': digest, 'stored_at': time.time()}, sort_keys=True))
    os.replace(temp, receipt)


def _receipt_sha(client, bucket, key, expected_bytes, etag):
    """Verify historical R2 uploader receipts when present; direct uploads are also allowed."""
    receipt_key = '_aivyla/receipts/' + key + '.json'
    try:
        raw = client.get_object(Bucket=bucket, Key=receipt_key)['Body'].read(8192)
    except Exception as e:
        # Only missing receipt may be ignored; never suppress network/auth errors.
        code = getattr(e, 'response', {}).get('Error', {}).get('Code')
        if isinstance(e, client.exceptions.NoSuchKey) or code in ('NoSuchKey', '404', 'NotFound'):
            return None
        raise
    data = json.loads(raw)
    if data.get('bytes') != expected_bytes or data.get('etag') != etag:
        raise ValueError(f'Remote receipt mismatch: {key}')
    digest = data.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError(f'Invalid receipt SHA256: {key}')
    return digest


def _fetch_one(model, dst, filename, categories, allow_fallback=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    fp = {'source': 'r2://' + registry().bucket + '/' + model.key,
          'source_size': model.bytes, 'source_etag': model.etag}
    if _local_ok(dst, fp):
        return 'hit'
    temp = dst.with_name(dst.name + '.aivyla-r2-part')
    temp.unlink(missing_ok=True)
    try:
        expected_sha = _receipt_sha(registry().client, registry().bucket, model.key, model.bytes, model.etag)
        from boto3.s3.transfer import TransferConfig
        concurrency = int(os.getenv('AIVYLA_R2_DOWNLOAD_STREAMS', '8'))
        concurrency = max(1, min(24, concurrency))
        config = TransferConfig(multipart_threshold=32*1024**2,
                                multipart_chunksize=32*1024**2,
                                max_concurrency=concurrency, use_threads=True)
        started = time.monotonic()
        registry().client.download_file(registry().bucket, model.key, str(temp), Config=config)
        if temp.stat().st_size != model.bytes:
            raise R2IntegrityError(f'R2 download incomplete: {model.key}')
        digest = _hash(temp)
        if expected_sha and digest != expected_sha:
            raise R2IntegrityError(f'R2 SHA256 mismatch: {model.key}')
        if temp.suffix.lower() == '.safetensors' or dst.suffix.lower() == '.safetensors':
            legacy._validate_safetensors(temp)
        # Refuse object updates during download: do not cache a mixed/version-changing file.
        head = registry().client.head_object(Bucket=registry().bucket, Key=model.key)
        if head['ContentLength'] != model.bytes or head['ETag'].strip('"') != model.etag:
            raise R2IntegrityError(f'R2 model changed while downloading: {model.key}')
        os.replace(temp, dst)
        _atomic_manifest(dst, fp, digest)
        print(f'Aivyla R2 MODEL {model.category}/{model.filename}: '
              f'{model.bytes/1024**3:.2f} GiB in {time.monotonic()-started:.1f}s (SHA256 OK)', flush=True)
        return 'r2'
    except Exception as exc:
        temp.unlink(missing_ok=True)
        # Only network/service failures may fall back; never bypass data integrity failures.
        if not allow_fallback or isinstance(exc, (ValueError, R2IntegrityError)):
            raise
        # Resolve remote FUSE path only if R2 actually failed; normal R2 requests
        # should not pay for slow Global Volume metadata operations.
        source = legacy._find_source(filename, categories)
        if source is None:
            raise
        print(f'Aivyla R2 network failure, trying Global Volume for {model.filename}: {type(exc).__name__}', flush=True)
        return _copy_volume(source[1], dst)


def _copy_volume(src, dst):
    """Stream-copy even when FUSE inaccurately reports a 0-byte source.

    Never use shutil.copyfile()/sendfile on zero-size FUSE files; stream via
    Python reads and compare against the validated safetensors header length.
    """
    fp = legacy._get_source_fingerprint(src)
    if fp['source_size'] == 0 and src.suffix.lower() == '.safetensors':
        import struct
        with src.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('Incomplete source safetensors header')
            n = struct.unpack('<Q', raw)[0]
            if n < 2 or n > 100_000_000:
                raise ValueError('Invalid safetensors header')
            header = json.loads(stream.read(n))
        offsets = [v['data_offsets'][1] for k, v in header.items()
                   if k != '__metadata__' and isinstance(v, dict) and 'data_offsets' in v]
        if not offsets:
            raise ValueError('Safetensors file has no tensor offsets')
        fp['source_size'] = 8 + n + max(offsets)
    if fp['source_size'] < 8:
        raise ValueError(f'Global Volume source has unknown/invalid size: {src}')
    if _local_ok(dst, fp):
        return 'hit'
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + '.aivyla-volume-part')
    temp.unlink(missing_ok=True)
    try:
        digest = hashlib.sha256()
        actual = 0
        with src.open('rb') as reader, temp.open('xb') as writer:
            while block := reader.read(8 * 1024 * 1024):
                actual += len(block)
                if actual > fp['source_size']:
                    raise IOError(f'Global Volume source larger than expected: {src}')
                writer.write(block)
                digest.update(block)
        if actual != fp['source_size']:
            raise IOError(f'Global Volume source incomplete: read {actual} expected {fp["source_size"]}')
        if dst.suffix.lower() == '.safetensors':
            legacy._validate_safetensors(temp)
        # Avoid publishing a cache built from a changing source.
        if src.stat().st_size not in (0, actual):
            raise IOError('Global source changed during transfer')
        os.replace(temp, dst)
        _atomic_manifest(dst, fp, digest.hexdigest())
        return 'global'
    finally:
        temp.unlink(missing_ok=True)


def _referenced(workflow):
    results = {}
    catalog = registry() if R2_ENABLED else None
    # Resolve R2 entries independently of FUSE metadata operations.
    for node in workflow.values():
        if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
            continue
        name = node.get('class_type', '')
        for field, filename in node['inputs'].items():
            if not legacy._model_filename(filename) or field.lower() in (
                    'prompt','text','negative','positive','image','audio','video','filename','file'):
                continue
            categories = legacy._possible_categories(name, field)
            try:
                model = catalog.find(filename, categories) if catalog else None
            except Exception:
                # A transient R2 catalog failure must not make a normally available
                # Global Volume workflow fail when fallback is enabled.
                if not R2_FALLBACK:
                    raise
                model = None
            if model:
                results[(model.category, filename)] = (model.category, filename, model, None, categories)
                continue
            source = legacy._find_source(filename, categories)
            if source:
                results[(source[0], filename)] = (source[0], filename, None, source[1], categories)
            elif name in legacy.KNOWN and field in legacy.KNOWN[name][1]:
                raise FileNotFoundError(f'Model not in R2 or Global Volume: {name}.{field}={filename}')
    if len(results) > legacy.MAX_REFERENCES:
        raise RuntimeError('Too many model references')
    return list(results.values())


def stage_workflow(workflow):
    if not R2_ENABLED:
        return legacy.stage_workflow(workflow)
    with LOCK:
        if not CACHE.is_absolute() or not str(CACHE).startswith('/tmp/'):
            raise RuntimeError('Model cache must be under /tmp')
        jobs = _referenced(workflow)
        free = shutil.disk_usage('/tmp').free
        required = 0
        for category, filename, model, src, categories in jobs:
            dst = CACHE/category/filename
            if model:
                fp = {'source': 'r2://' + registry().bucket + '/' + model.key,
                      'source_size': model.bytes, 'source_etag': model.etag}
            else:
                fp = legacy._get_source_fingerprint(src)
            if not _local_ok(dst, fp):
                required += model.bytes if model else max(fp['source_size'], 0)
        if required + RESERVE_GIB*1024**3 > free:
            raise RuntimeError(f'Insufficient local cache disk: need {required/1024**3:.1f} GiB + '
                               f'{RESERVE_GIB:g} GiB reserve, have {free/1024**3:.1f} GiB')
        counts = {'r2':0, 'global':0, 'hit':0}
        parallel = max(1, min(3, int(os.getenv('AIVYLA_R2_PARALLEL_MODELS', '2'))))
        def fetch(job):
            category, filename, model, source, categories = job
            dst = CACHE/category/filename
            return (_fetch_one(model, dst, filename, categories, R2_FALLBACK)
                    if model else _copy_volume(source, dst))
        # R2 download can be overlapped, but never run two staging jobs simultaneously.
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            for result in pool.map(fetch, jobs):
                counts[result] += 1
        print(f'Aivyla R2 stage: models={len(jobs)} r2={counts["r2"]} '
              f'global={counts["global"]} cached={counts["hit"]}', flush=True)
        return {'enabled': True, 'models': len(jobs), **counts}
