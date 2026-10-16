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
PRUNE_UNUSED = os.getenv('AIVYLA_R2_PRUNE_UNUSED', '1') == '1'
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
    if data.get('bytes') != expected_bytes:
        raise ValueError(f'Remote receipt size mismatch: {key}')
    # Cross-provider copy preserves trusted SHA256 receipt but may change
    # S3 multipart ETags; the full download is still hashed and verified.
    is_tigris = os.getenv('AIVYLA_R2_ENDPOINT', '').rstrip('/') == 'https://t3.storage.dev'
    if data.get('etag') != etag and not is_tigris:
        raise ValueError(f'Remote receipt ETag mismatch: {key}')
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
        start_total = time.monotonic()
        expected_sha = _receipt_sha(registry().client, registry().bucket, model.key, model.bytes, model.etag)
        receipt_s = time.monotonic() - start_total
        from boto3.s3.transfer import TransferConfig
        concurrency = int(os.getenv('AIVYLA_R2_DOWNLOAD_STREAMS', '12'))
        concurrency = max(1, min(24, concurrency))
        config = TransferConfig(multipart_threshold=32*1024**2,
                                multipart_chunksize=32*1024**2,
                                max_concurrency=concurrency, use_threads=True)
        started = time.monotonic()
        transport = 'direct_r2'
        if os.getenv('AIVYLA_CDN_V2_ENABLED', '0') == '1':
            try:
                from aivyla_cdn_v2_stage import download_file, CDNIntegrityError
                download_file(model, temp)
                transport = 'cdn_v2'
            except (CDNIntegrityError, ValueError) as exc:
                raise R2IntegrityError(str(exc)) from exc
            except Exception as exc:
                # Only availability/transport errors fall back to verified R2.
                temp.unlink(missing_ok=True)
                print(f'Aivyla CDN v2 unavailable ({type(exc).__name__}); using direct R2', flush=True)
        elif (os.getenv('AIVYLA_CDN_PILOT_ENABLED', '0') == '1'
                and model.key == 'models/split_files/diffusion_models/krea2_turbo_fp8_scaled.safetensors'):
            try:
                from aivyla_cdn_stage import download_file, CDNIntegrityError
                download_file(model, temp)
                transport = 'cdn'
            except CDNIntegrityError as exc:
                raise R2IntegrityError(str(exc)) from exc
            except Exception as exc:
                # Discard incomplete CDN bytes; retain verified direct-R2 fallback.
                temp.unlink(missing_ok=True)
                print(f'Aivyla CDN unavailable ({type(exc).__name__}); using direct R2', flush=True)
        if transport == 'direct_r2':
            registry().client.download_file(registry().bucket, model.key, str(temp), Config=config)
        transfer_s = time.monotonic() - started
        if temp.stat().st_size != model.bytes:
            raise R2IntegrityError(f'R2 download incomplete: {model.key}')
        hash_started = time.monotonic()
        digest = _hash(temp)
        hash_s = time.monotonic() - hash_started
        if expected_sha and digest != expected_sha:
            raise R2IntegrityError(f'R2 SHA256 mismatch: {model.key}')
        verify_started = time.monotonic()
        if temp.suffix.lower() == '.safetensors' or dst.suffix.lower() == '.safetensors':
            legacy._validate_safetensors(temp)
        # Refuse object updates during download: do not cache a mixed/version-changing file.
        head = registry().client.head_object(Bucket=registry().bucket, Key=model.key)
        if head['ContentLength'] != model.bytes or head['ETag'].strip('"') != model.etag:
            raise R2IntegrityError(f'R2 model changed while downloading: {model.key}')
        verify_s = time.monotonic() - verify_started
        os.replace(temp, dst)
        _atomic_manifest(dst, fp, digest)
        print(f'Aivyla R2 MODEL {model.category}/{model.filename}: '
              f'{model.bytes/1024**3:.2f} GiB; receipt={receipt_s:.2f}s '
              f'download={transfer_s:.2f}s sha256={hash_s:.2f}s '
              f'verify={verify_s:.2f}s total={time.monotonic()-start_total:.2f}s '
              f'checksum={"trusted_sha256_pass" if expected_sha else "local_sha256_only"} '
              f'workers={concurrency} transport={transport}', flush=True)
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



# Conservative ComfyUI execution-graph pruning. These are confirmed output
# classes. Unknown terminal classes are always retained: a custom node may
# have OUTPUT_NODE=True even if the worker does not know that class.
KNOWN_OUTPUTS = frozenset({
    'SaveImage', 'PreviewImage', 'SaveAnimatedPNG', 'SaveAnimatedWEBP',
    'SaveVideo', 'SaveAudio', 'SaveImageWebsocket', 'VHS_VideoCombine',
})
# Core ComfyUI classes known *not* to have side-effect output behavior.
# Only use this list to prune disconnected terminal branches. Do not guess
# about custom node classes or skip an unknown output node.
NON_OUTPUT_TERMINALS = frozenset({
    'CheckpointLoaderSimple', 'CheckpointLoader', 'UNETLoader', 'UnetLoaderGGUF',
    'LoraLoader', 'LoraLoaderModelOnly', 'VAELoader', 'CLIPLoader',
    'DualCLIPLoader', 'TripleCLIPLoader', 'UpscaleModelLoader',
    'KSampler', 'KSamplerAdvanced', 'CLIPTextEncode', 'CLIPTextEncodeSDXL',
    'VAEDecode', 'VAEEncode', 'EmptyLatentImage', 'EmptyImage',
    'LoadImage', 'LoadImageMask',
})


def _active_workflow(workflow):
    """Return a safe output-connected node subset; full workflow on ambiguity.

    Never remove custom/unknown terminal classes (potential OUTPUT_NODEs).
    With no recognized output we cannot tell which custom node executes, so
    retain the entire workflow. Graph cycles / malformed links also disable
    pruning rather than silently causing missing model downloads.
    """
    if not PRUNE_UNUSED or not isinstance(workflow, dict) or len(workflow) > 20000:
        return workflow, 0
    nodes = {str(k): v for k, v in workflow.items()
             if isinstance(v, dict) and isinstance(v.get('inputs'), dict)}
    if len(nodes) != len(workflow):
        return workflow, 0
    outputs = {k for k, v in nodes.items() if v.get('class_type') in KNOWN_OUTPUTS}
    if not outputs:
        return workflow, 0
    parents = {}
    referenced = set()

    def collect_dependencies(value, depth=0):
        """Follow ordinary and nested ComfyUI links; fail safe on ambiguity."""
        if depth > 16:
            return set(), True
        if isinstance(value, (list, tuple)):
            if (len(value) == 2 and isinstance(value[0], (str, int))
                    and isinstance(value[1], int) and not isinstance(value[1], bool)
                    and value[1] >= 0):
                name = str(value[0])
                return ({name}, False) if name in nodes else (set(), True)
            links = set()
            for subvalue in value:
                found, bad = collect_dependencies(subvalue, depth + 1)
                if bad:
                    return set(), True
                links.update(found)
            return links, False
        if isinstance(value, dict):
            links = set()
            for subvalue in value.values():
                found, bad = collect_dependencies(subvalue, depth + 1)
                if bad:
                    return set(), True
                links.update(found)
            return links, False
        return set(), False

    for node_id, node in nodes.items():
        deps = set()
        for value in node['inputs'].values():
            found, bad = collect_dependencies(value)
            if bad:
                return workflow, 0
            deps.update(found)
        parents[node_id] = deps
        referenced.update(deps)
    unknown_terminals = {k for k, v in nodes.items()
                         if k not in referenced and v.get('class_type') not in NON_OUTPUT_TERMINALS}
    stack = list(outputs | unknown_terminals)
    active = set()
    while stack:
        current = stack.pop()
        if current in active:
            continue
        active.add(current)
        stack.extend(parents[current] - active)
    if not active:
        return workflow, 0
    pruned = len(nodes) - len(active)
    return {k: v for k, v in workflow.items() if str(k) in active}, pruned


def _model_location(name, field):
    """Canonical loader destination; original R2 source may use a legacy category.

    The exact requested filename is mandatory; no model substitution.
    """
    if name == 'LatentUpscaleModelLoader' and field == 'model_name':
        return 'latent_upscale_models', ['latent_upscale_models', 'upscale_models'], True
    if name == 'LTXVAudioVAELoader' and field == 'ckpt_name':
        return 'checkpoints', ['checkpoints', 'diffusion_models'], True
    return None, legacy._possible_categories(name, field), False


def _referenced(workflow):
    results = {}
    catalog = registry() if R2_ENABLED else None
    # Resolve R2 entries independently of FUSE metadata operations.
    h3_pruned = False
    for node in workflow.values():
        if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
            continue
        name = node.get('class_type', '')
        for field, filename in node['inputs'].items():
            if not legacy._model_filename(filename) or field.lower() in (
                    'prompt','text','negative','positive','image','audio','video','filename','file'):
                continue
            target, categories, required = _model_location(name, field)
            lower = filename.lower()
            if 'minimax' in lower and 'h3' in lower and 'pruned' in lower:
                h3_pruned = True
            try:
                model = catalog.find(filename, categories) if catalog else None
            except Exception:
                # A transient R2 catalog failure must not make a normally available
                # Global Volume workflow fail when fallback is enabled.
                if not R2_FALLBACK:
                    raise
                model = None
            if model:
                destination = target or model.category
                results[(destination, filename)] = (
                    destination, filename, model, None, categories)
                continue
            source = legacy._find_source(filename, categories)
            if source:
                destination = target or source[0]
                results[(destination, filename)] = (
                    destination, filename, None, source[1], categories)
            elif required or (name in legacy.KNOWN and field in legacy.KNOWN[name][1]):
                raise FileNotFoundError(
                    f'Model not in R2 or Global Volume: {name}.{field}={filename} '
                    f'(searched categories: {categories})')
    # Optional basis for pruned MiniMax H3, not explicitly named in the workflow.
    # Never substitute an entire non-pruned checkpoint for this small weight.
    if h3_pruned:
        basis_name = 'h3_silu_temb_grid.safetensors'
        basis_categories = ['h3_adaln']
        try:
            basis = catalog.find(basis_name, basis_categories) if catalog else None
        except Exception:
            if not R2_FALLBACK:
                raise
            basis = None
        if basis:
            results[('h3_adaln', basis_name)] = (
                'h3_adaln', basis_name, basis, None, basis_categories)
        else:
            source = legacy._find_source(basis_name, basis_categories)
            if source:
                results[('h3_adaln', basis_name)] = (
                    'h3_adaln', basis_name, None, source[1], basis_categories)
            else:
                print('Aivyla R2 WARNING: H3 AdaLN basis unavailable in R2 '
                      'and Global Volume; pruned H3 may warn', flush=True)
    if len(results) > legacy.MAX_REFERENCES:
        raise RuntimeError('Too many model references')
    return list(results.values())


def stage_workflow(workflow):
    if not R2_ENABLED:
        return legacy.stage_workflow(workflow)
    start_total = time.monotonic()
    with LOCK:
        if not CACHE.is_absolute() or not str(CACHE).startswith('/tmp/'):
            raise RuntimeError('Model cache must be under /tmp')
        active_workflow, pruned_nodes = _active_workflow(workflow)
        start_discovery = time.monotonic()
        jobs = _referenced(active_workflow)
        discovery_s = time.monotonic() - start_discovery
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
        elapsed_s = time.monotonic() - start_total
        print(f'Aivyla R2 stage: models={len(jobs)} r2={counts["r2"]} '
              f'global={counts["global"]} cached={counts["hit"]} '
              f'pruned_nodes={pruned_nodes} discovery={discovery_s:.2f}s '
              f'total={elapsed_s:.2f}s', flush=True)
        return {'enabled': True, 'models': len(jobs), 'pruned_nodes': pruned_nodes,
                'discovery_s': round(discovery_s, 3), 'total_s': round(elapsed_s, 3), **counts}
