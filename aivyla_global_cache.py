"""Per-request local model staging for Aivyla Global serverless workers.

The Global Volume can provide inconsistent random reads to mmap/safetensors
loaders. Copy requested weights once to local worker disk; verify *before*
ComfyUI attempts to load them. This module never modifies the volume.

Local files are searched first by folder_paths via extra_model_paths.yaml.
A cache is intentionally PER WORKER; refreshing the worker discards it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import threading
import time

ROOT = Path(os.environ.get('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'))
VOLUME = Path('/runpod-volume')
CHUNK = 8 * 1024 * 1024
MIN_FREE_GIB = float(os.environ.get('AIVYLA_GLOBAL_CACHE_MIN_FREE_GIB', '6'))
MAX_REFERENCES = 80
MODEL_SUFFIXES = ('.safetensors', '.sft', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx')
CATEGORY_MAP = {
    'checkpoints': ['checkpoints'],
    'text_encoders': ['text_encoders'],
    'clip_vision': ['clip_vision'],
    'diffusion_models': ['diffusion_models'],
    'vae': ['vae'],
    'loras': ['loras'],
    'upscale_models': ['upscale_models'],
    'latent_upscale_models': ['latent_upscale_models', 'upscale_models'],
    'embeddings': ['embeddings'],
    'controlnet': ['controlnet'],
    'ipadapter': ['ipadapter'],
    'model_patches': ['model_patches'],
    'audio_encoders': ['audio_encoders'],
    'refmods': ['refmods'],
    'h3_adaln': ['h3_adaln'],
}
# Honor original YAML path preference: split_files first, then ComfyUI/models.
SEARCH_ROOTS = [VOLUME / 'split_files', VOLUME / 'ComfyUI' / 'models']
CANONICAL = {
    'clip': 'text_encoders', 'unet': 'diffusion_models',
    'checkpoint': 'checkpoints', 'lora': 'loras',
}
KNOWN = {
    'CheckpointLoaderSimple': ('checkpoints', ('ckpt_name',)),
    'LoraLoader': ('loras', ('lora_name',)),
    'LoraLoaderModelOnly': ('loras', ('lora_name',)),
    'VAELoader': ('vae', ('vae_name',)),
    'CLIPLoader': ('text_encoders', ('clip_name',)),
    'DualCLIPLoader': ('text_encoders', ('clip_name1', 'clip_name2')),
    'TripleCLIPLoader': ('text_encoders', ('clip_name1', 'clip_name2', 'clip_name3')),
    'UNETLoader': ('diffusion_models', ('unet_name',)),
    'UnetLoaderGGUF': ('diffusion_models', ('unet_name',)),
    'Hy3DModelLoader': ('diffusion_models', ('model',)),
    'UpscaleModelLoader': ('upscale_models', ('model_name',)),
}
NAME_HINTS = {
    'ckpt_name': 'checkpoints', 'checkpoint': 'checkpoints',
    'lora_name': 'loras', 'vae_name': 'vae',
    'clip_name': 'text_encoders', 'clip_name1': 'text_encoders',
    'clip_name2': 'text_encoders', 'clip_name3': 'text_encoders',
    'unet_name': 'diffusion_models', 'diffusion_model': 'diffusion_models',
    'refmod_name': 'refmods', 'controlnet_name': 'controlnet',
    'upscale_model': 'upscale_models', 'model_name': None,
}
LOCK = threading.RLock()


def _model_filename(s):
    if not isinstance(s, str) or not s.lower().endswith(MODEL_SUFFIXES):
        return False
    if '\\' in s or '\x00' in s or '://' in s or len(s) > 512:
        return False
    p = PurePosixPath(s)
    return (not p.is_absolute() and len(p.parts) > 0 and
            not any(part in ('', '.', '..') for part in p.parts))


def _possible_categories(class_type, field):
    known = KNOWN.get(class_type)
    if known and field in known[1]:
        return [known[0]]
    hint = NAME_HINTS.get(field.lower())
    if hint:
        return [hint]
    return list(CATEGORY_MAP)


def _find_source(filename, kinds):
    for kind in kinds:
        for root in SEARCH_ROOTS:
            for folder in CATEGORY_MAP[kind]:
                candidate = root / folder / filename
                if candidate.is_file():
                    # Don't allow links to escape the mounted volume.
                    try:
                        candidate.resolve().relative_to(VOLUME)
                    except ValueError:
                        raise RuntimeError('Model path escaped /runpod-volume')
                    return kind, candidate
    return None


def discover(workflow):
    if not isinstance(workflow, dict):
        return []
    refs = {}
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get('class_type', ''))
        for field, filename in inputs.items():
            if not _model_filename(filename):
                continue
            # Ignore prose and named input/output media references.
            if field.lower() in ('prompt', 'text', 'negative', 'positive',
                                 'image', 'audio', 'video', 'filename', 'file'):
                continue
            found = _find_source(filename, _possible_categories(class_type, field))
            if found:
                category, src = found
                refs[(category, filename)] = (src, ROOT / category / filename)
            elif (class_type in KNOWN and field in KNOWN[class_type][1]):
                raise FileNotFoundError(
                    f'Global model preflight: {class_type}.{field}={filename!r} not found')
            # Other custom-node file types may be managed internally; leave them to ComfyUI.
    if len(refs) > MAX_REFERENCES:
        raise RuntimeError(f'Too many model references ({len(refs)} > {MAX_REFERENCES})')
    return list(refs.items())


def _validate_safetensors(path):
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f'Safetensors too small: {path}')
    with path.open('rb') as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError('Safetensors incomplete first 8 bytes')
        n = struct.unpack('<Q', first)[0]
        if n < 2 or n > 100_000_000 or n + 8 > size:
            raise ValueError(f'Invalid safetensors header length={n} file size={size}')
        header = f.read(n)
        if len(header) != n:
            raise ValueError(f'Incomplete safetensors JSON header: {len(header)}/{n}')
        data = json.loads(header)
        if not isinstance(data, dict) or not any(k != '__metadata__' for k in data):
            raise ValueError('Safetensors has no tensors')
        max_end = max((info['data_offsets'][1] for key, info in data.items()
                       if key != '__metadata__'), default=0)
        if max_end + n + 8 > size:
            raise ValueError('Safetensors tensor data exceeds file size')
    try:
        from safetensors import safe_open
    except ImportError:
        return
    with safe_open(str(path), framework='pt', device='cpu') as f:
        if not f.keys():
            raise ValueError('safetensors.safe_open returned no tensors')


def _manifest(path):
    return path.with_name(path.name + '.aivyla-cache.json')


def _get_source_fingerprint(src):
    st = src.stat()
    return {'source': str(src), 'source_size': st.st_size,
            'source_mtime_ns': st.st_mtime_ns}


def _is_cached(dst, fingerprint):
    info_path = _manifest(dst)
    if not dst.is_file() or not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text())
        return (all(info.get(k) == v for k, v in fingerprint.items())
                and dst.stat().st_size == fingerprint['source_size']
                and info.get('sha256'))
    except (OSError, ValueError):
        return False


def _available_bytes():
    ROOT.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(ROOT).free


def _copy_and_verify(src, dst, fingerprint):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.aivyla-part')
    start = time.monotonic()
    try:
        if tmp.exists():
            tmp.unlink()
        h_src = hashlib.sha256()
        total = 0
        with src.open('rb', buffering=CHUNK) as reader, tmp.open('xb', buffering=CHUNK) as writer:
            while True:
                buf = reader.read(CHUNK)
                if not buf:
                    break
                writer.write(buf)
                h_src.update(buf)
                total += len(buf)
            writer.flush()
            os.fsync(writer.fileno())
        if total != fingerprint['source_size']:
            raise IOError(f'Short model copy: {total}/{fingerprint["source_size"]} bytes')
        # Confirm local bytes equal what the source stream returned.
        h_local = hashlib.sha256()
        with tmp.open('rb', buffering=CHUNK) as reader:
            while True:
                buf = reader.read(CHUNK)
                if not buf:
                    break
                h_local.update(buf)
        if h_local.digest() != h_src.digest():
            raise IOError('Model checksum mismatch after copying to local disk')
        if tmp.suffix == '.safetensors' or src.suffix.lower() in ('.safetensors', '.sft'):
            _validate_safetensors(tmp)
        # The original may have changed while we read it.
        if _get_source_fingerprint(src) != fingerprint:
            raise IOError('Global model changed during copy; refusing stale cache')
        os.replace(tmp, dst)
        info = {**fingerprint, 'sha256': h_src.hexdigest(),
                'copied_at': time.time()}
        m = _manifest(dst)
        mt = m.with_name(m.name + '.tmp')
        mt.write_text(json.dumps(info, sort_keys=True))
        os.replace(mt, m)
        seconds = time.monotonic() - start
        print(f'Aivyla Global CACHE MISS copied {src.name} {total / (1024**3):.2f} GiB '
              f'in {seconds:.1f}s ({total/max(seconds, .001)/1024**2:.1f} MiB/s)', flush=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def stage_workflow(workflow):
    """Prepare models for a single job; return dict with staging statistics."""
    if os.environ.get('AIVYLA_GLOBAL_CACHE', '1') != '1':
        return {'enabled': False, 'models': 0}
    if not ROOT.is_absolute() or not str(ROOT).startswith('/tmp/'):
        raise RuntimeError('AIVYLA_GLOBAL_MODEL_ROOT must be under /tmp')
    with LOCK:
        start = time.monotonic()
        refs = discover(workflow)
        prepared = []
        pending = []
        for (category, filename), (src, dst) in refs:
            fingerprint = _get_source_fingerprint(src)
            if _is_cached(dst, fingerprint):
                prepared.append((category, filename, 'hit'))
            else:
                pending.append((category, filename, src, dst, fingerprint))
        required = sum(fp['source_size'] for _category, _filename, _src, _dst, fp in pending)
        reserve = int(MIN_FREE_GIB * (1024**3))
        free = _available_bytes()
        if required + reserve > free:
            need = (required + reserve) / 1024**3
            have = free / 1024**3
            raise RuntimeError(
                f'Global local model cache requires {need:.1f} GiB available but has '
                f'{have:.1f} GiB. Increase Aivyla Global Container Disk; '
                f'never reduce reserve or copy partial models.')
        for category, filename, src, dst, fingerprint in pending:
            # Retry transient FUSE reads once, never expose incomplete files.
            for attempt in (1, 2):
                try:
                    _copy_and_verify(src, dst, fingerprint)
                    prepared.append((category, filename, 'miss'))
                    break
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    print(f'Aivyla Global cache attempt {attempt}/2 failed '
                          f'({category}/{filename}): {error}', flush=True)
                    if attempt == 2:
                        raise RuntimeError(
                            f'Unable to stage {category}/{filename} from Global Volume: {error}') from error
                    time.sleep(1)
        elapsed = time.monotonic() - start
        print(f'Aivyla Global cache ready: models={len(prepared)}, '
              f'hits={sum(x[2]=="hit" for x in prepared)}, '
              f'misses={len(pending)}, {elapsed:.1f}s', flush=True)
        return {'enabled': True, 'models': len(prepared), 'seconds': round(elapsed, 2),
                'hits': sum(x[2] == 'hit' for x in prepared), 'misses': len(pending)}
