"""Persist/restore MiniMax H3 RefMods through the S3-compatible model bucket (Tigris/R2).

RefMods created by MiniMaxH3RefModExtract live only on the ephemeral worker disk.
persist_refmods() uploads them after a job; ensure_refmods() downloads the ones a
workflow needs (RefModsLoader mod_*) before ComfyUI validates the prompt.
"""
import hashlib
import json
import os
import re
from pathlib import Path

EXTRACT_CLASS = 'MiniMaxH3RefModExtract'
SKIP_VALUES = {'', 'none', 'None', 'null', '-', 'disabled'}
LOADER_FIELD = re.compile(r'^mod_(\d+|a_\d+|b_\d+)$')
DOWNLOAD_DIR = Path(os.getenv('AIVYLA_REFMOD_DOWNLOAD_DIR', '/comfyui/models/refmods'))
SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$')


def _safe_name(name):
    name = str(name).strip()
    if name.endswith('.safetensors'):
        name = name[:-len('.safetensors')]
    if not SAFE.match(name) or '..' in name:
        raise ValueError(f'Unsafe RefMod name: {name!r}')
    return name


def _nodes(workflow):
    if isinstance(workflow, dict):
        for node in workflow.values():
            if isinstance(node, dict):
                yield node


def _extract_targets(workflow):
    out = []
    for node in _nodes(workflow):
        if node.get('class_type') != EXTRACT_CLASS:
            continue
        inp = node.get('inputs') or {}
        if inp.get('save') is False:
            continue
        name = inp.get('name')
        if isinstance(name, str) and name not in SKIP_VALUES:
            sub = str(inp.get('subfolder') or '').strip().strip('/')
            n = _safe_name(name)
            if sub:
                if not all(SAFE.match(p) for p in sub.split('/')):
                    raise ValueError(f'Unsafe RefMod subfolder: {sub!r}')
                n = f'{sub}/{n}'
            out.append(n)
    return out


def _loader_names(workflow):
    produced = set(_extract_targets(workflow))
    out = []
    for node in _nodes(workflow):
        if node.get('class_type') == EXTRACT_CLASS:
            continue
        for k, v in (node.get('inputs') or {}).items():
            if LOADER_FIELD.match(str(k)) and isinstance(v, str) and v not in SKIP_VALUES:
                n = v[:-len('.safetensors')] if v.endswith('.safetensors') else v
                parts = n.split('/')
                if not all(SAFE.match(p) for p in parts) or '..' in n:
                    raise ValueError(f'Unsafe RefMod name: {v!r}')
                if n not in produced and n not in out:
                    out.append(n)
    return out


def _candidate_dirs():
    dirs = [d for d in os.getenv('AIVYLA_REFMOD_DIRS', '').split(os.pathsep) if d]
    dirs.append(str(DOWNLOAD_DIR))
    dirs.append(os.path.join(os.getenv('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'), 'refmods'))
    dirs.append('/runpod-volume/ComfyUI/models/refmods')
    dirs.append('/comfyui/models/refmods')
    seen, res = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            res.append(Path(d))
    return res


def _find_local(rel):
    for d in _candidate_dirs():
        p = d / f'{rel}.safetensors'
        if p.is_file() and p.stat().st_size > 8:
            return p
    return None


def _client_and_layout(client=None, bucket=None, prefix=None):
    if client is None or bucket is None or prefix is None:
        from aivyla_r2_registry import Registry
        reg = Registry(client=client, bucket=bucket, prefix=prefix)
        client, bucket, prefix = reg.client, reg.bucket, reg.prefix
    # Registry prefix is the model prefix (e.g. "models/"); RefMods live beside it.
    root = prefix
    if root.endswith('models/'):
        root = root[:-len('models/')]
    return client, bucket, root


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def persist_refmods(workflow, client=None, bucket=None, prefix=None):
    """Upload every RefMod this workflow saves. Raises on any failure."""
    targets = _extract_targets(workflow)
    if not targets:
        return []
    client, bucket, root = _client_and_layout(client, bucket, prefix)
    saved = []
    for rel in targets:
        path = _find_local(rel)
        if path is None:
            raise FileNotFoundError(f'RefMod {rel!r} was not found on the worker disk after extraction')
        key = f'{root}refmods/{rel}.safetensors'
        size = path.stat().st_size
        digest = _sha256(path)
        client.upload_file(str(path), bucket, key)
        head = client.head_object(Bucket=bucket, Key=key)
        if int(head.get('ContentLength', -1)) != size:
            raise RuntimeError(f'RefMod upload size mismatch for {key}')
        receipt = {'bytes': size, 'etag': str(head.get('ETag', '')).strip('"'), 'sha256': digest}
        client.put_object(Bucket=bucket, Key='_aivyla/receipts/' + key + '.json',
                          Body=json.dumps(receipt).encode(), ContentType='application/json')
        saved.append({'name': rel, 'key': key, **receipt})
    return saved


def ensure_refmods(workflow, client=None, bucket=None, prefix=None):
    """Download RefMods referenced by loaders that are not on local disk."""
    needed = [n for n in _loader_names(workflow) if _find_local(n) is None]
    if not needed:
        return []
    client, bucket, root = _client_and_layout(client, bucket, prefix)
    restored = []
    for rel in needed:
        key = f'{root}refmods/{rel}.safetensors'
        dest = DOWNLOAD_DIR / f'{rel}.safetensors'
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        try:
            client.download_file(bucket, key, str(part))
            os.replace(part, dest)
        except Exception as exc:
            try:
                part.unlink()
            except OSError:
                pass
            raise FileNotFoundError(
                f"RefMod {rel!r} is not on the worker and not in storage ({key}): {exc}") from exc
        restored.append(rel)
    return restored
