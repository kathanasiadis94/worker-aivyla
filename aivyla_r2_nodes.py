"""Install explicitly enabled R2 custom-node tar.gz bundles on worker startup.

A new R2 upload is visible in inventory. Executable node code is NOT enabled
until nodes/registry.json includes enabled:true and a pinned SHA256.
Requires a new worker/restart for ComfyUI to discover new Python node classes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from aivyla_r2_registry import Registry

LOCAL_NODES = Path(os.getenv('AIVYLA_GLOBAL_NODE_ROOT', '/tmp/aivyla-node-cache'))
COMFY_NODES = Path('/comfyui/custom_nodes')
MAX_ARCHIVE_GIB = float(os.getenv('AIVYLA_R2_NODE_MAX_ARCHIVE_GIB', '2'))
ALLOW_PIP = os.getenv('AIVYLA_R2_NODE_PIP', '0') == '1'


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _inspect(archive, node_name):
    # Guard against tar traversal, links, devices and decompression bombs.
    total = 0
    with tarfile.open(archive, 'r:gz') as tf:
        seen_files = set()
        members = tf.getmembers()
        if len(members) > 25000:
            raise ValueError('Too many node archive entries')
        for member in members:
            name = member.name
            p = Path(name)
            if (p.is_absolute() or '..' in p.parts or not p.parts
                    or '\\' in name or '\x00' in name or len(name) > 1024
                    or p.parts[0] != node_name or not (member.isfile() or member.isdir())):
                raise ValueError(f'Unsafe node archive entry: {name!r}')
            if member.isfile():
                canonical = '/'.join(p.parts)
                if canonical in seen_files:
                    raise ValueError(f'Duplicate node archive file: {name!r}')
                seen_files.add(canonical)
            if member.isfile():
                total += member.size
                if total > 2*1024**3:
                    raise ValueError('Node uncompressed size exceeds 2 GiB')
        if not any(m.name == f'{node_name}/__init__.py' and m.isfile() for m in tf.getmembers()):
            raise ValueError(f'{node_name}/__init__.py missing')


def _install_requirements(node_dir):
    req = node_dir / 'requirements.txt'
    if not req.exists():
        return
    if not ALLOW_PIP:
        # Existing dependencies may already be present, but do not silently
        # enable a node whose declared requirements were never checked.
        raise RuntimeError(f'R2 node {node_dir.name} requires dependency approval; '
                           'set AIVYLA_R2_NODE_PIP=1 after reviewing requirements')
    # Limit requirements input to normal package specs, never remote scripts or URLs.
    lines = []
    for line in req.read_text().splitlines():
        item = line.strip()
        if not item or item.startswith('#'):
            continue
        if (item.startswith('-') or '://' in item or '/' in item or '\\' in item
                or not re.fullmatch(r'[A-Za-z0-9_.\[\],<>!=~ +;\-]+', item)):
            raise ValueError(f'Untrusted requirements expression: {item!r}')
        lines.append(item)
    if lines:
        subprocess.run(['python', '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--no-cache-dir', '-r', str(req)], check=True, timeout=300)


def install_enabled_nodes(registry=None, *, local=None, comfy=None):
    registry = registry or Registry()
    local = Path(local) if local is not None else LOCAL_NODES
    comfy = Path(comfy) if comfy is not None else COMFY_NODES
    if not local.is_dir() or not str(local).startswith('/tmp/'):
        raise RuntimeError('Custom node local root missing or unsafe')
    if not comfy.is_dir():
        raise RuntimeError('ComfyUI custom_nodes directory missing')
    all_nodes = registry.nodes()
    existing = set(x.name for x in local.iterdir() if x.is_dir())
    enabled = [x for x in all_nodes if x['enabled']]
    installed = []
    for node in enabled:
        name, key, expected = node['name'], node['key'], node['sha256']
        if name in existing or (comfy/name).exists() or (comfy/name).is_symlink():
            raise FileExistsError(f'R2 node collides with existing ComfyUI node: {name}')
        head = registry.client.head_object(Bucket=registry.bucket, Key=key)
        if head['ContentLength'] > MAX_ARCHIVE_GIB*1024**3 or head['ContentLength'] <= 0:
            raise ValueError(f'R2 node archive size invalid: {name}')
        with tempfile.TemporaryDirectory(prefix='aivyla-r2-node-', dir='/tmp') as work:
            src = Path(work)/'bundle.tar.gz'
            extract = Path(work)/'extract'
            extract.mkdir()
            registry.client.download_file(registry.bucket, key, str(src))
            if _hash(src) != expected:
                raise ValueError(f'Custom node SHA256 mismatch: {name}')
            _inspect(src, name)
            # tarfile.extractall(filter='data') requires a newer Python.
            # Extract only previously validated regular files and directories;
            # never restore links, special files, owners or unsafe permissions.
            with tarfile.open(src, 'r:gz') as tf:
                for member in tf.getmembers():
                    dest = extract.joinpath(*Path(member.name).parts)
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        reader = tf.extractfile(member)
                        if reader is None:
                            raise ValueError('Unable to read node archive member')
                        with reader, dest.open('xb') as writer:
                            shutil.copyfileobj(reader, writer, length=1024*1024)
                        dest.chmod(0o755 if member.mode & 0o111 else 0o644)
            target = local/name
            if target.exists():
                raise FileExistsError(f'Node target unexpectedly exists: {name}')
            shutil.move(str(extract/name), str(target))
        _install_requirements(target)
        (comfy/name).symlink_to(target, target_is_directory=True)
        installed.append(name)
        print(f'Aivyla R2 node installed: {name} ({node["version"]}) SHA256 PASS', flush=True)
    print(f'Aivyla R2 nodes: enabled={len(enabled)}, staged={len(installed)}, '
          f'catalog={len(all_nodes)}', flush=True)
    return {'listed':len(all_nodes), 'enabled':len(enabled), 'installed':installed}
