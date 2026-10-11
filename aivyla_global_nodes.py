"""Stage the verified Aivyla Global custom-node bundle onto worker-local disk.

The Global Volume is used for one sequential archive read, not thousands of
individual small-file reads. This module never writes to the Global Volume.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time

BUNDLE = Path('/runpod-volume/aivyla/custom_nodes_bundle_v1.tar.gz')
NODES_FILE = Path('/runpod-volume/aivyla/nodes.txt')
EXPECTED_SHA256 = '0f6d600452d04020fbe97886c657eb48ca6fc8ba6a548906193d0e419623c6cc'
LOCAL = Path(os.environ.get('AIVYLA_GLOBAL_NODE_ROOT', '/tmp/aivyla-node-cache'))
CHUNK_BYTES = 8 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
READY_MARKER = '.aivyla-node-bundle-sha256'


def _expected_nodes():
    nodes = []
    for line in NODES_FILE.read_text().splitlines():
        name = line.strip()
        if not name or name.startswith('#'):
            continue
        if name in ('.', '..') or re.fullmatch(r'[A-Za-z0-9._-]+', name) is None:
            raise ValueError(f'Unsafe custom-node name in nodes.txt: {name!r}')
        if name in nodes:
            raise ValueError(f'Duplicate custom-node name in nodes.txt: {name}')
        nodes.append(name)
    if len(nodes) != 16:
        raise ValueError(f'Expected 16 Aivyla Global custom nodes; found {len(nodes)}')
    return set(nodes)


def _check_root():
    if (not LOCAL.is_absolute() or not str(LOCAL).startswith('/tmp/') or
            LOCAL == Path('/tmp') or LOCAL.is_symlink()):
        raise ValueError('AIVYLA_GLOBAL_NODE_ROOT must be a non-symlink directory below /tmp')
    if not BUNDLE.is_file():
        raise FileNotFoundError(f'Global custom-node bundle missing: {BUNDLE}')


def _already_staged(expected):
    if not LOCAL.exists():
        return False
    marker = LOCAL / READY_MARKER
    if not LOCAL.is_dir() or LOCAL.is_symlink():
        raise RuntimeError(f'Unexpected local node cache type: {LOCAL}')
    if not marker.is_file() or marker.read_text().strip() != EXPECTED_SHA256:
        # v2 left no marker; delete only this isolated /tmp cache and restage.
        print('Aivyla Global: obsolete/incomplete local node cache; restaging', flush=True)
        shutil.rmtree(LOCAL)
        return False
    missing = sorted(name for name in expected if not (LOCAL / name / '__init__.py').is_file())
    if missing:
        print(f'Aivyla Global: incomplete local node cache ({missing}); restaging', flush=True)
        shutil.rmtree(LOCAL)
        return False
    print(f'Aivyla Global nodes already staged: {len(expected)}/{len(expected)} (local cache hit)', flush=True)
    return True


def _stage(expected):
    started = time.monotonic()
    free = shutil.disk_usage('/tmp').free
    # The archive is ~446MB and expands to ~604MB. Leave headroom for workers.
    if free < MIN_FREE_BYTES:
        raise OSError(f'Not enough local disk for nodes: {free / 2**30:.2f} GiB free')
    with tempfile.TemporaryDirectory(prefix='aivyla-node-stage-', dir='/tmp') as td:
        temp = Path(td)
        local_archive = temp / 'nodes.tar.gz'
        digest = hashlib.sha256()
        size = 0
        with BUNDLE.open('rb') as source, local_archive.open('wb') as sink:
            while block := source.read(CHUNK_BYTES):
                digest.update(block)
                sink.write(block)
                size += len(block)
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise ValueError(f'Global node bundle SHA-256 mismatch: {actual}')
        print(f'Aivyla Global bundle verified: {size / 2**20:.1f} MiB, SHA-256 OK', flush=True)

        unpack = temp / 'unpacked'
        unpack.mkdir()
        count = 0
        with tarfile.open(local_archive, mode='r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                p = PurePosixPath(member.name)
                if (p.is_absolute() or not p.parts or
                        any(part in ('..', '.') for part in p.parts) or
                        p.parts[0] not in expected or
                        not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unsafe or unexpected node archive entry: {member.name!r}')
                if member.isfile():
                    count += 1
            # The archive is pinned by SHA256, and entries above were verified
            # to be regular files/directories within expected node directories.
            # Manual extraction remains safe on Python 3.9 (no `filter` API).
            for member in members:
                target = unpack.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    reader = archive.extractfile(member)
                    if reader is None:
                        raise ValueError(f'Cannot read bundled node file: {member.name!r}')
                    with reader, target.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, length=1024*1024)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)

        missing = sorted(name for name in expected if not (unpack / name / '__init__.py').is_file())
        if missing:
            raise RuntimeError(f'Incomplete node bundle; missing __init__.py: {missing}')
        if LOCAL.exists():
            raise FileExistsError(f'Local node directory unexpectedly exists: {LOCAL}')
        (unpack / READY_MARKER).write_text(EXPECTED_SHA256 + '\n')
        unpack.rename(LOCAL)
        print(f'Aivyla Global nodes staged locally: {len(expected)}/{len(expected)}, '
              f'{count} files, {time.monotonic() - started:.1f}s', flush=True)


def main():
    if os.environ.get('AIVYLA_GLOBAL_LOCAL_NODES', '1') != '1':
        print('Aivyla Global: local node staging disabled; using original volume nodes', flush=True)
        return
    _check_root()
    expected = _expected_nodes()
    if not _already_staged(expected):
        _stage(expected)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Aivyla Global NODE BUNDLE FAILED: {type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        sys.exit(1)
