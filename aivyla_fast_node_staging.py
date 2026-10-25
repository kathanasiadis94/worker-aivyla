"""Opt-in single-pass, SHA-256-verified Aivyla node staging.

Only the FastStart candidate invokes this. Original v3 node staging remains
available with AIVYLA_FAST_NODE_STREAM=0. Never modifies the Global Volume.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time

import aivyla_global_nodes as original

# The source volume is treated as untrusted even though the expected archive
# SHA-256 is pinned. Nothing extracted is activated until validation passes.
class HashingStream:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.size = 0

    def read(self, length=-1):
        data = self.stream.read(length)
        self.sha.update(data)
        self.size += len(data)
        return data


def stage_stream(expected):
    started = time.monotonic()
    free = shutil.disk_usage('/tmp').free
    if free < original.MIN_FREE_BYTES:
        raise OSError(f'Not enough local disk for nodes: {free / 2**30:.2f} GiB free')

    with tempfile.TemporaryDirectory(prefix='aivyla-node-stage-', dir='/tmp') as td:
        unpack = Path(td) / 'unpacked'
        unpack.mkdir()
        count = 0
        total_uncompressed = 0
        # In streaming mode tarfile never seeks backwards and never needs a
        # second copy of the 446 MiB archive on local disk.
        with original.BUNDLE.open('rb', buffering=original.CHUNK_BYTES) as raw:
            source = HashingStream(raw)
            with tarfile.open(fileobj=source, mode='r|gz',
                              bufsize=original.CHUNK_BYTES) as archive:
                seen = set()
                for member in archive:
                    p = PurePosixPath(member.name)
                    if (not p.parts or p.is_absolute() or
                            any(part in ('..', '.') for part in p.parts) or
                            p.parts[0] not in expected or
                            not (member.isfile() or member.isdir()) or
                            member.name in seen):
                        raise ValueError(f'Unsafe or unexpected node archive entry: {member.name!r}')
                    seen.add(member.name)
                    if len(seen) > 100000:
                        raise ValueError('Node archive has too many entries')
                    dest = unpack.joinpath(*p.parts)
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                        continue
                    count += 1
                    total_uncompressed += member.size
                    if member.size < 0 or total_uncompressed > 4 * 2**30:
                        raise ValueError('Node archive exceeds extraction limit')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    reader = archive.extractfile(member)
                    if reader is None:
                        raise ValueError(f'Cannot read bundled node file: {member.name!r}')
                    # 'xb': do not overwrite any existing file, including files
                    # shadowing directory members.
                    with reader, dest.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    dest.chmod(0o755 if member.mode & 0o111 else 0o644)
            # tarfile may stop at its end-of-archive record before EOF. Hash
            # all trailing bytes too, matching the original full-file digest.
            while source.read(original.CHUNK_BYTES):
                pass
            actual = source.sha.hexdigest()
            if actual != original.EXPECTED_SHA256:
                raise ValueError(f'Global node bundle SHA-256 mismatch: {actual}')
            print(f'Aivyla Global bundle verified: {source.size / 2**20:.1f} MiB, SHA-256 OK', flush=True)

        missing = sorted(name for name in expected if not (unpack / name / '__init__.py').is_file())
        if missing:
            raise RuntimeError(f'Incomplete node bundle; missing __init__.py: {missing}')
        if original.LOCAL.exists():
            raise FileExistsError(f'Local node directory unexpectedly exists: {original.LOCAL}')
        (unpack / original.READY_MARKER).write_text(original.EXPECTED_SHA256 + '\n')
        unpack.rename(original.LOCAL)
        print(f'Aivyla Global nodes staged locally: {len(expected)}/{len(expected)}, '
              f'{count} files, {time.monotonic() - started:.1f}s (single-pass)', flush=True)


def main():
    if os.getenv('AIVYLA_GLOBAL_LOCAL_NODES', '1') != '1':
        return original.main()
    if os.getenv('AIVYLA_FAST_NODE_STREAM', '0') != '1':
        print('Aivyla FASTBOOT: original node staging selected', flush=True)
        return original.main()
    original._check_root()
    expected = original._expected_nodes()
    if not original._already_staged(expected):
        stage_stream(expected)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Aivyla Global NODE BUNDLE FAILED: {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        sys.exit(1)
