#!/usr/bin/env python3
"""Fail-closed, read-only export of original Tigris model SHA256 receipts.

Read from ORIGINAL `aivyla-models-global` (not the newer USA bucket suffering
EOF on tiny receipts). Output manifest remains bound to `aivyla-models-usa`.
Never reads or writes full model blobs; never writes to Tigris.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HEX = re.compile(r"[0-9a-f]{64}\Z")
SUFFIXES = ('.safetensors', '.sft', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx')


def export(output: Path, source_bucket: str = 'aivyla-models-global',
           target_bucket: str = 'aivyla-models-usa', min_records: int = 90):
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', source_bucket):
        raise ValueError('Invalid source bucket')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', target_bucket):
        raise ValueError('Invalid target bucket')
    if source_bucket == target_bucket:
        raise ValueError('Source bucket must differ from target; refusing broken USA receipt source')
    source = f'tigris:{source_bucket}/_aivyla/receipts/models'
    with tempfile.TemporaryDirectory(prefix='aivyla-trusted-original-') as tmp:
        # The old script used 12 simultaneous receipt GETs against a Tigris
        # bucket that currently returns EOF. Reduce fanout and use the original.
        cmd = ['rclone', 'copy', source, tmp,
               '--transfers', '2', '--checkers', '2',
               '--retries', '5', '--low-level-retries', '8',
               '--exclude', '*__aivyla_read_probe_20261022.json',
               '--stats', '0']
        print('Reading original trusted receipts from:', source, flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except FileNotFoundError as exc:
            raise RuntimeError('rclone is not installed on this server') from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Source receipt read timed out; NO manifest written') from exc
        if result.returncode:
            raise RuntimeError('Source receipts could not be read. NO manifest written.\n'
                               + result.stderr[-2400:])
        models: dict[str, dict] = {}
        invalid = []
        base = Path(tmp)
        for path in sorted(base.rglob('*.json')):
            rel = path.relative_to(base).as_posix()
            if rel.endswith('__aivyla_read_probe_20261022.json'):
                continue
            key = 'models/' + rel[:-5]
            if ('..' in key.split('/') or '\\' in key or not key.endswith(SUFFIXES)):
                invalid.append((rel, 'unsafe/non-model key'))
                continue
            try:
                receipt = json.loads(path.read_bytes())
                if not isinstance(receipt, dict):
                    raise ValueError('not an object')
                size, digest = receipt.get('bytes'), receipt.get('sha256')
                if type(size) is not int or size < 8 or not isinstance(digest, str) or not HEX.fullmatch(digest):
                    raise ValueError('invalid bytes or SHA256')
                if key in models:
                    raise ValueError('duplicate key')
                models[key] = {'bytes': size, 'sha256': digest}
            except (ValueError, TypeError, OSError, UnicodeError) as exc:
                invalid.append((rel, str(exc)[:160]))
        if invalid or len(models) < min_records:
            raise RuntimeError(f'ABORT: {len(invalid)} invalid receipts; '
                               f'{len(models)} valid (minimum {min_records}); '
                               f'examples: {invalid[:5]}. NO manifest written.')

    document = {'schema': 1, 'bucket': target_bucket, 'models': models}
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                         prefix='.aivyla-trusted-', suffix='.tmp', delete=False) as fp:
            scratch = Path(fp.name)
            json.dump(document, fp, sort_keys=True, separators=(',', ':'))
            fp.write('\n')
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(scratch, output)
    except BaseException:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
        raise
    print(f'PASS: pinned {len(models)} ORIGINAL SHA256 receipts for TARGET {target_bucket}', flush=True)
    print(f'CREATED: {output}', flush=True)
    print('The worker will still hash full USA model downloads and fail on mismatch.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/root/aivyla_trusted_receipts.json'))
    parser.add_argument('--source-bucket', default='aivyla-models-global')
    parser.add_argument('--target-bucket', default='aivyla-models-usa')
    parser.add_argument('--min-records', type=int, default=90)
    args = parser.parse_args()
    export(args.output, args.source_bucket, args.target_bucket, args.min_records)
