"""Copy node Python/code to local container disk while preserving heavy assets.

Never mutate /runpod-volume. Keep model weights and massive assets as symlinks
until a workflow requires them; only Python/code is eagerly staged.
"""
import os
import shutil
import stat
import sys
import time
from pathlib import Path

BASE = Path('/comfyui/custom_nodes')
VOLUME = Path('/runpod-volume/ComfyUI/custom_nodes')
LOCAL = Path(os.environ.get('AIVYLA_GLOBAL_NODE_ROOT', '/tmp/aivyla-node-cache'))
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules', 'venv', '.venv'}
ASSET_EXTS = {'.safetensors', '.pt', '.pth', '.ckpt', '.gguf', '.bin', '.onnx',
              '.mp4', '.mov', '.webm', '.wav', '.m4a', '.mp3', '.npy', '.npz', '.zip'}
ASSET_THRESHOLD = 4 * 1024 * 1024  # 4 MiB


def stage_one(name):
    source = VOLUME / name
    link = BASE / name
    if not source.is_dir() or not (source / '__init__.py').is_file():
        raise RuntimeError(f'Missing expected custom node {source}')
    if not link.is_symlink():
        raise RuntimeError(f'Expected custom node symlink for {name}: {link}')
    dest = LOCAL / name
    if dest.exists():
        shutil.rmtree(dest)
    copied = 0
    linked = 0
    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        rel = root_path.relative_to(source)
        out = dest / rel
        out.mkdir(parents=True, exist_ok=True)
        keep_dirs = []
        for d in dirs:
            p = root_path / d
            if d in SKIP_DIRS:
                continue
            if p.is_symlink():
                # Preserve linked subtrees without traversing remote model trees.
                (out / d).symlink_to(p.resolve(), target_is_directory=True)
                linked += 1
            else:
                keep_dirs.append(d)
        dirs[:] = keep_dirs
        for f in files:
            src = root_path / f
            dst = out / f
            try:
                size = src.stat().st_size
                if src.is_symlink() or (src.suffix.lower() in ASSET_EXTS and size > ASSET_THRESHOLD):
                    dst.symlink_to(src.resolve())
                    linked += 1
                else:
                    shutil.copy2(src, dst)
                    copied += 1
            except OSError as e:
                raise RuntimeError(f'Failed copying node {name}: {src}: {e}') from e
    if not (dest / '__init__.py').is_file():
        raise RuntimeError(f'Incomplete local custom node: {name}')
    link.unlink()
    link.symlink_to(dest, target_is_directory=True)
    return copied, linked


def main():
    if os.environ.get('AIVYLA_GLOBAL_LOCAL_NODES', '1') != '1':
        print('Aivyla Global: local node staging disabled')
        return
    if not VOLUME.is_dir():
        raise RuntimeError('Global volume custom-node directory missing')
    if not LOCAL.is_absolute() or not str(LOCAL).startswith('/tmp/'):
        raise RuntimeError('Local node cache must be under /tmp')
    LOCAL.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    staged = 0
    # Only nodes mounted from the volume (avoid touching built-in/image nodes).
    for node in sorted(BASE.iterdir()):
        if not node.is_symlink():
            continue
        if not (VOLUME / node.name).is_dir():
            continue
        files, syms = stage_one(node.name)
        print(f'Aivyla Global local node: {node.name}, copied={files}, linked_assets={syms}', flush=True)
        staged += 1
    print(f'Aivyla Global nodes staged: {staged} in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Aivyla Global NODE STAGING FAILED: {e}', file=sys.stderr, flush=True)
        sys.exit(1)
