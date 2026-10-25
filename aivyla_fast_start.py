"""Opt-in Aivyla Global R2 fast-start wrapper. Never patches production source.

Preserves R2 v4.2 handler, SageAttention GPU tests and full pip fallback.
Live previews are enabled only when AIVYLA_LIVE_PREVIEW_ENABLED=true on the
Global Runpod endpoint; otherwise the ComfyUI launcher remains unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time

BUNDLED_ROOT = Path('/opt/aivyla-global/baked-nodes')
BUNDLE_SHA256 = '0f6d600452d04020fbe97886c657eb48ca6fc8ba6a548906193d0e419623c6cc'

sys.path.insert(0, '/opt/aivyla-global')
import aivyla_global_r2_start as r2
from aivyla_global_r2_start import v3

ORIGINAL = 'python -m pip install --no-cache-dir -q -r "$target_dir/requirements.txt"'
FAST = 'python -u /opt/aivyla-global/aivyla_fast_requirements.py "$target_dir/requirements.txt"'


def patch_requirements(path: Path) -> None:
    """Fail closed on upstream launcher drift; idempotent in same container."""
    source = path.read_text()
    if source.count(FAST) == 1 and source.count(ORIGINAL) == 0:
        return
    if source.count(ORIGINAL) != 1 or source.count(FAST) != 0:
        raise RuntimeError('Aivyla startup requirements anchor drift; refuse fast-start patch')
    path.write_text(source.replace(ORIGINAL, FAST, 1))
    subprocess.run(['bash', '-n', str(path)], check=True)


def configure_comfy_preview(path: Path) -> None:
    """Patch only the Global container's runtime /start.sh, not the image source.

    The existing /aivyla-start.sh reads /start.sh to produce its final launcher.
    Its SageAttention/dynamic-VRAM logic preserves the extra preview arguments.
    """
    if os.getenv('AIVYLA_LIVE_PREVIEW_ENABLED', 'false').lower() != 'true':
        return

    source = path.read_text()
    pattern = re.compile(
        r'(?m)^(?P<prefix>[ \t]*python[ \t]+-u[ \t]+/comfyui/main\.py)(?P<args>[^\n]*)$'
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 2:
        raise RuntimeError('Global preview: expected exactly two ComfyUI launcher commands')

    flags = '--preview-method auto --preview-size 256'

    def patch(match):
        prefix, args = match.group('prefix'), match.group('args')
        if flags in args:
            return prefix + args  # Already enabled: idempotent.
        if '--preview-method' in args or '--preview-size' in args:
            raise RuntimeError('Global preview: conflicting ComfyUI preview flags')
        return prefix + ' ' + flags + args

    updated = pattern.sub(patch, source)
    # Validate before writing anything; this only changes the container's
    # launcher and never touches the network volume or repository.
    subprocess.run(['bash', '-n', '-'], input=updated, text=True, check=True)
    if updated != source:
        path.write_text(updated)
    print('Aivyla Global FastStart: low-resolution ComfyUI previews enabled', flush=True)


def baked_nodes_available(root: Path, nodes_file: Path) -> bool:
    """Allow baked nodes only when exact volume node list still matches.

    If user adds/removes a node, use the original dynamic volume staging.
    """
    if os.getenv('AIVYLA_FAST_BAKED_NODES', '0') != '1':
        return False
    try:
        if (root / '.aivyla-node-bundle-sha256').read_text().strip() != BUNDLE_SHA256:
            return False
        selected = [line.strip() for line in nodes_file.read_text().splitlines()
                    if line.strip() and not line.strip().startswith('#')]
        actual = sorted(p.name for p in root.iterdir()
                        if p.is_dir() and (p / '__init__.py').is_file())
        return len(selected) == 16 and sorted(selected) == actual
    except (OSError, UnicodeError):
        return False


def patch_baked_launcher(path: Path) -> None:
    """Use verified image-local node tree, including existence checks."""
    original = '"$VOL/ComfyUI/custom_nodes/$node'
    patched = '"$AIVYLA_GLOBAL_NODE_ROOT/$node'
    source = path.read_text()
    if source.count(patched) == 3 and source.count(original) == 0:
        return
    if source.count(original) != 3 or source.count(patched) != 0:
        raise RuntimeError('Aivyla baked launcher anchor drift')
    path.write_text(source.replace(original, patched))
    subprocess.run(['bash', '-n', str(path)], check=True)


def phase(label: str, start: float) -> None:
    print(f'Aivyla FASTBOOT {label}: {time.monotonic() - start:.3f}s', flush=True)


def main() -> None:
    total = time.monotonic()
    r2.validate_settings()
    phase('settings', total)

    started = time.monotonic()
    r2.prepare_handler()
    phase('handler', started)

    started = time.monotonic()
    baked = baked_nodes_available(BUNDLED_ROOT, Path('/runpod-volume/aivyla/nodes.txt'))
    if baked:
        print('Aivyla FASTBOOT: using SHA256-pinned baked node bundle', flush=True)
    elif v3.LOCAL_ENABLED:
        node_script = ('/opt/aivyla-global/aivyla_fast_node_staging.py'
                       if os.getenv('AIVYLA_FAST_NODE_STREAM', '0') == '1'
                       else '/opt/aivyla-global/aivyla_global_nodes.py')
        subprocess.run([sys.executable, '-u', node_script], check=True)
    phase('node_staging', started)

    started = time.monotonic()
    if baked:
        os.environ['AIVYLA_GLOBAL_NODE_ROOT'] = str(BUNDLED_ROOT)
        patch_baked_launcher(v3.AIVYLA_START)
    else:
        v3.patch_aivyla_launcher()
    if os.getenv('AIVYLA_R2_NODES_ENABLED', '0') == '1':
        from aivyla_r2_nodes import install_enabled_nodes
        install_enabled_nodes()
    patch_requirements(v3.AIVYLA_START)
    configure_comfy_preview(v3.START)
    phase('launcher_patch', started)
    phase('bootstrap_total_before_comfy', total)
    os.execv('/bin/bash', ['bash', str(v3.AIVYLA_START)])


if __name__ == '__main__':
    main()
