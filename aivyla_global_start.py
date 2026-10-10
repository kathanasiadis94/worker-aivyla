"""Isolated Aivyla Global bootstrap (v3).

Only runtime copies of the two launch scripts are patched, never production.
Both patches are idempotent: the same container can restart without crashing.
"""

from pathlib import Path
import os
import subprocess

START = Path('/start.sh')
AIVYLA_START = Path('/aivyla-start.sh')
CACHE = Path(os.environ.get('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'))
LOCAL_NODES = Path(os.environ.get('AIVYLA_GLOBAL_NODE_ROOT', '/tmp/aivyla-node-cache'))
LOCAL_ENABLED = os.environ.get('AIVYLA_GLOBAL_LOCAL_NODES', '1') == '1'

ANCHOR = 'echo "worker-comfyui: Starting ComfyUI"'
BASE_HANDLER = 'python -u /handler.py'
GLOBAL_HANDLER = 'python -u /opt/aivyla-global/aivyla_global_handler.py'
LOCAL_CACHE_MARKER = 'aivyla_global_local_cache:'


def _require_tmp_directory(p: Path, label: str):
    if (not p.is_absolute() or not str(p).startswith('/tmp/') or
            '\n' in str(p) or '\r' in str(p) or p == Path('/tmp')):
        raise RuntimeError(f'{label} must be an absolute directory below /tmp')


def patch_start():
    _require_tmp_directory(CACHE, 'AIVYLA_GLOBAL_MODEL_ROOT')
    source = START.read_text()
    if source.count(ANCHOR) != 1:
        raise RuntimeError('Unexpected /start.sh: ComfyUI anchor mismatch')

    original = source.count(BASE_HANDLER)
    patched = source.count(GLOBAL_HANDLER)
    cache_blocks = source.count(LOCAL_CACHE_MARKER)

    if (original, patched, cache_blocks) == (0, 2, 1):
        # A previous process in this same container already patched /start.sh.
        if f'  base_path: {CACHE}\n' not in source:
            raise RuntimeError('Previously patched /start.sh uses a different model-cache path')
        subprocess.run(['bash', '-n', str(START)], check=True)
        print('Aivyla Global: /start.sh already patched; keeping it', flush=True)
        return
    if (original, patched, cache_blocks) != (2, 0, 0):
        raise RuntimeError(
            f'Unexpected /start.sh state: base={original}, global={patched}, '
            f'cache={cache_blocks}; refusing partial/double patch')

    inject = '''cat >> /comfyui/extra_model_paths.yaml <<'AIVYLAGLOBALYAML'
aivyla_global_local_cache:
  is_default: true
  base_path: /tmp/aivyla-model-cache
  checkpoints: checkpoints
  clip: text_encoders
  text_encoders: text_encoders
  clip_vision: clip_vision
  diffusion_models: diffusion_models
  unet: diffusion_models
  vae: vae
  loras: loras
  upscale_models: upscale_models
  latent_upscale_models: latent_upscale_models
  embeddings: embeddings
  controlnet: controlnet
  ipadapter: ipadapter
  model_patches: model_patches
  audio_encoders: audio_encoders
  refmods: refmods
  h3_adaln: h3_adaln
AIVYLAGLOBALYAML
'''.replace('  base_path: /tmp/aivyla-model-cache', f'  base_path: {CACHE}')

    updated = source.replace(ANCHOR, inject + ANCHOR, 1)
    updated = updated.replace(BASE_HANDLER, GLOBAL_HANDLER)
    START.write_text(updated)
    subprocess.run(['bash', '-n', str(START)], check=True)
    print('Aivyla Global: /start.sh patched once', flush=True)


def patch_aivyla_launcher():
    if not LOCAL_ENABLED:
        return
    _require_tmp_directory(LOCAL_NODES, 'AIVYLA_GLOBAL_NODE_ROOT')
    source = AIVYLA_START.read_text()
    old = '"$VOL/ComfyUI/custom_nodes/$node'
    new = '"$AIVYLA_GLOBAL_NODE_ROOT/$node'
    original = source.count(old)
    patched = source.count(new)

    if (original, patched) == (0, 3):
        subprocess.run(['bash', '-n', str(AIVYLA_START)], check=True)
        print('Aivyla Global: node launcher already patched; keeping it', flush=True)
        return
    if (original, patched) != (3, 0):
        raise RuntimeError(
            f'Unexpected /aivyla-start.sh state: base={original}, local={patched}')

    AIVYLA_START.write_text(source.replace(old, new))
    subprocess.run(['bash', '-n', str(AIVYLA_START)], check=True)
    print('Aivyla Global: launcher resolves all 16 nodes from local disk', flush=True)


def main():
    os.environ.setdefault('AIVYLA_GLOBAL_NODE_ROOT', str(LOCAL_NODES))
    patch_start()
    if LOCAL_ENABLED:
        subprocess.run(['python', '-u', '/opt/aivyla-global/aivyla_global_nodes.py'], check=True)
    patch_aivyla_launcher()
    print('Aivyla Global v3: local-node bundle + model cache ready', flush=True)
    os.execv('/bin/bash', ['bash', str(AIVYLA_START)])


if __name__ == '__main__':
    main()
