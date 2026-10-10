"""Isolated Aivyla Global startup using a verified compressed node bundle.

Keep production /aivyla-start.sh unchanged in its base image; patch only its
container-local copy for this test image. Stage source before that launcher runs.
"""

from pathlib import Path
import os
import subprocess

START = Path('/start.sh')
AIVYLA_START = Path('/aivyla-start.sh')
CACHE = Path(os.environ.get('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'))
LOCAL_NODES = Path(os.environ.get('AIVYLA_GLOBAL_NODE_ROOT', '/tmp/aivyla-node-cache'))
LOCAL_ENABLED = os.environ.get('AIVYLA_GLOBAL_LOCAL_NODES', '1') == '1'


def _require_tmp_directory(p: Path, label: str):
    if not p.is_absolute() or not str(p).startswith('/tmp/') or '\n' in str(p):
        raise RuntimeError(f'{label} must be an absolute path under /tmp')


def patch_start():
    _require_tmp_directory(CACHE, 'AIVYLA_GLOBAL_MODEL_ROOT')
    source = START.read_text()
    anchor = 'echo "worker-comfyui: Starting ComfyUI"'
    if source.count(anchor) != 1 or source.count('python -u /handler.py') != 2:
        raise RuntimeError('Unexpected /start.sh; refusing unsafe Global patch')

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

    source = source.replace(anchor, inject + anchor, 1)
    source = source.replace('python -u /handler.py',
                            'python -u /opt/aivyla-global/aivyla_global_handler.py')
    START.write_text(source)
    subprocess.run(['bash', '-n', str(START)], check=True)


def patch_aivyla_launcher():
    if not LOCAL_ENABLED:
        return
    _require_tmp_directory(LOCAL_NODES, 'AIVYLA_GLOBAL_NODE_ROOT')
    source = AIVYLA_START.read_text()
    old = '"$VOL/ComfyUI/custom_nodes/$node'
    new = '"$AIVYLA_GLOBAL_NODE_ROOT/$node'
    if source.count(old) != 3:
        raise RuntimeError('Unexpected /aivyla-start.sh node mapping; refusing unsafe patch')
    source = source.replace(old, new)
    AIVYLA_START.write_text(source)
    subprocess.run(['bash', '-n', str(AIVYLA_START)], check=True)
    print('Aivyla Global: launcher resolves all 16 nodes from local disk', flush=True)


def main():
    os.environ.setdefault('AIVYLA_GLOBAL_NODE_ROOT', str(LOCAL_NODES))
    patch_start()
    if LOCAL_ENABLED:
        # Verify SHA-256, copy once sequentially, then unpack the archive locally.
        subprocess.run(['python', '-u', '/opt/aivyla-global/aivyla_global_nodes.py'], check=True)
    patch_aivyla_launcher()
    print('Aivyla Global v2: local-node bundle + model cache ready', flush=True)
    os.execv('/bin/bash', ['bash', str(AIVYLA_START)])


if __name__ == '__main__':
    main()
