"""Isolated Aivyla Global bootstrap; preserve base /aivyla-start.sh behavior.

The original production launcher generates /comfyui/extra_model_paths.yaml
and patches /start.sh at runtime.  We modify only the *global container copy*
of /start.sh first, so the same launcher can perform all GPU/Sage/node checks.
"""
from pathlib import Path
import os
import subprocess
import sys

START = Path('/start.sh')
PYDIR = Path('/opt/aivyla-global')
CACHE = Path(os.environ.get('AIVYLA_GLOBAL_MODEL_ROOT', '/tmp/aivyla-model-cache'))

# Python module cache + serving use the same directory; no secrets, no remote writes.
def patch_start():
    source = START.read_text()
    anchor = 'echo "worker-comfyui: Starting ComfyUI"'
    if source.count(anchor) != 1:
        raise RuntimeError('Unexpected /start.sh: cannot safely insert Global bootstrap')
    if source.count('python -u /handler.py') != 2:
        raise RuntimeError('Unexpected /start.sh: handler invocation count mismatch')

    # At this point the volume custom nodes have been installed and linked by
    # /aivyla-start.sh. Replace only their code symlinks with local code copies.
    # All heavy weights stay under /runpod-volume, until a job pre-stages them.
    inject = '''python -u /opt/aivyla-global/aivyla_global_nodes.py || exit 1
cat >> /comfyui/extra_model_paths.yaml <<'AIVYLAGLOBALYAML'
aivyla_global_local_cache:
  is_default: true
  base_path: ${AIVYLA_GLOBAL_MODEL_ROOT:-/tmp/aivyla-model-cache}
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
'''
    # ${...} inside quoted heredoc stays literal; create actual path from env
    # by replacing it before writing, with YAML-safe path validation.
    path = str(CACHE)
    if not CACHE.is_absolute() or not path.startswith('/tmp/') or '\n' in path:
        raise RuntimeError('AIVYLA_GLOBAL_MODEL_ROOT must be an absolute /tmp path')
    inject = inject.replace('${AIVYLA_GLOBAL_MODEL_ROOT:-/tmp/aivyla-model-cache}', path)
    source = source.replace(anchor, inject + anchor, 1)
    source = source.replace('python -u /handler.py',
                            'python -u /opt/aivyla-global/aivyla_global_handler.py')
    START.write_text(source)
    subprocess.run(['bash', '-n', str(START)], check=True)
    print('Aivyla Global: separate worker enabled; local model paths take priority', flush=True)


if __name__ == '__main__':
    patch_start()
    os.execv('/bin/bash', ['bash', '/aivyla-start.sh'])
