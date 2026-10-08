#!/usr/bin/env bash
# Aivyla worker start. Packages are preinstalled in the image; this wires the network volume.
set -Eeuo pipefail
trap 'echo "Aivyla startup failed at line $LINENO" >&2' ERR

VOL=/runpod-volume
PROD=0
[[ -d "$VOL/ComfyUI/custom_nodes" ]] && PROD=1
if [[ "$PROD" != 1 ]]; then
  echo "Aivyla: no network volume, starting in build-test mode (no custom nodes, no Sage)"
fi

# Optional handler diagnostics from the network volume
if [[ -s "$VOL/aivyla/handler-diagnostics.py" ]]; then
  python -m py_compile "$VOL/aivyla/handler-diagnostics.py"
  cp /handler.py /tmp/handler-before-diagnostics.py
  cp "$VOL/aivyla/handler-diagnostics.py" /handler.py
  echo "Aivyla handler diagnostics v1 installed"
fi

if [[ "$PROD" == 1 ]]; then
python - <<'PYGPU'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Aivyla: CUDA GPU unavailable")
p = torch.cuda.get_device_properties(0)
print(f"Aivyla GPU: {p.name}; VRAM={p.total_memory / 2**30:.2f} GiB; capability={p.major}.{p.minor}")
if p.total_memory < 90 * 2**30 or (p.major, p.minor) != (12, 0):
    raise SystemExit("Aivyla requires Blackwell with at least 90 GiB VRAM. Check endpoint GPU selection.")
PYGPU
fi

cat > /comfyui/extra_model_paths.yaml <<'YAML'
runpod_volume_wan:
  base_path: /runpod-volume/split_files
  checkpoints: checkpoints
  clip: text_encoders
  clip_vision: clip_vision
  diffusion_models: diffusion_models
  unet: diffusion_models
  vae: vae
  loras: loras
  upscale_models: upscale_models
  latent_upscale_models: upscale_models
  embeddings: embeddings
  controlnet: controlnet
  ipadapter: ipadapter
runpod_volume_h3:
  base_path: /runpod-volume/ComfyUI/models
  checkpoints: checkpoints
  diffusion_models: diffusion_models
  unet: diffusion_models
  text_encoders: text_encoders
  clip: text_encoders
  vae: vae
  loras: loras
  model_patches: model_patches
  audio_encoders: audio_encoders
  refmods: refmods
  h3_adaln: h3_adaln
  latent_upscale_models: latent_upscale_models
YAML

grep -qx "a8686f2b33fc540f137df50c0f0719953830a5e7" /comfyui/.aivyla-core-commit
grep -q 'class MiniMaxH3FunControlNetApply' /comfyui/comfy_extras/nodes_minimax_h3.py
echo "Aivyla: ComfyUI core a8686f2b (image)"

SAGE=0
if [[ "$PROD" == 1 ]]; then
  # SageAttention 2.2.0 wheel, built once by the old start script and kept on the volume
  WHEEL_DIR=""
  for d in $(ls -1dt "$VOL"/aivyla/sage-builds/*/ 2>/dev/null); do
    if [[ -f "$d/READY.sha256" ]] && (cd "$d" && sha256sum -c --status READY.sha256); then WHEEL_DIR="$d"; break; fi
  done
  if [[ -n "$WHEEL_DIR" ]]; then
    python -m pip install --no-cache-dir --no-deps --force-reinstall "$WHEEL_DIR"/sageattention-2.2.0-*.whl
python - <<'PYSAGECHECK'
import sys, logging
sys.path.insert(0, '/comfyui')
sys.argv = ['aivyla-sage-check', '--use-sage-attention', '--disable-dynamic-vram']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from importlib.metadata import version
from comfy.ldm.modules import attention
assert version('sageattention') == '2.2.0'
assert torch.cuda.get_device_capability(0) == (12, 0)
class FailOnFallback(logging.Handler):
    def emit(self, record):
        if 'Error running sage attention' in record.getMessage():
            raise RuntimeError(record.getMessage())
handler = FailOnFallback()
logging.getLogger().addHandler(handler)
original = attention.sageattn
calls = [0]
def counted(*args, **kwargs):
    calls[0] += 1
    return original(*args, **kwargs)
attention.sageattn = counted
with torch.inference_mode():
    for dtype in (torch.float16, torch.bfloat16):
        # H3 uses head_dim=128, 56 heads and HND tensors with transposed strides.
        base = [torch.randn(1, 257, 56, 128, device='cuda', dtype=dtype) for _ in range(3)]
        q, k, v = [x.transpose(1, 2) for x in base]
        before = calls[0]
        out = attention.optimized_attention(q, k, v, 56, skip_reshape=True)
        torch.cuda.synchronize()
        assert calls[0] == before + 1, 'ComfyUI did not call SageAttention'
        assert out.shape == (1, 257, 56 * 128), out.shape
        assert torch.isfinite(out).all().item(), 'Non-finite Sage output'
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(out)
        rel_error = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        # A smoke check catches gross kernel failure; not proof of H3 visual quality.
        assert rel_error.item() < 0.10, rel_error.item()
        print(f'Aivyla Sage: ComfyUI adapter passed; dtype={dtype}; relative_error={rel_error.item():.6f}')
attention.sageattn = original
logging.getLogger().removeHandler(handler)
print('Aivyla SageAttention 2.2.0 GPU CHECK PASSED; torch', torch.__version__)
PYSAGECHECK

mkdir -p /comfyui/custom_nodes/aivyla_sage_test_diagnostics
cat > /comfyui/custom_nodes/aivyla_sage_test_diagnostics/__init__.py <<'PYSAGEDIAG'
import logging
from comfy.ldm.modules import attention
_original = attention.sageattn
_seen = set()
def _observed(q, k, v, *args, **kwargs):
    result = _original(q, k, v, *args, **kwargs)
    layout = kwargs.get('tensor_layout', 'HND')
    heads = q.shape[1] if layout == 'HND' else q.shape[2]
    tag = 'H3-shaped' if heads == 56 and q.shape[-1] == 128 else 'first'
    if tag not in _seen:
        _seen.add(tag)
        logging.info('Aivyla Sage REAL CALL [%s]: q=%s dtype=%s layout=%s', tag, tuple(q.shape), q.dtype, layout)
    return result
attention.sageattn = _observed
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
PYSAGEDIAG
    SAGE=1
  else
    echo "Aivyla: WARNING: no verified Sage wheel on the volume; starting with PyTorch attention" >&2
  fi

  # Only the custom nodes the app workflows use
  nodes=(
    ComfyUI-AIvylaProduction
    aivyla_h3_first_frame_lock
    ComfyUI-BFSNodes
    aivyla_h3_identity_refs
    ComfyUI-KJNodes
    ComfyUI-MiniMaxH3Mod
    minimax-h3-audio-T8
    ComfyUI-Krea2-NAG
    ComfyUI-PlagueKind-Nodes
    ComfyUI-LTXVideo
  )
  for node in "${nodes[@]}"; do
    test -f "$VOL/ComfyUI/custom_nodes/$node/__init__.py"
  done
  test -f "$VOL/ComfyUI/custom_nodes/aivyla_h3_identity_refs/models/face_detection_yunet_2023mar.onnx"
  for node in "${nodes[@]}"; do
    target_dir="/comfyui/custom_nodes/$node"
    rm -rf -- "$target_dir"
    ln -s -- "$VOL/ComfyUI/custom_nodes/$node" "$target_dir"
    # Normally a no-op: the image already has these packages
    if [[ -f "$target_dir/requirements.txt" ]]; then
      python -m pip install --no-cache-dir -q -r "$target_dir/requirements.txt"
    fi
    echo "Aivyla node ready: $node"
  done
  python -m pip check
fi

# Launch flags: DynamicVRAM off unless AIVYLA_DYNAMIC_VRAM=1, Sage only when its wheel passed the check
export AIVYLA_SAGE="$SAGE"
python - <<'PYLAUNCH'
from pathlib import Path
import os, re
dynamic_vram = os.environ.get("AIVYLA_DYNAMIC_VRAM", "0") == "1"
sage = os.environ.get("AIVYLA_SAGE") == "1"
source = Path("/start.sh").read_text()
pattern = re.compile(r"(?m)^(?P<prefix>[ \t]*python[ \t]+-u[ \t]+/comfyui/main\.py)(?P<args>[^\n]*)$")
if len(pattern.findall(source)) != 2:
    raise SystemExit("Aivyla: expected 2 ComfyUI launcher lines in /start.sh")
def patch(m):
    args = [a for a in m.group("args").split() if a not in ("--disable-dynamic-vram", "--use-sage-attention")]
    if not dynamic_vram:
        args.insert(0, "--disable-dynamic-vram")
    if sage:
        args.insert(0, "--use-sage-attention")
    return m.group("prefix") + "".join(" " + a for a in args)
Path("/tmp/aivyla-worker-start.sh").write_text(pattern.sub(patch, source))
print(f"Aivyla: launcher ready; Sage={'ON' if sage else 'OFF'}; DynamicVRAM={'ON' if dynamic_vram else 'OFF'}")
PYLAUNCH
bash -n /tmp/aivyla-worker-start.sh
exec bash /tmp/aivyla-worker-start.sh
