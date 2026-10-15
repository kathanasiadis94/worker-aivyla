"""Real CPU runtime checks inside the exact candidate image, before publishing."""
from __future__ import annotations

import importlib
from importlib.metadata import distributions
import json
from pathlib import Path
import pkgutil
import sys


def main():
    import torch
    import ltx_core
    import ltx_pipelines
    import accelerate
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.loader.fuse_loras import apply_loras
    from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core.model.transformer.attention import PytorchAttention

    if not torch.__file__.startswith('/opt/venv/'):
        raise RuntimeError('Refuse a replacement CUDA torch: ' + torch.__file__)
    # Import every packaged module, including paths reached only at generation.
    imported = []
    for package in (ltx_core, ltx_pipelines):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            importlib.import_module(module.name)
            imported.append(module.name)
    q = torch.randn(1, 8, 16, device='cpu')
    out = PytorchAttention()(q, q, q, heads=2)
    if out.shape != q.shape or not torch.isfinite(out).all():
        raise RuntimeError('CPU attention smoke test failed')
    base = StateDict({'x.weight': torch.zeros((4, 8), dtype=torch.bfloat16)},
                     torch.device('cpu'), 64, {torch.bfloat16})
    lora = StateDict({'x.lora_A.weight': torch.ones((2, 8), dtype=torch.bfloat16),
                      'x.lora_B.weight': torch.ones((4, 2), dtype=torch.bfloat16)},
                     torch.device('cpu'), 48, {torch.bfloat16})
    fused = apply_loras(base, [LoraStateDictWithStrength(lora, .5)], dtype=torch.bfloat16)
    if not torch.equal(fused.sd['x.weight'], torch.ones((4, 8), dtype=torch.bfloat16)):
        raise RuntimeError('Actual upstream LoRA fusion smoke test failed')
    if LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key('diffusion_model.x.lora_A.weight') != 'x.lora_A.weight':
        raise RuntimeError('LoRA mapping smoke test failed')
    if not callable(ltx_pipelines.ARA2VidDistilledPipeline):
        raise RuntimeError('AvatarForever pipeline is unavailable')
    report = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  torch_path=torch.__file__, accelerate=accelerate.__version__,
                  imported_modules=len(imported), attention_cpu='passed', lora_fusion_cpu='passed')
    Path('/opt/avatarforever/runtime-check.json').write_text(json.dumps(report, indent=2))
    pinned = sorted({d.metadata['Name'] + '==' + d.version for d in distributions()})
    Path('/opt/avatarforever/resolved-requirements.txt').write_text('\n'.join(pinned) + '\n')
    print('PASS: complete AvatarForever CPU runtime ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
