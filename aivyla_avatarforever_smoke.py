"""Real CPU runtime checks inside the exact candidate image, before publishing."""
from __future__ import annotations

import importlib
from importlib.metadata import distributions, version
import json
from pathlib import Path
import pkgutil
import sys


def main(runtime_root=Path('/opt/avatarforever')):
    import torch
    import ltx_core
    import ltx_pipelines
    import accelerate
    from aivyla_avatarforever_gemma_check import check_gemma_runtime
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.loader.fuse_loras import apply_loras
    from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core.model.transformer.attention import PytorchAttention

    shared = json.loads((Path(sys.prefix) / 'aivyla_shared_runtime.json').read_text())
    torch_path = Path(torch.__file__).resolve()
    if not any(torch_path.is_relative_to(Path(site)) for site in shared['base_sites']):
        raise RuntimeError('Refuse a replacement CUDA torch: ' + str(torch_path))
    if any(str(Path(site)) in sys.path for site in shared['base_sites']):
        raise RuntimeError('ComfyUI site-packages leaked into AvatarForever sys.path')
    for name, expected in shared['shared_packages'].items():
        if version(name) != expected:
            raise RuntimeError('Shared CUDA dependency was replaced: ' + name)
    if importlib.util.find_spec('diffusers') is not None:
        raise RuntimeError('ComfyUI diffusers leaked into isolated AvatarForever environment')
    import torchaudio
    import torchvision
    for module in (torchaudio, torchvision):
        if not any(Path(module.__file__).resolve().is_relative_to(Path(site))
                   for site in shared['base_sites']):
            raise RuntimeError('Shared CUDA extension was replaced: ' + module.__name__)
    audio = torchaudio.functional.resample(torch.ones(1, 160), 16000, 8000)
    if audio.shape != (1, 80) or not torch.isfinite(audio).all():
        raise RuntimeError('Shared torchaudio CPU resampling failed')
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
                               torch.tensor([.9, .8]), .5)
    if kept.tolist() != [0]:
        raise RuntimeError('Shared torchvision native CPU operation failed')
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
    gemma_check = check_gemma_runtime()
    report = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  torch_path=str(torch_path), accelerate=accelerate.__version__,
                  imported_modules=len(imported), attention_cpu='passed', lora_fusion_cpu='passed',
                  gemma=gemma_check, isolated_dependencies=True, shared_native_ops_cpu='passed',
                  shared_packages=shared['shared_packages'])
    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root/'runtime-check.json').write_text(json.dumps(report, indent=2))
    pinned = sorted({d.metadata['Name'] + '==' + d.version for d in distributions()})
    (runtime_root/'resolved-requirements.txt').write_text('\n'.join(pinned) + '\n')
    print('PASS: complete AvatarForever CPU runtime ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path('/opt/avatarforever'))
    main(parser.parse_args().runtime_root)
