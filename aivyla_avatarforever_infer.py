"""Isolated adapter: required loras constructor argument and strict fusion audit.

Called in a fresh process per variant. Never imports into ComfyUI's process.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import time


def audit_lora(model_shapes, lora_shapes):
    """Refuse unmapped tensors, incomplete pairs and incompatible matrix sizes."""
    pairs = {}
    for key, shape in lora_shapes.items():
        suffix = next((s for s in ('.lora_A.weight', '.lora_B.weight') if key.endswith(s)), None)
        if suffix is None:
            raise ValueError(f'Unsupported mapped LoRA tensor: {key}')
        pairs.setdefault(key[:-len(suffix)], {})[suffix] = tuple(shape)
    if not pairs:
        raise ValueError('No LoRA pairs mapped; refuse a silently ignored LoRA')
    for prefix, values in pairs.items():
        if set(values) != {'.lora_A.weight', '.lora_B.weight'}:
            raise ValueError(f'Incomplete LoRA pair: {prefix}')
        a, b = values['.lora_A.weight'], values['.lora_B.weight']
        target = model_shapes.get(prefix + '.weight')
        if (target is None or len(a) != 2 or len(b) != 2 or a[0] != b[1]
                or tuple(target) != (b[0], a[1])):
            raise ValueError(f'LoRA target/shape mismatch: {prefix}')
    return len(pairs)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--lora', nargs=2, action='append', default=[])
    extra.add_argument('--aivyla-report', type=Path, required=True)
    options, remaining = extra.parse_known_args()
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    import torch
    import safetensors
    import ltx_pipelines
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP
    import ltx_core.loader.single_gpu_model_builder as builder

    loras = []
    audits = []
    for filename, strength in options.lora:
        path = Path(filename)
        strength = float(strength)
        if not path.is_file() or not math.isfinite(strength) or not 0 < strength <= 2:
            raise ValueError('Expected an existing LoRA and finite strength in (0,2]')
        loras.append(LoraPathStrengthAndSDOps(str(path), strength, LTXV_LORA_COMFY_RENAMING_MAP))

    # Check all raw keys survive the same renaming map used by the actual loader.
    checkpoint_arg = remaining.index('--distilled-checkpoint-path') + 1
    checkpoint = remaining[checkpoint_arg]
    with safetensors.safe_open(checkpoint, framework='pt', device='cpu') as src:
        model_shapes = {}
        for key in src.keys():
            mapped = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(key)
            if mapped is not None:
                model_shapes[mapped] = src.get_slice(key).get_shape()
    for lora in loras:
        shapes = {}
        with safetensors.safe_open(lora.path, framework='pt', device='cpu') as src:
            for key in src.keys():
                mapped = lora.sd_ops.apply_to_key(key)
                if mapped is None:
                    raise ValueError(f'Unmapped LoRA tensor {key}; no partial/silent loading allowed')
                if mapped in shapes:
                    raise ValueError(f'LoRA key mapping collision: {mapped}')
                shapes[mapped] = src.get_slice(key).get_shape()
        pairs = audit_lora(model_shapes, shapes)
        audits.append(dict(filename=Path(lora.path).name, strength=lora.strength, matched_pairs=pairs))
    del model_shapes
    print('AvatarForever LoRA preflight: ' + json.dumps(audits), flush=True)

    original_fuse = builder.apply_loras
    fusion_calls = []

    def checked_fuse(model_sd, lora_sd_and_strengths, **kwargs):
        target_shapes = {k: tuple(v.shape) for k, v in model_sd.sd.items() if v is not None}
        matched = [audit_lora(target_shapes, {k: tuple(v.shape) for k, v in sd.sd.items()})
                   for sd, strength in lora_sd_and_strengths if strength != 0]
        fusion_calls.append(matched)
        print('AvatarForever actual LoRA fusion pairs: ' + json.dumps(matched), flush=True)
        return original_fuse(model_sd, lora_sd_and_strengths, **kwargs)

    original_pipeline = ltx_pipelines.ARA2VidDistilledPipeline

    class TestPipeline(original_pipeline):
        def __init__(self, *args, **kwargs):
            if kwargs.get('loras'):
                raise ValueError('Unexpected upstream LoRAs; comparison requires an explicit stack')
            kwargs['loras'] = list(loras)
            super().__init__(*args, **kwargs)

    builder.apply_loras = checked_fuse
    ltx_pipelines.ARA2VidDistilledPipeline = TestPipeline
    previous = sys.argv
    started = time.monotonic()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    try:
        sys.argv = [str(repo / 'inference.py'), *remaining]
        runpy.run_path(str(repo / 'inference.py'), run_name='__main__')
        if loras and not fusion_calls:
            raise RuntimeError('LoRAs were requested but no model fusion occurred')
        options.aivyla_report.write_text(json.dumps(dict(
            lora_audit=audits, fusion_calls=fusion_calls,
            gpu_peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
            subprocess_s=round(time.monotonic()-started, 3))))
    finally:
        sys.argv = previous
        builder.apply_loras = original_fuse
        ltx_pipelines.ARA2VidDistilledPipeline = original_pipeline


if __name__ == '__main__':
    main()
