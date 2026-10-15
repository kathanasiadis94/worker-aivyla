"""Exercise Gemma's real builder, buffer population and text forward without weights."""
from __future__ import annotations

import copy
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import sys

from aivyla_avatarforever_env import TARGET_PACKAGE_PINS


def check_gemma_runtime():
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        GEMMA_MODEL_OPS, GemmaTextEncoderConfigurator, create_and_populate,
    )

    stack = {}
    for package, expected in TARGET_PACKAGE_PINS.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f'AvatarForever requires isolated {package}=={expected}; got {actual}')
        module = import_module(package.replace('-', '_'))
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f'AvatarForever {package} must load from its own venv: {module.__file__}')
        stack[package] = actual

    # This is the exact 12B configurator and module mutator that previously
    # crashed. Parameters stay on meta; no checkpoint or GPU is needed.
    builder = SingleGPUModelBuilder(
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_path='unused-gemma-build-check', module_ops=(GEMMA_MODEL_OPS,))
    model = builder.meta_model({}, builder.module_ops)
    if not all(p.is_meta for p in model.parameters()):
        raise RuntimeError('Full Gemma construction unexpectedly materialized parameters')
    if any(b.is_meta for b in model.buffers()):
        raise RuntimeError('Gemma buffer population left meta buffers')
    vision = model.model.model.vision_tower.vision_model
    language = model.model.model.language_model
    if vision.embeddings.position_ids.device.type != 'cpu':
        raise RuntimeError('Gemma vision position IDs were not populated')
    if language.rotary_emb.inv_freq.device.type != 'cpu':
        raise RuntimeError('Gemma rotary frequencies were not populated')
    parameter_count = sum(p.numel() for p in model.parameters())
    del model, builder

    # A tiny instance runs the real upstream text encoder with actual CPU
    # tensors, catching forward/hidden-state API changes beyond construction.
    config = copy.deepcopy(GEMMA3_CONFIG_FOR_LTX.to_dict())
    config['text_config'].update(vocab_size=128, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        sliding_window=16)
    config['vision_config'].update(hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, image_size=28, patch_size=14)
    config.update(mm_tokens_per_image=4, boi_token_index=124,
                  eoi_token_index=125, image_token_index=126, eos_token_id=1)

    class FixedTokens:
        def tokenize_with_weights(self, text):
            return {'gemma': [(2, 1), (3, 1), (4, 1), (0, 0)]}

    tiny = GemmaTextEncoder(model=Gemma3ForConditionalGeneration(
        Gemma3Config.from_dict(config)), tokenizer=FixedTokens())
    tiny = create_and_populate(tiny).eval()
    with torch.inference_mode():
        hidden, mask = tiny.encode('test')
    if len(hidden) != 3 or mask.shape != (1, 4):
        raise RuntimeError('Gemma text encoder returned unexpected hidden states')
    if any(h.shape != (1, 4, 32) or not torch.isfinite(h).all() for h in hidden):
        raise RuntimeError('Gemma text encoder CPU forward failed')
    return dict(packages=stack, full_model_meta_parameters=parameter_count,
                full_model_buffer_population='passed', text_encode_cpu='passed')


if __name__ == '__main__':
    import json
    print(json.dumps(check_gemma_runtime(), indent=2))
