import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).parent

def load(module_name):
    spec = importlib.util.spec_from_file_location(module_name, HERE / f'{module_name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGlobalCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.vol = self.base / 'vol'
        self.cache = self.base / 'local'
        self.vol.mkdir()
        self.cache.mkdir()
        self.mod = load('aivyla_global_cache')
        self.mod.VOLUME = self.vol
        self.mod.SEARCH_ROOTS = [self.vol / 'split_files', self.vol / 'ComfyUI' / 'models']
        self.mod.ROOT = self.cache
        self.mod.MIN_FREE_GIB = 0
        self.src = self.vol / 'split_files' / 'checkpoints' / 'test.safetensors'
        self.src.parent.mkdir(parents=True)
        blob = b'\x00' * 32
        head = json.dumps({'x': {'dtype': 'U8', 'shape': [len(blob)], 'data_offsets': [0, len(blob)]}}).encode()
        self.src.write_bytes(struct.pack('<Q', len(head)) + head + blob)
        self.workflow = {'7': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'test.safetensors'}}}

    def test_miss_then_hit(self):
        self.mod._validate_safetensors = self.mod._validate_safetensors
        stats = self.mod.stage_workflow(self.workflow)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual((self.cache / 'checkpoints' / 'test.safetensors').read_bytes(), self.src.read_bytes())
        stats = self.mod.stage_workflow(self.workflow)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 0)

    def test_corrupted_header_fails_and_no_partial_dest(self):
        self.src.write_bytes(b'\x00' * 32)
        with self.assertRaisesRegex(RuntimeError, 'Unable to stage'):
            self.mod.stage_workflow(self.workflow)
        self.assertFalse((self.cache / 'checkpoints' / 'test.safetensors').exists())

    def test_low_disk_fail_fast(self):
        self.mod.MIN_FREE_GIB = 10**5
        with self.assertRaisesRegex(RuntimeError, 'Increase Aivyla Global Container Disk'):
            self.mod.stage_workflow(self.workflow)
        self.assertFalse((self.cache / 'checkpoints' / 'test.safetensors').exists())

    def test_symlink_source_allowed_only_within_volume(self):
        alias = self.vol / 'ComfyUI' / 'models' / 'checkpoints'
        alias.mkdir(parents=True)
        (alias / 'evil.safetensors').symlink_to('/etc/passwd')
        with self.assertRaises(RuntimeError):
            self.mod._find_source('evil.safetensors', ['checkpoints'])


class TestBootstrap(unittest.TestCase):
    def test_patch_paths_and_handler_without_touching_base_image(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        mod = load('aivyla_global_start')
        path = Path(t.name) / 'start.sh'
        path.write_text('#!/bin/bash\necho "worker-comfyui: Starting ComfyUI"\nif [ "$SERVE_API_LOCALLY" == "true" ]; then\n python -u /comfyui/main.py --listen &\n python -u /handler.py --rp_serve_api\nelse\n python -u /comfyui/main.py &\n python -u /handler.py\nfi\n')
        mod.START = path
        mod.CACHE = Path('/tmp/aivyla-model-cache')
        mod.patch_start()
        text = path.read_text()
        self.assertEqual(text.count('aivyla_global_handler.py'), 2)
        self.assertEqual(text.count('aivyla_global_nodes.py'), 1)
        self.assertIn('is_default: true', text)
        self.assertLess(text.index('aivyla_global_nodes.py'), text.index('Starting ComfyUI'))


class TestNodeStaging(unittest.TestCase):
    def test_preserve_large_asset_link(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        m = load('aivyla_global_nodes')
        root = Path(t.name)
        m.BASE, m.VOLUME, m.LOCAL = (root / 'active', root / 'volume', root / 'local')
        (m.VOLUME / 'Sample').mkdir(parents=True)
        (m.VOLUME / 'Sample' / '__init__.py').write_text('NODE_CLASS_MAPPINGS = {}')
        (m.VOLUME / 'Sample' / 'weights.safetensors').write_bytes(b'X' * (m.ASSET_THRESHOLD + 1))
        m.BASE.mkdir()
        (m.BASE / 'Sample').symlink_to(m.VOLUME / 'Sample', target_is_directory=True)
        m.main()
        self.assertTrue((m.BASE / 'Sample').resolve().is_relative_to(m.LOCAL))
        self.assertTrue((m.BASE / 'Sample' / 'weights.safetensors').is_symlink())


if __name__ == '__main__':
    unittest.main(verbosity=2)
