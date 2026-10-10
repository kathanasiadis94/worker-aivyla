"""No-GPU regression suite for Aivyla Global v3 (runs before Docker build).

Tests both launch scripts from this repository if available; a small fallback
fixture permits the same suite to run independently during development.
"""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name + '_v3_test', HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_file(path, fallback):
    candidate = HERE / path
    if candidate.is_file():
        return candidate.read_text()
    if os.environ.get('CI'):
        raise AssertionError(f'CI source file missing: {candidate}')
    return fallback


START_FALLBACK = '''#!/bin/bash
echo "worker-comfyui: Starting ComfyUI"
python -u /comfyui/main.py &
python -u /handler.py --rp_serve_api
python -u /handler.py
'''
AIVYLA_FALLBACK = '''#!/bin/bash
for node in "${nodes[@]}"; do
  test -f "$VOL/ComfyUI/custom_nodes/$node/__init__.py"
  test -f "$VOL/ComfyUI/custom_nodes/$node/models/test.onnx"
  ln -s "$VOL/ComfyUI/custom_nodes/$node" /comfyui/custom_nodes/"$node"
done
'''


class TestLaunchPatches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.start = self.root / 'start.sh'
        self.aivyla = self.root / 'aivyla-start.sh'
        self.start.write_text(original_file('src/start.sh', START_FALLBACK))
        self.aivyla.write_text(original_file('aivyla/aivyla-start.sh', AIVYLA_FALLBACK))
        self.orig_start = self.start.read_bytes()
        self.orig_aivyla = self.aivyla.read_bytes()
        self.mod = load('aivyla_global_start')
        self.mod.START = self.start
        self.mod.AIVYLA_START = self.aivyla
        self.mod.CACHE = Path('/tmp/aivyla-model-cache')
        self.mod.LOCAL_NODES = Path('/tmp/aivyla-node-cache')
        self.mod.LOCAL_ENABLED = True

    def test_both_launchers_safe_to_patch_twice(self):
        self.mod.patch_start()
        self.mod.patch_aivyla_launcher()
        first_start, first_aivyla = self.start.read_bytes(), self.aivyla.read_bytes()
        self.mod.patch_start()
        self.mod.patch_aivyla_launcher()
        self.assertEqual(first_start, self.start.read_bytes())
        self.assertEqual(first_aivyla, self.aivyla.read_bytes())
        self.assertEqual(self.start.read_text().count('aivyla_global_local_cache:'), 1)
        self.assertEqual(self.start.read_text().count('python -u /opt/aivyla-global/aivyla_global_handler.py'), 2)
        self.assertNotIn('"$VOL/ComfyUI/custom_nodes/$node', self.aivyla.read_text())
        self.assertEqual(self.aivyla.read_text().count('"$AIVYLA_GLOBAL_NODE_ROOT/$node'), 3)
        self.assertEqual(self.orig_start, original_file('src/start.sh', START_FALLBACK).encode())
        self.assertEqual(self.orig_aivyla, original_file('aivyla/aivyla-start.sh', AIVYLA_FALLBACK).encode())

    def test_refuse_partially_patched_script(self):
        self.start.write_text(self.start.read_text().replace('python -u /handler.py', 'python -u /opt/aivyla-global/aivyla_global_handler.py', 1))
        with self.assertRaisesRegex(RuntimeError, 'refusing partial/double patch'):
            self.mod.patch_start()

    def test_refuse_cache_path_change_on_second_boot(self):
        self.mod.patch_start()
        self.mod.CACHE = Path('/tmp/aivyla-model-cache-different')
        with self.assertRaisesRegex(RuntimeError, 'different model-cache path'):
            self.mod.patch_start()


class TestLocalBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='aivyla-v3-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mod = load('aivyla_global_nodes')
        self.mod.BUNDLE = self.root / 'nodes.tar.gz'
        self.mod.NODES_FILE = self.root / 'nodes.txt'
        self.mod.LOCAL = self.root / 'node-cache'
        self.mod.MIN_FREE_BYTES = 0
        self.nodes = [f'Node-{i:02d}' for i in range(16)]
        self.mod.NODES_FILE.write_text('\n'.join(self.nodes) + '\n')
        self.make_archive()

    def make_archive(self, extra=None):
        with tarfile.open(self.mod.BUNDLE, 'w:gz') as f:
            for node in self.nodes:
                name = f'{node}/__init__.py'
                content = b'NODE_CLASS_MAPPINGS = {}\n'
                info = tarfile.TarInfo(name)
                info.size = len(content)
                f.addfile(info, io.BytesIO(content))
            if extra:
                name, content = extra
                info = tarfile.TarInfo(name)
                info.size = len(content)
                f.addfile(info, io.BytesIO(content))
        self.mod.EXPECTED_SHA256 = hashlib.sha256(self.mod.BUNDLE.read_bytes()).hexdigest()

    def test_first_boot_and_restart_reuses_local_cache(self):
        self.mod.main()
        self.assertEqual((self.mod.LOCAL / self.mod.READY_MARKER).read_text().strip(), self.mod.EXPECTED_SHA256)
        self.assertEqual(len(list(self.mod.LOCAL.glob('Node-*/__init__.py'))), 16)
        with mock.patch.object(self.mod, '_stage', side_effect=AssertionError('Must not re-extract bundle')):
            self.mod.main()

    def test_restart_after_v2_without_marker_repairs_cache(self):
        self.mod.main()
        (self.mod.LOCAL / self.mod.READY_MARKER).unlink()
        self.mod.main()
        self.assertTrue((self.mod.LOCAL / self.mod.READY_MARKER).exists())
        self.assertEqual(len(list(self.mod.LOCAL.glob('Node-*/__init__.py'))), 16)

    def test_partial_local_cache_recovered(self):
        self.mod.LOCAL.mkdir()
        (self.mod.LOCAL / 'broken.tmp').write_text('partial')
        self.mod.main()
        self.assertFalse((self.mod.LOCAL / 'broken.tmp').exists())
        self.assertEqual(len(list(self.mod.LOCAL.glob('Node-*/__init__.py'))), 16)

    def test_bad_archive_hash_rejected(self):
        self.mod.EXPECTED_SHA256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.mod.main()
        self.assertFalse(self.mod.LOCAL.exists())

    def test_path_traversal_in_tar_rejected(self):
        self.make_archive(extra=('../evil.txt', b'evil'))
        with self.assertRaisesRegex(ValueError, 'Unsafe or unexpected'):
            self.mod.main()
        self.assertFalse((self.root / 'evil.txt').exists())
        self.assertFalse(self.mod.LOCAL.exists())


class TestHandler(unittest.TestCase):
    def test_real_wrapper_imports_sibling_module_and_preserves_stage(self):
        with tempfile.TemporaryDirectory(prefix='aivyla-handler-') as tmp:
            root = Path(tmp)
            (root / 'network_volume.py').write_text('VALUE = "sibling works"\n')
            (root / 'handler.py').write_text(
                'from network_volume import VALUE\n'
                'def handler(job):\n'
                '    return {"base": VALUE, "job": job["id"]}\n')
            cache = types.ModuleType('aivyla_global_cache')
            staged = []
            cache.stage_workflow = lambda workflow: (staged.append(workflow), {'models': 1})[1]
            fake_runpod = types.ModuleType('runpod')
            fake_runpod.serverless = types.SimpleNamespace(start=lambda arg: None)
            old_modules = {name: sys.modules.get(name) for name in ('aivyla_global_cache', 'runpod', 'network_volume')}
            old_path = list(sys.path)
            try:
                sys.modules['aivyla_global_cache'] = cache
                sys.modules['runpod'] = fake_runpod
                source = (HERE / 'aivyla_global_handler.py').read_text()
                source = source.replace("Path('/handler.py')", f'Path({str(root / "handler.py")!r})')
                # Execute at a simulated /opt path (not the handler's sibling directory).
                wrapper = root / 'opt' / 'aivyla_global_handler.py'
                wrapper.parent.mkdir()
                wrapper.write_text(source)
                ns = runpy.run_path(str(wrapper), run_name='unit_test')
                job = {'id': 42, 'input': {'workflow': {'5': {'class_type': 'Sample', 'inputs': {}}}}}
                self.assertEqual(ns['handler'](job), {'base': 'sibling works', 'job': 42})
                self.assertEqual(staged, [job['input']['workflow']])
                self.assertEqual(ns['handler']({'id': 7, 'input': {'other': 1}})['job'], 7)
            finally:
                sys.path[:] = old_path
                for name, value in old_modules.items():
                    if value is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value

    def test_isolated_image_does_not_change_production_tag(self):
        docker = (HERE / 'Dockerfile.global').read_text()
        workflow = (HERE / '.github/workflows/build-aivyla-global.yml').read_text()
        self.assertIn('FROM ${AIVYLA_BASE_IMAGE}', docker)
        self.assertIn('worker-aivyla:aivyla-5', docker)
        self.assertIn('Global handler imports the real production handler', docker)
        self.assertIn('worker-aivyla:aivyla-global-v3', workflow)
        self.assertNotIn('worker-aivyla:aivyla-global-v2\n', workflow)
        self.assertNotIn('worker-aivyla:aivyla-5\n', workflow)


if __name__ == '__main__':
    unittest.main(verbosity=2)
