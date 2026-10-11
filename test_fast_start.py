import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys
import types

HERE = Path(__file__).parent
# Stub imported launchers: these tests do not need RunPod, ComfyUI or GPU.
r2 = types.ModuleType('aivyla_global_r2_start')
r2.v3 = types.SimpleNamespace()
sys.modules.setdefault('aivyla_global_r2_start', r2)
spec = importlib.util.spec_from_file_location('afs', HERE / 'aivyla_fast_start.py')
afs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afs)
spec = importlib.util.spec_from_file_location('afr', HERE / 'aivyla_fast_requirements.py')
afr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afr)

class FastStartTests(unittest.TestCase):
    def test_exact_patch_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'launcher.sh')
            p.write_text('#!/bin/bash\n' + afs.ORIGINAL + '\n')
            afs.patch_requirements(p)
            self.assertEqual(p.read_text().count(afs.FAST), 1)
            afs.patch_requirements(p)
            self.assertEqual(p.read_text().count(afs.FAST), 1)

    def test_refuse_launcher_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'launcher.sh')
            p.write_text('#!/bin/bash\necho changed\n')
            with self.assertRaisesRegex(RuntimeError, 'anchor drift'):
                afs.patch_requirements(p)

    def test_matching_requirements_fast_path(self):
        # The local Mac Python may not have the *distribution* named packaging,
        # even though pip has a vendored requirement parser. Mock a known
        # installed distribution instead of assuming the test host's packages.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'requirements.txt')
            p.write_text('aivyla-faststart-test-fixture>=1.0,<2\n')
            if afr._get_requirement_class() is None:
                self.skipTest('No PEP 508 parser in host Python; safe pip fallback tested separately')
            with mock.patch.object(afr.metadata, 'version', return_value='1.5'):
                self.assertTrue(afr.requirements_satisfied(p))
                with mock.patch.object(afr.subprocess, 'run') as run:
                    afr.ensure(p)
                    run.assert_not_called()

    def test_missing_requirement_parser_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'requirements.txt')
            p.write_text('aivyla-faststart-test-fixture>=1\n')
            with mock.patch.object(afr, '_get_requirement_class', return_value=None):
                self.assertFalse(afr.requirements_satisfied(p))
                with mock.patch.object(afr.subprocess, 'run') as run:
                    afr.ensure(p)
                    run.assert_called_once()

    def test_unsatisfied_requirement_uses_pip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'requirements.txt')
            p.write_text('nonexistent-aivyla-unittest-pkg>=9999\n')
            self.assertFalse(afr.requirements_satisfied(p))
            with mock.patch.object(afr.subprocess, 'run') as run:
                afr.ensure(p)
                self.assertIn('install', run.call_args.args[0])

    def test_complex_pip_options_fall_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'requirements.txt')
            for value in ('-r other.txt\n', 'packaging[docs]>=1\n', 'https://example.com/pkg.whl\n', 'packaging>=1 --hash=sha256:abcd\n'):
                p.write_text(value)
                self.assertFalse(afr.requirements_satisfied(p), value)

    def test_baked_nodes_match_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, 'nodes'); root.mkdir()
            (root / '.aivyla-node-bundle-sha256').write_text(afs.BUNDLE_SHA256)
            for i in range(16):
                node = root / f'node{i}'; node.mkdir(); (node / '__init__.py').write_text('')
            names = Path(d, 'nodes.txt')
            names.write_text('\n'.join(f'node{i}' for i in range(16)))
            with mock.patch.dict('os.environ', {'AIVYLA_FAST_BAKED_NODES':'1'}):
                self.assertTrue(afs.baked_nodes_available(root, names))
                names.write_text('extra-node\n' + names.read_text())
                self.assertFalse(afs.baked_nodes_available(root, names))

    def test_baked_patch_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, 'launcher.sh')
            path.write_text('#!/bin/bash\n' + '\n'.join(['echo "$VOL/ComfyUI/custom_nodes/$node"'] * 3))
            afs.patch_baked_launcher(path)
            afs.patch_baked_launcher(path)
            self.assertEqual(path.read_text().count('"$AIVYLA_GLOBAL_NODE_ROOT/$node'), 3)

    def test_env_disable_preserves_pip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'requirements.txt')
            p.write_text('packaging>=0\n')
            with mock.patch.dict('os.environ', {'AIVYLA_FAST_REQUIREMENTS': '0'}), mock.patch.object(afr.subprocess, 'run') as run:
                afr.ensure(p)
                run.assert_called_once()

if __name__ == '__main__':
    unittest.main(verbosity=2)
