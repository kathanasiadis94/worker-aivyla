"""CPU-only safety regression for the opt-in streaming node staging."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
legacy = types.ModuleType('aivyla_global_nodes')
sys.modules['aivyla_global_nodes'] = legacy
spec = importlib.util.spec_from_file_location('aivyla_fast_node_staging', HERE / 'aivyla_fast_node_staging.py')
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)


class FastNodeStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp', prefix='fast-nodes-unit-')
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {'AIVYLA_FAST_NODE_STREAM': '1'})
        env.start()
        self.addCleanup(env.stop)
        base = Path(self.tmp.name)
        self.names = [f'Node-{n:02}' for n in range(16)]
        legacy.BUNDLE = base / 'bundle.tar.gz'
        legacy.NODES_FILE = base / 'nodes.txt'
        legacy.NODES_FILE.write_text('\n'.join(self.names))
        legacy.LOCAL = base / 'cache'
        legacy.MIN_FREE_BYTES = 0
        legacy.CHUNK_BYTES = 8192
        legacy.READY_MARKER = '.aivyla-node-bundle-sha256'
        legacy._expected_nodes = lambda: set(self.names)
        legacy._check_root = lambda: None
        legacy._already_staged = lambda expected: legacy.LOCAL.exists()
        legacy.main = mock.Mock()
        self.build_archive()

    def build_archive(self, extra=None, append=b''):
        with tarfile.open(legacy.BUNDLE, 'w:gz') as arc:
            for name in self.names:
                payload = b'NODE_CLASS_MAPPINGS = {}\n'
                item = tarfile.TarInfo(name + '/__init__.py')
                item.size = len(payload)
                arc.addfile(item, io.BytesIO(payload))
            if extra is not None:
                name, payload, kind = extra
                item = tarfile.TarInfo(name)
                item.type = kind
                item.size = len(payload) if kind == tarfile.REGTYPE else 0
                arc.addfile(item, io.BytesIO(payload) if payload else None)
        if append:
            with legacy.BUNDLE.open('ab') as f:
                f.write(append)
        legacy.EXPECTED_SHA256 = hashlib.sha256(legacy.BUNDLE.read_bytes()).hexdigest()

    def test_stream_stages_all_nodes_and_marker(self):
        fast.main()
        for name in self.names:
            self.assertTrue((legacy.LOCAL / name / '__init__.py').is_file())
        self.assertEqual((legacy.LOCAL / legacy.READY_MARKER).read_text().strip(), legacy.EXPECTED_SHA256)
        with mock.patch.object(fast, 'stage_stream', side_effect=AssertionError('cache not reused')):
            fast.main()

    def test_wrong_digest_never_publishes_directory(self):
        legacy.EXPECTED_SHA256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            fast.main()
        self.assertFalse(legacy.LOCAL.exists())

    def test_rejects_traversal_before_publishing(self):
        self.build_archive(('../pwned', b'bad', tarfile.REGTYPE))
        with self.assertRaisesRegex(ValueError, 'Unsafe or unexpected'):
            fast.main()
        self.assertFalse(legacy.LOCAL.exists())

    def test_rejects_symlink_before_publishing(self):
        self.build_archive((self.names[0] + '/evil', b'', tarfile.SYMTYPE))
        with self.assertRaisesRegex(ValueError, 'Unsafe or unexpected'):
            fast.main()
        self.assertFalse(legacy.LOCAL.exists())

    def test_rejects_duplicate_file(self):
        self.build_archive((self.names[0] + '/__init__.py', b'again', tarfile.REGTYPE))
        with self.assertRaisesRegex(ValueError, 'Unsafe or unexpected'):
            fast.main()
        self.assertFalse(legacy.LOCAL.exists())

    def test_hash_covers_trailing_bytes(self):
        self.build_archive(append=b'after-end-of-tar')
        fast.main()
        self.assertTrue(legacy.LOCAL.is_dir())

    def test_tampered_trailing_bytes_fail(self):
        self.build_archive()
        with legacy.BUNDLE.open('ab') as f:
            f.write(b'evil')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            fast.main()
        self.assertFalse(legacy.LOCAL.exists())

    def test_environment_fallback_uses_unchanged_original(self):
        with mock.patch.dict(os.environ, {'AIVYLA_FAST_NODE_STREAM': '0'}):
            fast.main()
        legacy.main.assert_called_once_with()
        self.assertFalse(legacy.LOCAL.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
