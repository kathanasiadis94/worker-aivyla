"""CPU regressions: payload limits, exact R2 bundle, strict LoRAs and dispatch."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
import zipfile
from unittest import mock

import aivyla_avatarforever as av
from aivyla_avatarforever_infer import audit_lora
from avatarforever_global_test import resolve_endpoint
from aivyla_avatarforever_env import environment, freeze_base, link_dependencies


def payload(**extra):
    return dict(engine=av.ENGINE, output_transport='studio', image='eA==', audio='eA==', **extra)


class MissingObject(Exception):
    response = {'Error': {'Code': 'NoSuchKey'}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.downloads = []
        self.corrupt = False
        self.change = False

    def add(self, key, data):
        self.objects[key] = data

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject(Key)
        data = self.objects[Key]
        etag = hashlib.sha256(data).hexdigest()
        if self.change and Key in self.downloads:
            etag = 'changed'
        return dict(ContentLength=len(data), ETag='"' + etag + '"',
                    Metadata={'sha256': hashlib.sha256(data).hexdigest()})

    def download_file(self, bucket, key, filename, Config):
        self.downloads.append(key)
        data = self.objects[key]
        Path(filename).write_bytes(b'!' * len(data) if self.corrupt else data)


class EnvironmentBridgeTests(unittest.TestCase):
    def test_real_resolver_installs_missing_dependencies_and_preserves_base(self):
        # Fully offline wheelhouse: exercises the actual pip install command,
        # dependency resolution and constraints, rather than mocking a package.
        def wheel(directory, name, version, requires=()):
            module = name.replace('-', '_')
            info = f'{module}-{version}.dist-info'
            path = directory/f'{module}-{version}-py3-none-any.whl'
            metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
            metadata += ''.join(f'Requires-Dist: {requirement}\n' for requirement in requires)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(module+'.py', 'VALUE = '+repr(version)+'\n')
                archive.writestr(info+'/METADATA', metadata)
                archive.writestr(info+'/WHEEL', 'Wheel-Version: 1.0\nGenerator: regression-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
                archive.writestr(info+'/RECORD', '')
            return path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, target, wheels = root/'base', root/'target', root/'wheels'
            wheels.mkdir()
            subprocess.run([sys.executable, '-m', 'venv', str(base)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            subprocess.run([str(base/'bin/python'), '-m', 'venv', '--without-pip', str(target)],
                           check=True, timeout=60)
            basewheel = wheel(wheels, 'aivyla-base-lock-fixture', '1.2.3')
            subprocess.run([str(base/'bin/python'), '-m', 'pip', 'install', '--no-index', str(basewheel)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            link_dependencies(base/'bin/python', target/'bin/python')
            constraints = root/'base-constraints.txt'
            pins = freeze_base(base/'bin/python', constraints)
            self.assertEqual(pins['aivyla-base-lock-fixture'], '1.2.3')
            missing = wheel(wheels, 'aivyla-missing-dep-fixture', '4.5.6')
            consumer = wheel(wheels, 'aivyla-consumer-fixture', '1.0',
                             ['aivyla-base-lock-fixture>=1', 'aivyla-missing-dep-fixture>=4'])
            install = [str(target/'bin/python'), '-m', 'pip', 'install', '--no-index',
                       '--find-links', str(wheels), '--constraint', str(constraints)]
            subprocess.run([*install, str(consumer)], check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60)
            probe = ('from importlib.metadata import version; import aivyla_base_lock_fixture as b; '
                     'import aivyla_missing_dep_fixture as m; '
                     'print(version("aivyla-base-lock-fixture"), version("aivyla-missing-dep-fixture")); '
                     'print(b.__file__); print(m.__file__)')
            result = subprocess.check_output([str(target/'bin/python'), '-I', '-c', probe], text=True)
            self.assertEqual(result.splitlines()[0], '1.2.3 4.5.6')
            self.assertTrue(result.splitlines()[1].startswith(str(base)))
            self.assertTrue(result.splitlines()[2].startswith(str(target)))
            checked = subprocess.check_output([str(target/'bin/python'), '-m', 'pip', 'check'], text=True)
            self.assertIn('No broken requirements found', checked)
            # A conflicting upgrade must fail, rather than replace a CUDA/base package.
            wheel(wheels, 'aivyla-base-lock-fixture', '9.0')
            conflict = wheel(wheels, 'aivyla-conflict-fixture', '1.0', ['aivyla-base-lock-fixture>=9'])
            refused = subprocess.run([*install, str(conflict)], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=60)
            self.assertNotEqual(refused.returncode, 0)
            unchanged = subprocess.check_output([str(base/'bin/python'), '-I', '-c',
                       'from importlib.metadata import version; print(version("aivyla-base-lock-fixture"))'], text=True)
            self.assertEqual(unchanged.strip(), '1.2.3')

    def test_real_sibling_venvs_preserve_base_and_share_dependencies(self):
        # This reproduces the failed image build without requiring CUDA/Docker.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, target = root/'base', root/'target'
            subprocess.run([sys.executable, '-m', 'venv', str(base)],
                           check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run([str(base/'bin/python'), '-m', 'venv', '--without-pip',
                            '--system-site-packages', str(target)], check=True, timeout=60)
            source = Path(environment(base/'bin/python')['purelib'])
            package = source / 'aivyla_base_fixture.py'
            package.write_text('VALUE = 173\n')
            metadata = source / 'aivyla_base_fixture-1.2.3.dist-info'
            metadata.mkdir()
            (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: aivyla-base-fixture\nVersion: 1.2.3\n')
            target_site = Path(environment(target/'bin/python')['purelib'])
            dependent = target_site/'aivyla_target_fixture-1.0.dist-info'
            dependent.mkdir()
            (dependent/'METADATA').write_text('Metadata-Version: 2.1\nName: aivyla-target-fixture\n'
                                             'Version: 1.0\nRequires-Dist: aivyla-base-fixture>=1\n')
            before = {p.relative_to(base).as_posix(): p.read_bytes()
                      for p in base.rglob('*') if p.is_file() and not p.is_symlink()}
            probe = ('import aivyla_base_fixture; from importlib.metadata import version; '
                     'print(aivyla_base_fixture.VALUE, version("aivyla-base-fixture"))')
            missing = subprocess.run([str(target/'bin/python'), '-I', '-c', probe],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(missing.returncode, 0, 'Regression fixture must reproduce missing sibling deps')
            result = link_dependencies(base/'bin/python', target/'bin/python')
            output = subprocess.check_output([str(target/'bin/python'), '-I', '-c', probe], text=True)
            self.assertEqual(output.strip(), '173 1.2.3')
            checked = subprocess.check_output([str(target/'bin/python'), '-I', '-m', 'pip', 'check'], text=True)
            self.assertIn('No broken requirements found', checked)
            link_dependencies(base/'bin/python', target/'bin/python')
            self.assertEqual(Path(result['bridge']).read_text().count('addsitedir'), 1)
            # Imports can create bytecode; all original base files remain byte-identical.
            for name, data in before.items():
                self.assertEqual((base/name).read_bytes(), data, name)

    def test_refuse_base_environment_as_target(self):
        with self.assertRaisesRegex(ValueError, 'Refuse to modify'):
            link_dependencies(sys.executable, sys.executable)

    def test_refuse_cross_minor_version(self):
        with mock.patch('aivyla_avatarforever_env.environment',
                        side_effect=[dict(version=[3, 11]), dict(version=[3, 12])]):
            with self.assertRaisesRegex(ValueError, 'versions must match'):
                link_dependencies('base', 'target')

    def test_docker_installs_complete_dependencies_before_runtime_checks(self):
        docker = Path(__file__).with_name('Dockerfile.global-avatarforever').read_text()
        self.assertNotIn('--no-deps', docker)
        self.assertIn('--constraint /opt/avatarforever/base-constraints.txt', docker)
        self.assertLess(docker.index('--constraints /opt/avatarforever/base-constraints.txt'),
                        docker.index('-m pip install --no-cache-dir', docker.index('uv venv')))
        self.assertLess(docker.index('-m pip check', docker.index('uv venv')),
                        docker.index('python /opt/aivyla-global/aivyla_avatarforever_smoke.py'))


class PayloadTests(unittest.TestCase):
    def test_default_three_variants_and_limits(self):
        settings = av.validate_payload(payload())
        self.assertEqual(settings['variants'], ['BASE', 'OMNI1', 'OMNI2'])
        self.assertEqual(settings['seconds'], 10)
        self.assertEqual(av.validate_payload(payload(seconds=40))['seconds'], 40)
        for value in [0, -1, 40.01, True, float('nan'), float('inf'), '10']:
            with self.assertRaises(ValueError, msg=str(value)):
                av.validate_payload(payload(seconds=value))

    def test_reject_mixed_graph_or_unknown_lora(self):
        for change in [dict(workflow={}), dict(variants=['BASE', 'BASE']),
                       dict(variants=['FACEID']), dict(variants='BASE'),
                       dict(variants=[]), dict(seed=True), dict(seed=2**32),
                       dict(width=768, height=768), dict(width=769),
                       dict(width=128, height=512)]:
            with self.assertRaises(ValueError, msg=str(change)):
                av.validate_payload(payload(**change))

    def test_no_production_endpoint_fallback(self):
        with self.assertRaisesRegex(ValueError, 'fallback is disabled'):
            resolve_endpoint(None, {'RUNPOD_ENDPOINT_ID': 'production-only'})
        self.assertEqual(resolve_endpoint(None, {'AIVYLA_GLOBAL_ENDPOINT_ID': 'global-test'}), 'global-test')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            resolve_endpoint(None, {'AIVYLA_GLOBAL_ENDPOINT_ID': 'one', 'RUNPOD_GLOBAL_ENDPOINT_ID': 'two'})

    def test_decode_only_base64_media(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'input')
            av.decode_media('data:audio/wav;base64,eA==', path)
            self.assertEqual(path.read_bytes(), b'x')
            for value in ['https://example.com/audio.wav', 'file:///etc/passwd', '', 'data:audio/wav,abc']:
                with self.assertRaises(ValueError):
                    av.decode_media(value, path)

    def test_frame_alignment_and_padding_duration(self):
        for seconds in [0.01, 1, 7.5, 10, 30, 40]:
            frames = av.frame_count(seconds)
            self.assertEqual((frames - 1) % 8, 0)
            self.assertGreaterEqual(frames / 25, seconds)
            self.assertLess(frames / 25 - seconds, 8 / 25 + 1e-6)

    @unittest.skipUnless(__import__('shutil').which('ffmpeg'), 'ffmpeg not installed')
    def test_real_media_fit_pad_and_short_audio(self):
        import struct
        import zlib
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        png = b'\x89PNG\r\n\x1a\n'
        png += chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 64, 8, 2, 0, 0, 0))
        png += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 32) * 64))
        png += chunk(b'IEND', b'')
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\0\0' * 11200)
        data = payload()
        data.update(image=base64.b64encode(png).decode(), audio=base64.b64encode(stream.getvalue()).decode())
        settings = av.validate_payload(data)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            av.prepare_media(data, settings, work)
            self.assertAlmostEqual(settings['seconds'], .7)
            actual = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                           'format=duration', '-of', 'default=nw=1:nk=1', str(work/'conditioning.wav')]))
            self.assertAlmostEqual(actual, settings['frames']/25, places=3)
            dimensions = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                          'stream=width,height', '-of', 'csv=p=0', str(work/'conditioning.png')], text=True)
            self.assertEqual(dimensions.strip(), '768,512')


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.s3 = FakeS3()
        self.stager = av.BundleStager(self.directory.name, client=self.s3)
        transfer = types.ModuleType('boto3.s3.transfer')
        transfer.TransferConfig = lambda **kw: kw
        modules = {'boto3': types.ModuleType('boto3'), 'boto3.s3': types.ModuleType('boto3.s3'),
                   'boto3.s3.transfer': transfer}
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)

    def add_bundle(self):
        self.s3.add('models/split_files/checkpoints/' + av.CHECKPOINT, b'checkpoint')
        for name in av.GEMMA_FILES:
            data = b'{}'
            if name == 'model.safetensors.index.json':
                data = json.dumps({'weight_map': {str(i): f'model-{i:05d}-of-00005.safetensors'
                                                  for i in range(1, 6)}}).encode()
            self.s3.add(f'models/text_encoders/{av.GEMMA}/{name}', data)

    def test_exact_17_files_include_tokenizers_and_metadata(self):
        self.add_bundle()
        jobs = self.stager.plan(False)
        self.assertEqual(self.stager.bucket, 'aivyla-models-v2')
        self.assertEqual(len(jobs), 17)
        paths = {dst.relative_to(self.stager.root).as_posix() for _, dst in jobs}
        self.assertIn(f'checkpoints/{av.GEMMA}/tokenizer.model', paths)
        self.assertIn(f'checkpoints/{av.GEMMA}/config.json', paths)

    def test_cached_repeat_does_not_download(self):
        self.add_bundle()
        first = self.stager.stage(False)
        self.assertEqual(first['downloaded'], 17)
        second = self.stager.stage(False)
        self.assertEqual(second['cached'], 17)
        self.assertEqual(len(self.s3.downloads), 17)

    def test_checksum_failure_never_publishes_cache(self):
        key = 'models/test.bin'
        self.s3.add(key, b'good')
        self.s3.corrupt = True
        path = self.stager.root / 'test.bin'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.stager.fetch((self.stager.head(key), path))
        self.assertFalse(path.exists())
        self.assertFalse(self.stager.receipt(path).exists())
        self.assertFalse(path.with_name('test.bin.part').exists())

    def test_changed_remote_object_never_publishes_cache(self):
        key = 'models/test.bin'
        self.s3.add(key, b'good')
        self.s3.change = True
        path = self.stager.root / 'test.bin'
        with self.assertRaisesRegex(ValueError, 'changed during download'):
            self.stager.fetch((self.stager.head(key), path))
        self.assertFalse(path.exists())

    def test_preflight_missing_omni_has_no_download(self):
        self.add_bundle()
        with self.assertRaises(FileNotFoundError):
            self.stager.stage(True)
        self.assertEqual(self.s3.downloads, [])

    def test_omni_fallback_and_split_priority(self):
        self.add_bundle()
        key = f'models/loras/{av.OMNINFT}'
        self.s3.add(key, b'fallback')
        self.assertEqual(self.stager.plan(True)[-1][0].key, key)
        split = f'models/split_files/loras/{av.OMNINFT}'
        self.s3.add(split, b'preferred')
        self.assertEqual(self.stager.plan(True)[-1][0].key, split)

    def test_insufficient_disk_fails_before_download(self):
        self.add_bundle()
        with mock.patch.object(av.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, 'local disk needs'):
                self.stager.stage(False)
        self.assertEqual(self.s3.downloads, [])


class LoraAuditTests(unittest.TestCase):
    def test_valid_pairs(self):
        self.assertEqual(audit_lora({'x.weight': (8, 16)},
                                   {'x.lora_A.weight': (2, 16), 'x.lora_B.weight': (8, 2)}), 1)

    def test_no_silent_missing_partial_or_wrong_shape(self):
        for tensors in [{}, {'x.lora_A.weight': (2, 16)},
                        {'x.alpha': ()},
                        {'z.lora_A.weight': (2, 16), 'z.lora_B.weight': (8, 2)},
                        {'x.lora_A.weight': (3, 16), 'x.lora_B.weight': (8, 2)},
                        {'x.lora_A.weight': (2, 20), 'x.lora_B.weight': (8, 2)}]:
            with self.assertRaises(ValueError):
                audit_lora({'x.weight': (8, 16)}, tensors)


class ComfyReadinessTests(unittest.TestCase):
    def test_real_http_cold_start_waits_until_queue_api_is_available(self):
        # Reserve a real localhost port, but listen only after the first actual
        # connection refusal: reproduces the ordering in the worker's logs.
        class QueueHandler(BaseHTTPRequestHandler):
            paths = []
            def do_GET(self):
                self.paths.append(self.path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"queue_running": [], "queue_pending": []}')
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), QueueHandler, bind_and_activate=False)
        server.server_bind()
        address = 'http://127.0.0.1:' + str(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        from urllib.request import urlopen
        from urllib.error import URLError
        attempts = []
        def probe(*args, **kwargs):
            try:
                response = urlopen(*args, **kwargs)
                attempts.append('ready')
                return response
            except URLError:
                attempts.append('refused')
                server.server_activate()
                thread.start()
                raise
        try:
            with mock.patch.object(av, 'COMFY_API', address), \
                    mock.patch.object(av, 'comfy_process_alive', return_value=True), \
                    mock.patch('urllib.request.urlopen', side_effect=probe):
                queue = av.wait_for_comfy_queue(timeout_seconds=2, poll_seconds=.01)
            self.assertEqual(attempts, ['refused', 'ready'])
            self.assertEqual(queue, {'queue_running': [], 'queue_pending': []})
            self.assertEqual(QueueHandler.paths, ['/queue'])
        finally:
            if thread.is_alive():
                server.shutdown()
                thread.join(timeout=2)
            server.server_close()

    def test_dead_comfy_process_fails_without_waiting_or_http(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'comfy.pid'
            child = subprocess.Popen([sys.executable, '-c', 'pass'])
            child.wait(timeout=5)
            pidfile.write_text(str(child.pid))
            with mock.patch.object(av, 'COMFY_PID_FILE', pidfile), \
                    mock.patch('urllib.request.urlopen') as request:
                with self.assertRaisesRegex(RuntimeError, 'process exited'):
                    av.wait_for_comfy_queue(timeout_seconds=2)
                request.assert_not_called()

    def test_unavailable_api_has_bounded_wait_when_pid_is_missing(self):
        from urllib.error import URLError
        with mock.patch.object(av, 'comfy_process_alive', return_value=None), \
                mock.patch('urllib.request.urlopen', side_effect=URLError('connection refused')):
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'did not become ready'):
                av.wait_for_comfy_queue(timeout_seconds=.05, poll_seconds=.01)
            self.assertLess(time.monotonic() - started, 1)

    def test_busy_comfy_queue_is_never_unloaded(self):
        requests = types.ModuleType('requests')
        requests.post = mock.Mock()
        with mock.patch.dict(sys.modules, {'requests': requests}), \
                mock.patch.object(av, 'wait_for_comfy_queue', return_value={
                    'queue_running': [['running-job']], 'queue_pending': []}):
            with self.assertRaisesRegex(RuntimeError, 'active/queued work'):
                av.free_comfy_gpu()
            requests.post.assert_not_called()

    def test_invalid_queue_is_rejected_before_unloading(self):
        response = mock.MagicMock()
        response.__enter__.return_value = io.BytesIO(b'{}')
        with mock.patch.object(av, 'comfy_process_alive', return_value=True), \
                mock.patch('urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'invalid response'):
                av.wait_for_comfy_queue(timeout_seconds=2)


class JobTests(unittest.TestCase):
    def test_build_import_from_unrelated_directory_uses_real_sibling_modules(self):
        # Build uses python -c + runpy from /, whereas runtime executes the
        # wrapper file directly. Only the latter adds its parent to sys.path.
        root = Path(__file__).resolve().parent
        preamble = '''import runpy, sys, types
sys.modules['runpod'] = types.ModuleType('runpod')
def validate_workflow_models(workflow):
    return None
def handler(job):
    return {}
original_run_path = runpy.run_path
def intercepted(path, **kwargs):
    if path == '/handler.py':
        return {'handler': handler}
    return original_run_path(path, **kwargs)
runpy.run_path = intercepted
'''
        command = ("scope=runpy.run_path(" + repr(str(root/'aivyla_global_r2_handler.py'))
                   + ", run_name='avatarforever_build_smoke'); "
                   + "assert callable(scope['handler']); print('PASS')")
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            old = subprocess.run([sys.executable, '-c', preamble+command], cwd=directory,
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(old.returncode, 0)
            self.assertIn("No module named 'aivyla_r2_stage'", old.stderr)
            fixed = preamble + 'sys.path.insert(0, ' + repr(str(root)) + ')\n' + command
            new = subprocess.run([sys.executable, '-c', fixed], cwd=directory,
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(new.returncode, 0, new.stderr)
            self.assertEqual(new.stdout.strip(), 'PASS')
        docker = (root/'Dockerfile.global-avatarforever').read_text()
        self.assertIn("sys.path.insert(0, '/opt/aivyla-global'); scope=runpy.run_path", docker)

    def fake_comparison(self, fail_on=None):
        with tempfile.TemporaryDirectory() as directory:
            transport = types.ModuleType('aivyla_output_upload')
            transport.check_transport_ready = mock.Mock()
            transport.upload_comfy_output = mock.Mock(side_effect=lambda job, filename, subfolder, kind, index:
                                                     {'filename': filename, 'type': 'aivyla_upload',
                                                      'data': f'{job}:{index}'})
            calls = []
            def media(data, settings, work):
                settings['frames'] = 257
                return work / 'audio'
            def inference(command, cwd, log, timeout):
                raw = Path(command[command.index('--output-path')+1])
                name = raw.name.split('-')[0]
                calls.append((name, command))
                if name == fail_on:
                    raise RuntimeError('synthetic inference failure')
                report = Path(command[command.index('--aivyla-report')+1])
                report.write_text(json.dumps({'lora_audit': [], 'fusion_calls': [], 'gpu_peak_allocated_gib': 1}))
            stager = mock.Mock()
            stager.stage.return_value = {'files': 18}
            with mock.patch.dict(sys.modules, {'aivyla_output_upload': transport}), \
                    mock.patch.object(av, 'COMFY_OUTPUT_ROOT', Path(directory)), \
                    mock.patch.object(av, 'runtime_preflight', return_value=(Path('/python'), Path('/repo'))), \
                    mock.patch.object(av, 'prepare_media', side_effect=media), \
                    mock.patch.object(av, 'free_comfy_gpu', return_value=80000), \
                    mock.patch.object(av, 'progress'), \
                    mock.patch.object(av, 'BundleStager', return_value=stager), \
                    mock.patch.object(av, 'run_logged', side_effect=inference), \
                    mock.patch.object(av.subprocess, 'run'):
                data = payload()
                result = av._run_comparison({'id': 'test-job'}, data, av.validate_payload(data))
            return result, calls, transport

    def test_one_request_three_outputs_separate_processes(self):
        result, calls, transport = self.fake_comparison()
        self.assertEqual(result['status'], 'success')
        self.assertEqual([item['variant'] for item in result['images']], ['BASE', 'OMNI1', 'OMNI2'])
        self.assertEqual(transport.upload_comfy_output.call_count, 3)
        self.assertNotIn('--lora', calls[0][1])
        self.assertEqual(calls[1][1][-1], '1.0')
        self.assertEqual(calls[2][1][-1], '2.0')

    def test_failed_variant_preserves_delivered_baseline(self):
        result, calls, transport = self.fake_comparison(fail_on='OMNI1')
        self.assertIn('OMNI1 failed', result['error'])
        self.assertEqual(len(result['images']), 1)
        self.assertEqual(result['images'][0]['variant'], 'BASE')
        self.assertEqual(len(calls), 2)
        self.assertEqual(transport.upload_comfy_output.call_count, 1)

    def test_disabled_route_does_not_start_runtime(self):
        with mock.patch.dict(os.environ, {'AIVYLA_AVATARFOREVER_ENABLED': '0'}), \
                mock.patch.object(av, 'runtime_preflight') as runtime:
            self.assertIn('disabled', av.run_comparison({'id': 'job'}, payload())['error'])
            runtime.assert_not_called()

    def test_capabilities_does_not_stage_or_generate(self):
        with mock.patch.dict(os.environ, {'AIVYLA_AVATARFOREVER_ENABLED': '1'}), \
                mock.patch.object(av, 'runtime_preflight'), \
                mock.patch.object(av, 'wait_for_comfy_queue') as ready, \
                mock.patch.object(av, 'BundleStager') as staging:
            result = av.run_comparison({'id': 'job'}, {'action': 'capabilities'})
            self.assertEqual(result['bucket'], 'aivyla-models-v2')
            self.assertFalse(result['generation_started'])
            self.assertTrue(result['comfy_api_ready'])
            ready.assert_called_once_with()
            staging.assert_not_called()

    def test_concurrent_comparison_refused(self):
        av.JOB_LOCK.acquire()
        try:
            with mock.patch.dict(os.environ, {'AIVYLA_AVATARFOREVER_ENABLED': '1'}):
                with self.assertRaisesRegex(RuntimeError, 'Another'):
                    av.run_comparison({'id': 'job'}, payload())
        finally:
            av.JOB_LOCK.release()

    def test_failure_releases_lock(self):
        with mock.patch.dict(os.environ, {'AIVYLA_AVATARFOREVER_ENABLED': '1'}), \
                mock.patch.object(av, '_run_comparison', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                av.run_comparison({'id': 'job'}, payload())
        self.assertFalse(av.JOB_LOCK.locked())

    def test_timeout_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock(pid=99999)
            process.wait.side_effect = [subprocess.TimeoutExpired('test', 1), 0]
            with mock.patch.object(av.subprocess, 'Popen', return_value=process), \
                    mock.patch.object(os, 'killpg') as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    av.run_logged(['test'], cwd=directory, log=Path(directory, 'log'), timeout=1)
                kill.assert_called_once()

    def test_dispatch_keeps_comfy_requests_on_original_handler(self):
        def original(job):
            return {'legacy': True}
        original.__globals__['validate_workflow_models'] = lambda workflow: None
        stage = types.ModuleType('aivyla_r2_stage')
        stage.stage_workflow = mock.Mock(return_value={})
        runpod = types.ModuleType('runpod')
        with mock.patch.dict(sys.modules, {'runpod': runpod, 'aivyla_r2_stage': stage}), \
                mock.patch.object(runpy, 'run_path', return_value={'handler': original}):
            spec = importlib.util.spec_from_file_location('test_global_avatar_dispatch',
                                                         Path(__file__).with_name('aivyla_global_r2_handler.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with mock.patch.object(av, 'run_comparison', return_value={'avatar': True}) as compare, \
                mock.patch.dict(os.environ, {'AIVYLA_AVATARFOREVER_ENABLED': '1'}):
            self.assertEqual(module.handler({'id': 'job', 'input': payload()}), {'avatar': True})
            self.assertEqual(module.handler({'input': {'workflow': {}}}), {'legacy': True})
            compare.assert_called_once()
            stage.stage_workflow.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
