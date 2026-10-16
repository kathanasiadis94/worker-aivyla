"""CPU-only R2 catalog, lazy staging and custom-node security regression tests."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# These are FakeS3 unit tests: boto3 is provided by the Docker image, not
# required on the developer's Mac. Stub only its TransferConfig for tests.
try:
    from boto3.s3.transfer import TransferConfig as _TransferConfig
except ImportError:
    boto3_stub = types.ModuleType('boto3')
    s3_stub = types.ModuleType('boto3.s3')
    transfer_stub = types.ModuleType('boto3.s3.transfer')
    class _TransferConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    transfer_stub.TransferConfig = _TransferConfig
    boto3_stub.s3 = s3_stub
    s3_stub.transfer = transfer_stub
    sys.modules['boto3'] = boto3_stub
    sys.modules['boto3.s3'] = s3_stub
    sys.modules['boto3.s3.transfer'] = transfer_stub

# A minimal stand-in for the original v3 module to exercise R2-only changes.
legacy = types.ModuleType('aivyla_global_cache')
legacy._model_filename = lambda s: isinstance(s, str) and s.endswith(('.safetensors','.gguf'))
legacy._possible_categories = lambda node, field: ['loras'] if field=='lora_name' else ['diffusion_models']
legacy._find_source = lambda filename, categories: None
legacy._get_source_fingerprint = lambda src: {'source':str(src),'source_size':src.stat().st_size, 'source_mtime_ns':src.stat().st_mtime_ns}
legacy._validate_safetensors = lambda f: valid_sf(f)
legacy.KNOWN = {'LoraLoader':('loras',('lora_name',)),'UNETLoader':('diffusion_models',('unet_name',))}
legacy.MAX_REFERENCES = 80
sys.modules['aivyla_global_cache'] = legacy
from aivyla_r2_registry import Registry, parse_model, safe_relative
import aivyla_r2_stage as stage
import aivyla_r2_nodes as node_stage


def sf(payload=b'X'*128):
    header = json.dumps({'x':{'dtype':'U8', 'shape':[len(payload)], 'data_offsets':[0,len(payload)]}}).encode()
    return struct.pack('<Q',len(header))+header+payload


def valid_sf(f):
    b=Path(f).read_bytes()
    n=struct.unpack('<Q',b[:8])[0]
    j=json.loads(b[8:8+n])
    assert len(b)==n+8+j['x']['data_offsets'][1]


class FakeErrors:
    class NoSuchKey(Exception): pass


class FakeS3:
    exceptions=FakeErrors
    def __init__(self):
        self.objects={}
        self.calls=[]
    def add(self,key,data):
        self.objects[key]=bytes(data)
    def list_objects_v2(self,**kwargs):
        key=kwargs['Prefix']
        records=[{'Key':x,'Size':len(y),'ETag':'"'+hashlib.md5(y).hexdigest()+'"'}
                 for x,y in self.objects.items() if x.startswith(key)]
        return {'Contents':records}
    def get_object(self,**kwargs):
        try: b=self.objects[kwargs['Key']]
        except KeyError: raise FakeErrors.NoSuchKey()
        return {'Body':io.BytesIO(b)}
    def head_object(self,**kwargs):
        b=self.objects[kwargs['Key']]
        return {'ContentLength':len(b),'ETag':'"'+hashlib.md5(b).hexdigest()+'"'}
    def download_file(self,bucket,key,path,Config=None):
        self.calls.append(key)
        Path(path).write_bytes(self.objects[key])


class TestRegistry(unittest.TestCase):
    def test_model_keys_and_safety(self):
        self.assertFalse(safe_relative('../escape'))
        self.assertFalse(safe_relative('loras/../escape'))
        self.assertIsNone(parse_model('models/../bad.safetensors'))
        self.assertIsNone(parse_model('models/quarantine/foo.safetensors'))
        self.assertEqual(parse_model('models/split_files/diffusion_models/model.safetensors').filename,'model.safetensors')
        self.assertEqual(parse_model('models/loras/minimax_h3/test.safetensors').filename,'minimax_h3/test.safetensors')

    def test_dynamic_listing_refresh_new_upload(self):
        fake=FakeS3()
        fake.add('models/loras/first.safetensors',sf())
        r=Registry(fake, ttl=60)
        self.assertEqual(len(r.catalog()),1)
        fake.add('models/loras/new-lora.safetensors',sf())
        self.assertEqual(len(r.catalog()),1)  # bounded TTL, no spam
        self.assertEqual(len(r.refresh(force=True)),2)  # new upload discovered
        self.assertIsNotNone(r.find('new-lora.safetensors',['loras']))

    def test_manifest_rejects_bad_digest_and_unsafe_paths(self):
        fake=FakeS3()
        fake.add('nodes/registry.json', json.dumps({'schema':1,'nodes':[{
            'name':'Node', 'key':'nodes/packages/Node/v1.tar.gz', 'sha256':'bad','enabled':True}]}).encode())
        r=Registry(fake)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            r.nodes()


class TestStaging(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='aivyla-r2-stage-',dir='/tmp')
        self.addCleanup(t.cleanup)
        self.tmp=Path(t.name)
        self.fake=FakeS3()
        stage._registry=Registry(self.fake, ttl=0)
        stage.CACHE=self.tmp/'cache'
        stage.R2_ENABLED=True
        stage.R2_FALLBACK=False
        stage.RESERVE_GIB=0
        self.payload=sf()
        self.fake.add('models/loras/NewNodeLoRA.safetensors',self.payload)
        self.workflow={'15':{'class_type':'LoraLoader','inputs':{'lora_name':'NewNodeLoRA.safetensors'}}}

    def test_first_download_then_local_cache_hit(self):
        result=stage.stage_workflow(self.workflow)
        self.assertEqual(result['r2'],1)
        p=stage.CACHE/'loras/NewNodeLoRA.safetensors'
        self.assertEqual(p.read_bytes(),self.payload)
        result=stage.stage_workflow(self.workflow)
        self.assertEqual(result['hit'],1)
        self.assertEqual(self.fake.calls,['models/loras/NewNodeLoRA.safetensors'])

    def test_receipt_sha_mismatch_refuses_corrupt_model(self):
        key='models/loras/NewNodeLoRA.safetensors'
        fake_receipt={'bytes':len(self.payload),'etag':hashlib.md5(self.payload).hexdigest(),
                      'sha256':'0'*64}
        self.fake.add('_aivyla/receipts/'+key+'.json',json.dumps(fake_receipt).encode())
        with self.assertRaisesRegex(stage.R2IntegrityError,'SHA256 mismatch'):
            stage.stage_workflow(self.workflow)
        self.assertFalse((stage.CACHE/'loras/NewNodeLoRA.safetensors').exists())

    def test_tigris_copy_accepts_changed_etag_with_trusted_sha256(self):
        key='models/loras/NewNodeLoRA.safetensors'
        receipt={'bytes':len(self.payload),'etag':'etag-from-original-r2-object',
                 'sha256':hashlib.sha256(self.payload).hexdigest()}
        self.fake.add('_aivyla/receipts/'+key+'.json',json.dumps(receipt).encode())
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENDPOINT':'https://t3.storage.dev'}):
            result=stage.stage_workflow(self.workflow)
        self.assertEqual(result['r2'],1)
        self.assertEqual((stage.CACHE/'loras/NewNodeLoRA.safetensors').read_bytes(),self.payload)

    def test_cloudflare_still_rejects_changed_receipt_etag(self):
        key='models/loras/NewNodeLoRA.safetensors'
        receipt={'bytes':len(self.payload),'etag':'different',
                 'sha256':hashlib.sha256(self.payload).hexdigest()}
        self.fake.add('_aivyla/receipts/'+key+'.json',json.dumps(receipt).encode())
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENDPOINT':'https://example.r2.cloudflarestorage.com'}):
            with self.assertRaisesRegex(ValueError,'receipt ETag mismatch'):
                stage.stage_workflow(self.workflow)

    def test_tigris_still_rejects_wrong_sha256(self):
        key='models/loras/NewNodeLoRA.safetensors'
        receipt={'bytes':len(self.payload),'etag':'old-r2-etag', 'sha256':'0'*64}
        self.fake.add('_aivyla/receipts/'+key+'.json',json.dumps(receipt).encode())
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENDPOINT':'https://t3.storage.dev'}):
            with self.assertRaisesRegex(stage.R2IntegrityError,'SHA256 mismatch'):
                stage.stage_workflow(self.workflow)

    def test_changed_remote_etag_redownloads(self):
        stage.stage_workflow(self.workflow)
        self.fake.add('models/loras/NewNodeLoRA.safetensors',sf(b'Y'*128))
        result=stage.stage_workflow(self.workflow)
        self.assertEqual(result['r2'],1)
        self.assertEqual(len(self.fake.calls),2)


class TestNodes(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='aivyla-r2-nodes-',dir='/tmp')
        self.addCleanup(t.cleanup)
        self.tmp=Path(t.name)
        self.local=self.tmp/'local'
        self.comfy=self.tmp/'custom_nodes'
        self.local.mkdir(); self.comfy.mkdir()
        self.fake=FakeS3()
        self.registry=Registry(self.fake)
        self.bundle=self.make_tar('../none')

    def make_tar(self, extra=None):
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w:gz') as tf:
            b=b'NODE_CLASS_MAPPINGS = {}\n'
            info=tarfile.TarInfo('MyNode/__init__.py')
            info.size=len(b)
            tf.addfile(info,io.BytesIO(b))
            if extra and extra!='../none':
                b=b'BAD'
                info=tarfile.TarInfo(extra)
                info.size=len(b)
                tf.addfile(info,io.BytesIO(b))
        return out.getvalue()

    def load_manifest(self,blob,enabled):
        key='nodes/packages/MyNode/v1.tar.gz'
        self.fake.add(key,blob)
        self.fake.add('nodes/registry.json',json.dumps({'schema':1,'nodes':[
            {'name':'MyNode','key':key,'sha256':hashlib.sha256(blob).hexdigest(),
             'version':'v1','enabled':enabled}]}).encode())

    def test_enabled_node_installs_and_links(self):
        self.load_manifest(self.bundle,True)
        result=node_stage.install_enabled_nodes(self.registry,local=self.local,comfy=self.comfy)
        self.assertEqual(result['installed'],['MyNode'])
        self.assertTrue((self.comfy/'MyNode').is_symlink())
        self.assertIn('NODE_CLASS', (self.comfy/'MyNode/__init__.py').read_text())

    def test_disabled_node_is_cataloged_not_executed(self):
        self.load_manifest(self.bundle,False)
        result=node_stage.install_enabled_nodes(self.registry,local=self.local,comfy=self.comfy)
        self.assertEqual(result['listed'],1)
        self.assertFalse((self.comfy/'MyNode').exists())

    def test_tar_traversal_denied(self):
        blob=self.make_tar('MyNode/../../escaped.py')
        self.load_manifest(blob,True)
        with self.assertRaisesRegex(ValueError,'Unsafe node archive'):
            node_stage.install_enabled_nodes(self.registry,local=self.local,comfy=self.comfy)
        self.assertFalse((self.tmp/'escaped.py').exists())

    def test_existing_node_name_collision(self):
        self.load_manifest(self.bundle,True)
        (self.local/'MyNode').mkdir()
        with self.assertRaisesRegex(FileExistsError,'collides'):
            node_stage.install_enabled_nodes(self.registry,local=self.local,comfy=self.comfy)


class TestV4Integration(unittest.TestCase):
    def test_v4_startup_patch_is_idempotent(self):
        import importlib.util
        fake=types.ModuleType('aivyla_global_start')
        fake.patch_start=lambda: None
        fake.START=Path('/dev/null')
        with mock.patch.dict(sys.modules, {'aivyla_global_start':fake}):
            path=HERE/'aivyla_global_r2_start.py'
            spec=importlib.util.spec_from_file_location('r2_start_test',path)
            start=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(start)
        with tempfile.TemporaryDirectory(prefix='aivyla-r2-boot-') as td:
            f=Path(td)/'start.sh'
            f.write_text('#!/bin/bash\n# aivyla_global_local_cache:\n'
                         'python -u /opt/aivyla-global/aivyla_global_handler.py\n'
                         'python -u /opt/aivyla-global/aivyla_global_handler.py\n')
            old=start.v3.START
            try:
                start.v3.START=f
                start.prepare_handler()
                first=f.read_text()
                start.prepare_handler()
                self.assertEqual(f.read_text(),first)
                self.assertEqual(first.count('aivyla_global_r2_handler.py'),2)
            finally:
                start.v3.START=old

    def test_r2_model_hit_never_stats_remote_volume(self):
        fake=FakeS3()
        fake.add('models/loras/online.safetensors',sf())
        with tempfile.TemporaryDirectory(prefix='aivyla-no-fuse-',dir='/tmp') as temp:
            stage._registry=Registry(fake,ttl=0)
            with mock.patch.object(stage, 'CACHE', Path(temp)/'cache'), \
                    mock.patch.object(stage,'R2_ENABLED',True), \
                    mock.patch.object(stage,'RESERVE_GIB',0), \
                    mock.patch.object(legacy,'_find_source',side_effect=AssertionError('Unexpected Global Volume stat')):
                result=stage.stage_workflow({'1':{'class_type':'LoraLoader','inputs':{'lora_name':'online.safetensors'}}})
                self.assertEqual(result['r2'],1)

    def test_zero_size_global_volume_fingerprint_still_streams(self):
        with tempfile.TemporaryDirectory(prefix='aivyla-fuse-zero-',dir='/tmp') as td:
            root=Path(td)
            source=root/'source.safetensors'
            source.write_bytes(sf(b'A'*2048))
            dst=root/'out.safetensors'
            with mock.patch.object(legacy,'_get_source_fingerprint',return_value={
                    'source':str(source),'source_size':0,'source_mtime_ns':source.stat().st_mtime_ns}):
                self.assertEqual(stage._copy_volume(source,dst),'global')
                self.assertEqual(dst.read_bytes(),source.read_bytes())
                self.assertEqual(stage._copy_volume(source,dst),'hit')

    def test_network_download_failure_fallback_when_source_exists(self):
        fake=FakeS3()
        fake.add('models/loras/fallback.safetensors',sf(b'B'*90))
        def fail_download(*args,**kwargs):
            raise ConnectionError('network down')
        fake.download_file=fail_download
        with tempfile.TemporaryDirectory(prefix='aivyla-fallback-',dir='/tmp') as td:
            root=Path(td)
            source=root/'volume.safetensors'
            source.write_bytes(sf(b'C'*90))
            stage._registry=Registry(fake,ttl=0)
            with mock.patch.object(stage,'CACHE',root/'cache'), \
                mock.patch.object(stage,'R2_ENABLED',True), \
                mock.patch.object(stage,'R2_FALLBACK',True), \
                mock.patch.object(stage,'RESERVE_GIB',0), \
                mock.patch.object(legacy,'_find_source',return_value=('loras',source)):
                result=stage.stage_workflow({'1':{'class_type':'LoraLoader',
                            'inputs':{'lora_name':'fallback.safetensors'}}})
                self.assertEqual(result['global'],1)
                self.assertEqual((root/'cache/loras/fallback.safetensors').read_bytes(),source.read_bytes())

    def test_node_requirements_denied_without_approval(self):
        with tempfile.TemporaryDirectory(prefix='aivyla-nodes-req-',dir='/tmp') as td:
            root=Path(td)
            (root/'requirements.txt').write_text('torch>=2.1\n')
            with mock.patch.object(node_stage,'ALLOW_PIP',False):
                with self.assertRaisesRegex(RuntimeError,'requires dependency approval'):
                    node_stage._install_requirements(root)


class TestV42GraphPruning(unittest.TestCase):
    """Disconnected model branches must not cause R2 download when outputs are known."""
    def setUp(self):
        self.fake = FakeS3()
        self.fake.add('models/diffusion_models/huge.safetensors', sf())
        self.fake.add('models/loras/used.safetensors', sf())
        stage._registry = Registry(self.fake, ttl=0)
        temp = tempfile.TemporaryDirectory(prefix='aivyla-r2-prune-', dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        self.old = (stage.CACHE, stage.R2_ENABLED, stage.RESERVE_GIB, stage.PRUNE_UNUSED)
        self.addCleanup(lambda: self._restore())
        stage.CACHE = self.tmp/'cache'
        stage.R2_ENABLED = True
        stage.RESERVE_GIB = 0
        stage.PRUNE_UNUSED = True

    def _restore(self):
        stage.CACHE, stage.R2_ENABLED, stage.RESERVE_GIB, stage.PRUNE_UNUSED = self.old

    def test_exact_smoke_unused_unet_is_not_fetched(self):
        workflow = {
            '1': {'class_type':'UNETLoader', 'inputs':{'unet_name':'huge.safetensors'}},
            '2': {'class_type':'EmptyImage', 'inputs':{'width':128,'height':128}},
            '3': {'class_type':'SaveImage', 'inputs':{'images':['2',0]}},
        }
        result = stage.stage_workflow(workflow)
        self.assertEqual(result['models'],0)
        self.assertEqual(result['pruned_nodes'],1)
        self.assertEqual(self.fake.calls,[])
        self.assertEqual(len(workflow),3, 'never mutate original workflow')

    def test_connected_model_still_downloads(self):
        workflow = {
            '1':{'class_type':'LoraLoader', 'inputs':{'lora_name':'used.safetensors'}},
            '2':{'class_type':'SaveImage', 'inputs':{'images':['1',0]}},
        }
        result = stage.stage_workflow(workflow)
        self.assertEqual(result['r2'],1)
        self.assertEqual(result['pruned_nodes'],0)
        self.assertEqual(self.fake.calls,['models/loras/used.safetensors'])

    def test_unknown_custom_output_branch_is_preserved(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'MyCustomVideoSaver','inputs':{'frames':['1',0]}},
            '3':{'class_type':'EmptyImage','inputs':{}},
            '4':{'class_type':'SaveImage','inputs':{'images':['3',0]}},
        }
        result = stage.stage_workflow(workflow)
        self.assertEqual(result['r2'],1)
        self.assertEqual(result['pruned_nodes'],0)

    def test_unknown_only_outputs_fall_back_to_full_graph(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'MyCustomVideoSaver','inputs':{}},
        }
        selected, pruned = stage._active_workflow(workflow)
        self.assertEqual(pruned,0)
        self.assertEqual(selected, workflow)

    def test_disabled_pruning_preserves_old_behavior(self):
        stage.PRUNE_UNUSED = False
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'SaveImage','inputs':{'images':['3',0]}},
            '3':{'class_type':'EmptyImage','inputs':{}},
        }
        result = stage.stage_workflow(workflow)
        self.assertEqual(result['r2'],1)
        self.assertEqual(result['pruned_nodes'],0)

    def test_nested_custom_links_keep_model_dependency(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'SaveImage','inputs':{'images':{'groups':[['1',0]]}}},
        }
        selected, pruned = stage._active_workflow(workflow)
        self.assertEqual(pruned,0)
        self.assertIn('1',selected)

    def test_nested_missing_link_disables_pruning(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'SaveImage','inputs':{'images':{'groups':[['99',0]]}}},
        }
        selected, pruned = stage._active_workflow(workflow)
        self.assertEqual(pruned,0)
        self.assertEqual(selected,workflow)

    def test_missing_link_preserves_full_graph(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'SaveImage','inputs':{'images':['99',0]}},
        }
        selected, pruned = stage._active_workflow(workflow)
        self.assertEqual(pruned,0)
        self.assertEqual(selected,workflow)

    def test_disconnected_non_output_sampler_chain_removed(self):
        workflow = {
            '1':{'class_type':'UNETLoader','inputs':{'unet_name':'huge.safetensors'}},
            '2':{'class_type':'KSampler','inputs':{'model':['1',0]}},
            '3':{'class_type':'EmptyImage','inputs':{}},
            '4':{'class_type':'SaveImage','inputs':{'images':['3',0]}},
        }
        selected, pruned = stage._active_workflow(workflow)
        self.assertEqual(pruned,2)
        self.assertEqual(list(selected),['3','4'])

    def test_metrics_have_nonnegative_times(self):
        workflow = {'1':{'class_type':'LoraLoader','inputs':{'lora_name':'used.safetensors'}},
                    '2':{'class_type':'SaveImage','inputs':{'images':['1',0]}}}
        result = stage.stage_workflow(workflow)
        self.assertGreaterEqual(result['total_s'],0)
        self.assertGreaterEqual(result['discovery_s'],0)


class TestV42FastPreflight(unittest.TestCase):
    def test_skip_only_when_r2_enabled(self):
        """Load the wrapper with a fake handler; never import production code."""
        import importlib.util
        scope = {}
        source = (
            "def validate_workflow_models(workflow):\n"
            "    return 'original'\n"
            "def handler(job):\n"
            "    return validate_workflow_models(job['input']['workflow'])\n"
        )
        exec(source, scope)
        base = scope['handler']
        fake_runpod = types.ModuleType('runpod')
        with mock.patch.dict(sys.modules, {'runpod':fake_runpod}), \
             mock.patch('runpy.run_path',return_value={'handler':base}):
            spec=importlib.util.spec_from_file_location(
                'aivyla_r2_handler_v42_test', HERE/'aivyla_global_r2_handler.py')
            mod=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        payload={'input':{'workflow':{}}}
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENABLED':'1','AIVYLA_R2_SKIP_OBJECT_INFO':'1'}):
            self.assertIsNone(base(payload))
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENABLED':'0','AIVYLA_R2_SKIP_OBJECT_INFO':'1'}):
            self.assertEqual(base(payload),'original')
        with mock.patch.dict(os.environ, {'AIVYLA_R2_ENABLED':'1','AIVYLA_R2_SKIP_OBJECT_INFO':'0'}):
            self.assertEqual(base(payload),'original')


if __name__=='__main__':
    unittest.main(verbosity=2)
