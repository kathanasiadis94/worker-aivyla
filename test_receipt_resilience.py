import io, sys, json, tempfile, types, hashlib, os, importlib.util
from pathlib import Path
from unittest import mock

stub=types.ModuleType('aivyla_r2_registry');stub.Registry=object;stub.env_int=lambda *a:None
sys.modules['aivyla_r2_registry']=stub
legacy=types.ModuleType('aivyla_global_cache');sys.modules['aivyla_global_cache']=legacy
spec=importlib.util.spec_from_file_location('stagefix',str(Path(__file__).with_name('aivyla_r2_stage.py')))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
from botocore.exceptions import ResponseStreamingError
key='models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors'
sha=hashlib.sha256(b'MODEL').hexdigest()
raw=json.dumps({'bytes':777,'etag':'old-etag','sha256':sha}).encode()
class FakeErrors:
    class NoSuchKey(Exception): pass
class FakeClient:
    exceptions=FakeErrors
    def __init__(self,fail=0): self.fail=fail;self.calls=[];self.managed_calls=0
    def get_object(self,**kw):
        self.calls.append(kw)
        if len(self.calls)<=self.fail: raise ResponseStreamingError(error=RuntimeError('IncompleteRead(0 bytes read)'))
        return {'Body':io.BytesIO(raw),'ContentLength':len(raw)}
    def download_fileobj(self,b,k,sink,Config=None):
        self.managed_calls+=1
        sink.write(raw)

with mock.patch.dict(os.environ,{'AIVYLA_TRUSTED_RECEIPTS_PATH':'/tmp/__does_not_exist__','AIVYLA_R2_ENDPOINT':'https://t3.storage.dev'}),mock.patch.object(mod.time,'sleep',lambda n:None):
    c=FakeClient();assert mod._receipt_sha(c,'aivyla-models-usa',key,777,'tigris-etag')==sha;assert len(c.calls)==1
    print('PASS regular GET / digest validation')
    c=FakeClient(fail=2);assert mod._receipt_sha(c,'aivyla-models-usa',key,777,'tigris-etag')==sha
    assert len(c.calls)==3 and c.calls[-1]['Range']=='bytes=0-8192'
    print('PASS range fallback')
    c=FakeClient(fail=4);assert mod._receipt_sha(c,'aivyla-models-usa',key,777,'tigris-etag')==sha
    assert c.managed_calls==1
    print('PASS managed transfer fallback')
    c=FakeClient(fail=4)
    def fail(*a,**kw): raise ResponseStreamingError(error=RuntimeError('broken stream'))
    c.download_fileobj=fail
    try:mod._receipt_sha(c,'aivyla-models-usa',key,777,'x');assert False
    except OSError: print('PASS fail closed after all read paths fail')
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'trusted.json';p.write_text(json.dumps({'schema':1,'bucket':'aivyla-models-usa','models':{key:{'bytes':777,'sha256':sha}}}))
    with mock.patch.dict(os.environ,{'AIVYLA_TRUSTED_RECEIPTS_PATH':str(p)}):
        c=FakeClient(fail=999)
        assert mod._receipt_sha(c,'aivyla-models-usa',key,777,'whatever')==sha and c.calls==[]
        print('PASS pinned trusted manifest completely bypasses broken receipt GET')
        try:mod._receipt_sha(c,'aivyla-models-usa',key,778,'whatever');assert False
        except mod.R2IntegrityError:print('PASS pinned size mismatch rejected')
print('ALL TESTS PASSED')
