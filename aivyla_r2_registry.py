"""R2 catalog for Aivyla Global. Read-only; no fixed model/node allowlist.

Only the trusted backend/worker may use S3 credentials.  Never expose them to UI.
Model keys: models/<category>/<filename> or models/split_files/<category>/<filename>.
Node keys: nodes/packages/<node>/<version>.tar.gz; activation: nodes/registry.json.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import PurePosixPath
import re
import threading
import time

_MODEL_SUFFIXES = ('.safetensors', '.sft', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx')
_CATEGORIES = {
    'checkpoints', 'diffusion_models', 'text_encoders', 'vae', 'loras',
    'latent_upscale_models', 'upscale_models', 'clip_vision', 'embeddings',
    'controlnet', 'model_patches', 'audio_encoders', 'refmods', 'h3_adaln',
    'ipadapter',
}


def safe_relative(value: str) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or '\\' in value or '\x00' in value or '://' in value:
        return False
    p = PurePosixPath(value)
    return not p.is_absolute() and all(x not in ('', '.', '..') for x in value.split('/'))


@dataclass(frozen=True)
class Model:
    key: str
    category: str
    filename: str
    bytes: int
    etag: str


def parse_model(key: str, prefix='models/') -> Model | None:
    if not key.startswith(prefix):
        return None
    path = key[len(prefix):]
    if not safe_relative(path):
        return None
    parts = path.split('/')
    # Preserve legacy split_files priority and the existing upload naming scheme.
    if parts[0] == 'split_files':
        parts = parts[1:]
    if len(parts) < 2 or parts[0] not in _CATEGORIES:
        return None
    filename = '/'.join(parts[1:])
    if not filename.lower().endswith(_MODEL_SUFFIXES):
        return None
    return Model(key=key, category=parts[0], filename=filename, bytes=0, etag='')


class Registry:
    def __init__(self, client=None, *, bucket=None, prefix=None, ttl=None):
        self.bucket = bucket or os.getenv('AIVYLA_R2_BUCKET', 'aivyla-models')
        self.prefix = prefix or os.getenv('AIVYLA_R2_MODEL_PREFIX', 'models/')
        self.ttl = float(ttl if ttl is not None else os.getenv('AIVYLA_R2_CATALOG_TTL', '60'))
        if not safe_relative(self.prefix.rstrip('/')) or not self.prefix.endswith('/'):
            raise ValueError('Unsafe R2 model prefix')
        self.client = client or self._client()
        self._models = {}
        self._stamp = 0
        self._lock = threading.RLock()

    @staticmethod
    def _client():
        import boto3
        from botocore.config import Config
        endpoint = os.getenv('AIVYLA_R2_ENDPOINT', '')
        if not (endpoint.startswith('https://') and endpoint.endswith('.r2.cloudflarestorage.com')):
            raise ValueError('AIVYLA_R2_ENDPOINT must be a Cloudflare R2 S3 endpoint')
        # Credentials come ONLY from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (secrets).
        return boto3.client('s3', endpoint_url=endpoint, region_name='auto',
                            config=Config(signature_version='s3v4',
                                          max_pool_connections=24,
                                          retries={'max_attempts': 3}))

    def refresh(self, force=False):
        with self._lock:
            if not force and self._stamp and time.monotonic() - self._stamp < self.ttl:
                return self._models
            discovered = {}
            marker = None
            while True:
                kwargs = {'Bucket': self.bucket, 'Prefix': self.prefix, 'MaxKeys': 1000}
                if marker:
                    kwargs['ContinuationToken'] = marker
                page = self.client.list_objects_v2(**kwargs)
                for obj in page.get('Contents', []):
                    item = parse_model(obj['Key'], self.prefix)
                    if not item or int(obj.get('Size', 0)) < 8:
                        continue
                    item = Model(item.key, item.category, item.filename,
                                 int(obj['Size']), obj.get('ETag', '').strip('"'))
                    identity = item.category, item.filename
                    # split_files previously wins over ComfyUI/models where names overlap
                    if identity not in discovered or '/split_files/' in item.key:
                        discovered[identity] = item
                marker = page.get('NextContinuationToken')
                if not marker:
                    break
            self._models = discovered
            self._stamp = time.monotonic()
            return self._models

    def find(self, filename, categories):
        models = self.refresh()
        for category in categories:
            item = models.get((category, filename))
            if item:
                return item
        return None

    def catalog(self):
        return [{'category': x.category, 'filename': x.filename, 'bytes': x.bytes,
                 'source': 'r2'} for x in sorted(self.refresh().values(), key=lambda i: (i.category, i.filename))]

    def nodes(self):
        """Controlled code execution: only enabled, sha256-pinned packages in manifest."""
        key = os.getenv('AIVYLA_R2_NODES_MANIFEST', 'nodes/registry.json')
        if not safe_relative(key):
            raise ValueError('Unsafe node manifest key')
        try:
            blob = self.client.get_object(Bucket=self.bucket, Key=key)['Body'].read(1024 * 1024 + 1)
        except self.client.exceptions.NoSuchKey:
            return []
        if len(blob) > 1024 * 1024:
            raise ValueError('Node registry too large')
        manifest = json.loads(blob)
        if manifest.get('schema') != 1 or not isinstance(manifest.get('nodes'), list):
            raise ValueError('Invalid nodes/registry.json schema')
        names = set()
        entries = []
        for item in manifest['nodes']:
            if not isinstance(item, dict):
                raise ValueError('Invalid node registry entry')
            name = item.get('name')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,90}', name):
                raise ValueError('Unsafe node name')
            if name in names:
                raise ValueError('Duplicate node name')
            names.add(name)
            key = item.get('key')
            digest = item.get('sha256', '')
            if not safe_relative(key) or not key.startswith('nodes/packages/') or not key.endswith('.tar.gz'):
                raise ValueError(f'Invalid node archive path: {name}')
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError(f'Node must specify SHA256: {name}')
            entries.append({'name': name, 'key': key, 'sha256': digest,
                            'enabled': item.get('enabled') is True,
                            'version': str(item.get('version', ''))})
        return entries
