"""R2 v4 test bootstrap on top of existing validated Global v3 image.

No source-volume writes, no production image patches; runtime /start.sh only.
"""
from pathlib import Path
import os
import subprocess
import sys

sys.path.insert(0, '/opt/aivyla-global')
import aivyla_global_start as v3

R2_HANDLER = '/opt/aivyla-global/aivyla_global_r2_handler.py'
V3_HANDLER = '/opt/aivyla-global/aivyla_global_handler.py'


def prepare_handler():
    # A repeated bootstrap must not call v3.patch_start on an already v4-patched
    # launcher: v3 expects its own handler and would reject the v4 handler.
    source = v3.START.read_text()
    if source.count(R2_HANDLER) == 2 and source.count(V3_HANDLER) == 0:
        if source.count('aivyla_global_local_cache:') != 1:
            raise RuntimeError('Unexpected R2 startup state: missing local cache path')
        subprocess.run(['bash', '-n', str(v3.START)], check=True)
        return
    v3.patch_start()
    source = v3.START.read_text()
    r2_count, old_count = source.count(R2_HANDLER), source.count(V3_HANDLER)
    if (r2_count, old_count) != (0, 2):
        raise RuntimeError('Unexpected Global handler launcher; aborting R2 patch')
    v3.START.write_text(source.replace(V3_HANDLER, R2_HANDLER))
    subprocess.run(['bash', '-n', str(v3.START)], check=True)


def validate_settings():
    if os.getenv('AIVYLA_R2_ENABLED', '0') != '1' and os.getenv('AIVYLA_R2_NODES_ENABLED', '0') != '1':
        return
    endpoint = os.getenv('AIVYLA_R2_ENDPOINT', '')
    endpoint = endpoint.rstrip('/')
    is_r2 = endpoint.startswith('https://') and endpoint.endswith('.r2.cloudflarestorage.com')
    if not (is_r2 or endpoint == 'https://t3.storage.dev'):
        raise RuntimeError('AIVYLA_R2_ENDPOINT must be Cloudflare R2 or Tigris in test endpoint')
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        raise RuntimeError('R2 credentials missing from Runpod test endpoint secrets')


def main():
    validate_settings()
    prepare_handler()
    if v3.LOCAL_ENABLED:
        subprocess.run([sys.executable, '-u', '/opt/aivyla-global/aivyla_global_nodes.py'], check=True)
    v3.patch_aivyla_launcher()
    if os.getenv('AIVYLA_R2_NODES_ENABLED', '0') == '1':
        from aivyla_r2_nodes import install_enabled_nodes
        install_enabled_nodes()
    print('Aivyla Global R2 v4 bootstrap: complete', flush=True)
    os.execv('/bin/bash', ['bash', str(v3.AIVYLA_START)])


if __name__ == '__main__':
    main()
