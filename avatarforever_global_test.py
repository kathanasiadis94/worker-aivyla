#!/usr/bin/env python3
"""Submit one AvatarForever comparison job to an explicitly selected Global endpoint.

Run from Studio with its existing venv; no keys enter requests/results/logs.
Preview is the default. Use --execute for generation, --check for capabilities,
or --resume JOB_ID to retrieve an already-submitted job without resubmitting.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time

GLOBAL_ENDPOINT_VARS = ('AIVYLA_GLOBAL_ENDPOINT_ID', 'RUNPOD_GLOBAL_ENDPOINT_ID',
                        'RUNPOD_ENDPOINT_ID_GLOBAL', 'AIVYLA_GLOBAL_RUNPOD_ENDPOINT_ID')


def resolve_endpoint(explicit, env):
    endpoint = explicit
    if not endpoint:
        values = {env[name].strip() for name in GLOBAL_ENDPOINT_VARS if env.get(name, '').strip()}
        if len(values) > 1:
            raise ValueError('Conflicting Global endpoint settings; supply --endpoint explicitly')
        endpoint = next(iter(values), '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,100}', endpoint or ''):
        raise ValueError('Supply --endpoint with the Aivyla Global endpoint ID; production fallback is disabled')
    return endpoint


def make_payload(args):
    if not args.image or not args.audio:
        raise ValueError('Supply --image and --audio')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 40:
        raise ValueError('seconds must be >0 and <=40')
    payload = dict(engine='avatarforever_test', output_transport='studio',
                   variants=args.variants.split(','), seconds=args.seconds,
                   width=args.width, height=args.height, seed=args.seed)
    for name in ('image', 'audio'):
        path = getattr(args, name)
        if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024**2:
            raise ValueError(f'{name} must be an existing file <=32 MiB')
        payload[name] = base64.b64encode(path.read_bytes()).decode('ascii')
    if args.prompt_file:
        payload['prompt'] = args.prompt_file.read_text().strip()
    # Keep validation identical without importing R2/worker dependencies on Studio.
    names = payload['variants']
    if (not 1 <= len(names) <= 3 or any(x not in ('BASE', 'OMNI1', 'OMNI2') for x in names)
            or len(set(names)) != len(names)):
        raise ValueError('Use unique BASE,OMNI1,OMNI2 variants')
    if (type(args.seed) is not int or not 0 <= args.seed < 2**32
            or min(args.width, args.height) < 256 or max(args.width, args.height) > 1024
            or args.width % 32 or args.height % 32 or args.width * args.height > 768 * 512):
        raise ValueError('Invalid seed or dimensions')
    return payload


def save_outputs(output, job_id, destination):
    # Reuse the exact server-side transport used by ltx_service.py.
    from app.runpod_output_transport import read_output_bytes
    for index, info in enumerate(output.get('images', [])):
        raw = read_output_bytes(info, job_id)
        if inspect.isawaitable(raw):
            import asyncio
            raw = asyncio.run(raw)
        if not isinstance(raw, bytes) or not raw:
            raise RuntimeError('Studio transport returned no output bytes')
        if info.get('sha256') and hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise RuntimeError('Downloaded output checksum mismatch')
        name = info.get('variant')
        if name not in ('BASE', 'OMNI1', 'OMNI2'):
            name = str(index)
        path = destination / (name + '.mp4')
        path.write_bytes(raw)
        print('VIDEO=' + str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studio-root', type=Path, default=Path('/var/www/studio-fastapi'))
    parser.add_argument('--endpoint')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', default='BASE,OMNI1,OMNI2')
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('/var/tmp/avatarforever-tests'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    sys.path.insert(0, str(args.studio_root.resolve()))
    from dotenv import load_dotenv
    load_dotenv(args.studio_root / '.env', override=False)
    endpoint = resolve_endpoint(args.endpoint, os.environ)
    payload = ({'engine': 'avatarforever_test', 'action': 'capabilities'}
               if args.check else None if args.resume else make_payload(args))
    if not (args.execute or args.check or args.resume):
        plan = {k: v for k, v in payload.items() if k not in ('image', 'audio')}
        plan.update(endpoint=endpoint, image=str(args.image), audio=str(args.audio))
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print('Preview only. Add --execute to submit ONE comparison job.')
        return
    import requests
    api_key = os.getenv('RUNPOD_API_KEY', '')
    if not api_key:
        raise ValueError('RUNPOD_API_KEY is missing from the Studio environment')
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + api_key
    api = f'https://api.runpod.ai/v2/{endpoint}'
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = args.output_root / stamp
    destination.mkdir()
    job_id = args.resume
    if not job_id:
        # Never retry POST /run automatically: a lost response may have accepted a job.
        response = session.post(api + '/run', json={'input': payload}, timeout=(15, 90))
        response.raise_for_status()
        job_id = response.json()['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', job_id):
        raise ValueError('Invalid RunPod job ID')
    (destination / 'job.json').write_text(json.dumps(dict(endpoint=endpoint, job_id=job_id)))
    print('JOB=' + job_id, flush=True)
    print('RESULTS=' + str(destination), flush=True)
    print('Resume without resubmitting: --endpoint ' + endpoint + ' --resume ' + job_id, flush=True)
    deadline = time.monotonic() + 7200
    last_status = None
    while True:
        response = session.get(api + '/status/' + job_id, timeout=(15, 60))
        response.raise_for_status()
        result = response.json()
        status = result.get('status')
        if status != last_status:
            print('STATUS=' + str(status), flush=True)
            last_status = status
        if status in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Polling timed out; resume this job to retrieve its result')
        time.sleep(5)
    (destination / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    output = result.get('output')
    if isinstance(output, dict) and output.get('images'):
        save_outputs(output, job_id, destination)
    if status != 'COMPLETED' or not isinstance(output, dict) or output.get('error'):
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise RuntimeError('AvatarForever job failed; inspect result.json (partial outputs were retained)')
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.check and output.get('engine') != 'avatarforever_test':
        raise RuntimeError('The selected endpoint does not report AvatarForever support')


if __name__ == '__main__':
    main()
