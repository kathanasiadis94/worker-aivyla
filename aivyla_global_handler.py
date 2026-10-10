"""Wrap the production handler with per-request verified model staging.

/handler.py is a sibling of /network_volume.py and /aivyla_output_upload.py.
Unlike `python /handler.py`, running a wrapper under /opt does not put `/`
on sys.path. Explicitly restore that path before loading the original file.
"""

import json
from pathlib import Path
import runpy
import sys
import traceback

import runpod

from aivyla_global_cache import stage_workflow

BASE_HANDLER_PATH = Path('/handler.py')
BASE_DIRECTORY = str(BASE_HANDLER_PATH.parent)
if BASE_DIRECTORY not in sys.path:
    sys.path.insert(0, BASE_DIRECTORY)

# A diagnostic handler from the network volume may have replaced /handler.py
# during /aivyla-start.sh. Preserve exactly the handler active in this worker.
BASE = runpy.run_path(str(BASE_HANDLER_PATH), run_name='aivyla_global_base_handler')
base_handler = BASE['handler']
if not callable(base_handler):
    raise RuntimeError('Original /handler.py does not expose a callable handler')


def handler(job):
    try:
        job_input = job.get('input', {})
        if isinstance(job_input, str):
            try:
                job_input = json.loads(job_input)
            except ValueError:
                return base_handler(job)
        if isinstance(job_input, dict):
            workflow = job_input.get('workflow')
            if isinstance(workflow, dict):
                stats = stage_workflow(workflow)
                print('Aivyla Global model staging: ' + json.dumps(stats), flush=True)
    except Exception as exc:
        print(f'Aivyla Global cache failed: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()
        return {'error': 'Aivyla Global model staging failed',
                'details': [f'{type(exc).__name__}: {exc}']}
    return base_handler(job)


if __name__ == '__main__':
    print('Aivyla Global cached handler ready', flush=True)
    runpod.serverless.start({'handler': handler})
