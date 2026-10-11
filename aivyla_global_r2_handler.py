"""R2-enabled handler. Does not modify production handler.py or Global v3."""
import json
from pathlib import Path
import runpy
import sys
import traceback

import runpod
from aivyla_r2_stage import stage_workflow

BASE = Path('/handler.py')
if str(BASE.parent) not in sys.path:
    sys.path.insert(0, str(BASE.parent))
_original = runpy.run_path(str(BASE), run_name='aivyla_r2_base_handler')['handler']


def handler(job):
    try:
        payload = job.get('input', {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict) and isinstance(payload.get('workflow'), dict):
            summary = stage_workflow(payload['workflow'])
            print('Aivyla R2 staging: ' + json.dumps(summary), flush=True)
    except Exception as exc:
        traceback.print_exc()
        return {'error': 'R2 model staging failed', 'details': [f'{type(exc).__name__}: {exc}']}
    return _original(job)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
