"""R2-enabled handler. Does not modify production handler.py or Global v3."""
import json
import os
import time
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


def install_fast_preflight(original):
    """Avoid 10s /object_info timeout on R2 jobs after verified staging.

    This only replaces the optional validation step in the R2-only handler
    process, never the production or Global v3 handler. ComfyUI still validates
    the entire workflow when it is enqueued. Setting the env var to 0 restores
    original behavior for debugging missing model/LoadImage references.
    """
    globals_ = original.__globals__
    previous = globals_.get('validate_workflow_models')
    if not callable(previous):
        raise RuntimeError('Base handler preflight hook unavailable; refuse unsafe patch')

    def validate(workflow):
        if (os.getenv('AIVYLA_R2_ENABLED', '0') == '1'
                and os.getenv('AIVYLA_R2_SKIP_OBJECT_INFO', '1') == '1'):
            print('Aivyla R2: skip redundant /object_info preflight; '
                  'ComfyUI queue validation remains active', flush=True)
            return None
        return previous(workflow)

    globals_['validate_workflow_models'] = validate


install_fast_preflight(_original)


def handler(job):
    started = time.monotonic()
    try:
        payload = job.get('input', {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict) and isinstance(payload.get('workflow'), dict):
            summary = stage_workflow(payload['workflow'])
            print('Aivyla R2 staging: ' + json.dumps(summary) +
                  f' wrapper_s={time.monotonic()-started:.2f}', flush=True)
    except Exception as exc:
        traceback.print_exc()
        return {'error': 'R2 model staging failed', 'details': [f'{type(exc).__name__}: {exc}']}
    return _original(job)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
