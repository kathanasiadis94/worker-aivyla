"""Run the existing production handler unchanged, after Global local staging.

The stock /aivyla-start.sh may substitute /handler.py with diagnostics. Loading
it at runtime preserves that behavior without duplicating 1170 lines of code.
"""
import json
import runpy
import time
import traceback

import runpod

from aivyla_global_cache import stage_workflow

BASE = runpy.run_path('/handler.py', run_name='aivyla_global_base_handler')
base_handler = BASE['handler']


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
