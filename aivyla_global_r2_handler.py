"""R2-enabled handler. Does not modify production handler.py or Global v3."""
import json
import os
import time
from pathlib import Path
import runpy
import sys
import traceback

# Optional for CPU-only CI tests; the runtime image supplies requests.
try:
    import requests
except ModuleNotFoundError as exc:
    if exc.name != "requests":
        raise
    requests = None
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


# This is a Global-only image wrapper. Main production never imports it.
def _maybe_attach_animated_ltx_preview(payload):
    """Opt-in KJNodes preview injection, without changing Studio's shared graphs.

    Fail open for preview *only*: a missing/older KJNodes node cannot break
    the generation. This path is called only for authenticated Studio jobs
    already opted in by the Studio's endpoint-ID gate.
    """
    if not (os.getenv("AIVYLA_ANIMATED_PREVIEW_ENABLED", "false").lower() == "true"
            and os.getenv("AIVYLA_LIVE_PREVIEW_ENABLED", "false").lower() == "true"
            and payload.get("live_preview") is True):
        return
    if requests is None:
        print("Aivyla animated preview: requests unavailable; still preview only", flush=True)
        return
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict):
        return
    kinds = {n.get("class_type") for n in workflow.values() if isinstance(n, dict)}
    # Only LTX 2.5 T2V, I2V, MSR; not lipsync (2.3) or MiniMax.
    if not {"EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio"}.issubset(kinds):
        return
    if any(isinstance(n, dict) and n.get("class_type") == "LTXVAudioVAEEncode"
           for n in workflow.values()):
        return
    if any(node_id.startswith("aivyla_kj_anim_") for node_id in workflow):
        return  # A duplicate RunPod job should never wrap the model twice.
    try:
        response = requests.get(
            "http://127.0.0.1:8188/object_info/ModelPreviewOverrideKJ", timeout=(2, 4)
        )
        response.raise_for_status()
        schema = response.json().get("ModelPreviewOverrideKJ", {}).get("input", {})
        fields = set(schema.get("required", {})) | set(schema.get("optional", {}))
        if not {"model", "preview_frames", "preview_fps", "max_resolution"}.issubset(fields):
            print("Aivyla animated preview: KJNodes version lacks expected inputs; still preview only", flush=True)
            return
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        print(f"Aivyla animated preview: node unavailable ({type(exc).__name__}); still preview only", flush=True)
        return

    sources = {}
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") not in ("CFGGuider", "LTXVDualCFGGuider"):
            continue
        inp = node.get("inputs")
        if not isinstance(inp, dict):
            continue
        link = inp.get("model")
        if not (isinstance(link, list) and len(link) == 2
                and isinstance(link[0], str) and type(link[1]) is int
                and link[0] in workflow):
            continue
        # Only wrap an LTX model path and use one KJ wrapper for both passes.
        key = (link[0], link[1])
        if key not in sources:
            new_id = f"aivyla_kj_anim_{len(sources)}"
            if new_id in workflow:
                print("Aivyla animated preview: ID conflict; skipped", flush=True)
                return
            sources[key] = new_id
        inp["model"] = [sources[key], 0]

    if not sources:
        print("Aivyla animated preview: no suitable LTX guider; still preview only", flush=True)
        return
    for (source_id, output_index), new_id in sources.items():
        workflow[new_id] = {
            "class_type": "ModelPreviewOverrideKJ",
            "inputs": {
                "model": [source_id, output_index],
                "max_resolution": 192,
                "jpeg_quality": 65,
                "suppress_default_preview": False,
                "preview_frames": 8,
                "preview_fps": 8,
            },
            "_meta": {"title": "Aivyla Global Animated Live Preview"},
        }
    print(f"Aivyla animated preview: attached {len(sources)} KJ wrapper(s), 8 frames at 8 fps", flush=True)


def handler(job):
    started = time.monotonic()
    try:
        payload = job.get('input', {})
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict) and isinstance(payload.get('workflow'), dict):
            _maybe_attach_animated_ltx_preview(payload)
            summary = stage_workflow(payload['workflow'])
            print('Aivyla R2 staging: ' + json.dumps(summary) +
                  f' wrapper_s={time.monotonic()-started:.2f}', flush=True)
    except Exception as exc:
        traceback.print_exc()
        return {'error': 'R2 model staging failed', 'details': [f'{type(exc).__name__}: {exc}']}
    return _original(job)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
    # rebuild v4.2
