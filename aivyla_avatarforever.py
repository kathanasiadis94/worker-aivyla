"""Opt-in AvatarForever comparison jobs on the Global worker only.

The isolated runtime is baked at build time. Models are read from R2; no
runtime package installs, source updates, network-volume writes or S3 writes.
"""
from __future__ import annotations

import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time

from aivyla_r2_registry import Registry

ENGINE = 'avatarforever_test'
UPSTREAM_REVISION = 'de4deb6a55c9567a17dad58d67390fe089993630'
CHECKPOINT = 'avatarforever-ltx-2.3-22b.safetensors'
GEMMA = 'gemma-3-12b-it-qat-q4_0-unquantized'
OMNINFT = 'LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors'
GEMMA_FILES = (
    'added_tokens.json', 'chat_template.json', 'config.json',
    'generation_config.json',
    *(f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)),
    'model.safetensors.index.json', 'preprocessor_config.json',
    'processor_config.json', 'special_tokens_map.json', 'tokenizer.json',
    'tokenizer.model', 'tokenizer_config.json',
)
VARIANTS = {'BASE': None, 'OMNI1': 1.0, 'OMNI2': 2.0}
DEFAULT_PROMPT = (
    'A person speaks naturally to the camera in one continuous shot. '
    'Accurate lip synchronization with the supplied speech audio. '
    'Natural blinking, restrained head movement and relaxed conversational gestures. '
    'Stable facial appearance, clothing, lighting, background and framing throughout. '
    'During pauses the mouth relaxes and the person remains in the same scene.'
)
JOB_LOCK = threading.Lock()
MAX_MEDIA_BYTES = 32 * 1024**2
COMFY_OUTPUT_ROOT = Path('/comfyui/output')
COMFY_API = 'http://127.0.0.1:8188'
COMFY_PID_FILE = Path('/tmp/comfyui.pid')


def validate_payload(payload):
    if not isinstance(payload, dict) or payload.get('engine') != ENGINE:
        raise ValueError('Expected an avatarforever_test input')
    if payload.get('output_transport') != 'studio':
        raise ValueError('AvatarForever tests require output_transport=studio')
    if 'workflow' in payload:
        raise ValueError('AvatarForever test must not include a ComfyUI workflow')
    names = payload.get('variants', list(VARIANTS))
    if (not isinstance(names, list) or not 1 <= len(names) <= 3
            or any(not isinstance(x, str) or x not in VARIANTS for x in names)
            or len(set(names)) != len(names)):
        raise ValueError('variants must contain unique BASE, OMNI1 or OMNI2 names')
    seconds = payload.get('seconds', 10)
    if (type(seconds) not in (float, int) or not math.isfinite(seconds)
            or not 0 < seconds <= 40):
        raise ValueError('seconds must be finite, >0 and <=40')
    width, height = payload.get('width', 768), payload.get('height', 512)
    if (type(width) is not int or type(height) is not int
            or min(width, height) < 256 or max(width, height) > 1024
            or width % 32 or height % 32 or width * height > 768 * 512):
        raise ValueError('Use multiples of 32, >=256, and at most 768x512 pixels')
    seed = payload.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    prompt = payload.get('prompt', DEFAULT_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError('prompt must contain 1..12000 characters')
    for field in ('image', 'audio'):
        value = payload.get(field)
        if not isinstance(value, str) or not value or len(value) > MAX_MEDIA_BYTES * 4 // 3 + 1024:
            raise ValueError(f'{field} must be base64 data <=32 MiB')
    return dict(variants=names, seconds=float(seconds), width=width,
                height=height, seed=seed, prompt=prompt.strip())


def decode_media(value, destination):
    # Accept base64/data URIs only, never fetch job-supplied URLs or local paths.
    raw = value
    if value.startswith('data:'):
        header, sep, raw = value.partition(',')
        if not sep or not header.endswith(';base64'):
            raise ValueError('Expected a base64 data URI')
    try:
        data = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Invalid base64 media') from exc
    if not 0 < len(data) <= MAX_MEDIA_BYTES:
        raise ValueError('Decoded media must contain 1 byte..32 MiB')
    destination.write_bytes(data)


def frame_count(seconds):
    return 8 * math.ceil(max(0, seconds * 25 - 1) / 8) + 1


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    size: int
    etag: str
    sha256: str | None


class BundleStager:
    """Fetch exact bundle keys, including metadata omitted by the model catalog."""

    def __init__(self, root, client=None, bucket=None):
        self.root = Path(root)
        self.client = client if client is not None else Registry._client()
        self.bucket = bucket or os.getenv('AIVYLA_AVATARFOREVER_R2_BUCKET', 'aivyla-models-v2')

    def head(self, key):
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        size, etag = int(head['ContentLength']), head['ETag'].strip('"')
        if size <= 0 or not etag:
            raise ValueError(f'Empty or unversioned R2 object: {key}')
        metadata = head.get('Metadata', {})
        digest = metadata.get('sha256') or metadata.get('source-sha256')
        if digest is not None and not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError(f'Invalid R2 SHA256 metadata: {key}')
        return ObjectSpec(key, size, etag, digest)

    def plan(self, with_omni):
        entries = [(f'models/split_files/checkpoints/{CHECKPOINT}', Path('checkpoints') / CHECKPOINT)]
        entries += [(f'models/text_encoders/{GEMMA}/{name}', Path('checkpoints') / GEMMA / name)
                    for name in GEMMA_FILES]
        if with_omni:
            # Use the catalog's split_files priority without a full-bucket listing.
            for key in (f'models/split_files/loras/{OMNINFT}', f'models/loras/{OMNINFT}'):
                try:
                    spec = self.head(key)
                except Exception as exc:
                    code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
                    if code in ('NoSuchKey', '404', 'NotFound'):
                        continue
                    raise
                break
            else:
                raise FileNotFoundError(f'OmniNFT is missing from {self.bucket}')
            omni = (spec, self.root / 'loras' / OMNINFT)
        else:
            omni = None
        jobs = [(self.head(key), self.root / rel) for key, rel in entries]
        if omni:
            jobs.append(omni)
        return jobs

    def fingerprint(self, spec):
        return dict(bucket=self.bucket, key=spec.key, bytes=spec.size,
                    etag=spec.etag, source_sha256=spec.sha256)

    @staticmethod
    def receipt(dst):
        return dst.with_name(dst.name + '.avatarforever-cache.json')

    def cached(self, spec, dst):
        try:
            data = json.loads(self.receipt(dst).read_text())
            return (dst.is_file() and dst.stat().st_size == spec.size
                    and data.get('source') == self.fingerprint(spec)
                    and re.fullmatch(r'[0-9a-f]{64}', data.get('sha256', '')) is not None
                    and (not spec.sha256 or data['sha256'] == spec.sha256))
        except (OSError, ValueError, TypeError):
            return False

    def fetch(self, job):
        from boto3.s3.transfer import TransferConfig
        spec, dst = job
        if self.cached(spec, dst):
            return 'cached'
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + '.part')
        receipt_tmp = self.receipt(dst).with_name(self.receipt(dst).name + '.part')
        try:
            streams = max(1, min(12, int(os.getenv('AIVYLA_R2_DOWNLOAD_STREAMS', '12'))))
            config = TransferConfig(max_concurrency=streams, multipart_threshold=32 * 1024**2,
                                    multipart_chunksize=32 * 1024**2, use_threads=True)
            started = time.monotonic()
            self.client.download_file(self.bucket, spec.key, str(tmp), Config=config)
            if tmp.stat().st_size != spec.size:
                raise ValueError(f'Incomplete R2 object: {spec.key}')
            digest = sha256_file(tmp)
            if spec.sha256 and digest != spec.sha256:
                raise ValueError(f'R2 SHA256 mismatch: {spec.key}')
            if self.head(spec.key) != spec:
                raise ValueError(f'R2 object changed during download: {spec.key}')
            os.replace(tmp, dst)
            receipt_tmp.write_text(json.dumps(dict(source=self.fingerprint(spec), sha256=digest)))
            os.replace(receipt_tmp, self.receipt(dst))
            print(f'AvatarForever R2: {spec.key} bytes={spec.size} '
                  f'seconds={time.monotonic()-started:.2f} '
                  f'checksum={"source_sha256" if spec.sha256 else "local_sha256_etag"}', flush=True)
            return 'downloaded'
        finally:
            tmp.unlink(missing_ok=True)
            receipt_tmp.unlink(missing_ok=True)

    def stage(self, with_omni):
        self.root.mkdir(parents=True, exist_ok=True)
        jobs = self.plan(with_omni)
        required = sum(spec.size for spec, dst in jobs if not self.cached(spec, dst))
        free = shutil.disk_usage(self.root).free
        reserve = 8 * 1024**3
        if required + reserve > free:
            raise RuntimeError(f'AvatarForever local disk needs {required/1024**3:.1f} GiB '
                               f'+8 GiB reserve; available {free/1024**3:.1f} GiB')
        started = time.monotonic()
        parallel = max(1, min(2, int(os.getenv('AIVYLA_R2_PARALLEL_MODELS', '2'))))
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            outcomes = list(pool.map(self.fetch, jobs))
        # Detect an incomplete/incorrect shard index before launching the model.
        index = json.loads((self.root / 'checkpoints' / GEMMA / 'model.safetensors.index.json').read_text())
        shards = set(index.get('weight_map', {}).values())
        expected_shards = {f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)}
        if shards != expected_shards:
            raise ValueError('Gemma index does not reference the expected five shards')
        return dict(bucket=self.bucket, files=len(jobs), cached=outcomes.count('cached'),
                    downloaded=outcomes.count('downloaded'), seconds=round(time.monotonic()-started, 3))


def runtime_preflight():
    root = Path(os.getenv('AIVYLA_AVATARFOREVER_RUNTIME', '/opt/avatarforever'))
    python = root / 'venv/bin/python'
    repo = root / 'repo'
    for path in (python, repo / 'inference.py', root / 'revision.txt'):
        if not path.is_file():
            raise RuntimeError('AvatarForever runtime is missing; deploy the AvatarForever Global image first')
    if (root / 'revision.txt').read_text().strip() != UPSTREAM_REVISION:
        raise RuntimeError('AvatarForever runtime revision mismatch')
    for command in ('ffmpeg', 'ffprobe', 'nvidia-smi'):
        if not shutil.which(command):
            raise RuntimeError(f'AvatarForever runtime missing {command}')
    return python, repo


def comfy_process_alive():
    """None means the launcher has not supplied a PID; the deadline still applies."""
    try:
        pid = int(COMFY_PID_FILE.read_text().strip())
        if pid <= 0:
            raise ValueError('non-positive PID')
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise RuntimeError('Invalid ComfyUI PID file; inspect worker startup logs') from exc
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A dead child can remain a zombie while its shell parent runs the handler.
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0]
        return state not in ('Z', 'X')
    except FileNotFoundError:
        return False


def wait_for_comfy_queue(*, timeout_seconds=None, poll_seconds=1.0):
    """Wait for the actual API, without unloading models or submitting any work."""
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
    seconds = float(timeout_seconds if timeout_seconds is not None else
                    os.getenv('AIVYLA_AVATARFOREVER_COMFY_READY_TIMEOUT_SECONDS', '180'))
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError('ComfyUI readiness timeout must be >0 and <=600 seconds')
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('ComfyUI readiness poll interval must be positive')
    started = time.monotonic()
    deadline = started + seconds
    print('AvatarForever: waiting for ComfyUI API readiness', flush=True)
    while True:
        if comfy_process_alive() is False:
            raise RuntimeError('ComfyUI process exited before API readiness; inspect worker startup logs')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'ComfyUI API did not become ready within {seconds:g}s; inspect worker startup logs')
        try:
            with urlopen(COMFY_API + '/queue', timeout=min(5, remaining)) as response:
                queue = json.load(response)
        except HTTPError:
            raise  # An actual server error is not an unopened startup socket.
        except (URLError, TimeoutError):
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
            continue
        if (not isinstance(queue, dict) or
                any(not isinstance(queue.get(key), list) for key in ('queue_running', 'queue_pending'))):
            raise RuntimeError('ComfyUI /queue returned an invalid response')
        print(f'AvatarForever: ComfyUI API ready after {time.monotonic() - started:.2f}s', flush=True)
        return queue


def free_comfy_gpu():
    import requests
    queue = wait_for_comfy_queue()
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('ComfyUI has active/queued work; retry when idle')
    response = requests.post(COMFY_API + '/free',
                             json={'unload_models': True, 'free_memory': True}, timeout=(3, 10))
    response.raise_for_status()
    # /free is serviced asynchronously by ComfyUI. Wait for actual VRAM release.
    deadline = time.monotonic() + 30
    minimum = int(os.getenv('AIVYLA_AVATARFOREVER_MIN_FREE_MIB', '70000'))
    while True:
        data = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=10)
        available = [int(x.strip()) for x in data.splitlines() if x.strip()]
        if len(available) != 1:
            raise RuntimeError('AvatarForever pilot currently requires a single visible GPU')
        if available[0] >= minimum:
            return available[0]
        if time.monotonic() >= deadline:
            raise RuntimeError(f'AvatarForever GPU preflight needs {minimum} MiB free; available {available[0]} MiB')
        time.sleep(1)


def run_logged(command, *, cwd, log, timeout):
    # Each variant has its own process, so LoRA weights/VRAM cannot leak to the next.
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            result = proc.wait(timeout=timeout)
        except BaseException:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
            raise
    if result:
        tail = log.read_text(errors='replace')[-4000:]
        raise RuntimeError(f'AvatarForever inference exited {result}:\n{tail}')


def prepare_media(payload, settings, work):
    image, audio = work / 'input.image', work / 'input.audio'
    decode_media(payload['image'], image)
    decode_media(payload['audio'], audio)
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
        '-of', 'json', str(audio)], text=True, timeout=30)
    probe = json.loads(raw)
    duration = float(probe.get('format', {}).get('duration', 0))
    if (not math.isfinite(duration) or duration <= 0
            or not any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))):
        raise ValueError('Input has no valid audio stream/duration')
    settings['seconds'] = min(settings['seconds'], duration)
    settings['frames'] = frame_count(settings['seconds'])
    seconds = settings['seconds']
    frames_seconds = settings['frames'] / 25
    # Pad audio through the last valid 8n+1 frame, then remux/trim to requested duration.
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(audio), '-vn',
                    '-af', f'atrim=duration={seconds},apad=whole_dur={frames_seconds}',
                    '-t', str(frames_seconds), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                    str(work / 'conditioning.wav')], check=True, timeout=60)
    scale = (f'scale={settings["width"]}:{settings["height"]}:force_original_aspect_ratio=decrease,'
             f'pad={settings["width"]}:{settings["height"]}:(ow-iw)/2:(oh-ih)/2:black,setsar=1')
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(image), '-vf', scale,
                    '-frames:v', '1', str(work / 'conditioning.png')], check=True, timeout=60)
    return audio


def progress(job, stage, **extra):
    print('AvatarForever: ' + json.dumps(dict(stage=stage, **extra)), flush=True)
    try:
        import runpod
        runpod.serverless.progress_update(job, dict(stage=stage, **extra))
    except Exception as exc:
        print(f'AvatarForever progress delivery unavailable ({type(exc).__name__})', flush=True)


def run_comparison(job, payload):
    if os.getenv('AIVYLA_AVATARFOREVER_ENABLED', '0') != '1':
        return {'error': 'AvatarForever test is disabled on this worker'}
    if payload.get('action') == 'capabilities':
        runtime_preflight()
        wait_for_comfy_queue()
        return dict(engine=ENGINE, enabled=True, upstream_revision=UPSTREAM_REVISION,
                    bucket=os.getenv('AIVYLA_AVATARFOREVER_R2_BUCKET', 'aivyla-models-v2'),
                    variants=list(VARIANTS), max_seconds=40,
                    comfy_api_ready=True, model_downloads_started=False, generation_started=False)
    settings = validate_payload(payload)
    job_id = job.get('id')
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', job_id):
        raise ValueError('A valid RunPod job id is required')
    if not JOB_LOCK.acquire(blocking=False):
        raise RuntimeError('Another AvatarForever comparison is running on this worker')
    try:
        return _run_comparison(job, payload, settings)
    finally:
        JOB_LOCK.release()


def _run_comparison(job, payload, settings):
    from aivyla_output_upload import check_transport_ready, upload_comfy_output
    started = time.monotonic()
    python, repo = runtime_preflight()
    check_transport_ready()
    # Check input files before downloading 70 GB or unloading any ComfyUI models.
    output_root = COMFY_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    images, measurements = [], []
    with tempfile.TemporaryDirectory(prefix='aivyla-avatarforever-', dir=output_root) as directory:
        work = Path(directory)
        audio = prepare_media(payload, settings, work)
        progress(job, 'gpu_preflight')
        free_mib = free_comfy_gpu()
        progress(job, 'r2_staging')
        root = Path(os.getenv('AIVYLA_AVATARFOREVER_MODEL_ROOT', '/tmp/aivyla-avatarforever-models'))
        if not root.is_absolute() or not str(root).startswith('/tmp/') or '..' in root.parts:
            raise ValueError('AvatarForever model cache must be below /tmp')
        staging = BundleStager(root).stage(any(VARIANTS[x] is not None for x in settings['variants']))
        deadline = started + int(os.getenv('AIVYLA_AVATARFOREVER_TIMEOUT_SECONDS', '3300'))
        for name in settings['variants']:
            progress(job, 'generating', variant=name, seconds=settings['seconds'])
            variant_started = time.monotonic()
            raw, final = work / (name + '-raw.mp4'), work / (name + '.mp4')
            report, log = work / (name + '.json'), work / (name + '.log')
            command = [str(python), '-u', str(Path(__file__).with_name('aivyla_avatarforever_infer.py')),
                       '--distilled-checkpoint-path', str(root / 'checkpoints' / CHECKPOINT),
                       '--gemma-root', str(root / 'checkpoints' / GEMMA),
                       '--audio-path', str(work / 'conditioning.wav'),
                       '--first-frame-condition-image-path', str(work / 'conditioning.png'),
                       '--prompt', settings['prompt'], '--width', str(settings['width']),
                       '--height', str(settings['height']), '--seed', str(settings['seed']),
                       '--num-frames', str(settings['frames']), '--frame-rate', '25',
                       '--ar-history-feature-cache', '--output-path', str(raw),
                       '--aivyla-report', str(report)]
            if VARIANTS[name] is not None:
                command.extend(['--lora', str(root / 'loras' / OMNINFT), str(VARIANTS[name])])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('AvatarForever comparison exceeded its time budget')
            try:
                run_logged(command, cwd=repo, log=log, timeout=remaining)
                inference_s = time.monotonic() - variant_started
                details = json.loads(report.read_text())
                subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(raw), '-i', str(audio),
                                '-map', '0:v:0', '-map', '1:a:0', '-t', str(settings['seconds']),
                                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
                                str(final)], check=True, timeout=120)
                # Existing Studio output transport, unchanged, serves files through ComfyUI /view.
                progress(job, 'uploading', variant=name)
                output = upload_comfy_output(job['id'], final.name, work.name, 'output', len(images))
                output['variant'] = name
                images.append(output)
                measurements.append(dict(variant=name, strength=VARIANTS[name],
                                         inference_s=round(inference_s, 3),
                                         total_variant_s=round(time.monotonic()-variant_started, 3),
                                         **details))
            except Exception as exc:
                # Preserve already-delivered outputs; never report a failed variant as a completed comparison.
                return dict(error=f'AvatarForever {name} failed', details=[str(exc)],
                            images=images, measurements=measurements, staging=staging)
            finally:
                raw.unlink(missing_ok=True)
        return dict(status='success', engine=ENGINE, images=images,
                    measurements=measurements, staging=staging, settings=settings,
                    gpu_free_mib_before=free_mib, upstream_revision=UPSTREAM_REVISION,
                    total_s=round(time.monotonic()-started, 3))
