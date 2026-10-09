"""Worker-side direct output delivery to the Aivyla Studio HTTPS server.

Requires AIVYLA_STUDIO_BASE_URL and AIVYLA_OUTPUT_SHARED_SECRET in the RunPod
worker environment. Never embeds credentials in the RunPod job JSON/result.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.parse
from pathlib import Path

import requests

CHUNK_BYTES = 4 * 1024 * 1024
MAX_BYTES = int(os.environ.get("AIVYLA_OUTPUT_MAX_BYTES", str(2 * 1024**3)))


def _settings():
    base = os.environ.get("AIVYLA_STUDIO_BASE_URL", "").rstrip("/")
    secret = os.environ.get("AIVYLA_OUTPUT_SHARED_SECRET", "")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("AIVYLA_STUDIO_BASE_URL must be an HTTPS origin")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise RuntimeError("AIVYLA_STUDIO_BASE_URL must not contain a path or query")
    if len(secret) < 32:
        raise RuntimeError("AIVYLA_OUTPUT_SHARED_SECRET is missing or too short")
    return base, secret


def check_transport_ready():
    base, secret = _settings()
    url = base + "/api/internal/runpod-outputs/health"
    response = requests.put(url, headers={"X-Aivyla-Output-Key": secret}, data=b"ping", timeout=(15, 30))
    response.raise_for_status()
    result = response.json()
    if result.get("transport") != "studio-chunks-v1":
        raise RuntimeError("The Studio server is not running the expected output transport")
    return True


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def upload_comfy_output(job_id, filename, subfolder, image_type, output_index):
    base, secret = _settings()
    if image_type not in ("output", "temp"):
        raise ValueError("Unsupported ComfyUI output location")
    if not filename or os.path.basename(filename) != filename:
        raise ValueError("Invalid output filename")
    url = "http://127.0.0.1:8188/view?" + urllib.parse.urlencode({
        "filename": filename, "subfolder": subfolder or "", "type": image_type
    })
    fd, tmp_path = tempfile.mkstemp(prefix="aivyla-result-", suffix=Path(filename).suffix)
    os.close(fd)
    try:
        size = 0
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as target:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    size += len(block)
                    if size > MAX_BYTES:
                        raise ValueError("RunPod output exceeded AIVYLA_OUTPUT_MAX_BYTES")
                    target.write(block)
        if size <= 0:
            raise ValueError("ComfyUI returned an empty output")
        checksum = _sha256_file(tmp_path)
        endpoint = f"{base}/api/internal/runpod-outputs/{job_id}/{int(output_index)}"
        with open(tmp_path, "rb") as source:
            index = 0
            while True:
                chunk = source.read(CHUNK_BYTES)
                if not chunk:
                    break
                offset = index * CHUNK_BYTES
                headers = {
                    "X-Aivyla-Output-Key": secret,
                    "X-Chunk-Index": str(index),
                    "X-Chunk-Length": str(len(chunk)),
                    "X-Chunk-Sha256": hashlib.sha256(chunk).hexdigest(),
                    "X-Output-Sha256": checksum,
                    "X-Output-Size": str(size),
                    "X-Output-Name": filename,
                    "X-Final-Chunk": "1" if offset + len(chunk) == size else "0",
                    "Content-Type": "application/octet-stream",
                }
                last_error = None
                for attempt in range(3):
                    try:
                        res = requests.put(endpoint, data=chunk, headers=headers, timeout=(20, 90))
                        res.raise_for_status()
                        if not res.json().get("ok"):
                            raise RuntimeError("Studio upload did not acknowledge chunk")
                        last_error = None
                        break
                    except (requests.RequestException, ValueError, RuntimeError) as exc:
                        last_error = exc
                        if attempt < 2:
                            time.sleep(2 * (attempt + 1))
                if last_error:
                    raise RuntimeError(f"Direct upload of output chunk {index} failed") from last_error
                index += 1
        return {"filename": filename, "type": "aivyla_upload",
                "data": f"{job_id}:{int(output_index)}", "size": size,
                "sha256": checksum}
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
