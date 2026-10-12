"""Opt-in, best-effort ComfyUI preview relay for Aivyla Global.

The Studio must implement POST /api/internal/runpod-previews/{job_id},
authenticate the internal secret, validate job associations, and expire data.
Failure to transmit a preview must never fail a generation.
"""
import base64
import logging
import os
import queue
import threading
import time
from urllib.parse import quote

import requests
from aivyla_output_upload import _settings

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 240_000
MIN_IMAGE_INTERVAL_S = 2.5
MIN_PROGRESS_INTERVAL_S = 1.0


def decode_preview(packet):
    """Decode only ComfyUI binary type=1 JPEG/PNG preview frames."""
    if not isinstance(packet, bytes) or len(packet) < 12:
        return None
    if int.from_bytes(packet[:4], "big") != 1:
        return None
    image_type = int.from_bytes(packet[4:8], "big")
    data = packet[8:]
    if not data or len(data) > MAX_IMAGE_BYTES:
        return None
    if image_type == 1 and data.startswith(b"\xff\xd8"):
        return "image/jpeg", data
    if image_type == 2 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", data
    return None


class PreviewPublisher:
    def __init__(self, job_id, enabled=False):
        self.enabled = (
            enabled is True
            and os.environ.get("AIVYLA_LIVE_PREVIEW_ENABLED", "false").lower()
            == "true"
        )
        self.prompt_id = None
        self.node = None
        self.step = None
        self.total = None
        self.last_image_at = 0.0
        self.last_progress_at = 0.0
        self._queue = queue.Queue(maxsize=2)
        self._thread = None
        if not self.enabled:
            return
        try:
            base, secret = _settings()
            self.url = (
                base + "/api/internal/runpod-previews/"
                + quote(str(job_id), safe="")
            )
            self.secret = secret
        except (ValueError, RuntimeError) as exc:
            log.warning("Live preview is unavailable: %s", type(exc).__name__)
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._sender, daemon=True)
        self._thread.start()

    def _push(self, item):
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def event(self, kind, data):
        if not self.enabled or not isinstance(data, dict):
            return
        actual_prompt = data.get("prompt_id")
        if self.prompt_id and actual_prompt != self.prompt_id:
            return
        if kind == "executing":
            self.node = data.get("node")
        elif kind == "progress":
            now = time.monotonic()
            if now - self.last_progress_at < MIN_PROGRESS_INTERVAL_S:
                return
            self.last_progress_at = now
            self.step = data.get("value")
            self.total = data.get("max")
        else:
            return
        self._push({
            "type": "progress",
            "prompt_id": self.prompt_id,
            "node": self.node,
            "step": self.step,
            "total": self.total,
        })

    def image(self, packet):
        if not self.enabled or not self.prompt_id:
            return
        now = time.monotonic()
        if now - self.last_image_at < MIN_IMAGE_INTERVAL_S:
            return
        decoded = decode_preview(packet)
        if decoded is None:
            return
        mime, data = decoded
        self.last_image_at = now
        self._push({
            "type": "preview",
            "prompt_id": self.prompt_id,
            "node": self.node,
            "step": self.step,
            "total": self.total,
            "mime": mime,
            "image_base64": base64.b64encode(data).decode("ascii"),
        })

    def _sender(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                response = requests.post(
                    self.url,
                    json=item,
                    headers={"X-Aivyla-Output-Key": self.secret},
                    timeout=(3, 5),
                )
                if response.status_code in (401, 403, 404, 405, 413):
                    log.warning(
                        "Disabling preview relay: Studio HTTP %s",
                        response.status_code,
                    )
                    self.enabled = False
                    return
                response.raise_for_status()
            except requests.RequestException as exc:
                log.debug(
                    "Skipping preview update due to relay failure: %s",
                    type(exc).__name__,
                )

    def close(self):
        if self._thread is None:
            return
        self.enabled = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=0.2)
