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
MAX_ANIMATION_BYTES = 900_000  # 1.2 MB base64 max, bounded POST to Studio
MIN_IMAGE_INTERVAL_S = 1.0
MIN_ANIMATION_INTERVAL_S = 0.8
MIN_PROGRESS_INTERVAL_S = 0.5


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
        self.last_animation_at = 0.0
        self.last_progress_at = 0.0
        self._queue = queue.Queue(maxsize=2)
        self._thread = None
        self._relay_failures = 0
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
        # KJNodes sends this event to the current ComfyUI WebSocket client,
        # omitting prompt_id. The handler owns one WebSocket per RunPod request;
        # relay only after that request's prompt_id is known.
        if kind == "kj_preview_override":
            self._kj_preview(data)
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

    def _kj_preview(self, data):
        if not self.enabled or not self.prompt_id:
            return
        now = time.monotonic()
        node_id = data.get("node_id")
        if not isinstance(node_id, str) or not node_id.startswith("aivyla_kj_anim_"):
            return  # Ignore unrelated custom KJ previews from other workflows.
        if node_id is not None:
            self.node = str(node_id)[:128]
        step, total = data.get("step"), data.get("total")
        if type(step) is int and type(total) is int and 0 <= step <= total <= 100_000:
            self.step, self.total = step, total
        mime = data.get("mime")
        encoded = data.get("image")
        if mime in ("video/mp4", "image/webp") and isinstance(encoded, str):
            if now - self.last_animation_at < MIN_ANIMATION_INTERVAL_S:
                return
            # The Studio independently checks base64 and container signatures.
            # No binary conversion in the websocket receive thread.
            if not 0 < len(encoded) <= ((MAX_ANIMATION_BYTES + 2) // 3) * 4 + 4:
                log.warning("Skipping oversized animated preview (%d base64 chars)", len(encoded))
                return
            if self.last_animation_at == 0:
                print(f"Aivyla animation relay: first {mime} clip, {len(encoded)} base64 chars", flush=True)
            self.last_animation_at = now
            self._push({
                "type": "animation",
                "prompt_id": self.prompt_id,
                "node": self.node,
                "step": self.step,
                "total": self.total,
                "mime": mime,
                "animation_base64": encoded,
                "fps": data.get("fps") or 8,
            })
        elif mime == "image/jpeg" and isinstance(encoded, str):
            if now - self.last_image_at < MIN_IMAGE_INTERVAL_S:
                return
            if 0 < len(encoded) <= ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
                self.last_image_at = now
                self._push({
                    "type": "preview", "prompt_id": self.prompt_id,
                    "node": self.node, "step": self.step, "total": self.total,
                    "mime": "image/jpeg", "image_base64": encoded,
                })
        elif now - self.last_progress_at >= MIN_PROGRESS_INTERVAL_S:
            self.last_progress_at = now
            self._push({
                "type": "progress", "prompt_id": self.prompt_id,
                "node": self.node, "step": self.step, "total": self.total,
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
                if response.status_code == 413 and item.get("type") == "animation":
                    log.warning("Studio rejected oversized animated preview; keeping progress relay")
                    continue
                if response.status_code in (401, 403, 404, 405, 413):
                    log.warning(
                        "Disabling preview relay: Studio HTTP %s",
                        response.status_code,
                    )
                    self.enabled = False
                    return
                response.raise_for_status()
                self._relay_failures = 0
            except requests.RequestException as exc:
                self._relay_failures += 1
                if self._relay_failures == 1 or self._relay_failures % 10 == 0:
                    log.warning(
                        "Preview relay failure (%d): %s; generation continues",
                        self._relay_failures, type(exc).__name__,
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
