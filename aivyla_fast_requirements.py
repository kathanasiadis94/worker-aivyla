"""Conservative startup fast-path for already installed ComfyUI node requirements.

Unknown pip constructs fall back to pip; never silently skip unresolved dependencies.
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import time


def _get_requirement_class():
    """Use the public parser, or pip's vendored parser in minimal Python envs.

    If neither exists, return None and use the original pip resolver. Never
    guess how to parse PEP 508 specifiers using ad-hoc string comparisons.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
        except ImportError:
            return None
    return Requirement


def requirements_satisfied(path: Path) -> bool:
    Requirement = _get_requirement_class()
    if Requirement is None:
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            # URLs, includes, editable refs, hashes, index options, and continuations
            # must be delegated to pip for correct behavior.
            if (raw.startswith(("-", ".", "/", "git+", "http:", "https:", "file:"))
                    or "\\" in raw or " --" in raw or " #" not in raw and "#" in raw):
                return False
            raw = raw.split(" #", 1)[0].strip()
            req = Requirement(raw)
            if req.url or req.extras:
                return False
            if req.marker is not None and not req.marker.evaluate():
                continue
            version = metadata.version(req.name)
            if not req.specifier.contains(version, prereleases=True):
                return False
        return True
    except Exception:
        # Defensive: unusual requirement lines should always trigger ordinary pip.
        return False


def ensure(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    started = time.monotonic()
    enabled = os.getenv("AIVYLA_FAST_REQUIREMENTS", "1") == "1"
    if enabled and requirements_satisfied(path):
        print(f"Aivyla fast-start: requirements satisfied; pip skipped: {path} ({time.monotonic()-started:.3f}s)", flush=True)
        return
    print(f"Aivyla fast-start: using original pip install: {path}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", str(path)], check=True)
    print(f"Aivyla fast-start: pip completed: {path} ({time.monotonic()-started:.2f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: aivyla_fast_requirements.py REQUIREMENTS_PATH")
    ensure(Path(sys.argv[1]))
