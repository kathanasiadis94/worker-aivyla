"""Expose the immutable base venv's dependencies to the AvatarForever venv.

--system-site-packages exposes the *system* interpreter, not a sibling venv.
Create an explicit .pth bridge in the destination only. Never install into or
modify the base environment. Its CUDA torch/native extensions remain intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

PROBE = (
    "import json,sys,sysconfig; "
    "print(json.dumps(dict(prefix=sys.prefix,version=list(sys.version_info[:2]),"
    "purelib=sysconfig.get_path('purelib'),platlib=sysconfig.get_path('platlib'))))"
)


def environment(python):
    return json.loads(subprocess.check_output([str(python), '-I', '-c', PROBE],
                                              text=True, timeout=30))


def link_dependencies(base_python, target_python):
    base, target = environment(base_python), environment(target_python)
    if base['version'] != target['version']:
        raise ValueError('Base and AvatarForever Python minor versions must match')
    if Path(base['prefix']).resolve() == Path(target['prefix']).resolve():
        raise ValueError('Refuse to modify the base Python environment')
    target_site = Path(target['purelib']).resolve()
    base_sites = list(dict.fromkeys(str(Path(base[key]).resolve()) for key in ('purelib', 'platlib')))
    if not target_site.is_dir() or any(not Path(path).is_dir() for path in base_sites):
        raise ValueError('Missing Python site-packages directory')
    if str(target_site) in base_sites:
        raise ValueError('Base and target site-packages must differ')
    bridge = target_site / 'aivyla_base_dependencies.pth'
    # addsitedir also processes the base's existing .pth files, including
    # editable/native extension package paths. Base files are never changed.
    source = ''.join(f'import site; site.addsitedir({path!r})\n' for path in base_sites)
    bridge.write_text(source, encoding='utf-8')
    visible = json.loads(subprocess.check_output(
        [str(target_python), '-I', '-c', 'import json,sys; print(json.dumps(sys.path))'],
        text=True, timeout=30))
    if not all(path in visible for path in base_sites):
        raise RuntimeError('Base dependencies are not visible in the target interpreter')
    return {'base_sites': base_sites, 'target_site': str(target_site), 'bridge': str(bridge)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path, required=True)
    parser.add_argument('--target-python', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(link_dependencies(args.base_python, args.target_python), indent=2))


if __name__ == '__main__':
    main()
