"""Share only the base CUDA distribution closure with an isolated AvatarForever venv.

No base site-packages directory or its .pth files enter the target's sys.path.
The base package files and metadata are linked read-only by convention; target
bytecode writes are disabled and exact constraints prevent shared upgrades.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

PROBE = (
    "import json,sys,sysconfig; "
    "print(json.dumps(dict(prefix=sys.prefix,version=list(sys.version_info[:2]),"
    "purelib=sysconfig.get_path('purelib'),platlib=sysconfig.get_path('platlib'))))"
)

SHARED_ROOTS = ('pip', 'torch', 'torchaudio', 'torchvision', 'triton')
TARGET_PACKAGE_PINS = {
    'transformers': '4.57.6',
    'huggingface-hub': '0.36.2',
    'tokenizers': '0.22.2',
}

# Run under the base interpreter so requirement markers and metadata describe
# the actual image, not the helper's calling environment.
SNAPSHOT = r"""
import json, re, sys
from importlib.metadata import distribution
from pip._vendor.packaging.requirements import Requirement
canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()
pending = [(name, set()) for name in json.loads(sys.argv[1])]
selected = {}
extras_seen = {}
while pending:
    name, extras = pending.pop()
    name = canonical(name)
    previous = extras_seen.get(name)
    if previous is not None and extras <= previous:
        continue
    extras_seen[name] = (previous or set()) | extras
    dist = distribution(name)
    files = dist.files
    if not files:
        raise RuntimeError('Shared distribution has no file manifest: ' + name)
    top = sorted({str(path).split('/')[0] for path in files
                  if str(path).split('/')[0] not in ('..', '.', '', '__pycache__')})
    selected[name] = dict(name=name, version=dist.version,
                         site=str(dist.locate_file('')), top=top)
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker and not any(req.marker.evaluate({'extra': extra})
                                  for extra in ({''} | extras_seen[name])):
            continue
        dependency = distribution(req.name)
        if not req.specifier.contains(dependency.version, prereleases=True):
            raise RuntimeError('Base CUDA dependency conflict: ' + raw
                               + '; installed ' + dependency.version)
        pending.append((req.name, set(req.extras)))
print(json.dumps(selected, sort_keys=True))
"""


def environment(python):
    return json.loads(subprocess.check_output([str(python), '-I', '-B', '-c', PROBE],
                                              text=True, timeout=30))


def shared_distributions(base_python, *, roots=SHARED_ROOTS):
    selected = json.loads(subprocess.check_output(
        [str(base_python), '-I', '-B', '-c', SNAPSHOT, json.dumps(list(roots))],
        text=True, timeout=60))
    if not selected or set(selected) & set(TARGET_PACKAGE_PINS):
        raise RuntimeError('Invalid shared CUDA closure: Gemma packages must remain isolated')
    return selected


def link_dependencies(base_python, target_python, *, roots=SHARED_ROOTS):
    base, target = environment(base_python), environment(target_python)
    if base['version'] != target['version']:
        raise ValueError('Base and AvatarForever Python minor versions must match')
    if Path(base['prefix']).resolve() == Path(target['prefix']).resolve():
        raise ValueError('Refuse to modify the base Python environment')
    target_site = Path(target['purelib']).resolve()
    base_sites = list(dict.fromkeys(str(Path(base[key]).resolve())
                                    for key in ('purelib', 'platlib')))
    if not target_site.is_dir() or any(not Path(path).is_dir() for path in base_sites):
        raise ValueError('Missing Python site-packages directory')
    if str(target_site) in base_sites:
        raise ValueError('Base and target site-packages must differ')
    config = Path(target['prefix']) / 'pyvenv.cfg'
    if re.search(r'include-system-site-packages\s*=\s*true', config.read_text(), re.I):
        raise ValueError('Target venv must not include system site-packages')

    selected = shared_distributions(base_python, roots=roots)
    links = {}
    for package in selected.values():
        site = Path(package['site']).resolve()
        if str(site) not in base_sites:
            raise RuntimeError('Shared distribution is outside base site-packages: ' + package['name'])
        for top in package['top']:
            source = site / top
            if not source.exists():
                raise RuntimeError('Missing shared package path: ' + str(source))
            if top in links and links[top] != source:
                raise RuntimeError('Conflicting shared package path: ' + top)
            links[top] = source
    for top, source in links.items():
        destination = target_site / top
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError('Refuse to overwrite target package: ' + str(destination))

    # Remove only the obsolete bridge owned by the previous helper.
    (target_site / 'aivyla_base_dependencies.pth').unlink(missing_ok=True)
    guard = target_site / 'aivyla_shared_runtime.pth'
    guard.write_text('import sys; sys.dont_write_bytecode = True\n')
    for top, source in links.items():
        destination = target_site / top
        if not destination.is_symlink():
            destination.symlink_to(source, target_is_directory=source.is_dir())
    report = dict(base_sites=base_sites, target_site=str(target_site),
                  shared_packages={name: item['version'] for name, item in selected.items()},
                  shared_roots=list(roots), links=sorted(links))
    manifest = Path(target['prefix']) / 'aivyla_shared_runtime.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    visible = json.loads(subprocess.check_output(
        [str(target_python), '-I', '-c', 'import json,sys; print(json.dumps(sys.path))'],
        text=True, timeout=30))
    if any(path in visible for path in base_sites):
        raise RuntimeError('Base site-packages leaked into isolated AvatarForever sys.path')
    return dict(report, manifest=str(manifest))


def freeze_base(base_python, destination, *, roots=SHARED_ROOTS):
    """Constrain only shared CUDA dependencies and the target Gemma stack."""
    selected = shared_distributions(base_python, roots=roots)
    versions = {name: item['version'] for name, item in selected.items()}
    versions.update(TARGET_PACKAGE_PINS)
    for name, version in versions.items():
        if (not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name)
                or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version)):
            raise ValueError('Invalid shared package name/version: ' + name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(f'{name}=={version}\n'
                                   for name, version in sorted(versions.items())))
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path, required=True)
    parser.add_argument('--target-python', type=Path, required=True)
    parser.add_argument('--constraints', type=Path)
    args = parser.parse_args()
    report = link_dependencies(args.base_python, args.target_python)
    if args.constraints:
        versions = freeze_base(args.base_python, args.constraints)
        report['constrained_packages'] = len(versions)
        report['constraints'] = str(args.constraints)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
