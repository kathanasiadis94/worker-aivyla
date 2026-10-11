#!/usr/bin/env python3
"""Single-command R2 publishing for new models, LoRAs and trusted custom nodes.

Install locally: python3 -m pip install boto3
Set R2 S3 endpoint + AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in environment.
Credentials should never be committed to GitHub or included in output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile

from aivyla_r2_registry import Registry, _CATEGORIES, safe_relative


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def object_exists(registry, key):
    try:
        return registry.client.head_object(Bucket=registry.bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
        if code in ('404','NoSuchKey','NotFound') or isinstance(exc, registry.client.exceptions.NoSuchKey):
            return None
        raise


def upload(registry, src, key, *, allow_same=False):
    src = Path(src)
    head = object_exists(registry, key)
    if head:
        if allow_same and head['ContentLength'] == src.stat().st_size:
            print('EXISTS: object with matching size (verify manually if needed):', key)
            return
        raise RuntimeError(f'R2 key already exists, refusing overwrite: {key}')
    registry.client.upload_file(str(src), registry.bucket, key)
    remote = registry.client.head_object(Bucket=registry.bucket, Key=key)
    if remote['ContentLength'] != src.stat().st_size:
        raise IOError('Upload size mismatch')
    receipt = {'bytes': src.stat().st_size,
               'etag': remote['ETag'].strip('"'), 'sha256': digest(src)}
    receipt_key = '_aivyla/receipts/'+key+'.json'
    registry.client.put_object(Bucket=registry.bucket, Key=receipt_key,
                               Body=json.dumps(receipt,sort_keys=True).encode(),
                               ContentType='application/json')
    print('UPLOADED:', key, f'({src.stat().st_size/1024**3:.2f} GiB; SHA256 receipt saved)')


def read_node_manifest(registry):
    try:
        return registry.client.get_object(Bucket=registry.bucket,Key='nodes/registry.json')['Body'].read()
    except Exception as e:
        code = getattr(e,'response',{}).get('Error',{}).get('Code')
        if code in ('404','NoSuchKey','NotFound') or isinstance(e, registry.client.exceptions.NoSuchKey):
            return b'{"schema":1,"nodes":[]}'
        raise


def save_node_manifest(registry, manifest):
    # Explicit manifest update; use one publisher at a time.
    registry.client.put_object(Bucket=registry.bucket,Key='nodes/registry.json',
        Body=json.dumps(manifest,indent=2,sort_keys=True).encode(),
        ContentType='application/json')


def pack_node(folder, name, dest):
    folder = Path(folder).resolve()
    if not (folder.is_dir() and (folder/'__init__.py').is_file()):
        raise ValueError('Node directory must contain __init__.py')
    excluded_dirs={'.git','.hg','.svn','__pycache__','.venv','venv','node_modules','.pytest_cache'}
    excluded_files={'.env','.env.local','.DS_Store'}
    with tarfile.open(dest,'w:gz') as tf:
        for path in sorted(folder.rglob('*')):
            rel=path.relative_to(folder)
            if any(component in excluded_dirs for component in rel.parts) or path.name in excluded_files:
                continue
            if path.is_symlink():
                raise ValueError('Symlinks cannot be packaged as R2 nodes')
            if not (path.is_dir() or path.is_file()):
                raise ValueError('Unsupported special file in custom node')
            tf.add(path,arcname=str(Path(name)/rel),recursive=False)
    print('PACKAGED:',name, 'SHA256:',digest(dest))


def node_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,90}', value):
        raise argparse.ArgumentTypeError('Node name may only contain letters, digits, _, ., -')
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('list-models')
    sub.add_parser('list-nodes')
    m=sub.add_parser('upload-model')
    m.add_argument('file',type=Path)
    m.add_argument('--category',required=True,choices=sorted(_CATEGORIES))
    m.add_argument('--name',help='R2 filename/subpath instead of source basename')
    n=sub.add_parser('publish-node')
    n.add_argument('directory',type=Path)
    n.add_argument('--name',type=node_name,required=True)
    n.add_argument('--version',required=True)
    n.add_argument('--enable',action='store_true',help='Explicitly enable executable code on next worker start')
    t=sub.add_parser('set-node')
    t.add_argument('name',type=node_name)
    t.add_argument('state',choices=['enable','disable'])
    args=parser.parse_args(argv)
    r=Registry()
    if args.cmd=='list-models':
        for m in r.catalog():
            print(f'{m["category"]}/{m["filename"]} ({m["bytes"]/1024**3:.2f} GiB)')
    elif args.cmd=='list-nodes':
        for n in r.nodes():
            print(f'{n["name"]} {n["version"]}: {"ENABLED" if n["enabled"] else "AVAILABLE"}')
    elif args.cmd=='upload-model':
        name=args.name or args.file.name
        if not safe_relative(name) or not name.lower().endswith(('.safetensors','.sft','.ckpt','.pt','.pth','.bin','.gguf','.onnx')):
            raise ValueError('Unsafe model filename or unsupported format')
        upload(r,args.file, f'models/{args.category}/{name}')
    elif args.cmd=='publish-node':
        if not safe_relative(args.version) or '/' in args.version or len(args.version)>80:
            raise ValueError('Unsafe node version')
        name=args.name
        key=f'nodes/packages/{name}/{args.version}.tar.gz'
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'node.tar.gz'
            pack_node(args.directory,name,archive)
            checksum=digest(archive)
            upload(r,archive,key)
        doc=json.loads(read_node_manifest(r))
        if doc.get('schema') != 1 or not isinstance(doc.get('nodes'),list):
            raise ValueError('Invalid node registry; refusing modification')
        doc['nodes']=[entry for entry in doc['nodes'] if entry['name']!=name]
        doc['nodes'].append({'name':name,'version':args.version,'key':key,
                             'sha256':checksum,'enabled':bool(args.enable)})
        save_node_manifest(r,doc)
        print(f'NODE CATALOG UPDATED: {name}; enabled={args.enable}. '
              'An enabled node is loaded on the NEXT worker start.')
    elif args.cmd=='set-node':
        doc=json.loads(read_node_manifest(r))
        found=False
        for item in doc.get('nodes',[]):
            if item['name']==args.name:
                item['enabled']=args.state=='enable'
                found=True
        if not found:
            raise RuntimeError('Node not in registry; publish it first')
        save_node_manifest(r,doc)
        print('NODE STATUS UPDATED; takes effect on next worker start:',args.name,args.state)


if __name__=='__main__':
    main()
