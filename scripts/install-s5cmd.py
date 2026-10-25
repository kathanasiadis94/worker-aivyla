"""Install the official pinned Linux amd64 release during the image build."""
import hashlib
import io
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request


def main(destination='/usr/local/bin/s5cmd'):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
        raise RuntimeError('Build this RunPod image with --platform linux/amd64')
    name = 's5cmd_2.3.0_Linux-64bit.tar.gz'
    base = 'https://github.com/peak/s5cmd/releases/download/v2.3.0/'
    # Published in the official v2.3.0 s5cmd_checksums.txt.
    expected = 'de0fdbfa3aceae55e069ba81a0fc17b2026567637603734a387b2fca06c299b4'
    with urllib.request.urlopen(base + name, timeout=120) as response:
        archive = response.read(20 * 1024**2)
    if hashlib.sha256(archive).hexdigest() != expected:
        raise RuntimeError('Official s5cmd release checksum mismatch')
    target = Path(destination)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as package:
        member = package.getmember('s5cmd')
        if not member.isfile():
            raise RuntimeError('Expected a regular s5cmd executable')
        with package.extractfile(member) as source:
            target.write_bytes(source.read())
    target.chmod(0o755)
    subprocess.run([str(target), 'version'], check=True)


if __name__ == '__main__':
    main()
