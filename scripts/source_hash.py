#!/usr/bin/env python3
"""Hash renderer build inputs independently of documentation/evidence commits."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    paths = [ROOT / 'CMakeLists.txt']
    for directory in ('src', 'stream_compaction', 'external', 'cmake'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(ROOT))):
        name = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, 'little'))
        digest.update(name)
        digest.update(len(content).to_bytes(8, 'little'))
        digest.update(content)
    return digest.hexdigest()


if __name__ == '__main__':
    print(source_hash())
