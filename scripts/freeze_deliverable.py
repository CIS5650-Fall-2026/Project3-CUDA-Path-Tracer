#!/usr/bin/env python3
"""Hash a clean committed deliverable and keep audit artifacts outside it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, default=ROOT.parent / 'project_3_reviews')
    args = parser.parse_args()
    if git('status', '--porcelain'):
        raise RuntimeError('Commit the complete deliverable before freezing it; working tree must be clean.')
    revision = git('rev-parse', 'HEAD')
    tree = git('rev-parse', 'HEAD^{tree}')
    rows = []
    for name in git('ls-tree', '-r', '--name-only', 'HEAD').splitlines():
        data = (ROOT / name).read_bytes()
        rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    audit = args.audit_root.resolve() / digest
    if audit == ROOT or ROOT in audit.parents:
        raise RuntimeError('Review reports must live outside the frozen repository.')
    audit.mkdir(parents=True, exist_ok=True)
    (audit / 'bundle.json').write_text(json.dumps(dict(revision=revision, tree=tree, bundle_sha256=digest, files=rows), indent=2) + '\n')
    print(json.dumps(dict(revision=revision, tree=tree, bundle_sha256=digest, audit_directory=str(audit)), indent=2))


if __name__ == '__main__':
    main()
