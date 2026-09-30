#!/usr/bin/env python3
"""Record file-level provenance against the authorized official baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '8695a5af18ae6fc130d1d38713be82bbbff7f38f'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/change-manifest.json')
    args = parser.parse_args()
    baseline = set(git('ls-tree', '-r', '--name-only', BASELINE))
    candidates = set(git('ls-files', '--cached', '--others', '--exclude-standard'))
    rows = []
    for name in sorted(candidates):
        path = ROOT / name
        if not path.is_file() or path == args.output:
            continue
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if name in baseline:
            original = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASELINE + ':' + name])
            category = 'inherited-starter' if content == original else 'modified-starter'
            origin = BASELINE
        elif name == 'scripts/remote.py':
            category, origin = 'authorized-own-project2-adaptation', '6a13e6fbe3a998c67fdbbdbe6643a5b4c4233b31:scripts/remote.py'
        elif name.startswith('stream_compaction/'):
            category, origin = 'authorized-own-project2-reuse', '6a13e6fbe3a998c67fdbbdbe6643a5b4c4233b31; see source headers and provenance.md'
        elif name.startswith(('scenes/assets/', 'scenes/')):
            category, origin = 'original-generated-asset', 'scripts/generate_assets.py'
        elif name.startswith(('results/', 'img/')):
            category, origin = 'generated-validation-evidence', 'documented CUDA build and scripts'
        else:
            category, origin = 'new-original-project-content', 'independently authored in this project'
        rows.append(dict(path=name, sha256=sha, category=category, origin=origin))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(baseline=BASELINE, files=rows), indent=2) + '\n')
    print(f'Recorded {len(rows)} files in {args.output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
