#!/usr/bin/env python3
"""Retain original media helper snapshots and exact, fresh command receipts.

The renderer and recorder run separately on the authorized existing workstation.
This private driver validates downloaded evidence and prepares or publishes local
media. Publication requires the root review bound to the eight actual frames.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time

PREFIX = 'submission-checkpoint-final'
EXPECTED_SOURCE = 'a1979f061a22d1a7e815bb906ba62a2eeb1f0ae65f1c7a51ecbb19aa36bdd639'
ROOT = Path.cwd().resolve()
RESULT = ROOT / f'results/{PREFIX}-interaction'
TOOLS = RESULT / 'tools'
RECEIPTS = TOOLS / 'command-receipts'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.read_bytes() == source.read_bytes(), f'Existing snapshot differs: {destination}'
    else:
        shutil.copy2(source, destination)


def snapshots():
    sources = {
        '.local/run_final_interaction_remote.py': 'run_final_interaction_remote.py',
        '.local/setup_final_display_remote.py': 'setup_final_display_remote.py',
        '.local/run_checkpoint_graphical_step.py': 'run_checkpoint_graphical_step.py',
        '.local/validate_checkpoint_preview.py': 'validate_checkpoint_preview.py',
        '.local/finalize_interaction.py': 'finalize_interaction_artifacts.py',
        '.local/convert_final_interaction.py': 'convert_final_interaction.py',
        '.local/interaction_pixel_compare.cpp': 'interaction_pixel_compare.cpp',
        '.local/run_checkpoint_media_local.py': 'run_checkpoint_media_local.py',
    }
    records = []
    for private, name in sources.items():
        source, destination = ROOT / private, TOOLS / name
        exact_copy(source, destination)
        records.append({'path': str(destination.relative_to(ROOT)), 'original_private_path': private,
                        'bytes': destination.stat().st_size, 'sha256': sha(destination)})
    manifest = TOOLS / 'source-snapshots.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'schema_version': 1,
            'scope': 'Original media helper source snapshots; actual execution status is recorded separately in command receipts.',
            'recorded_at_utc': now(), 'renderer_source_sha256': EXPECTED_SOURCE,
            'evidence_prefix': PREFIX, 'compiled_executables_delivered': False,
            'controller_executed_remote_sha256': sha(ROOT / '.local/run_final_interaction_remote.py'),
            'sources': records}, indent=2) + '\n')
    else:
        assert json.loads(manifest.read_text())['sources'] == records, 'Source snapshot inventory differs'
    RECEIPTS.mkdir(exist_ok=True)


def run(command, stem, extra=None):
    record = RECEIPTS / f'{stem}.command.json'
    stdout, stderr = RECEIPTS / f'{stem}.stdout.json', RECEIPTS / f'{stem}.stderr.log'
    assert not any(path.exists() for path in (record, stdout, stderr)), 'Actual command receipt already exists'
    receipt = {'argv': command, 'cwd': str(ROOT), 'shell': False,
               'renderer_source_sha256': EXPECTED_SOURCE, 'evidence_prefix': PREFIX,
               'started_at_utc': now(), 'status': 'running', **(extra or {})}
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    start = time.monotonic()
    with stdout.open('x') as so, stderr.open('x') as se:
        process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=so, stderr=se, shell=False, check=False)
    receipt.update({'exit_code': process.returncode, 'ended_at_utc': now(),
                    'elapsed_seconds': time.monotonic() - start,
                    'stdout_log': stdout.name, 'stderr_log': stderr.name,
                    'status': 'succeeded' if process.returncode == 0 else 'failed'})
    if stem == 'full-artifact-finalize-v2' and process.returncode == 0:
        receipt['summary_sha256'] = sha(RESULT / 'summary.json')
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'exit_code', 'ended_at_utc')}))
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('validate-preview', 'prepare', 'publish', 'finalize'))
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    assert args.source == EXPECTED_SOURCE, 'Explicit source differs from the frozen corrected renderer'
    snapshots()
    relative = lambda name: str((TOOLS / name).relative_to(ROOT))
    common = ['--root', str(ROOT), '--expected-source-sha256', args.source,
              '--prefix', PREFIX, '--renderer-build-dir', 'build-submission']
    if args.phase == 'validate-preview':
        command = [sys.executable, '-B', relative('validate_checkpoint_preview.py'),
                   '--source', args.source, '--prefix', PREFIX]
        code = run(command, 'preview-validation')
        if code == 0:
            exact_copy(ROOT / '.local/checkpoint-preview-validation.json', RECEIPTS / 'checkpoint-preview-validation.json')
        return code
    if args.phase in ('prepare', 'publish'):
        command = [sys.executable, '-B', relative('convert_final_interaction.py'), args.phase, *common,
                   '--finalizer', relative('finalize_interaction_artifacts.py'),
                   '--pixel-source', relative('interaction_pixel_compare.cpp'),
                   '--verification-build-dir', '.local/interaction-helper-build/checkpoint-current']
        extra = {}
        if args.phase == 'publish':
            review = ROOT / '.local/checkpoint-interaction-root-review.json'
            assert review.exists(), 'Root has not completed the required eight-frame review'
            exact_copy(review, RESULT / 'root-visual-review.json')
            command.extend(['--visual-verification', str((RESULT / 'root-visual-review.json').relative_to(ROOT))])
            extra['root_review_sha256'] = sha(review)
        return run(command, 'media-' + args.phase, extra)
    command = [sys.executable, '-B', relative('finalize_interaction_artifacts.py'), *common,
               '--pixel-source', relative('interaction_pixel_compare.cpp'),
               '--build-dir', '.local/interaction-helper-build/checkpoint-current']
    return run(command, 'full-artifact-finalize-v2')


if __name__ == '__main__':
    raise SystemExit(main())
