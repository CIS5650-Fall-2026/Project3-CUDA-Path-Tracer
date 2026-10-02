#!/usr/bin/env python3
"""Original sequential controller for evidence after the checker boundary fix."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    assert len(args.source) == 64 and all(c in '0123456789abcdef' for c in args.source)
    root = Path.cwd()
    output = root / 'results/submission-final-controller'
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    script = Path(__file__).resolve()
    snapshot = output / 'executed-controller.py'
    snapshot.write_bytes(script.read_bytes())
    assert digest(snapshot) == digest(script)
    state = {'status': 'running', 'started_utc': now(), 'expected_renderer_source_sha256': args.source,
             'driver_sha256': digest(script), 'executed_driver': str(snapshot.relative_to(root)),
             'sequential_gpu_phases': True, 'phases': [], 'errors': []}
    state_path = output / 'summary.json'
    def persist():
        temp = output / '.summary.pending'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(state_path)
    def source_hash():
        return subprocess.check_output([sys.executable, 'scripts/source_hash.py'], text=True).strip()
    suite = lambda phase: [sys.executable, 'scripts/run_hedge_suite.py', '--phase', phase,
                          '--build', 'build-submission', '--prefix', 'submission-final']
    phases = [(name, suite(name), 'results/submission-final-' + ('rebuild' if name == 'build' else 'suite-' + name) + '/summary.json')
              for name in ('build', 'new-validation', 'sanitizers')]
    phases.append(('gltf', [sys.executable, 'scripts/measure_gltf.py', '--binary',
                  'build-submission/bin/cis565_path_tracer_headless', '--build', 'build-submission',
                  '--output', 'results/submission-final-gltf', '--phase', 'all'],
                  'results/submission-final-gltf/summary.json'))
    phases.extend((name, suite(name), 'results/submission-final-suite-' + name + '/summary.json')
                  for name in ('baseline-validation', 'baseline-measurements', 'new-measurements', 'gallery'))
    persist()
    try:
        assert source_hash() == args.source, 'source mismatch before refresh'
        for name, argv, summary_relative in phases:
            assert source_hash() == args.source, 'source changed before ' + name
            item = {'name': name, 'argv': argv, 'started_utc': now(), 'status': 'running',
                    'stdout': 'logs/' + name + '.stdout.log', 'stderr': 'logs/' + name + '.stderr.log',
                    'summary': summary_relative}
            state['phases'].append(item)
            persist()
            tick = time.monotonic()
            with (output / item['stdout']).open('xb') as stdout, (output / item['stderr']).open('xb') as stderr:
                code = subprocess.call(argv, stdout=stdout, stderr=stderr)
            item.update(returncode=code, ended_utc=now(), elapsed_seconds=time.monotonic()-tick)
            summary = json.loads((root / summary_relative).read_text())
            assert source_hash() == args.source, 'source changed after ' + name
            if name == 'gltf':
                assert code == 0 and summary['status'] == 'completed' and not summary['errors'], 'glTF phase failed'
            else:
                assert summary['source_input_sha256_start'] == summary['source_input_sha256_end'] == args.source
                if name != 'build':
                    assert summary['binary_sha256_start'] == summary['binary_sha256_end']
                assert not summary['errors'], 'suite phase errors'
                allowed = code == 0 and summary['status'] == 'passed'
                allowed |= name == 'new-measurements' and code == 2 and summary['status'] == 'completed_not_qualified'
                assert allowed, 'suite phase failed: ' + name
            item['status'] = 'passed' if code == 0 else 'completed_not_qualified'
            item['summary_sha256'] = digest(root / summary_relative)
            persist()
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'
        state['errors'].append(type(error).__name__ + ': ' + str(error))
        if state['phases'] and state['phases'][-1]['status'] == 'running':
            state['phases'][-1]['status'] = 'failed'
        raise
    finally:
        state['finished_utc'] = now()
        persist()
    print(json.dumps({'status': state['status'], 'phases': len(state['phases']), 'source': args.source}))

if __name__ == '__main__':
    main()
