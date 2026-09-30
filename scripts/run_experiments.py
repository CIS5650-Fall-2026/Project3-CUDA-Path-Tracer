#!/usr/bin/env python3
"""Run controlled Release-renderer experiments without external Python packages."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three measured repeats are required')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('experiments-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    experiments = []
    for label, scene, feature in [('compact-open', 'open-benchmark', 'compact'),
                                  ('compact-closed', 'closed-benchmark', 'compact'),
                                  ('sort-uniform', 'uniform-benchmark', 'sort'),
                                  ('sort-mixed', 'optical-workbench', 'sort')]:
        for value in ('off', 'on'):
            experiments.append((label, value, scene, ['--' + feature, value], '160x106', 16))
    for level in ('low', 'medium', 'high'):
        for value in ('off', 'on'):
            experiments.append(('bvh-' + level, value, 'complexity-' + level, ['--bvh', value], '96x64', 8))
    for value, scene in [('file', 'workbench-no-bump'), ('procedural', 'workbench-procedural')]:
        experiments.append(('texture', value, scene, [], '160x106', 16))
    for label, variants in [
        ('bump', [('off', 'workbench-no-bump', []), ('on', 'optical-workbench', [])]),
        ('refraction', [('mirror', 'workbench-mirror', []), ('glass', 'optical-workbench', [])]),
        ('dof', [('off', 'optical-workbench', []), ('on', 'workbench-dof', [])]),
        ('nee', [('off', 'optical-workbench', ['--nee', 'off']), ('on', 'optical-workbench', [])]),
        ('checkpoint', [('off', 'optical-workbench', []), ('on', 'optical-workbench', [])])]:
        for value, scene, switches in variants:
            experiments.append((label, value, scene, switches, '160x106', 16))
    rows = []
    for label, variant, scene, switches, resolution, samples in experiments:
        for repeat in range(-3, args.repeats):
            phase = 'warmup' if repeat < 0 else 'measured'
            prefix = output / f'{label}-{variant}-{phase}-{abs(repeat)}'
            toggles = dict(compact='on', sort='off', bvh='on', aa='on', nee='on')
            for key, value in zip(switches[::2], switches[1::2]):
                toggles[key[2:]] = value
            command = [str(args.binary.resolve()), str(ROOT / 'scenes' / (scene + '.json')),
                       '--samples', str(samples), '--seed', '947', '--output', str(prefix),
                       '--resolution', resolution, '--depth', '10', '--benchmark',
                       *[item for key, value in toggles.items() for item in ('--' + key, value)]]
            if label == 'checkpoint' and variant == 'on':
                command.extend(['--checkpoint', str(prefix) + '.checkpoint', '--checkpoint-every', '8'])
            start = time.perf_counter()
            run = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            prefix.with_suffix('.stdout.txt').write_text(run.stdout)
            prefix.with_suffix('.stderr.txt').write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(f'{label}/{variant} failed: {run.stderr}')
            rows.append(dict(experiment=label, variant=variant, phase=phase, repeat=abs(repeat),
                             process_seconds=elapsed, scene=scene, resolution=resolution,
                             samples=samples, seed=947, prefix=prefix.name))
            print(f'{label} {variant} {phase} {abs(repeat)}: {elapsed:.4f}s', flush=True)
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label, variant, *_ in experiments:
        values = [r['process_seconds'] for r in rows if r['experiment'] == label and r['variant'] == variant and r['phase'] == 'measured']
        summary.append(dict(experiment=label, variant=variant, repeats=len(values),
                            process_median_seconds=statistics.median(values),
                            process_min_seconds=min(values), process_max_seconds=max(values)))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
