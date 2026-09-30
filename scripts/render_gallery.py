#!/usr/bin/env python3
"""Render the showcase and feature comparisons with fixed seeds and settings."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cover-samples', type=int, default=2048)
    parser.add_argument('--comparison-samples', type=int, default=512)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('gallery-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    configurations = [
        ('workbench', 'optical-workbench', args.cover_samples, '960x640', {}),
        ('comparison-bump', 'optical-workbench', args.comparison_samples, '640x424', {}),
        ('comparison-no-bump', 'workbench-no-bump', args.comparison_samples, '640x424', {}),
        ('comparison-procedural', 'workbench-procedural', args.comparison_samples, '640x424', {}),
        ('comparison-glass', 'optical-workbench', args.comparison_samples, '640x424', {}),
        ('comparison-mirror', 'workbench-mirror', args.comparison_samples, '640x424', {}),
        ('comparison-dof', 'workbench-dof', args.comparison_samples, '640x424', {}),
        ('comparison-nee', 'optical-workbench', 64, '640x424', {}),
        ('comparison-no-nee', 'optical-workbench', 64, '640x424', {'nee': 'off'}),
        ('comparison-aa', 'optical-workbench', args.comparison_samples, '640x424', {}),
        ('comparison-no-aa', 'optical-workbench', args.comparison_samples, '640x424', {'aa': 'off'}),
        ('restart-partial', 'optical-workbench', 16, '640x424', {}),
        ('restart-resumed', 'optical-workbench', args.comparison_samples, '640x424', {}),
        ('debug-normals', 'optical-workbench', 1, '640x424', {'debug': 'normals'}),
        ('debug-uv', 'optical-workbench', 1, '640x424', {'debug': 'uv'}),
        ('debug-depth', 'optical-workbench', 1, '640x424', {'debug': 'depth'}),
    ]
    records = []
    for name, scene, samples, resolution, overrides in configurations:
        toggles = dict(aa='on', nee='on', bvh='on', compact='on', sort='off')
        toggles.update(overrides)
        prefix = output / name
        command = [str(args.binary.resolve()), str(ROOT / 'scenes' / (scene + '.json')),
                   '--samples', str(samples), '--resolution', resolution, '--seed', '947',
                   '--depth', '10', '--output', str(prefix),
                   *[x for key, value in toggles.items() for x in ('--' + key, value)]]
        if 'debug' not in toggles:
            command.append('--benchmark')
        if name == 'restart-partial':
            command.extend(['--checkpoint', str(output / 'restart.checkpoint')])
        elif name == 'restart-resumed':
            command.extend(['--resume', str(output / 'restart.checkpoint')])
        run = subprocess.run(command, capture_output=True, text=True)
        prefix.with_suffix('.stdout.txt').write_text(run.stdout)
        prefix.with_suffix('.stderr.txt').write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name}: {run.stderr}')
        records.append(dict(name=name, scene=scene, samples=samples, resolution=resolution,
                            seed=947, settings=toggles, metadata=prefix.with_suffix('.json').name))
        print(f'Rendered {name}', flush=True)
    (output / 'gallery.json').write_text(json.dumps(records, indent=2) + '\n')
    # Both processes use the same sampling configuration as comparison-glass.
    # Header and checksum also match because completed sample targets are equal.
    uninterrupted = (output / 'comparison-glass.sum.bin').read_bytes()
    resumed = (output / 'restart-resumed.sum.bin').read_bytes()
    if uninterrupted != resumed:
        raise AssertionError('showcase restart differs from uninterrupted accumulation')
    print(output)


if __name__ == '__main__':
    main()
