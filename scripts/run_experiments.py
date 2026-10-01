#!/usr/bin/env python3
"""Run controlled Release-renderer experiments without external Python packages."""
import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SEED = 947
WARMUP_RUNS = 3
DEFAULT_RR_START = 3


@dataclass(frozen=True)
class Experiment:
    experiment: str
    variant: str
    scene: str
    settings: tuple
    resolution: str
    samples: int
    depth: int


def experiment_matrix():
    """Return the ordered 48-variant experiment matrix."""
    experiments = []

    # Preserve the original 26 variants. Wavefront and roulette are disabled
    # explicitly so these remain measurements of the original feature set.
    for label, scene, feature in (
        ('compact-open', 'open-benchmark', 'compact'),
        ('compact-closed', 'closed-benchmark', 'compact'),
        ('sort-uniform', 'uniform-benchmark', 'sort'),
        ('sort-mixed', 'optical-workbench', 'sort'),
    ):
        for value in ('off', 'on'):
            experiments.append(Experiment(
                label, value, scene, ((feature, value),), '160x106', 16, 10))

    for level in ('low', 'medium', 'high'):
        for value in ('off', 'on'):
            experiments.append(Experiment(
                'bvh-' + level, value, 'complexity-' + level,
                (('bvh', value),), '96x64', 8, 10))

    for value, scene in (
        ('file', 'workbench-no-bump'),
        ('procedural', 'workbench-procedural'),
    ):
        experiments.append(Experiment(
            'texture', value, scene, (), '160x106', 16, 10))

    for label, variants in (
        ('bump', (('off', 'workbench-no-bump', ()),
                  ('on', 'optical-workbench', ()))),
        ('refraction', (('mirror', 'workbench-mirror', ()),
                        ('glass', 'optical-workbench', ()))),
        ('dof', (('off', 'optical-workbench', ()),
                 ('on', 'workbench-dof', ()))),
        ('nee', (('off', 'optical-workbench', (('nee', 'off'),)),
                 ('on', 'optical-workbench', ()))),
        ('checkpoint', (('off', 'optical-workbench', ()),
                        ('on', 'optical-workbench', ()))),
    ):
        for value, scene, settings in variants:
            experiments.append(Experiment(
                label, value, scene, settings, '160x106', 16, 10))

    # Compare both unified schedules with the material-queue schedule on four
    # materially different workloads. Roulette remains fixed off so it cannot
    # change queue survival or confound the scheduling comparison.
    wave_scenes = (
        ('uniform', 'uniform-benchmark'),
        ('mixed', 'optical-workbench'),
        ('closed', 'closed-benchmark'),
        ('mesh-heavy', 'complexity-high'),
    )
    wave_modes = (
        ('legacy-unsorted', (('wavefront', 'off'), ('sort', 'off'))),
        ('legacy-sorted', (('wavefront', 'off'), ('sort', 'on'))),
        ('wavefront', (('wavefront', 'on'), ('sort', 'off'))),
    )
    for workload, scene in wave_scenes:
        for variant, settings in wave_modes:
            experiments.append(Experiment(
                'wave-' + workload, variant, scene,
                settings + (('rr', 'off'),), '512x336', 32, 16))

    # Roulette comparisons use the same wavefront schedule and differ only in
    # whether survival termination is enabled.
    for workload, scene in (
        ('open', 'open-benchmark'),
        ('closed', 'closed-benchmark'),
    ):
        for value in ('off', 'on'):
            experiments.append(Experiment(
                'rr-' + workload, value, scene,
                (('wavefront', 'on'), ('sort', 'off'), ('rr', value)),
                '512x336', 32, 32))

    # Generated scene names encode both shape and material so every procedural
    # timing is backed by a separately retained render.
    for shape, stem in (('body', 'procedural-fluted-body'),
                        ('knot', 'procedural-swept-knot')):
        for material in ('plain', 'marble', 'cells'):
            experiments.append(Experiment(
                'procedural-' + shape, material, stem + '-' + material,
                (('wavefront', 'on'), ('sort', 'off'), ('rr', 'on')),
                '160x106', 16, 10))

    if len(experiments) != 48:
        raise AssertionError(
            f'experiment matrix has {len(experiments)} variants, expected 48')
    identities = {(item.experiment, item.variant) for item in experiments}
    if len(identities) != len(experiments):
        raise AssertionError(
            'experiment matrix contains a duplicate experiment/variant')
    return experiments


def renderer_settings(experiment):
    settings = {
        'compact': 'on',
        'sort': 'off',
        'bvh': 'on',
        'aa': 'on',
        'nee': 'on',
        'wavefront': 'off',
        'rr': 'off',
        'rr-start': str(DEFAULT_RR_START),
    }
    settings.update(experiment.settings)
    if settings['wavefront'] == 'on' and settings['sort'] == 'on':
        raise AssertionError('--sort on requires --wavefront off')
    return settings


def command_for(binary, output, experiment):
    settings = renderer_settings(experiment)
    command = [
        str(binary.resolve()),
        str(ROOT / 'scenes' / (experiment.scene + '.json')),
        '--samples', str(experiment.samples),
        '--seed', str(SEED),
        '--output', str(output),
        '--resolution', experiment.resolution,
        '--depth', str(experiment.depth),
        '--benchmark',
    ]
    command.extend(
        item
        for key, value in settings.items()
        for item in ('--' + key, value)
    )
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path,
                        default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3,
                        help='measured repeats (fixed at 3)')
    args = parser.parse_args()
    if args.repeats != 3:
        parser.error('the fixed experiment design requires exactly three measured repeats')

    experiments = experiment_matrix()
    if not args.binary.is_file():
        parser.error(f'renderer binary does not exist: {args.binary}')
    missing_scenes = sorted({
        experiment.scene for experiment in experiments
        if not (ROOT / 'scenes' / (experiment.scene + '.json')).is_file()
    })
    if missing_scenes:
        parser.error('missing experiment scenes: ' + ', '.join(missing_scenes))
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('experiments-' + stamp)
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    schedule = ([('warmup', repeat)
                 for repeat in range(WARMUP_RUNS, 0, -1)]
                + [('measured', repeat) for repeat in range(args.repeats)])
    for experiment in experiments:
        for phase, repeat in schedule:
            prefix = output / (
                f'{experiment.experiment}-{experiment.variant}-{phase}-{repeat}')
            command = command_for(args.binary, prefix, experiment)
            if (experiment.experiment == 'checkpoint'
                    and experiment.variant == 'on'):
                command.extend([
                    '--checkpoint', str(prefix) + '.checkpoint',
                    '--checkpoint-every', '8',
                ])
            start = time.perf_counter()
            run = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            prefix.with_suffix('.stdout.txt').write_text(run.stdout)
            prefix.with_suffix('.stderr.txt').write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(
                    f'{experiment.experiment}/{experiment.variant} failed: '
                    f'{run.stderr}')
            # Keep the original runs.csv columns and order. Renderer-specific
            # configuration is retained authoritatively in each prefix JSON.
            rows.append(dict(
                experiment=experiment.experiment,
                variant=experiment.variant,
                phase=phase,
                repeat=repeat,
                process_seconds=elapsed,
                scene=experiment.scene,
                resolution=experiment.resolution,
                samples=experiment.samples,
                seed=SEED,
                prefix=prefix.name,
            ))
            print(
                f'{experiment.experiment} {experiment.variant} '
                f'{phase} {repeat}: {elapsed:.4f}s',
                flush=True,
            )

    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for experiment in experiments:
        values = [
            row['process_seconds'] for row in rows
            if row['experiment'] == experiment.experiment
            and row['variant'] == experiment.variant
            and row['phase'] == 'measured'
        ]
        summary.append(dict(
            experiment=experiment.experiment,
            variant=experiment.variant,
            repeats=len(values),
            process_median_seconds=statistics.median(values),
            process_min_seconds=min(values),
            process_max_seconds=max(values),
        ))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
