#!/usr/bin/env python3
"""Verify experiment coverage and exact scheduling/serialization agreement."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from run_experiments import experiment_matrix, renderer_settings
from validate_render import read_sum


EXACT_GROUPS = (
    ('compact-open', ('off', 'on'), ('compact',)),
    ('compact-closed', ('off', 'on'), ('compact',)),
    ('sort-uniform', ('off', 'on'), ('sort',)),
    ('sort-mixed', ('off', 'on'), ('sort',)),
    ('bvh-low', ('off', 'on'), ('bvh',)),
    ('bvh-medium', ('off', 'on'), ('bvh',)),
    ('bvh-high', ('off', 'on'), ('bvh',)),
    ('checkpoint', ('off', 'on'), ()),
    ('wave-uniform',
     ('legacy-unsorted', 'legacy-sorted', 'wavefront'),
     ('sort', 'wavefront')),
    ('wave-mixed',
     ('legacy-unsorted', 'legacy-sorted', 'wavefront'),
     ('sort', 'wavefront')),
    ('wave-closed',
     ('legacy-unsorted', 'legacy-sorted', 'wavefront'),
     ('sort', 'wavefront')),
    ('wave-mesh-heavy',
     ('legacy-unsorted', 'legacy-sorted', 'wavefront'),
     ('sort', 'wavefront')),
)


CONFIG_ALIASES = {
    'compact': ('compact',),
    'sort': ('sort',),
    'bvh': ('bvh',),
    'aa': ('aa',),
    'nee': ('nee',),
    'wavefront': ('wavefront',),
    'rr': ('russian_roulette', 'rr'),
    'rr-start': ('rr_start', 'rrStart'),
}


def config_value(config, setting):
    for key in CONFIG_ALIASES[setting]:
        if key in config:
            return config[key]
    names = ', '.join(CONFIG_ALIASES[setting])
    raise AssertionError(f'metadata config lacks required field ({names})')


def load_metadata(directory, record):
    path = directory / (record['prefix'] + '.json')
    with path.open() as stream:
        return json.load(stream)


def physical_signature(metadata, varying):
    """Describe all retained rendering inputs except the tested scheduler flag."""
    ignored = set(varying)
    config = {
        key: value for key, value in metadata['config'].items()
        if key not in ignored
    }
    fingerprints = metadata.get('fingerprints', {})
    return {
        'scene': Path(metadata['scene']).name,
        'resolution': metadata['resolution'],
        'target_samples': metadata['target_samples'],
        'seed': metadata['seed'],
        'sampler_version': metadata['sampler_version'],
        'camera': metadata['camera'],
        'config': config,
        'scene_file_fingerprint': fingerprints.get('scene_file'),
        'asset_fingerprint': fingerprints.get('assets'),
        'material_fingerprint': fingerprints.get('materials'),
        'geometry_fingerprint': fingerprints.get('geometry'),
    }


def assert_matrix(runs, directory):
    expected_matrix = experiment_matrix()
    expected = {
        (item.experiment, item.variant): item for item in expected_matrix
    }
    actual = {(row['experiment'], row['variant']) for row in runs}
    assert actual == set(expected), (
        'runs.csv experiment matrix differs: '
        f'missing={sorted(set(expected) - actual)}, '
        f'unexpected={sorted(actual - set(expected))}')
    prefixes = [row['prefix'] for row in runs]
    assert len(prefixes) == len(set(prefixes)), 'runs.csv contains duplicate prefixes'

    metadata_by_prefix = {}
    for identity, experiment in expected.items():
        records = [
            row for row in runs
            if (row['experiment'], row['variant']) == identity
        ]
        warmups = [row for row in records if row['phase'] == 'warmup']
        measured = [row for row in records if row['phase'] == 'measured']
        assert len(warmups) == 3, f'{identity}: expected three warmups'
        assert len(measured) == 3, f'{identity}: expected exactly three measured runs'
        assert len(records) == len(warmups) + len(measured), (
            f'{identity}: unknown phase in runs.csv')

        expected_settings = renderer_settings(experiment)
        for record in records:
            assert record['scene'] == experiment.scene, (
                f'{record["prefix"]}: unexpected scene')
            assert record['resolution'] == experiment.resolution, (
                f'{record["prefix"]}: unexpected resolution')
            assert int(record['samples']) == experiment.samples, (
                f'{record["prefix"]}: unexpected sample count')
            assert int(record['seed']) == 947, (
                f'{record["prefix"]}: unexpected seed')

            metadata = load_metadata(directory, record)
            metadata_by_prefix[record['prefix']] = metadata
            config = metadata['config']
            assert config['max_depth'] == experiment.depth, (
                f'{record["prefix"]}: unexpected max depth')
            for setting, expected_text in expected_settings.items():
                actual_value = config_value(config, setting)
                expected_value = (expected_text == 'on'
                                  if expected_text in ('on', 'off')
                                  else int(expected_text))
                assert actual_value == expected_value, (
                    f'{record["prefix"]}: {setting} is {actual_value!r}, '
                    f'expected {expected_value!r}')
    return metadata_by_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path)
    args = parser.parse_args()
    with (args.experiments / 'runs.csv').open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == 288, (
        f'runs.csv contains {len(runs)} processes, expected exactly 288')
    metadata_by_prefix = assert_matrix(runs, args.experiments)

    checks = []
    for name, expected_variants, varying in EXACT_GROUPS:
        records = [row for row in runs if row['experiment'] == name]
        variants = {row['variant'] for row in records}
        assert variants == set(expected_variants), (
            f'{name}: variants {sorted(variants)} differ from '
            f'{list(expected_variants)}')
        assert len(records) >= 6 * len(expected_variants), (
            f'{name}: missing warmup/measured processes')

        reference_record = records[0]
        reference_metadata = metadata_by_prefix[reference_record['prefix']]
        reference_signature = physical_signature(reference_metadata, varying)
        reference_rr = config_value(reference_metadata['config'], 'rr')
        reference_rr_start = config_value(
            reference_metadata['config'], 'rr-start')
        assert reference_rr is False, (
            f'{name}: exact scheduling comparisons must keep roulette off')

        reference = read_sum(
            args.experiments / (reference_record['prefix'] + '.sum.bin'))
        compared = []
        for record in records:
            metadata = metadata_by_prefix[record['prefix']]
            assert physical_signature(metadata, varying) == reference_signature, (
                f'{name}/{record["prefix"]}: physical render inputs differ')
            assert config_value(metadata['config'], 'rr') == reference_rr, (
                f'{name}/{record["prefix"]}: roulette setting differs')
            assert config_value(metadata['config'], 'rr-start') == reference_rr_start, (
                f'{name}/{record["prefix"]}: roulette start differs')

            path = args.experiments / (record['prefix'] + '.sum.bin')
            data = read_sum(path)
            assert (data['width'], data['height'], data['samples']) == (
                reference['width'], reference['height'], reference['samples'])
            assert data['pixels'] == reference['pixels'], (
                f'{name}/{record["prefix"]}: accumulation differs')
            compared.append(dict(
                prefix=record['prefix'],
                variant=record['variant'],
                raw_sha256=data['sha256'],
            ))
        checks.append(dict(
            experiment=name,
            variants=list(expected_variants),
            passed=True,
            processes=len(records),
            resolution=[reference['width'], reference['height']],
            samples=reference['samples'],
            russian_roulette=reference_rr,
            rr_start=reference_rr_start,
            pixel_sha256=hashlib.sha256(reference['pixels']).hexdigest(),
            compared=compared,
        ))

    output = args.experiments / 'exact-comparison.json'
    output.write_text(json.dumps(dict(
        passed=True,
        experiment_variants=len(experiment_matrix()),
        processes=len(runs),
        checks=checks,
    ), indent=2) + '\n')
    print(
        f'Exact agreement: {len(checks)} comparisons, '
        f'{sum(check["processes"] for check in checks)} processes; '
        f'full matrix: {len(runs)} processes')


if __name__ == '__main__':
    main()
