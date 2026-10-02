#!/usr/bin/env python3
"""Validate actual CUDA outputs, continuation, and transport estimators."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RR_SEEDS = (947, 104729, 130363, 155921, 196613, 262147, 314159, 524287)
RR_SAMPLES = 8192

CHECKPOINT_HEADER_BYTES = 24
CHECKPOINT_SAMPLE_METADATA_BYTES = 20
CHECKPOINT_V1_CONFIG_BYTES = 17
CHECKPOINT_V2_CONFIG_BYTES = 23
CHECKPOINT_V3_CONFIG_BYTES = 24
CHECKPOINT_V2_CAMERA_BYTES = 84


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def fnv1a(data):
    digest = 14695981039346656037
    for byte in data:
        digest = ((digest ^ byte) * 1099511628211) & 0xffffffffffffffff
    return digest


def refresh_checkpoint_checksum(encoded):
    if len(encoded) < CHECKPOINT_HEADER_BYTES + 8:
        raise AssertionError('checkpoint is too short to contain a checksum')
    struct.pack_into('<Q', encoded, len(encoded) - 8, fnv1a(encoded[:-8]))


def version_two_checkpoint(version_three):
    """Remove the v3 motion/shutter fields to form a legacy-v2 encoding."""
    encoded = bytearray(version_three)
    assert encoded[:8] == b'C565CKPT'
    assert struct.unpack_from('<I', encoded, 8)[0] == 3
    config_start = CHECKPOINT_HEADER_BYTES + CHECKPOINT_SAMPLE_METADATA_BYTES
    shutter_start = (config_start + CHECKPOINT_V3_CONFIG_BYTES
                     + CHECKPOINT_V2_CAMERA_BYTES)
    del encoded[shutter_start:shutter_start + 8]
    del encoded[config_start + CHECKPOINT_V2_CONFIG_BYTES:
                config_start + CHECKPOINT_V3_CONFIG_BYTES]
    removed = 8 + CHECKPOINT_V3_CONFIG_BYTES - CHECKPOINT_V2_CONFIG_BYTES
    struct.pack_into('<I', encoded, 8, 2)
    payload_bytes = struct.unpack_from('<Q', encoded, 16)[0]
    struct.pack_into('<Q', encoded, 16, payload_bytes - removed)
    refresh_checkpoint_checksum(encoded)
    return encoded


def version_one_checkpoint(version_two):
    """Construct a structurally valid legacy-v1 encoding from a v2 checkpoint."""
    encoded = bytearray(version_two)
    assert encoded[:8] == b'C565CKPT'
    assert struct.unpack_from('<I', encoded, 8)[0] == 2
    extension = (CHECKPOINT_HEADER_BYTES + CHECKPOINT_SAMPLE_METADATA_BYTES
                 + CHECKPOINT_V1_CONFIG_BYTES)
    del encoded[extension:extension
                + CHECKPOINT_V2_CONFIG_BYTES - CHECKPOINT_V1_CONFIG_BYTES]
    struct.pack_into('<I', encoded, 8, 1)
    payload_bytes = struct.unpack_from('<Q', encoded, 16)[0]
    struct.pack_into('<Q', encoded, 16,
                     payload_bytes - (CHECKPOINT_V2_CONFIG_BYTES
                                      - CHECKPOINT_V1_CONFIG_BYTES))
    refresh_checkpoint_checksum(encoded)
    return encoded


def read_sum(path):
    data = path.read_bytes()
    if len(data) < 40:
        raise AssertionError('truncated raw accumulation')
    magic, version, width, height, channels, samples = struct.unpack_from(
        '<8sIIIIQ', data)
    assert magic == b'C565SUM\0' and version == 1 and channels == 3
    assert len(data) == 32 + width * height * 12 + 8
    assert fnv1a(data[:-8]) == struct.unpack_from('<Q', data, len(data) - 8)[0]
    values = struct.unpack_from('<' + str(width * height * 3) + 'f', data, 32)
    assert samples > 0 and all(math.isfinite(value) and value >= 0
                               for value in values)
    return dict(width=width, height=height, samples=samples, values=values,
                pixels=data[32:-8], encoded=data,
                sha256=hashlib.sha256(data).hexdigest())


def image_mean(record):
    return math.fsum(record['values']) / len(record['values']) / record['samples']


def mean_and_sample_variance(values):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path,
                        default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lighting-samples', type=positive_int, default=8192)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('validation-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def run(label, extra=(), scene='optical-workbench', samples=17,
            resolution='17x13', depth=7, seed=947, wavefront=False,
            rr=False, rr_start=3, fail=False, expected_error=None,
            inherit_config=False, expected=None):
        prefix = output / label
        if inherit_config:
            if expected is None:
                raise AssertionError(
                    f'{label}: inherited invocation needs expected metadata')
            options = {
                '--samples': str(samples),
                '--output': str(prefix),
            }
        else:
            options = {
                '--samples': str(samples),
                '--seed': str(seed),
                '--resolution': resolution,
                '--depth': str(depth),
                '--compact': 'on',
                '--sort': 'off',
                '--bvh': 'on',
                '--aa': 'on',
                '--nee': 'on',
                '--wavefront': 'on' if wavefront else 'off',
                '--rr': 'on' if rr else 'off',
                '--rr-start': str(rr_start),
                '--output': str(prefix),
            }
        extras = list(map(str, extra))
        flags = {'--validate', '--benchmark'}
        cursor = 0
        while cursor < len(extras):
            key = extras[cursor]
            if key in flags:
                options[key] = None
                cursor += 1
            else:
                if cursor + 1 >= len(extras):
                    raise AssertionError(f'{label}: missing value after {key}')
                options[key] = extras[cursor + 1]
                cursor += 2
        if inherit_config:
            assert set(options) == {'--samples', '--output', '--resume'}, (
                f'{label}: minimal resume unexpectedly supplies renderer options')
        command = [str(args.binary.resolve()),
                   str(ROOT / 'scenes' / (scene + '.json'))]
        for key, value in options.items():
            command.append(key)
            if value is not None:
                command.append(value)
        result = subprocess.run(command, capture_output=True, text=True)
        prefix.with_suffix('.stdout.txt').write_text(result.stdout)
        prefix.with_suffix('.stderr.txt').write_text(result.stderr)
        if fail:
            assert result.returncode != 0, f'{label}: invalid invocation was accepted'
            rejection = (result.stdout + '\n' + result.stderr).strip()
            if expected_error is not None:
                assert expected_error in rejection, (
                    f'{label}: rejection did not contain {expected_error!r}: {rejection}')
            checks.append(dict(name=label, passed=True, rejection=rejection))
            return None
        assert result.returncode == 0, f'{label}: {result.stderr}'
        record = read_sum(prefix.with_suffix('.sum.bin'))
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        assert metadata['completed_samples'] == samples
        assert metadata['resolution'] == [record['width'], record['height']]
        config = metadata['config']
        if expected is None:
            assert metadata['seed'] == seed
            assert config['wavefront'] == (options['--wavefront'] == 'on')
            assert config['rr'] == (options['--rr'] == 'on')
            assert config['rr_start'] == int(options['--rr-start'])
        else:
            assert metadata['seed'] == expected['seed']
            assert metadata['resolution'] == expected['resolution']
            assert config == expected['config'], (
                f'{label}: resumed configuration was not inherited exactly')
        assert prefix.with_suffix('.png').is_file()
        record['metadata'] = metadata
        record['prefix'] = prefix
        record['command'] = command
        return record

    def exact(name, expected, actual, detail):
        assert actual['encoded'] == expected['encoded'], f'{name}: {detail}'
        checks.append(dict(name=name, passed=True, detail=detail,
                           raw_sha256=actual['sha256']))

    def validate_benchmark_counters(mode, record, wavefront, rr):
        csv_path = record['prefix'].with_suffix('.benchmark.csv')
        with csv_path.open(newline='') as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row['record_type'] == 'bounce']
        assert rows, f'{mode}: benchmark contained no bounce records'
        queue_fields = ('queued_diffuse', 'queued_mirror',
                        'queued_dielectric', 'queued_emission',
                        'queued_terminal')
        queue_total = 0
        dispatches = 0
        rr_tested = 0
        rr_terminated = 0
        for row in rows:
            queued = sum(int(row[field]) for field in queue_fields)
            active_before = int(row['active_before'])
            row_dispatches = int(row['kernel_dispatches'])
            tested = int(row['rr_tested'])
            terminated = int(row['rr_terminated'])
            assert 0 <= terminated <= tested <= active_before, (
                f'{mode}: invalid per-bounce RR counters')
            if wavefront:
                assert queued == active_before, (
                    f'{mode}: queues contain {queued} of {active_before} live inputs')
                assert row_dispatches > 0, f'{mode}: no specialized kernel dispatched'
            else:
                assert queued == 0 and row_dispatches == 0, (
                    f'{mode}: unified mode reported wavefront queue work')
            queue_total += queued
            dispatches += row_dispatches
            rr_tested += tested
            rr_terminated += terminated
        if rr:
            assert rr_tested > 0, f'{mode}: enabled RR never tested a path'
        else:
            assert rr_tested == 0 and rr_terminated == 0, (
                f'{mode}: disabled RR reported tests or terminations')
        checks.append(dict(name='benchmark-counters-' + mode, passed=True,
                           bounce_records=len(rows), queued_paths=queue_total,
                           kernel_dispatches=dispatches,
                           rr_tested=rr_tested, rr_terminated=rr_terminated))

    diagnostic = subprocess.run([
        str(args.binary.resolve()), str(ROOT / 'scenes/optical-workbench.json'),
        '--seed', '947', '--resolution', '17x13', '--depth', '7',
        '--wavefront', 'off', '--rr', 'off', '--rr-start', '3', '--validate'],
        capture_output=True, text=True)
    (output / 'gpu-validation.stdout.txt').write_text(diagnostic.stdout)
    (output / 'gpu-validation.stderr.txt').write_text(diagnostic.stderr)
    assert diagnostic.returncode == 0, diagnostic.stderr
    checks.append(dict(name='gpu-equivalence-and-edge-validation', passed=True))

    references = {}
    checkpoints = {}
    for wavefront in (False, True):
        for rr in (False, True):
            mode = f'wavefront-{"on" if wavefront else "off"}-rr-{"on" if rr else "off"}'
            reference = run('reference-' + mode, wavefront=wavefront, rr=rr)
            references[(wavefront, rr)] = reference

            scheduling = [
                ('compact-off', ['--compact', 'off']),
                ('brute-force', ['--bvh', 'off']),
            ]
            if wavefront:
                scheduling.append(
                    ('all-valid-scheduling-toggles', ['--compact', 'off', '--bvh', 'off']))
            else:
                scheduling.extend([
                    ('sort-on', ['--sort', 'on']),
                    ('all-valid-scheduling-toggles',
                     ['--compact', 'off', '--sort', 'on', '--bvh', 'off']),
                ])
            for variant, extra in scheduling:
                candidate = run(f'{variant}-{mode}', extra,
                                wavefront=wavefront, rr=rr)
                exact(f'{variant}-{mode}', reference, candidate,
                      'valid scheduling changes must preserve the complete raw sum file')

            continuation_rr_start = 11 if (not wavefront and not rr) else 3
            continuation_reference = reference
            if continuation_rr_start != 3:
                continuation_reference = run(
                    'continuation-reference-' + mode,
                    wavefront=wavefront, rr=rr,
                    rr_start=continuation_rr_start)

            checkpoint = output / ('partial-' + mode + '.checkpoint')
            checkpoints[(wavefront, rr)] = checkpoint
            run('partial-' + mode, ['--checkpoint', checkpoint], samples=5,
                wavefront=wavefront, rr=rr,
                rr_start=continuation_rr_start)
            inherited_expected = dict(
                seed=continuation_reference['metadata']['seed'],
                resolution=list(
                    continuation_reference['metadata']['resolution']),
                config=dict(continuation_reference['metadata']['config']),
            )
            resumed = run('resumed-' + mode, ['--resume', checkpoint],
                          inherit_config=True, expected=inherited_expected)
            exact('fresh-process-exact-resume-' + mode,
                  continuation_reference, resumed,
                  '5-to-17 continuation must equal an uninterrupted 17-sample render')
            checks[-1].update(
                inherited_config=inherited_expected,
                resume_arguments=resumed['command'][2:],
            )

            instrumentation_reference = reference
            instrumentation_scene = 'optical-workbench'
            instrumentation_depth = 7
            if rr:
                # The closed chamber deterministically retains paths long
                # enough for the instrumentation to observe RR work.
                instrumentation_scene = 'closed-benchmark'
                instrumentation_depth = 8
                instrumentation_reference = run(
                    'instrumentation-reference-' + mode,
                    scene=instrumentation_scene, depth=instrumentation_depth,
                    wavefront=wavefront, rr=rr)
            instrumented = run(
                'instrumented-' + mode, ['--benchmark'],
                scene=instrumentation_scene, depth=instrumentation_depth,
                wavefront=wavefront, rr=rr)
            exact('instrumented-identity-' + mode,
                  instrumentation_reference, instrumented,
                  'benchmark instrumentation must not alter rendering bytes')
            validate_benchmark_counters(mode, instrumented, wavefront, rr)

    for rr in (False, True):
        label = 'rr-on' if rr else 'rr-off'
        exact('wavefront-legacy-equivalence-' + label,
              references[(False, rr)], references[(True, rr)],
              'legacy and queue-based material scheduling must be byte-identical')

    for wavefront in (False, True):
        label = 'wavefront-on' if wavefront else 'wavefront-off'
        inert = run('rr-off-alternate-start-' + label, wavefront=wavefront,
                    rr=False, rr_start=4096)
        exact('rr-off-start-identity-' + label,
              references[(wavefront, False)], inert,
              'rr-start must not affect bytes while Russian roulette is disabled')
        deferred = run('rr-on-deferred-' + label, wavefront=wavefront,
                       rr=True, rr_start=7)
        exact('rr-depth-not-reached-identity-' + label,
              references[(wavefront, False)], deferred,
              'RR on/off must be byte-identical when max depth is at most rr-start')

    checkpoint = checkpoints[(True, True)]
    run('reject-seed', ['--resume', checkpoint, '--seed', '948'],
        wavefront=True, rr=True, fail=True)
    run('reject-resolution', ['--resume', checkpoint], resolution='19x13',
        wavefront=True, rr=True, fail=True)
    run('reject-feature', ['--resume', checkpoint, '--nee', 'off'],
        wavefront=True, rr=True, fail=True)
    run('reject-wavefront', ['--resume', checkpoint],
        wavefront=False, rr=True, fail=True)
    run('reject-rr', ['--resume', checkpoint],
        wavefront=True, rr=False, fail=True)
    run('reject-rr-start', ['--resume', checkpoint],
        wavefront=True, rr=True, rr_start=4, fail=True)
    run('reject-wavefront-sort-conflict', ['--sort', 'on'],
        wavefront=True, rr=False, fail=True,
        expected_error='--sort on requires --wavefront off')
    run('reject-rr-start-zero', ['--rr-start', '0'],
        wavefront=False, rr=False, fail=True)
    run('reject-rr-start-oversized', ['--rr-start', '4097'],
        wavefront=False, rr=False, fail=True)

    broken = output / 'corrupt.checkpoint'
    corrupt = bytearray(checkpoint.read_bytes())
    corrupt[len(corrupt) // 2] ^= 0x40
    broken.write_bytes(corrupt)
    run('reject-corruption', ['--resume', broken],
        wavefront=True, rr=True, fail=True)

    legacy = output / 'version-1.checkpoint'
    legacy.write_bytes(version_one_checkpoint(
        version_two_checkpoint(checkpoint.read_bytes())))
    run('reject-checkpoint-version-1', ['--resume', legacy],
        wavefront=True, rr=True, fail=True,
        expected_error='unsupported checkpoint format version 1')

    legacy_two = output / 'version-2.checkpoint'
    legacy_two.write_bytes(version_two_checkpoint(checkpoint.read_bytes()))
    run('reject-checkpoint-version-2', ['--resume', legacy_two],
        wavefront=True, rr=True, fail=True,
        expected_error='unsupported checkpoint format version 2')

    config_extension = (CHECKPOINT_HEADER_BYTES + CHECKPOINT_SAMPLE_METADATA_BYTES
                        + CHECKPOINT_V1_CONFIG_BYTES)
    for setting, relative_offset in (('wavefront', 0), ('rr', 1),
                                     ('motion', CHECKPOINT_V2_CONFIG_BYTES
                                      - CHECKPOINT_V1_CONFIG_BYTES)):
        malformed = bytearray(checkpoint.read_bytes())
        bool_offset = config_extension + relative_offset
        assert malformed[bool_offset] in (0, 1)
        malformed[bool_offset] = 2
        refresh_checkpoint_checksum(malformed)
        path = output / ('noncanonical-' + setting + '.checkpoint')
        path.write_bytes(malformed)
        run('reject-noncanonical-' + setting, ['--resume', path],
            wavefront=True, rr=True, fail=True, expected_error='non-canonical')

    # Alter one texture byte without changing the scene path; restore exactly.
    asset = ROOT / 'scenes/assets/glaze.bmp'
    original = asset.read_bytes()
    altered = bytearray(original)
    altered[64] ^= 0x01
    try:
        asset.write_bytes(altered)
        run('reject-asset-change', ['--resume', checkpoint],
            wavefront=True, rr=True, fail=True)
    finally:
        asset.write_bytes(original)

    # Keep RR disabled so these tests isolate depth semantics and the two
    # independent direct-light estimators.
    for kind in ('cube', 'sphere', 'mesh'):
        shallow_on = run('light-' + kind + '-depth1-nee',
                         scene='light-test-' + kind, samples=32,
                         resolution='48x32', depth=1,
                         wavefront=True, rr=False)
        shallow_off = run('light-' + kind + '-depth1-bsdf', ['--nee', 'off'],
                          scene='light-test-' + kind, samples=32,
                          resolution='48x32', depth=1,
                          wavefront=True, rr=False)
        exact(kind + '-depth-one-exact', shallow_on, shallow_off,
              'depth-one NEE must not add an untraced transport vertex')
        on = run('light-' + kind + '-nee', scene='light-test-' + kind,
                 samples=args.lighting_samples, resolution='48x32', depth=4,
                 wavefront=True, rr=False)
        off = run('light-' + kind + '-bsdf', ['--nee', 'off'],
                  scene='light-test-' + kind, samples=args.lighting_samples,
                  resolution='48x32', depth=4, wavefront=True, rr=False)
        mean_on = image_mean(on)
        mean_off = image_mean(off)
        relative = abs(mean_on - mean_off) / max(mean_on, mean_off, 1e-8)
        assert relative < .05, f'{kind} MIS mean differs by {relative:.2%}'
        checks.append(dict(name=kind + '-mis-brightness', passed=True,
                           nee_mean=mean_on, bsdf_mean=mean_off,
                           relative_difference=relative,
                           samples=args.lighting_samples, rr=False))

    # These exact seeds are part of the test design.  Every paired estimate is
    # retained and used; no seed is selected or discarded after observation.
    for scene_label, scene in (('closed', 'closed-benchmark'),
                               ('glass', 'optical-workbench')):
        off_means = []
        on_means = []
        paired = []
        for seed in RR_SEEDS:
            off = run(f'rr-{scene_label}-seed-{seed}-off', scene=scene,
                      samples=RR_SAMPLES, resolution='64x48', depth=32,
                      seed=seed, wavefront=True, rr=False, rr_start=3)
            on = run(f'rr-{scene_label}-seed-{seed}-on', scene=scene,
                     samples=RR_SAMPLES, resolution='64x48', depth=32,
                     seed=seed, wavefront=True, rr=True, rr_start=3)
            mean_off = image_mean(off)
            mean_on = image_mean(on)
            off_means.append(mean_off)
            on_means.append(mean_on)
            paired.append(dict(seed=seed, rr_off_mean=mean_off,
                               rr_on_mean=mean_on,
                               difference=mean_on - mean_off))

        reference_mean, off_variance = mean_and_sample_variance(off_means)
        rr_mean, on_variance = mean_and_sample_variance(on_means)
        differences = [entry['difference'] for entry in paired]
        mean_difference, difference_variance = mean_and_sample_variance(differences)
        standard_error = math.sqrt(difference_variance / len(differences))
        allowed_error = max(4 * standard_error, .01 * reference_mean) + 1e-6
        absolute_error = abs(mean_difference)
        assert absolute_error <= allowed_error, (
            f'{scene_label} RR paired mean difference {mean_difference:.9g} '
            f'exceeds {allowed_error:.9g}')
        checks.append(dict(
            name='rr-' + scene_label + '-paired-statistical-agreement',
            passed=True,
            seeds=list(RR_SEEDS),
            samples_per_seed=RR_SAMPLES,
            resolution=[64, 48],
            max_depth=32,
            rr_start=3,
            rr_off_mean=reference_mean,
            rr_on_mean=rr_mean,
            rr_off_sample_variance=off_variance,
            rr_on_sample_variance=on_variance,
            paired_mean_difference=mean_difference,
            paired_sample_variance=difference_variance,
            paired_standard_error=standard_error,
            absolute_mean_error=absolute_error,
            allowed_error=allowed_error,
            criterion='abs(mean(on-off)) <= max(4*SE, 1%*mean(off)) + 1e-6',
            paired_estimates=paired))

    summary = dict(
        passed=True,
        checks=checks,
        fixed_rr_seeds=list(RR_SEEDS),
        compatibility=references[(True, True)]['metadata']['compatibility'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
