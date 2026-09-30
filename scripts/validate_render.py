#!/usr/bin/env python3
"""Validate actual CUDA outputs, continuation, and independent lighting estimators."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_sum(path):
    data = path.read_bytes()
    if len(data) < 40:
        raise AssertionError('truncated raw accumulation')
    magic, version, w, h, channels, samples = struct.unpack_from('<8sIIIIQ', data)
    assert magic == b'C565SUM\0' and version == 1 and channels == 3
    assert len(data) == 32 + w * h * 12 + 8
    digest = 14695981039346656037
    for byte in data[:-8]:
        digest = ((digest ^ byte) * 1099511628211) & 0xffffffffffffffff
    assert digest == struct.unpack_from('<Q', data, len(data) - 8)[0]
    values = struct.unpack_from('<' + str(w * h * 3) + 'f', data, 32)
    assert samples > 0 and all(math.isfinite(v) and v >= 0 for v in values)
    return dict(width=w, height=h, samples=samples, values=values,
                pixels=data[32:-8], sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lighting-samples', type=int, default=8192)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('validation-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    checks = []

    def run(label, extra=(), scene='optical-workbench', samples=17, resolution='17x13', depth=7, fail=False):
        prefix = output / label
        options = {'--samples': str(samples), '--seed': '947', '--resolution': resolution,
                   '--depth': str(depth), '--compact': 'on', '--sort': 'off', '--bvh': 'on',
                   '--aa': 'on', '--nee': 'on', '--output': str(prefix)}
        extras = list(map(str, extra))
        cursor = 0
        while cursor < len(extras):
            key = extras[cursor]
            options[key] = None if key in ('--validate', '--benchmark') else extras[cursor + 1]
            cursor += 1 if options[key] is None else 2
        command = [str(args.binary.resolve()), str(ROOT / 'scenes' / (scene + '.json'))]
        for key, value in options.items():
            command.append(key)
            if value is not None:
                command.append(value)
        result = subprocess.run(command, capture_output=True, text=True)
        prefix.with_suffix('.stdout.txt').write_text(result.stdout)
        prefix.with_suffix('.stderr.txt').write_text(result.stderr)
        if fail:
            assert result.returncode != 0, f'{label}: incompatible checkpoint was accepted'
            checks.append(dict(name=label, passed=True, rejection=result.stderr.strip()))
            return None
        assert result.returncode == 0, f'{label}: {result.stderr}'
        record = read_sum(prefix.with_suffix('.sum.bin'))
        metadata = json.loads(prefix.with_suffix('.json').read_text())
        assert metadata['completed_samples'] == samples
        assert metadata['resolution'] == [record['width'], record['height']]
        assert prefix.with_suffix('.png').is_file()
        return record

    diagnostic = subprocess.run([
        str(args.binary.resolve()), str(ROOT / 'scenes/optical-workbench.json'),
        '--seed', '947', '--resolution', '17x13', '--depth', '7', '--validate'],
        capture_output=True, text=True)
    (output / 'gpu-validation.stdout.txt').write_text(diagnostic.stdout)
    (output / 'gpu-validation.stderr.txt').write_text(diagnostic.stderr)
    assert diagnostic.returncode == 0, diagnostic.stderr
    checks.append(dict(name='gpu-equivalence-and-edge-validation', passed=True))
    reference = run('reference')
    for name, extra in [('compact-off', ['--compact', 'off']),
                        ('sort-on', ['--sort', 'on']),
                        ('brute-force', ['--bvh', 'off']),
                        ('all-scheduling-toggles', ['--compact', 'off', '--sort', 'on', '--bvh', 'off'])]:
        candidate = run(name, extra)
        assert candidate['pixels'] == reference['pixels'], f'{name}: scheduling changed accumulation'
        checks.append(dict(name=name, passed=True, raw_sha256=candidate['sha256']))
    checkpoint = output / 'partial.checkpoint'
    run('partial', ['--checkpoint', checkpoint], samples=5)
    resumed = run('resumed', ['--resume', checkpoint])
    assert resumed['pixels'] == reference['pixels'], 'fresh-process resume differs'
    checks.append(dict(name='fresh-process-exact-resume', passed=True, raw_sha256=resumed['sha256']))
    run('reject-seed', ['--resume', checkpoint, '--seed', '948'], fail=True)
    run('reject-resolution', ['--resume', checkpoint], resolution='19x13', fail=True)
    run('reject-feature', ['--resume', checkpoint, '--nee', 'off'], fail=True)
    broken = output / 'corrupt.checkpoint'
    corrupt = bytearray(checkpoint.read_bytes())
    corrupt[len(corrupt) // 2] ^= 0x40
    broken.write_bytes(corrupt)
    run('reject-corruption', ['--resume', broken], fail=True)
    # Alter one texture byte without changing the scene path; restore exactly.
    asset = ROOT / 'scenes/assets/glaze.bmp'
    original = asset.read_bytes()
    altered = bytearray(original)
    altered[64] ^= 0x01
    try:
        asset.write_bytes(altered)
        run('reject-asset-change', ['--resume', checkpoint], fail=True)
    finally:
        asset.write_bytes(original)
    # Repeated independent image-level estimates should agree in expectation.
    # Raw means include direct emitters consistently in both estimators.
    for kind in ('cube', 'sphere', 'mesh'):
        shallow_on = run('light-' + kind + '-depth1-nee', scene='light-test-' + kind,
                         samples=32, resolution='48x32', depth=1)
        shallow_off = run('light-' + kind + '-depth1-bsdf', ['--nee', 'off'],
                          scene='light-test-' + kind, samples=32, resolution='48x32', depth=1)
        assert shallow_on['pixels'] == shallow_off['pixels'], f'{kind}: depth-one NEE added an untraced vertex'
        checks.append(dict(name=kind + '-depth-one-exact', passed=True))
        on = run('light-' + kind + '-nee', scene='light-test-' + kind,
                 samples=args.lighting_samples, resolution='48x32', depth=4)
        off = run('light-' + kind + '-bsdf', ['--nee', 'off'], scene='light-test-' + kind,
                  samples=args.lighting_samples, resolution='48x32', depth=4)
        mean_on = sum(on['values']) / len(on['values']) / on['samples']
        mean_off = sum(off['values']) / len(off['values']) / off['samples']
        relative = abs(mean_on - mean_off) / max(mean_on, mean_off, 1e-8)
        assert relative < .05, f'{kind} MIS mean differs by {relative:.2%}'
        checks.append(dict(name=kind + '-mis-brightness', passed=True, nee_mean=mean_on,
                           bsdf_mean=mean_off, relative_difference=relative, samples=args.lighting_samples))
    summary = dict(passed=True, checks=checks,
                   compatibility=json.loads((output / 'reference.json').read_text())['compatibility'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
