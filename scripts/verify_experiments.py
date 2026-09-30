#!/usr/bin/env python3
"""Verify exact scheduling/serialization agreement in the measured experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from validate_render import read_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path)
    args = parser.parse_args()
    with (args.experiments / 'runs.csv').open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    checks = []
    groups = ('compact-open', 'compact-closed', 'sort-uniform', 'sort-mixed',
              'bvh-low', 'bvh-medium', 'bvh-high', 'checkpoint')
    for name in groups:
        records = [r for r in runs if r['experiment'] == name]
        assert len(records) >= 12, f'{name}: missing warmup/measured processes'
        reference = read_sum(args.experiments / (records[0]['prefix'] + '.sum.bin'))
        compared = []
        for record in records:
            path = args.experiments / (record['prefix'] + '.sum.bin')
            data = read_sum(path)
            assert (data['width'], data['height'], data['samples']) == (
                reference['width'], reference['height'], reference['samples'])
            assert data['pixels'] == reference['pixels'], f'{name}/{record["prefix"]}: accumulation differs'
            compared.append(dict(prefix=record['prefix'], raw_sha256=data['sha256']))
        checks.append(dict(experiment=name, passed=True, processes=len(records),
                           resolution=[reference['width'], reference['height']],
                           samples=reference['samples'],
                           pixel_sha256=hashlib.sha256(reference['pixels']).hexdigest(),
                           compared=compared))
    output = args.experiments / 'exact-comparison.json'
    output.write_text(json.dumps(dict(passed=True, checks=checks), indent=2) + '\n')
    print(f'Exact agreement: {len(checks)} comparisons, {sum(c["processes"] for c in checks)} processes')


if __name__ == '__main__':
    main()
