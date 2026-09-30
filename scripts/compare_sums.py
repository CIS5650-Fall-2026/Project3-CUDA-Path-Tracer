#!/usr/bin/env python3
"""Compare lossless accumulation files, reporting exact and numerical differences."""
import argparse
import json
from pathlib import Path

from validate_render import read_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()
    a, b = read_sum(args.first), read_sum(args.second)
    assert (a['width'], a['height'], a['samples']) == (b['width'], b['height'], b['samples'])
    different = [(i, x, y) for i, (x, y) in enumerate(zip(a['values'], b['values'])) if x != y]
    first = []
    for i, x, y in different[:12]:
        pixel = i // 3
        first.append(dict(pixel=pixel, x=pixel % a['width'], y=pixel // a['width'], channel=i % 3, first=x, second=y, difference=y-x))
    print(json.dumps(dict(exact=a['pixels'] == b['pixels'], different_components=len(different),
                          max_abs_difference=max((abs(y-x) for _, x, y in different), default=0), first_differences=first), indent=2))
    if args.exact and a['pixels'] != b['pixels']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
