#!/usr/bin/env python3
"""Measure original OBJ scene loading (including BVH) and BVH rebuilding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = os.environ.get('CXX', 'c++')
    executable = output / 'profile-scene-load'
    source = ROOT / 'scripts/profile_scene_load.cpp'
    command = [compiler, '-std=c++17', '-O3', '-DNDEBUG', '-I' + str(ROOT / 'src'),
               '-I' + str(ROOT / 'external/include'), str(source),
               str(args.build.resolve() / 'lib/libcis565_core.a'), '-o', str(executable)]
    build = subprocess.run(command, capture_output=True, text=True)
    (output / 'compile.stdout.txt').write_text(build.stdout)
    (output / 'compile.stderr.txt').write_text(build.stderr)
    build.check_returncode()
    scenes = []
    for shape in ('fluted-body', 'swept-knot'):
        for level in ('low', 'medium', 'high'):
            scene = dict(Materials=dict(surface=dict(TYPE='Diffuse', RGB=[1, 1, 1])),
                         Objects=[dict(TYPE='obj', MATERIAL='surface',
                                       FILE=str(ROOT / 'scenes/assets' / (shape + '-' + level + '.obj')),
                                       TRANS=[0, 0, 0], ROTAT=[0, 0, 0], SCALE=[1, 1, 1])],
                         Camera=dict(RES=[1, 1], FOVY=20, ITERATIONS=1, DEPTH=1,
                                     FILE='profile', EYE=[0, 1, 5], LOOKAT=[0, 1, 0], UP=[0, 1, 0]))
            path = output / (shape + '-' + level + '.json')
            path.write_text(json.dumps(scene, indent=2) + '\n')
            scenes.append(path)
    measured = subprocess.run([str(executable), *map(str, scenes)], capture_output=True, text=True)
    (output / 'measure.stderr.txt').write_text(measured.stderr)
    measured.check_returncode()
    records = json.loads(measured.stdout)
    report = dict(records=records, warmups=1, measured_repeats=3,
                  units='milliseconds', compile_command=command,
                  compiler=subprocess.check_output([compiler, '--version'], text=True),
                  helper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  note='CPU scene load includes parsing, transforms, camera setup, and initial BVH; separate BVH measurement rebuilds from loaded primitives. Files are warm in the OS cache. These are not GPU generation timings.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    # Retain source and measurements, not a workstation-specific executable.
    executable.unlink()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
