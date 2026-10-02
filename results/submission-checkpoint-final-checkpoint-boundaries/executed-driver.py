#!/usr/bin/env python3
"""Original actual CLI regressions for the reviewed checkpoint boundaries."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.source) == 64 and set(args.source) <= set('0123456789abcdef')
    assert args.prefix.replace('-', '').replace('_', '').isalnum()
    root = Path.cwd()
    binary = args.binary.resolve()
    result = root / 'results' / (args.prefix + '-checkpoint-boundaries')
    result.mkdir(exist_ok=False)
    driver = Path(__file__).resolve()
    shutil.copyfile(driver, result / 'executed-driver.py')
    source_hash = lambda: subprocess.check_output([sys.executable, 'scripts/source_hash.py'], text=True).strip()
    scene = root / 'scenes/cornell.json'
    before = {'source_sha256': source_hash(), 'binary_sha256': sha(binary), 'scene_sha256': sha(scene)}
    assert before['source_sha256'] == args.source
    report = {'status': 'running', 'started_utc': utc(), 'renderer_source_sha256': args.source,
              'driver_sha256': sha(driver), 'executed_driver': 'executed-driver.py',
              'inputs_before': before, 'commands': [], 'checks': [], 'errors': []}
    def persist():
        (result / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def check(label, condition):
        assert condition, label
        report['checks'].append({'label': label, 'passed': True})
    def run(label, argv, *, rejected=False):
        item = {'label': label, 'argv': [str(v) for v in argv], 'cwd': str(root),
                'shell': False, 'started_utc': utc()}
        tick = time.monotonic()
        process = subprocess.run(item['argv'], cwd=root, capture_output=True, text=True, check=False)
        for channel in ('stdout', 'stderr'):
            path = result / (label + '.' + channel + '.log')
            path.write_text(getattr(process, channel))
            item[channel] = {'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size}
        item.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - tick, finished_utc=utc())
        report['commands'].append(item)
        if rejected:
            check(label + ': rejected overlapping output', process.returncode != 0 and 'overlap' in process.stderr)
        else:
            check(label + ': actual command succeeded', process.returncode == 0)
        persist()
        return process
    common = ['--seed', '1', '--resolution', '32x24', '--depth', '4', '--sampler', 'counter',
              '--motion', 'off', '--aa', 'on', '--nee', 'on', '--rr', 'off']
    persist()
    try:
        real = result / 'real'
        real.mkdir()
        alias = result / 'alias'
        alias.symlink_to(real, target_is_directory=True)
        try:
            output = real / 'parent-case'
            run('reject-parent-symlink', [binary, scene, *common, '--samples', '1', '--output', output,
                '--checkpoint', alias / 'parent-case.json'], rejected=True)
            check('parent alias creates no render or checkpoint', not list(real.iterdir()))
        finally:
            alias.unlink()
        output = real / 'final-case'
        final_alias = result / 'final-alias.json'
        final_alias.symlink_to(real / 'final-case.json')
        try:
            run('reject-final-symlink', [binary, scene, *common, '--samples', '1', '--output', output,
                '--checkpoint', final_alias], rejected=True)
            check('final alias creates no render output', not list(real.iterdir()))
        finally:
            final_alias.unlink()
        marker = real / 'hardlink-case.json'
        marker.write_bytes(b'original independently authored collision-test marker\n')
        hardlink = result / 'hardlink-checkpoint'
        hardlink.hardlink_to(marker)
        marker_sha = sha(marker)
        try:
            run('reject-hardlink', [binary, scene, *common, '--samples', '1', '--output', real / 'hardlink-case',
                '--checkpoint', hardlink], rejected=True)
            check('existing hardlinked files remain byte-identical', sha(marker) == sha(hardlink) == marker_sha)
            check('hardlink rejection creates no new outputs', set(p.name for p in real.iterdir()) == {'hardlink-case.json'})
        finally:
            hardlink.unlink()
            marker.unlink()
        document = json.loads(scene.read_text())
        document['Camera']['FOVY'] = 0.0
        zero = result / 'zero-fov-scene.json'
        zero.write_text(json.dumps(document, indent=2) + '\n')
        checkpoint = result / 'zero-fov.checkpoint'
        run('zero-fov-save', [binary, zero, *common, '--samples', '3', '--output', result / 'zero-fov-split',
                             '--checkpoint', checkpoint])
        check('zero-FOV valid checkpoint actually saved', checkpoint.is_file() and checkpoint.stat().st_size > 0)
        checkpoint_sha = sha(checkpoint)
        run('zero-fov-resume', [binary, zero, *common, '--samples', '8', '--resume', checkpoint,
                               '--output', result / 'zero-fov-resumed'])
        run('zero-fov-direct', [binary, zero, *common, '--samples', '8', '--output', result / 'zero-fov-direct'])
        check('resuming preserves source checkpoint', sha(checkpoint) == checkpoint_sha)
        run('zero-fov-exact-sums', [sys.executable, 'scripts/compare_sums.py',
            result / 'zero-fov-resumed.sum.bin', result / 'zero-fov-direct.sum.bin', '--exact'])
        check('zero-FOV direct and resumed PNGs are byte-identical',
              (result / 'zero-fov-resumed.png').read_bytes() == (result / 'zero-fov-direct.png').read_bytes())
        for label in ('zero-fov-split', 'zero-fov-resumed', 'zero-fov-direct'):
            metadata = json.loads((result / (label + '.json')).read_text())
            check(label + ': actual final source identity',
                  '_source-' + args.source + '_' in metadata['compatibility']['build_identifier'])
        report['inputs_after'] = {'source_sha256': source_hash(), 'binary_sha256': sha(binary), 'scene_sha256': sha(scene)}
        check('renderer, binary and canonical scene unchanged', report['inputs_after'] == before)
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['errors'].append(type(error).__name__ + ': ' + str(error))
        raise
    finally:
        report['finished_utc'] = utc()
        report['artifacts'] = [{'path': p.relative_to(result).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                               for p in sorted(result.rglob('*')) if p.is_file() and p.name != 'summary.json']
        persist()
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'commands': len(report['commands'])}))

if __name__ == '__main__':
    main()
