#!/usr/bin/env python3
"""Exercise the preview on its validation workstation and preserve evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def rr_start(value):
    result = int(value)
    if not 1 <= result <= 4096:
        raise argparse.ArgumentTypeError('must be in [1, 4096]')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-binary', type=Path, default=ROOT / 'build/bin/cis565_path_tracer')
    parser.add_argument('--headless-binary', type=Path, default=ROOT / 'build/bin/cis565_path_tracer_headless')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wavefront', choices=('on', 'off'), default='on')
    parser.add_argument('--rr', choices=('on', 'off'), default='on')
    parser.add_argument('--rr-start', type=rr_start, default=3)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT / 'results' / ('preview-' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')
    actions = []
    renderer_options = [
        '--wavefront', args.wavefront,
        '--rr', args.rr,
        '--rr-start', str(args.rr_start),
    ]

    def command(argv, cwd, log, timeout=45):
        with log.open('w') as stream:
            result = subprocess.run(list(map(str, argv)), cwd=cwd, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        assert result.returncode == 0, f'{argv[0]} exited {result.returncode}; see {log}'

    def headless(scene, samples, prefix):
        command([args.headless_binary.resolve(), ROOT / 'scenes' / scene,
                 '--samples', samples, '--seed', '1', '--output', prefix,
                 *renderer_options],
                ROOT, prefix.with_suffix('.log'))

    smoke = output / 'smoke'
    smoke.mkdir()
    command([args.preview_binary.resolve(), ROOT / 'scenes/preview-smoke.json',
             *renderer_options],
            smoke, smoke / 'preview.log')
    saved = list(smoke.glob('*.png'))
    assert len(saved) == 1 and '.16samp.' in saved[0].name, 'smoke did not complete exactly 16 samples'
    prefix = smoke / 'headless'
    headless('preview-smoke.json', 16, prefix)
    assert saved[0].read_bytes() == prefix.with_suffix('.png').read_bytes(), 'preview/headless PNG mismatch'

    interactive = output / 'interactive'
    interactive.mkdir()
    process = None
    reset_image = None
    moved_images = {}
    log_path = interactive / 'preview.log'
    with log_path.open('w') as log:
        try:
            process = subprocess.Popen([str(args.preview_binary.resolve()),
                str(ROOT / 'scenes/optical-workbench.json'),
                *renderer_options], cwd=interactive,
                env=environment, stdout=log, stderr=subprocess.STDOUT)

            def xdo(*arguments, allow_failure=False):
                result = subprocess.run(['xdotool', *map(str, arguments)], env=environment,
                    capture_output=True, text=True, timeout=12)
                actions.append(dict(command=['xdotool', *map(str, arguments)],
                                    returncode=result.returncode, stdout=result.stdout.strip(),
                                    stderr=result.stderr.strip()))
                if not allow_failure:
                    assert result.returncode == 0, f'xdotool failed: {result.stderr}'
                return result.stdout.strip()

            deadline = time.monotonic() + 20
            window = ''
            while time.monotonic() < deadline and not window:
                assert process.poll() is None, 'preview exited before creating its window'
                window = xdo('search', '--onlyvisible', '--pid', process.pid, allow_failure=True).splitlines()
                if window:
                    window = window[0]
                    break
                time.sleep(.1)
            assert window, 'preview window was not found'
            xdo('windowactivate', '--sync', window)

            def progress(label):
                time.sleep(.4)
                deadline = time.monotonic() + 15
                previous = None
                while time.monotonic() < deadline:
                    assert process.poll() is None, 'preview exited during interaction'
                    title = xdo('getwindowname', window)
                    match = re.search(r'\| (\d+) Iterations', title)
                    if match:
                        current = int(match.group(1))
                        if previous is not None and current > previous:
                            actions.append(dict(action=label, title=title, frame_progress=True))
                            return
                        previous = current
                    time.sleep(.1)
                raise AssertionError(f'{label}: preview made no frame progress')

            def save(label, key='s'):
                before = set(interactive.glob('*.png'))
                xdo('key', '--window', window, key)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    added = set(interactive.glob('*.png')) - before
                    if len(added) == 1:
                        path = added.pop()
                        previous_size = -1
                        for _ in range(20):
                            size = path.stat().st_size
                            if size == previous_size and size > 1000:
                                actions.append(dict(action=label, image=path.name, bytes=size,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                                return path
                            previous_size = size
                            time.sleep(.1)
                    time.sleep(.1)
                raise AssertionError(f'{label}: no unique completed PNG save')

            progress('initial-frame-progress')
            save('initial-save')
            for label, button, start, end in (
                ('horizontal-orbit', 1, (700, 400), (810, 400)),
                ('vertical-zoom', 3, (700, 400), (700, 450)),
                ('screen-plane-pan', 2, (700, 400), (740, 425))):
                if label != 'horizontal-orbit':
                    xdo('key', '--window', window, 'space')
                    progress('restore-before-' + label)
                xdo('mousemove', '--window', window, *start)
                progress(label + '-pointer-ready')
                xdo('mousedown', button)
                progress(label + '-button-held')
                xdo('mousemove', '--window', window, *end)
                progress(label + '-motion-delivered')
                xdo('mouseup', button)
                progress(label)
                moved_images[label] = save(label + '-save')
            xdo('key', '--window', window, 'space')
            progress('space-restore')
            reset_image = save('space-restore-save')
            save('escape-save', 'Escape')
            assert process.wait(timeout=20) == 0, 'Escape did not exit successfully'
            assert len(list(interactive.glob('*.png'))) == 6, 'unexpected unique save count'
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            (output / 'actions.json').write_text(json.dumps(actions, indent=2) + '\n')

    samples = int(re.search(r'\.(\d+)samp\.', reset_image.name).group(1))
    reference = interactive / 'reset-reference'
    headless('optical-workbench.json', samples, reference)
    assert reset_image.read_bytes() == reference.with_suffix('.png').read_bytes(), 'Space did not restore exact original camera'
    for label, image in moved_images.items():
        target = int(re.search(r'\.(\d+)samp\.', image.name).group(1))
        baseline = interactive / (label + '-original-reference')
        headless('optical-workbench.json', target, baseline)
        assert image.read_bytes() != baseline.with_suffix('.png').read_bytes(), f'{label}: input did not change the camera image'
    summary = dict(passed=True, smoke_samples=16, preview_headless_png_exact=True,
                   interaction_saves=6, frame_progress=True, escape_exit_code=0,
                   space_restore_png_exact=True, restored_samples=samples,
                   independent_camera_motion_checks=list(moved_images),
                   renderer_config=dict(wavefront=args.wavefront, rr=args.rr,
                                        rr_start=args.rr_start),
                   compatibility=json.loads(prefix.with_suffix('.json').read_text())['compatibility'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == '__main__':
    main()
