#!/usr/bin/env python3
"""Run one immutable Project 3 evidence phase on the validation host."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = ROOT / 'scripts' / 'source_hash.py'
PYTHON = str(Path(sys.executable).resolve())
PHASES = ('build', 'baseline-validation', 'new-validation', 'baseline-measurements', 'new-measurements', 'gallery', 'sanitizers', 'preview')
PREFIX_RE = re.compile('^[A-Za-z0-9_-]+$')
SHA256_RE = re.compile('^[0-9a-f]{64}$')
ERROR_SUMMARY_RE = re.compile('(?mi)^=+\\s*ERROR SUMMARY:\\s*(\\d+)\\s+errors?\\s*$')
RACECHECK_SUMMARY_RE = re.compile('(?mi)^=+\\s*RACECHECK SUMMARY:\\s*(\\d+)\\s+hazards? displayed\\s*\\(\\s*(\\d+)\\s+errors?\\s*,\\s*(\\d+)\\s+warnings?\\s*\\)\\s*$')
GEOMETRY_SUMMARY_RE = re.compile(
    r'(?m)^Executed\s+(\d+)\s+test groups,\s+(\d+)\s+repository scene\(s\),\s+'
    r'and\s+(\d+)\s+assertions;\s+(\d+)\s+group\(s\) failed\.\s*$'
)

class SuiteError(RuntimeError):
    pass


class SuiteInterrupted(SuiteError):
    pass

def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path, value):
    """Atomically publish JSON without ever replacing an existing record."""
    pending = path.with_name('.' + path.name + '.pending')
    payload = (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + '\n').encode('utf-8')
    try:
        with pending.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)

def reject_json_constant(value):
    raise ValueError(f'non-finite JSON constant: {value}')

def prefix_value(value):
    if not PREFIX_RE.fullmatch(value):
        raise argparse.ArgumentTypeError('must contain only ASCII letters, numbers, underscores, and dashes')
    return value

def explicit_bool(value):
    values = {'1': True, 'true': True, 'yes': True, 'on': True, '0': False, 'false': False, 'no': False, 'off': False}
    try:
        return values[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError('expected true or false') from error

def binaries_for(phase, build):
    headless = build / 'bin' / 'cis565_path_tracer_headless'
    preview = build / 'bin' / 'cis565_path_tracer'
    if phase == 'build':
        return {'headless': headless, 'preview': preview}
    if phase == 'preview':
        return {'headless': headless, 'preview': preview}
    return {'headless': headless}

def phase_layout(phase, prefix, build, root=ROOT):
    results = root / 'results'
    suite = results / f'{prefix}-rebuild' if phase == 'build' else results / f'{prefix}-suite-{phase}'
    layout = dict(suite=suite, fresh=[suite], prepare=[], metadata_roots=[], expected=[suite])

    def output(stem):
        path = results / f'{prefix}-{stem}'
        layout['fresh'].append(path)
        layout['metadata_roots'].append(path)
        layout['expected'].append(path)
        return path
    if phase == 'build':
        layout['expected'] += [
            suite / 'system.log',
            *binaries_for(phase, build).values(),
            build / 'bin' / 'geometry_tests',
        ]
    elif phase == 'baseline-validation':
        path = output('validation')
        layout['expected'].append(path / 'summary.json')
    elif phase == 'new-validation':
        path = output('feature-validation')
        layout['expected'].append(path / 'summary.json')
    elif phase == 'baseline-measurements':
        host, mesh, experiments, plots = (output('host'), output('mesh-load'), output('experiments'), output('baseline-plots'))
        layout['prepare'].append(host)
        layout['expected'] += [host / 'procedural-generation.json', mesh / 'summary.json', experiments / 'runs.csv', experiments / 'summary.json', experiments / 'exact-comparison.json', experiments / 'timing-summary.json']
        layout.update(host=host, mesh=mesh, experiments=experiments, plots=plots)
    elif phase == 'new-measurements':
        quality, timing, gallery, plots = (output('quality'), output('timing'), output('feature-gallery'), output('hedge-plots'))
        layout['expected'] += [quality / 'summary.json', timing / 'summary.json', gallery / 'summary.json', plots / 'plot-manifest.json']
        layout.update(quality=quality, timing=timing, gallery=gallery, plots=plots)
    elif phase == 'gallery':
        path = output('gallery')
        layout['expected'] += [path / 'gallery.json', path / 'restart.checkpoint']
    elif phase == 'sanitizers':
        path = output('sanitizers')
        renders = path / 'renders'
        layout['prepare'] += [path, renders]
        layout['expected'].append(renders)
        layout.update(sanitizers=path, renders=renders)
    elif phase == 'preview':
        path = output('preview')
        layout['expected'] += [path / 'actions.json', path / 'summary.json']
    return layout

def phase_commands(phase, layout, build, source_hash, root=ROOT):
    headless = build / 'bin' / 'cis565_path_tracer_headless'
    preview = build / 'bin' / 'cis565_path_tracer'
    script = lambda name: str(root / 'scripts' / name)
    if phase == 'build':
        scenes = [str(path.resolve()) for path in sorted((root / 'scenes').glob('*.json'))]
        return [
            ('system-nvcc', ['nvcc', '--version'], True),
            ('system-gpu', [
                'nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader',
            ], True),
            ('system-cpu', ['lscpu'], True),
            ('system-os', ['cat', '/etc/os-release'], True),
            ('system-memory', ['free', '-b'], True),
            ('system-cmake', ['cmake', '--version'], True),
            ('system-gcc', ['gcc', '--version'], True),
            ('configure', [
                'cmake', '-S', str(root), '-B', str(build),
                '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON',
                '-DBUILD_RENDERER=ON', '-DBUILD_PREVIEW=ON',
                '-DCMAKE_CUDA_ARCHITECTURES=86',
                f'-DPROJECT3_BUILD_ID={source_hash}',
            ], False),
            ('compile', ['cmake', '--build', str(build), '-j4'], False),
            ('test', ['ctest', '--test-dir', str(build), '--output-on-failure'], False),
            ('geometry-scenes', [str(build / 'bin' / 'geometry_tests'), *scenes], False),
        ]
    if phase == 'baseline-validation':
        return [('validate-baseline', [PYTHON, script('validate_render.py'), '--binary', str(headless), '--output', str(layout['metadata_roots'][0])], False)]
    if phase == 'new-validation':
        return [('validate-hedge', [
            PYTHON, script('validate_hedge.py'), '--binary', str(headless),
            '--output', str(layout['metadata_roots'][0]), '--benchmark',
        ], False)]
    if phase == 'baseline-measurements':
        return [('profile-procedural', [PYTHON, script('generate_assets.py'), '--profile', str(layout['host'] / 'procedural-generation.json'), '--profile-only'], False), ('profile-mesh-load', [PYTHON, script('profile_mesh_load.py'), '--build', str(build), '--output', str(layout['mesh'])], False), ('run-experiments', [PYTHON, script('run_experiments.py'), '--binary', str(headless), '--output', str(layout['experiments'])], False), ('verify-experiments', [PYTHON, script('verify_experiments.py'), str(layout['experiments'])], False), ('plot-experiments', [PYTHON, script('plot_experiments.py'), str(layout['experiments']), '--procedural-cpu-profile', str(layout['mesh'] / 'summary.json'), '--output', str(layout['plots'])], False)]
    if phase == 'new-measurements':
        commands = []
        for subphase, key in (('quality', 'quality'), ('timing', 'timing'), ('gallery', 'gallery')):
            commands.append((f'measure-{subphase}', [PYTHON, script('measure_hedge.py'), '--binary', str(headless), '--output', str(layout[key]), '--phase', subphase], False))
        commands.append(('plot-hedge', [PYTHON, script('plot_hedge.py'), '--quality', str(layout['quality']), '--timing', str(layout['timing']), '--output', str(layout['plots'])], False))
        return commands
    if phase == 'gallery':
        return [('render-gallery', [PYTHON, script('render_gallery.py'), '--binary', str(headless), '--output', str(layout['metadata_roots'][0])], False)]
    if phase == 'preview':
        return [('verify-preview', [PYTHON, script('verify_preview.py'), '--preview-binary', str(preview), '--headless-binary', str(headless), '--output', str(layout['metadata_roots'][0]), '--wavefront', 'on', '--rr', 'on', '--rr-start', '3'], False)]
    return []

def sanitizer_cases(layout, build, root=ROOT):
    cases = []
    for scene in ('optical-workbench', 'hedge-motion-validation'):
        for sampler in ('counter', 'halton'):
            for schedule, sort, wavefront in (('sorted-unified', 'on', 'off'), ('wavefront', 'off', 'on')):
                for tool in ('memcheck', 'racecheck'):
                    label = f'{scene}-{sampler}-{schedule}-{tool}'
                    cases.append(dict(scene=scene, sampler=sampler, schedule=schedule, sort=sort, wavefront=wavefront, tool=tool, prefix=layout['renders'] / label, binary=build / 'bin' / 'cis565_path_tracer_headless', scene_path=root / 'scenes' / f'{scene}.json'))
    return cases

def sanitizer_command(case):
    return ['compute-sanitizer', '--error-exitcode', '90', '--target-processes', 'all', '--tool', case['tool'], str(case['binary']), str(case['scene_path']), '--resolution', '67x35', '--samples', '3', '--depth', '8', '--seed', '947', '--rr', 'on', '--rr-start', '3', '--motion', 'on', '--sampler', case['sampler'], '--sort', case['sort'], '--wavefront', case['wavefront'], '--compact', 'on', '--bvh', 'on', '--aa', 'on', '--nee', 'on', '--output', str(case['prefix'])]

def run_self_test():
    import tempfile

    root, build = Path('/suite-root'), Path('/suite-build')
    for phase in PHASES:
        layout = phase_layout(phase, 'hedge_1', build, root)
        expected = 'hedge_1-rebuild' if phase == 'build' else f'hedge_1-suite-{phase}'
        assert layout['suite'].name == expected
        assert len(layout['fresh']) == len(set(layout['fresh']))

    plan = phase_layout('new-measurements', 'safe-name', build, root)
    commands = phase_commands('new-measurements', plan, build, 'a' * 64, root)
    assert len(commands) == 4 and commands[0][1][-2:] == ['--phase', 'quality']
    validation = phase_commands(
        'new-validation', phase_layout('new-validation', 'x', build, root),
        build, 'a' * 64, root,
    )[0][1]
    assert '--benchmark' in validation

    sanitizer_plan = phase_layout('sanitizers', 'x', build, root)
    cases = sanitizer_cases(sanitizer_plan, build, root)
    assert len(cases) == 16
    identities = {
        (case['scene'], case['sampler'], case['schedule'], case['tool'])
        for case in cases
    }
    assert len(identities) == 16
    assert {case['tool'] for case in cases} == {'memcheck', 'racecheck'}
    assert {case['sampler'] for case in cases} == {'counter', 'halton'}
    assert {case['scene'] for case in cases} == {'optical-workbench', 'hedge-motion-validation'}
    assert all(
        (case['sort'], case['wavefront']) in (('on', 'off'), ('off', 'on'))
        for case in cases
    )
    for case in cases:
        command = sanitizer_command(case)
        assert command[:7] == [
            'compute-sanitizer', '--error-exitcode', '90',
            '--target-processes', 'all', '--tool', case['tool'],
        ]
        assert command[command.index('--motion') + 1] == 'on'
        assert command[command.index('--sampler') + 1] == case['sampler']

    for invalid in ('', '../hedge', 'a/b', 'a.b', 'a b'):
        try:
            prefix_value(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f'invalid prefix accepted: {invalid!r}')

    mapped = {
        phase: [path.name for path in phase_layout(phase, 'p', build, root)['fresh']]
        for phase in PHASES
    }
    assert mapped == {
        'build': ['p-rebuild'],
        'baseline-validation': ['p-suite-baseline-validation', 'p-validation'],
        'new-validation': ['p-suite-new-validation', 'p-feature-validation'],
        'baseline-measurements': [
            'p-suite-baseline-measurements', 'p-host', 'p-mesh-load',
            'p-experiments', 'p-baseline-plots',
        ],
        'new-measurements': [
            'p-suite-new-measurements', 'p-quality', 'p-timing',
            'p-feature-gallery', 'p-hedge-plots',
        ],
        'gallery': ['p-suite-gallery', 'p-gallery'],
        'sanitizers': ['p-suite-sanitizers', 'p-sanitizers'],
        'preview': ['p-suite-preview', 'p-preview'],
    }

    assert classify_stage('measure-quality', 2) == 'completed_not_qualified'
    assert classify_stage('measure-timing', 2) == 'failed'
    assert classify_stage('anything', 130) == 'interrupted'
    assert resumable_exit_code(
        {'status': 'passed', 'exit_code': 0, 'errors': []}, 'gallery'
    ) == 0
    assert resumable_exit_code(
        {
            'status': 'completed_not_qualified',
            'exit_code': 2,
            'errors': [],
        },
        'new-measurements',
    ) == 2
    try:
        resumable_exit_code(
            {'status': 'failed', 'exit_code': 1, 'errors': ['x']}, 'gallery'
        )
    except SuiteError:
        pass
    else:
        raise AssertionError('failed summary was accepted for resume')

    with tempfile.TemporaryDirectory(prefix='run-hedge-suite-') as temporary:
        fixture = Path(temporary)
        fake_root = fixture / 'root'
        scenes = fake_root / 'scenes'
        scenes.mkdir(parents=True)
        for name in ('z-scene.json', 'a-scene.json'):
            (scenes / name).write_text('{}\n', encoding='utf-8')
        fake_build = fixture / 'build'
        build_plan = phase_layout('build', 'x', fake_build, fake_root)
        build_commands = phase_commands('build', build_plan, fake_build, 'b' * 64, fake_root)
        configure = next(argv for name, argv, _ in build_commands if name == 'configure')
        geometry = next(argv for name, argv, _ in build_commands if name == 'geometry-scenes')
        assert '-DBUILD_TESTING=ON' in configure
        assert '-DPROJECT3_BUILD_ID=' + 'b' * 64 in configure
        assert geometry[1:] == [str(path.resolve()) for path in sorted(scenes.glob('*.json'))]

        suite = fixture / 'suite'
        (suite / 'logs').mkdir(parents=True)
        fake_summary = {'suite_path': suite, 'stages': []}
        failure = run_logged(
            fake_summary,
            'fake-failure',
            [PYTHON, '-c', "import sys\nprint('fake stdout')\nprint('fake stderr', file=sys.stderr)\nsys.exit(7)"],
        )
        assert failure['status'] == 'failed' and failure['returncode'] == 7
        assert (suite / failure['stdout_log']).read_text().strip() == 'fake stdout'
        assert (suite / failure['stderr_log']).read_text().strip() == 'fake stderr'

        geometry_record = run_logged(
            fake_summary,
            'fake-geometry',
            [PYTHON, '-c', "print('Executed 4 test groups, 2 repository scene(s), and 9 assertions; 0 group(s) failed.')"],
        )
        counts = parse_geometry_summary(geometry_record, suite, 2)
        validate_geometry_evidence(counts)
        assert counts['repository_scenes'] == 2 and counts['failed_groups'] == 0
        invalid_counts = dict(counts, failed_groups=1)
        try:
            validate_geometry_evidence(invalid_counts)
        except SuiteError:
            pass
        else:
            raise AssertionError('failed geometry counts were accepted')

        quality_root = fixture / 'quality-output'
        (quality_root / 'quality').mkdir(parents=True)
        qualification = {'status': 'not-qualified', 'criteria': {'example': False}}
        phase_summary = {
            'phase': 'quality', 'status': 'completed',
            'qualification': qualification,
        }
        session_summary = {
            'execution_status': 'completed', 'phase_requested': 'quality',
            'status': 'not-qualified', 'phases': {'quality': phase_summary},
        }
        (quality_root / 'summary.json').write_text(
            json.dumps(session_summary), encoding='utf-8'
        )
        (quality_root / 'quality' / 'summary.json').write_text(
            json.dumps(phase_summary), encoding='utf-8'
        )
        outcome = read_quality_outcome(quality_root, require_negative=True)
        assert outcome['qualification'] == qualification
        try:
            read_quality_outcome(quality_root, require_negative=False)
        except SuiteError:
            pass
        else:
            raise AssertionError('negative quality evidence was accepted for return code 0')

        resume_plan = expected_stage_plan(
            'new-measurements', plan, build, 'a' * 64
        )
        resumed_stages = [
            {
                'name': name,
                'argv': argv,
                'cwd': str(ROOT),
                'returncode': 0,
                'status': 'passed',
            }
            for name, argv in resume_plan
        ]
        resumed_stages[1]['returncode'] = 2
        resumed_stages[1]['status'] = 'completed_not_qualified'
        validate_resumed_stage_plan(resumed_stages, resume_plan, 2)
        try:
            validate_resumed_stage_plan([], resume_plan, 2)
        except SuiteError:
            pass
        else:
            raise AssertionError('empty resume stage list was accepted')

    print('run_hedge_suite.py self-test passed')
    return 0


def classify_stage(name, returncode):
    if returncode == 0:
        return 'passed'
    if name == 'measure-quality' and returncode == 2:
        return 'completed_not_qualified'
    if returncode in (-2, 130):
        return 'interrupted'
    return 'failed'

def run_logged(summary, name, argv):
    number = len(summary['stages']) + 1
    safe = re.sub('[^A-Za-z0-9_-]+', '-', name).strip('-')
    stdout_rel = Path('logs') / f'{number:03d}-{safe}.stdout.log'
    stderr_rel = Path('logs') / f'{number:03d}-{safe}.stderr.log'
    stdout_path, stderr_path = (summary['suite_path'] / stdout_rel, summary['suite_path'] / stderr_rel)
    started, tick = (utc_now(), time.perf_counter())
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr, shell=False, check=False)
            returncode = result.returncode
        except KeyboardInterrupt:
            stderr.write(b'suite command interrupted by KeyboardInterrupt\n')
            returncode = 130
        except OSError as error:
            stderr.write(f'suite could not start command: {error}\n'.encode('utf-8', 'replace'))
            returncode = 127
        except BaseException as error:
            stderr.write(f'suite command interrupted by {type(error).__name__}\n'.encode())
            returncode = 125
    record = dict(
        name=name,
        argv=list(argv),
        cwd=str(ROOT),
        started_at_utc=started,
        ended_at_utc=utc_now(),
        elapsed_seconds=time.perf_counter() - tick,
        returncode=returncode,
        status=classify_stage(name, returncode),
        stdout_log=stdout_rel.as_posix(),
        stderr_log=stderr_rel.as_posix(),
        stdout_sha256=sha256_file(stdout_path),
        stderr_sha256=sha256_file(stderr_path),
    )
    summary['stages'].append(record)
    return record

def require_stage(record):
    if record['status'] == 'interrupted':
        raise SuiteInterrupted(f"stage {record['name']} was interrupted")
    if record['status'] != 'passed':
        raise SuiteError(f"stage {record['name']} exited {record['returncode']}; see its logs")

def parse_source_hash(record, suite):
    if record['status'] == 'interrupted':
        raise SuiteInterrupted(f"stage {record['name']} was interrupted")
    text = (suite / record['stdout_log']).read_text(encoding='utf-8').strip()
    if record['returncode'] != 0 or not SHA256_RE.fullmatch(text):
        raise SuiteError(f"stage {record['name']} did not emit exactly one SHA-256")
    return text


def read_quality_outcome(output, require_negative=False):
    session_path = output / 'summary.json'
    phase_path = output / 'quality' / 'summary.json'
    try:
        session = json.loads(session_path.read_text(encoding='utf-8'))
        phase = json.loads(phase_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SuiteError(f'quality return code 2 lacks readable completed evidence: {error}') from error
    qualification = phase.get('qualification') if isinstance(phase, dict) else None
    qualification_status = qualification.get('status') if isinstance(qualification, dict) else None
    expected_session_status = 'passed' if qualification_status == 'qualified' else qualification_status
    if not (
        isinstance(session, dict)
        and session.get('execution_status') == 'completed'
        and session.get('phase_requested') == 'quality'
        and session.get('status') == expected_session_status
        and session.get('phases', {}).get('quality') == phase
        and phase.get('status') == 'completed'
        and qualification_status in ('qualified', 'not-qualified', 'inconclusive')
        and (
            qualification_status in ('not-qualified', 'inconclusive')
            if require_negative
            else qualification_status == 'qualified'
        )
    ):
        raise SuiteError('quality summary does not describe a completed measurement outcome')
    return {
        'session_summary_path': str(session_path),
        'session_summary_sha256': sha256_file(session_path),
        'session_status': session.get('status'),
        'execution_status': session.get('execution_status'),
        'phase_requested': session.get('phase_requested'),
        'phase_summary_path': str(phase_path),
        'phase_summary_sha256': sha256_file(phase_path),
        'phase_status': phase.get('status'),
        'qualification': qualification,
    }


def parse_geometry_summary(record, suite, expected_scenes):
    text = (suite / record['stdout_log']).read_text(encoding='utf-8')
    matches = GEOMETRY_SUMMARY_RE.findall(text)
    if len(matches) != 1:
        raise SuiteError('geometry_tests output lacks one recognizable final summary')
    groups, scenes, assertions, failed = map(int, matches[0])
    counts = {
        'test_groups': groups,
        'repository_scenes': scenes,
        'assertions': assertions,
        'failed_groups': failed,
        'expected_repository_scenes': expected_scenes,
    }
    return counts


def geometry_evidence(record, suite):
    scene_paths = record['argv'][1:]
    counts = parse_geometry_summary(record, suite, len(scene_paths))
    executable = Path(record['argv'][0])
    return {
        **counts,
        'executable': str(executable),
        'executable_sha256': sha256_file(executable),
        'scene_paths': scene_paths,
    }


def validate_geometry_evidence(evidence):
    if (
        evidence['expected_repository_scenes'] <= 0
        or evidence['repository_scenes'] != evidence['expected_repository_scenes']
        or evidence['failed_groups'] != 0
    ):
        raise SuiteError(f'geometry_tests reported invalid counts: {evidence}')

def metadata_record(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_json_constant)
    except (OSError, UnicodeError, ValueError):
        return None
    if not (isinstance(value, dict) and isinstance(value.get('compatibility'), dict) and isinstance(value.get('completed_samples'), int) and (not isinstance(value.get('completed_samples'), bool)) and isinstance(value.get('resolution'), list) and (len(value['resolution']) == 2)):
        return None
    linear = value.get('linear_sum_format')
    camera = value.get('camera')
    return dict(path=str(path), sha256=sha256_file(path), completed_samples=value['completed_samples'], resolution=value['resolution'], compatibility=value['compatibility'], config=value.get('config'), sampler=value.get('sampler'), sampler_id=value.get('sampler_id'), sampler_version=value.get('sampler_version'), camera_shutter=camera.get('shutter') if isinstance(camera, dict) else None, format_version=value.get('format_version'), linear_sum_format_version=linear.get('version') if isinstance(linear, dict) else None)

def first_metadata(roots):
    for root in roots:
        if root.is_dir():
            for path in sorted(root.rglob('*.json')):
                record = metadata_record(path)
                if record is not None:
                    return record
    return None

def validate_metadata_source(records, source_hash):
    for record in records:
        identifier = record.get('compatibility', {}).get('build_identifier')
        if not isinstance(identifier, str) or source_hash not in identifier:
            raise SuiteError('renderer build_identifier does not name the phase source hash')

def emitted_files(layout):
    files = []
    for root in layout['fresh']:
        if root.is_dir():
            files.extend((path for path in root.rglob('*') if path.is_file()))
    return sorted(files)

def expected_stage_plan(phase, layout, build, source_hash):
    if phase == 'sanitizers':
        middle = [
            ('sanitize-' + case['prefix'].name, sanitizer_command(case))
            for case in sanitizer_cases(layout, build)
        ]
    else:
        middle = [
            (name, argv)
            for name, argv, _ in phase_commands(phase, layout, build, source_hash)
        ]
    source = [PYTHON, str(SOURCE_HASH)]
    return [('source-hash-start', source), *middle, ('source-hash-end', source)]

def parse_sanitizer_log(record, suite, tool):
    text = '\n'.join(((suite / record[key]).read_bytes().decode('utf-8', 'replace') for key in ('stdout_log', 'stderr_log')))
    errors = [int(value) for value in ERROR_SUMMARY_RE.findall(text)]
    races = [tuple(map(int, values)) for values in RACECHECK_SUMMARY_RE.findall(text)]
    if tool == 'memcheck' and (not errors):
        raise SuiteError('memcheck log lacks a recognizable ERROR SUMMARY')
    if tool == 'racecheck' and (not races):
        raise SuiteError('racecheck log lacks a recognizable RACECHECK SUMMARY')
    if (
        text.count('ERROR SUMMARY:') != len(errors)
        or text.count('RACECHECK SUMMARY:') != len(races)
    ):
        raise SuiteError('sanitizer log contains an unrecognized summary format')
    warning_lines = [line.strip() for line in text.splitlines() if re.search('\\bwarnings?\\b', line, re.IGNORECASE)]
    return dict(error_counts=errors, racecheck_counts=[dict(hazards=a, errors=b, warnings=c) for a, b, c in races], warning_lines=warning_lines, clean=not (any(errors) or any((any(counts) for counts in races))))

def inspect_sanitizer_artifact(case):
    from validate_render import read_sum
    prefix = case['prefix']
    paths = {'png': prefix.with_suffix('.png'), 'sum': prefix.with_suffix('.sum.bin'), 'metadata': prefix.with_suffix('.json')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SuiteError('sanitizer renderer outputs missing: ' + ', '.join(missing))
    raw = read_sum(paths['sum'])
    metadata = metadata_record(paths['metadata'])
    if metadata is None:
        raise SuiteError(f"unrecognized renderer metadata: {paths['metadata']}")
    if [raw['width'], raw['height']] != [67, 35] or raw['samples'] != 3 or metadata['resolution'] != [raw['width'], raw['height']] or (metadata['completed_samples'] != raw['samples']):
        raise SuiteError(f'sanitizer renderer output contract mismatch: {prefix}')
    config = metadata.get('config')
    sampler_id = {'counter': 2, 'halton': 3}[case['sampler']]
    shutter = metadata.get('camera_shutter')
    if not isinstance(config, dict) or metadata.get('sampler') != case['sampler'] or config.get('sampler') != case['sampler'] or (metadata.get('sampler_id') != sampler_id) or (metadata.get('sampler_version') != sampler_id) or (config.get('sampler_id') != sampler_id) or (config.get('motion') is not True) or (config.get('sort') != (case['sort'] == 'on')) or (config.get('wavefront') != (case['wavefront'] == 'on')) or any((config.get(key) is not True for key in ('compact', 'bvh', 'aa', 'nee', 'rr'))) or (config.get('rr_start') != 3) or (config.get('max_depth') != 8) or (not (isinstance(shutter, list) and len(shutter) == 2 and all((type(value) in (int, float) for value in shutter)) and (0 <= shutter[0] <= shutter[1] <= 1))) or (metadata.get('linear_sum_format_version') != 1):
        raise SuiteError(f'sanitizer metadata flags do not match invocation: {prefix}')
    return (paths, raw, metadata)

def run_sanitizers(summary, layout, build, source_hash, binary_hash):
    groups = {}
    summary['sanitizer_runs'] = []
    summary['sanitizer_exact_comparisons'] = []
    for case in sanitizer_cases(layout, build):
        stage = run_logged(summary, 'sanitize-' + case['prefix'].name, sanitizer_command(case))
        run = dict(scene=case['scene'], sampler=case['sampler'], schedule=case['schedule'], tool=case['tool'], source_input_sha256=source_hash, headless_binary_sha256=binary_hash, stage=stage['name'])
        summary['sanitizer_runs'].append(run)
        problem = None
        try:
            run['sanitizer_summary'] = parse_sanitizer_log(stage, summary['suite_path'], case['tool'])
            if not run['sanitizer_summary']['clean']:
                problem = SuiteError('sanitizer reported nonzero errors, hazards, or warnings')
        except Exception as error:
            run['sanitizer_summary_error'] = str(error)
            problem = error
        try:
            paths, raw, metadata = inspect_sanitizer_artifact(case)
            run['metadata'] = metadata
            run['artifacts'] = {name: dict(path=str(path), sha256=sha256_file(path)) for name, path in paths.items()}
            summary['renderer_metadata'].append(metadata)
            groups.setdefault((case['scene'], case['sampler']), []).append((str(paths['sum']), raw['encoded']))
        except Exception as error:
            run['artifact_error'] = str(error)
            problem = problem or error
        require_stage(stage)
        if problem is not None:
            raise problem
    comparisons = []
    for (scene, sampler), items in groups.items():
        if len(items) != 4 or any((encoded != items[0][1] for _, encoded in items[1:])):
            raise SuiteError(f'sanitizer sums differ for {scene}/{sampler}')
        comparisons.append(dict(scene=scene, sampler=sampler, byte_exact=True, paths=[path for path, _ in items], sha256=hashlib.sha256(items[0][1]).hexdigest()))
    summary['sanitizer_exact_comparisons'] = comparisons
    return [Path(item['path']) for record in summary['sanitizer_runs'] for item in record.get('artifacts', {}).values()]

def write_system_log(summary, records):
    target = summary['suite_path'] / 'system.log'
    with target.open('xb') as output:
        for record in records:
            output.write(f"## {record['name']}\n$ {shlex.join(record['argv'])}\nreturncode: {record['returncode']}\n[stdout]\n".encode())
            output.write((summary['suite_path'] / record['stdout_log']).read_bytes())
            output.write(b'\n[stderr]\n')
            output.write((summary['suite_path'] / record['stderr_log']).read_bytes())
            output.write(b'\n\n')

def validate_contract(phase, layout, build):
    missing = [str(path) for path in layout['expected'] if not os.path.lexists(path)]
    if missing:
        raise SuiteError('expected outputs missing: ' + ', '.join(missing))
    directories = set(layout['fresh'] + layout['prepare'])
    invalid_directories = [str(path) for path in directories if not path.is_dir()]
    if invalid_directories:
        raise SuiteError(
            'expected output directories are invalid: '
            + ', '.join(invalid_directories)
        )
    invalid = [str(path) for path in layout['expected'] if path not in directories and (not path.is_file())]
    if invalid:
        raise SuiteError('expected file outputs are not regular files: ' + ', '.join(invalid))
    if phase == 'build':
        executables = [*binaries_for(phase, build).values(), build / 'bin' / 'geometry_tests']
        bad = [str(path) for path in executables if not path.is_file() or not os.access(path, os.X_OK)]
        if bad:
            raise SuiteError('built renderer binary missing or not executable: ' + ', '.join(bad))
    elif phase == 'baseline-measurements':
        profile = json.loads((layout['host'] / 'procedural-generation.json').read_text())
        if len(profile.get('records', [])) != 6:
            raise SuiteError('procedural profile does not contain six records')
        with (layout['experiments'] / 'runs.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 288 or len({row.get('prefix') for row in rows}) != 288:
            raise SuiteError('experiment evidence is not 288 unique fresh runs')
    elif phase == 'gallery':
        records = json.loads((layout['metadata_roots'][0] / 'gallery.json').read_text())
        if not isinstance(records, list) or len(records) != 23 or len({record.get('name') for record in records}) != 23:
            raise SuiteError('gallery evidence is not 23 unique records')
    elif phase == 'sanitizers':
        groups = {}
        for case in sanitizer_cases(layout, build):
            _, raw, _ = inspect_sanitizer_artifact(case)
            groups.setdefault((case['scene'], case['sampler']), []).append(raw['encoded'])
        if any((len(values) != 4 or len(set(values)) != 1 for values in groups.values())):
            raise SuiteError('sanitizer raw outputs are incomplete or not byte-exact')

def snapshot_binaries(paths):
    return {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}

def current_source_hash():
    result = subprocess.run([PYTHON, str(SOURCE_HASH)], cwd=ROOT, capture_output=True, text=True, shell=False, check=False)
    value = result.stdout.strip()
    if result.returncode != 0 or not SHA256_RE.fullmatch(value):
        raise SuiteError('could not obtain current source identity from source_hash.py')
    return value


def resumable_exit_code(summary, phase):
    if not isinstance(summary, dict) or summary.get('errors') != []:
        raise SuiteError('resume rejected: suite summary is incomplete or failed')
    status = summary.get('status')
    if status == 'passed' and summary.get('exit_code') == 0:
        return 0
    if (
        status == 'completed_not_qualified'
        and summary.get('exit_code') == 2
        and phase == 'new-measurements'
    ):
        return 2
    raise SuiteError('resume rejected: suite summary is incomplete or failed')


def validate_resumed_stage_plan(stages, plan, outcome_code):
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise SuiteError('resume rejected: stage record is incomplete')
    if [(stage.get('name'), stage.get('argv')) for stage in stages] != plan:
        raise SuiteError('resume rejected: stage command sequence does not match')

    quality_code_twos = 0
    for stage in stages:
        is_quality_two = (
            stage.get('name') == 'measure-quality'
            and stage.get('returncode') == 2
            and stage.get('status') == 'completed_not_qualified'
        )
        if is_quality_two:
            quality_code_twos += 1
        elif stage.get('status') != 'passed' or stage.get('returncode') != 0:
            raise SuiteError('resume rejected: summary contains an unsuccessful stage')
        if stage.get('cwd') != str(ROOT):
            raise SuiteError('resume rejected: stage working directory does not match')
    if quality_code_twos != (1 if outcome_code == 2 else 0):
        raise SuiteError('resume rejected: quality outcome and stage return code disagree')


def verify_resume(phase, prefix, build, layout):
    existing = [path for path in layout['fresh'] if os.path.lexists(path)]
    if not existing:
        return None
    summary_path = layout['suite'] / 'summary.json'
    if not summary_path.is_file():
        raise SuiteError('resume rejected: existing outputs lack the suite summary')
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteError(f'resume rejected: unreadable suite summary: {error}') from error
    outcome_code = resumable_exit_code(summary, phase)
    if summary.get('schema_version') != 1:
        raise SuiteError('resume rejected: unsupported suite summary schema')
    if (summary.get('phase'), summary.get('prefix'), summary.get('build')) != (phase, prefix, str(build)):
        raise SuiteError('resume rejected: phase, prefix, or build does not match')

    current = current_source_hash()
    if not summary.get('source_input_sha256_start') == summary.get('source_input_sha256_end') == current:
        raise SuiteError('resume rejected: source identity does not match')

    current_binaries = snapshot_binaries(binaries_for(phase, build))
    if summary.get('binary_sha256_end') != current_binaries:
        raise SuiteError('resume rejected: relevant binary identity does not match')
    if phase != 'build' and summary.get('binary_sha256_start') != current_binaries:
        raise SuiteError('resume rejected: runtime binary changed during the recorded phase')
    helper_hash = sha256_file(SOURCE_HASH)
    if not summary.get('source_hash_script_sha256_start') == summary.get('source_hash_script_sha256_end') == helper_hash:
        raise SuiteError('resume rejected: source_hash.py identity does not match')

    output_paths = summary.get('output_paths')
    if not isinstance(output_paths, list) or not all((isinstance(path, str) for path in output_paths)) or any((not os.path.exists(path) for path in output_paths)):
        raise SuiteError('resume rejected: a recorded output is missing')
    if any((str(path) not in output_paths for path in layout['expected'])):
        raise SuiteError('resume rejected: summary omits an expected output')

    stages = summary.get('stages')
    plan = expected_stage_plan(phase, layout, build, current)
    validate_resumed_stage_plan(stages, plan, outcome_code)
    for stage in stages:
        for key, hash_key in (('stdout_log', 'stdout_sha256'), ('stderr_log', 'stderr_sha256')):
            relative = stage.get(key)
            if not isinstance(relative, str):
                raise SuiteError('resume rejected: stage log path is invalid')
            path = layout['suite'] / relative
            try:
                path.resolve().relative_to(layout['suite'].resolve())
            except ValueError as error:
                raise SuiteError('resume rejected: unsafe recorded log path') from error
            if not path.is_file() or sha256_file(path) != stage.get(hash_key):
                raise SuiteError('resume rejected: a stage log is missing or changed')

    if phase == 'new-measurements':
        actual_quality = read_quality_outcome(layout['quality'], require_negative=outcome_code == 2)
        if summary.get('quality_outcome') != actual_quality:
            raise SuiteError('resume rejected: recorded quality outcome changed')
    if phase == 'build':
        geometry_stage = next(stage for stage in stages if stage['name'] == 'geometry-scenes')
        actual_geometry = geometry_evidence(geometry_stage, layout['suite'])
        validate_geometry_evidence(actual_geometry)
        if summary.get('geometry_tests') != actual_geometry:
            raise SuiteError('resume rejected: geometry_tests evidence changed')

    metadata_records = summary.get('renderer_metadata')
    if phase != 'build' and (not isinstance(metadata_records, list) or not metadata_records):
        raise SuiteError('resume rejected: renderer metadata record is missing')
    for record in metadata_records or []:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise SuiteError('resume rejected: invalid renderer metadata record')
        path = Path(record['path'])
        if not path.is_file() or sha256_file(path) != record.get('sha256'):
            raise SuiteError('resume rejected: renderer metadata is missing or changed')
    if metadata_records:
        validate_metadata_source(metadata_records, current)
    runs = summary.get('sanitizer_runs', [])
    comparisons = summary.get('sanitizer_exact_comparisons', [])
    if phase == 'sanitizers' and (not isinstance(runs, list) or len(runs) != 16 or (not isinstance(comparisons, list)) or (len(comparisons) != 4)):
        raise SuiteError('resume rejected: sanitizer record set is incomplete')
    if not isinstance(runs, list) or not all((isinstance(run, dict) for run in runs)):
        raise SuiteError('resume rejected: sanitizer records are invalid')
    for run in runs:
        artifacts = run.get('artifacts')
        if not isinstance(artifacts, dict) or set(artifacts) != {'png', 'sum', 'metadata'}:
            raise SuiteError('resume rejected: sanitizer artifact record is incomplete')
        for artifact in artifacts.values():
            if not isinstance(artifact, dict) or not isinstance(artifact.get('path'), str):
                raise SuiteError('resume rejected: sanitizer artifact record is invalid')
            path = Path(artifact.get('path', ''))
            if not path.is_file() or sha256_file(path) != artifact.get('sha256'):
                raise SuiteError('resume rejected: sanitizer artifact is missing or changed')
    try:
        validate_contract(phase, layout, build)
    except Exception as error:
        if isinstance(error, SuiteError):
            raise
        raise SuiteError(f'resume rejected: output validation failed: {error}') from error
    print(f'completed {phase} evidence already verified; nothing was executed')
    return outcome_code

def execute_phase(phase, prefix, build, layout):
    suite = layout['suite']
    started_utc, tick = utc_now(), time.perf_counter()
    summary = {
        'schema_version': 1,
        'phase': phase,
        'prefix': prefix,
        'build': str(build),
        'suite_path': suite,
        'started_at_utc': started_utc,
        'ended_at_utc': None,
        'elapsed_seconds': None,
        'status': 'failed',
        'exit_code': 1,
        'source_input_sha256_start': None,
        'source_input_sha256_end': None,
        'source_hash_script_sha256_start': None,
        'source_hash_script_sha256_end': None,
        'binary_paths': {
            name: str(path) for name, path in binaries_for(phase, build).items()
        },
        'binary_sha256_start': {},
        'binary_sha256_end': {},
        'output_paths': [str(path) for path in layout['expected']],
        'renderer_metadata': [],
        'stages': [],
        'errors': [],
    }
    completed = False
    interrupted = False
    quality_not_qualified = False

    def note_error(error, prefix=''):
        nonlocal interrupted
        if isinstance(error, (KeyboardInterrupt, SuiteInterrupted)):
            interrupted = True
        message = str(error) or type(error).__name__
        summary['errors'].append(prefix + message)

    def run_phase_stage(name, argv):
        nonlocal quality_not_qualified
        record = run_logged(summary, name, argv)
        if record['status'] == 'interrupted':
            require_stage(record)
        if name == 'measure-quality' and record['returncode'] in (0, 2):
            is_negative = record['returncode'] == 2
            summary['quality_outcome'] = read_quality_outcome(
                layout['quality'], require_negative=is_negative
            )
            quality_not_qualified = is_negative
            return record
        if name == 'geometry-scenes':
            try:
                evidence = geometry_evidence(record, suite)
                summary['geometry_tests'] = evidence
            except SuiteError as error:
                summary['geometry_tests'] = {'parse_error': str(error)}
                require_stage(record)
                raise
            require_stage(record)
            validate_geometry_evidence(evidence)
            return record
        require_stage(record)
        return record

    setup_error = None
    try:
        suite.mkdir(parents=True, exist_ok=False)
        (suite / 'logs').mkdir()
    except KeyboardInterrupt as error:
        if not suite.is_dir():
            raise
        (suite / 'logs').mkdir(exist_ok=True)
        setup_error = error
    except OSError as error:
        raise SuiteError(f'could not create fresh suite directory: {error}') from error

    try:
        if setup_error is not None:
            raise setup_error
        summary['binary_sha256_start'] = snapshot_binaries(binaries_for(phase, build))
        summary['source_hash_script_sha256_start'] = sha256_file(SOURCE_HASH)
        start_record = run_logged(summary, 'source-hash-start', [PYTHON, str(SOURCE_HASH)])
        summary['source_input_sha256_start'] = parse_source_hash(start_record, suite)
        if phase != 'build' and any((value is None for value in summary['binary_sha256_start'].values())):
            raise SuiteError('a required renderer binary is missing')
        for path in layout['prepare']:
            path.mkdir(exist_ok=False)
        if phase == 'sanitizers':
            run_sanitizers(
                summary,
                layout,
                build,
                summary['source_input_sha256_start'],
                summary['binary_sha256_start']['headless'],
            )
        else:
            commands = phase_commands(phase, layout, build, summary['source_input_sha256_start'])
            if phase == 'build':
                system_records = []
                for name, argv, system in commands:
                    if not system:
                        continue
                    record = run_logged(summary, name, argv)
                    system_records.append(record)
                    if record['status'] == 'interrupted':
                        write_system_log(summary, system_records)
                        require_stage(record)
                write_system_log(summary, system_records)
                failed = [record for record in system_records if record['returncode']]
                if failed:
                    names = ', '.join(record['name'] for record in failed)
                    raise SuiteError('system inventory command failed: ' + names)
                commands = [item for item in commands if not item[2]]
            for name, argv, _ in commands:
                run_phase_stage(name, argv)
        validate_contract(phase, layout, build)
        completed = True
    except BaseException as error:
        note_error(error)
    finally:
        try:
            end_record = run_logged(summary, 'source-hash-end', [PYTHON, str(SOURCE_HASH)])
            summary['source_input_sha256_end'] = parse_source_hash(end_record, suite)
        except BaseException as error:
            note_error(error)
        try:
            summary['source_hash_script_sha256_end'] = sha256_file(SOURCE_HASH)
            summary['binary_sha256_end'] = snapshot_binaries(binaries_for(phase, build))
        except BaseException as error:
            note_error(error, 'final identity snapshot failed: ')
        if summary['source_input_sha256_start'] is None or summary['source_input_sha256_start'] != summary['source_input_sha256_end']:
            summary['errors'].append('source input identity changed or was unavailable')
        if summary['source_hash_script_sha256_start'] is None or summary['source_hash_script_sha256_start'] != summary['source_hash_script_sha256_end']:
            summary['errors'].append('source_hash.py changed or was unavailable')
        if phase != 'build' and summary['binary_sha256_start'] != summary['binary_sha256_end']:
            summary['errors'].append('a relevant renderer binary changed during the phase')
        try:
            if phase != 'build' and (not summary['renderer_metadata']):
                record = first_metadata(layout['metadata_roots'])
                if record is not None:
                    summary['renderer_metadata'] = [record]
                elif completed:
                    raise SuiteError('no recognizable renderer metadata was generated')
            if summary['renderer_metadata'] and summary['source_input_sha256_start']:
                validate_metadata_source(summary['renderer_metadata'], summary['source_input_sha256_start'])
        except BaseException as error:
            note_error(error)
        try:
            inventory = [str(path) for path in emitted_files(layout)] + [str(suite / 'summary.json')]
            summary['output_paths'] = list(dict.fromkeys(summary['output_paths'] + inventory))
        except BaseException as error:
            note_error(error, 'output inventory failed: ')
        if not completed and (not summary['errors']):
            summary['errors'].append('phase did not complete')
        if interrupted:
            summary['status'] = 'interrupted'
            summary['exit_code'] = 130
        elif not completed or summary['errors']:
            summary['status'] = 'failed'
            summary['exit_code'] = 1
        elif quality_not_qualified:
            summary['status'] = 'completed_not_qualified'
            summary['exit_code'] = 2
        else:
            summary['status'] = 'passed'
            summary['exit_code'] = 0
        summary['ended_at_utc'] = utc_now()
        summary['elapsed_seconds'] = time.perf_counter() - tick
        summary['suite_path'] = str(suite)
        summary_path = suite / 'summary.json'
        try:
            write_json_exclusive(summary_path, summary)
        except KeyboardInterrupt as error:
            if summary_path.exists():
                (suite / '.summary.json.pending').unlink(missing_ok=True)
            else:
                note_error(error, 'summary persistence interrupted: ')
                summary['status'] = 'interrupted'
                summary['exit_code'] = 130
                summary['ended_at_utc'] = utc_now()
                summary['elapsed_seconds'] = time.perf_counter() - tick
                (suite / '.summary.json.pending').unlink(missing_ok=True)
                write_json_exclusive(summary_path, summary)
    print(suite)
    return summary['exit_code']

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, help='single evidence phase to execute')
    parser.add_argument(
        '--build', type=Path, default=ROOT / 'build',
        help='build directory (default: ROOT/build)',
    )
    parser.add_argument('--prefix', type=prefix_value, default='hedge', help='safe results directory stem')
    parser.add_argument(
        '--resume-completed', nargs='?', const=True, type=explicit_bool,
        default=False, metavar='BOOL',
        help='verify and skip matching completed evidence',
    )
    parser.add_argument('--self-test', action='store_true', help='run local pure construction assertions and exit')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.phase is None:
        parser.error('--phase is required unless --help or --self-test is used')
    build = args.build.expanduser().resolve()
    layout = phase_layout(args.phase, args.prefix, build)
    try:
        if args.resume_completed:
            resumed = verify_resume(args.phase, args.prefix, build, layout)
            if resumed is not None:
                return resumed
        conflicts = [str(path) for path in layout['fresh'] if os.path.lexists(path)]
        if conflicts:
            raise SuiteError('refusing to overwrite existing phase path(s): ' + ', '.join(conflicts))
        return execute_phase(args.phase, args.prefix, build, layout)
    except KeyboardInterrupt:
        print('run_hedge_suite.py: interrupted', file=sys.stderr)
        return 130
    except SuiteError as error:
        print(f'run_hedge_suite.py: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
