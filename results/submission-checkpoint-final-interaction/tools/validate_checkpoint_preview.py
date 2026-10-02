#!/usr/bin/env python3
"""Validate downloaded genuine preview-suite evidence without running a renderer."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json

ROOT = Path.cwd()
REMOTE_ROOT = Path('/home/ubuntu/cis5650/project3')
SOURCE = None
PREFIX = None


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f'Expected a JSON object: {path}')
    return value


def main():
    global SOURCE, PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    SOURCE, PREFIX = args.source, args.prefix
    need(len(SOURCE) == 64 and set(SOURCE) <= set('0123456789abcdef'), 'Explicit source SHA256 required')
    need(PREFIX == 'submission-checkpoint-final', 'Unexpected evidence prefix')
    controller_path = ROOT / f'results/{PREFIX}-controller/summary.json'
    controller = load(controller_path)
    expected_phases = ['build', 'new-validation', 'sanitizers', 'gltf', 'baseline-validation', 'baseline-measurements', 'new-measurements', 'gallery']
    need(controller.get('status') == 'completed' and controller.get('errors') == [], 'Headless completion gate did not pass')
    need(controller.get('expected_renderer_source_sha256') == SOURCE, 'Headless source identity differs')
    need([phase.get('name') for phase in controller['phases']] == expected_phases, 'Expected all eight headless phases')
    for phase in controller['phases']:
        allowed = (phase.get('status') == 'passed' and phase.get('returncode') == 0) or (phase['name'] == 'new-measurements' and phase.get('status') == 'completed_not_qualified' and phase.get('returncode') == 2)
        need(allowed and sha(ROOT / phase['summary']) == phase.get('summary_sha256'), 'Actual phase status or summary hash differs')
    build_summary = load(ROOT / controller['phases'][0]['summary'])
    expected_binaries = build_summary['binary_sha256_end']
    need(build_summary['source_input_sha256_start'] == build_summary['source_input_sha256_end'] == SOURCE, 'Actual build source identities changed')
    need(all(isinstance(value, str) and len(value) == 64 for value in expected_binaries.values()), 'Actual built binary identities missing')
    suite_path = ROOT / f'results/{PREFIX}-suite-preview/summary.json'
    preview_path = ROOT / f'results/{PREFIX}-preview/summary.json'
    suite, preview = load(suite_path), load(preview_path)
    need((suite.get('schema_version'), suite.get('phase'), suite.get('prefix'), suite.get('status'), suite.get('exit_code'))
         == (1, 'preview', PREFIX, 'passed', 0) and suite.get('errors') == [], 'Preview suite did not pass its actual phase')
    need(suite['source_input_sha256_start'] == suite['source_input_sha256_end'] == SOURCE, 'Preview suite source identity differs')
    hasher_sha = sha(ROOT / 'scripts/source_hash.py')
    need(suite['source_hash_script_sha256_start'] == suite['source_hash_script_sha256_end'] == hasher_sha, 'Source hashing script changed')
    need(suite['binary_sha256_start'] == suite['binary_sha256_end'] == expected_binaries, 'Preview binary identities differ from the actual frozen build')
    build = REMOTE_ROOT / 'build-submission'
    need(suite['build'] == str(build), 'Preview build path differs')
    need(suite['binary_paths'] == {'headless': str(build / 'bin/cis565_path_tracer_headless'), 'preview': str(build / 'bin/cis565_path_tracer')}, 'Preview binary paths differ')
    stages = suite['stages']
    need([stage['name'] for stage in stages] == ['source-hash-start', 'verify-preview', 'source-hash-end'], 'Preview suite stage set differs')
    need(all(stage['status'] == 'passed' and stage['returncode'] == 0 for stage in stages), 'An actual preview stage did not pass')
    records = suite['renderer_metadata']
    need(isinstance(records, list) and records, 'Preview renderer metadata is missing')
    for record in records:
        relative = Path(record['path']).relative_to(REMOTE_ROOT)
        need(relative.is_relative_to(Path(f'results/{PREFIX}-preview')), 'Unexpected preview metadata location')
        path = ROOT / relative
        need(sha(path) == record['sha256'], 'Preview renderer metadata differs from the suite inventory')
        metadata = load(path)
        need(metadata['compatibility'] == record['compatibility'] and SOURCE in metadata['compatibility']['build_identifier'], 'Preview renderer metadata identifies another build')
    expected = {'passed': True, 'smoke_samples': 16, 'preview_headless_png_exact': True,
                'interaction_saves': 6, 'frame_progress': True, 'escape_exit_code': 0,
                'space_restore_png_exact': True,
                'independent_camera_motion_checks': ['horizontal-orbit', 'vertical-zoom', 'screen-plane-pan'],
                'renderer_config': {'wavefront': 'on', 'rr': 'on', 'rr_start': 3}}
    for key, value in expected.items():
        need(type(preview.get(key)) is type(value) and preview[key] == value, f'Actual preview check failed: {key}')
    need(SOURCE in preview['compatibility']['build_identifier'], 'Preview summary identifies another renderer build')
    need(preview['compatibility'] == load(ROOT / f'results/{PREFIX}-preview/smoke/headless.json')['compatibility'], 'Preview smoke compatibility differs')
    report = {'scope': 'Downloaded actual preview suite only; no new renderer execution or media publication',
              'passed': True, 'source_sha256': SOURCE, 'binary_sha256': expected_binaries,
              'headless_controller_sha256': sha(controller_path), 'evidence_prefix': PREFIX,
              'suite_summary_sha256': sha(suite_path), 'preview_summary_sha256': sha(preview_path),
              'renderer_metadata_records_validated': len(records),
              'validated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    (ROOT / '.local/checkpoint-preview-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
