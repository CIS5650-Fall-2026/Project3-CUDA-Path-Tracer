#!/usr/bin/env python3
"""Validate downloaded genuine preview-suite evidence without running a renderer."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT = Path.cwd()
REMOTE_ROOT = Path('/home/ubuntu/cis5650/project3')
SOURCE = '07233055f7de64aac672f3b5f173066827eac89b27c3fc12f4b7f35982847703'


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
    preflight = load(ROOT / '.local/submission-preview-preflight.json')
    need(preflight['preflight_passed'] is True and preflight['renderer_source_sha256'] == SOURCE, 'Preflight source identity differs')
    expected_binaries = preflight['binary_sha256']
    suite_path = ROOT / 'results/submission-final-suite-preview/summary.json'
    preview_path = ROOT / 'results/submission-final-preview/summary.json'
    suite, preview = load(suite_path), load(preview_path)
    need((suite.get('schema_version'), suite.get('phase'), suite.get('prefix'), suite.get('status'), suite.get('exit_code'))
         == (1, 'preview', 'submission-final', 'passed', 0) and suite.get('errors') == [], 'Preview suite did not pass its actual phase')
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
        need(relative.is_relative_to(Path('results/submission-final-preview')), 'Unexpected preview metadata location')
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
    need(preview['compatibility'] == load(ROOT / 'results/submission-final-preview/smoke/headless.json')['compatibility'], 'Preview smoke compatibility differs')
    report = {'scope': 'Downloaded actual preview suite only; no new renderer execution or media publication',
              'passed': True, 'source_sha256': SOURCE, 'binary_sha256': expected_binaries,
              'suite_summary_sha256': sha(suite_path), 'preview_summary_sha256': sha(preview_path),
              'renderer_metadata_records_validated': len(records),
              'validated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    (ROOT / '.local/submission-preview-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
