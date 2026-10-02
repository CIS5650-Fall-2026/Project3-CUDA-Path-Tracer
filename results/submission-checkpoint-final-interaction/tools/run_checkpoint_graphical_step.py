#!/usr/bin/env python3
"""Execute one authorized graphical step, preserving real command receipts.

This private driver checks the existing headless completion gate again remotely
before each step. It never provisions resources, alters deadline units, converts
media, or publishes a final success report.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys

SOURCE = None
REMOTE_ROOT = '/home/ubuntu/cis5650/project3'
WRAPPER = '/home/ubuntu/cis5650/with-display'
UNITS = {
    '/etc/systemd/system/cis5650-autostop.timer': '8e778aeda0a4d573f6cb0a888faa765889161f2d42a71c043d96f082e2f4077e',
    '/etc/systemd/system/cis5650-deadline.service': '83759763bfbe69915c05ce7147639bf1359db0254d2b37823bcc0217f0ec8a4f',
    '/usr/local/sbin/cis5650-refresh-deadline': '73782530a355c5159993e7d8030190beab4dc15ca83426d2e93d149b38acbc2d',
}
HELPERS = {
    '.local/setup_final_display_remote.py': '747ed64117703f644649645c4c55088b84c2cbcbef63c07a617f5ac8dfd1904f',
    '.local/run_final_interaction_remote.py': 'e1b709e6a756c403dd6c9e0c39e7e76007bb33f1af31b427479adf5887ddd1ee',
}
SCRIPTS = {
    'scripts/record_interaction_video.py': '6b3d74194e2b2916eb204cf5aec5a1e9bda4bcf5c60084f5db846662b983f395',
    'scripts/source_hash.py': 'dfd60869e393d6b49d4bbcbf9091d67e4fb569a38b50fb478dfd74cf5b10a295',
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def remote_code(step, command):
    constants = repr({'step': step, 'command': command, 'source': SOURCE,
                      'root': REMOTE_ROOT, 'units': UNITS, 'helpers': HELPERS,
                      'scripts': SCRIPTS})
    return '''from pathlib import Path
import datetime as dt,hashlib,json,os,subprocess,sys
settings = ''' + constants + '''
root=Path(settings['root']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((root/'results/submission-checkpoint-final-controller/summary.json').read_text())
expected=['build','new-validation','sanitizers','gltf','baseline-validation','baseline-measurements','new-measurements','gallery']
assert summary.get('status')=='completed' and summary.get('errors')==[], 'Headless controller is incomplete or failed'
assert summary.get('expected_renderer_source_sha256')==settings['source'], 'Headless controller source identity differs'
assert [p.get('name') for p in summary['phases']]==expected, 'Expected all eight headless phases'
for phase in summary['phases']:
 allowed=(phase.get('status')=='passed' and phase.get('returncode')==0) or (phase['name']=='new-measurements' and phase.get('status')=='completed_not_qualified' and phase.get('returncode')==2)
 assert allowed and sha(root/phase['summary'])==phase.get('summary_sha256'), 'Phase approval or summary hash differs'
active=[]
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:
   binary=Path(os.readlink(proc/'exe')).name
   if binary in ('cis565_path_tracer_headless','cis565_path_tracer'):active.append({'pid':int(proc.name),'binary':binary})
  except OSError:pass
assert not active, 'A renderer is still active'
assert subprocess.check_output(['python3','scripts/source_hash.py'],cwd=root,text=True).strip()==settings['source'], 'Frozen renderer source changed'
build_summary=json.loads((root/summary['phases'][0]['summary']).read_text())
binary_paths={'headless':root/'build-submission/bin/cis565_path_tracer_headless','preview':root/'build-submission/bin/cis565_path_tracer'}
binaries_before={name:sha(path) for name,path in binary_paths.items()}
assert binaries_before==build_summary['binary_sha256_end'], 'Frozen build binary identities changed'
units_before={name:sha(Path(name)) for name in settings['units']}
assert units_before==settings['units'], 'Deadline unit bytes changed'
assert subprocess.run(['systemctl','is-active','--quiet','cis5650-autostop.timer']).returncode==0, 'Authorized deadline timer is inactive'
deadline=subprocess.check_output(['systemctl','show','cis5650-autostop.timer','--property=NextElapseUSecRealtime','--value'],text=True).strip()
assert '2026-10-02' in deadline and '04:38:03' in deadline, 'Authorized session deadline differs'
for name,digest in settings['helpers'].items():assert sha(root/name)==digest, 'Transferred helper source differs'
for name,digest in settings['scripts'].items():assert sha(root/name)==digest, 'Approved recording or hashing script differs'
if settings['step']=='setup':
 assert subprocess.run(['systemctl','is-active','--quiet','gdm3']).returncode!=0, 'Display setup was already applied'
if settings['step']=='preview':
 assert not (root/'results/submission-checkpoint-final-suite-preview').exists() and not (root/'results/submission-checkpoint-final-preview').exists(), 'Preview output is not fresh'
if settings['step'] in ('display-reset','capture'):
 preview_suite=json.loads((root/'results/submission-checkpoint-final-suite-preview/summary.json').read_text())
 preview=json.loads((root/'results/submission-checkpoint-final-preview/summary.json').read_text())
 assert preview_suite.get('status')=='passed' and preview_suite.get('exit_code')==0 and preview_suite.get('errors')==[], 'Preview suite did not pass'
 assert preview_suite.get('source_input_sha256_start')==preview_suite.get('source_input_sha256_end')==settings['source'], 'Preview suite belongs to another source'
 assert preview_suite.get('binary_sha256_start')==preview_suite.get('binary_sha256_end')==binaries_before, 'Preview suite binary identities differ'
 assert preview.get('passed') is True and preview.get('preview_headless_png_exact') is True and preview.get('space_restore_png_exact') is True, 'Actual preview parity checks did not pass'
if settings['step']=='capture':
 result=root/'results/submission-checkpoint-final-interaction'
 assert not any((result/name).exists() for name in ('capture','capture-run.json','capture-provenance-before.json','post-space-reference.png','post-space-reference.json','post-space-reference.sum.bin')), 'Interaction evidence output is not fresh'
 clock=subprocess.check_output(['/home/ubuntu/cis5650/with-display','xrandr','--current'],text=True)
 assert any(line.startswith('DVI-D-0 connected') and '1920x1080+0+0' in line for line in clock.splitlines()), 'Capture display is not1920x1080'
 glx=subprocess.check_output(['/home/ubuntu/cis5650/with-display','glxinfo','-B'],text=True)
 assert 'A10G' in glx and 'direct rendering: Yes' in glx, 'Expected direct NVIDIA A10G rendering'
receipt={'schema_version':1,'step':settings['step'],'argv':settings['command'],'cwd':str(root),'shell':False,'source_sha256':settings['source'],'binary_sha256_before':binaries_before,'headless_controller_sha256':sha(root/'results/submission-checkpoint-final-controller/summary.json'),'deadline_units_before':units_before,'timer_deadline':deadline,'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
out=root/'.local/checkpoint-graphical-logs';out.mkdir(parents=True,exist_ok=True)
receipt_file=out/(settings['step']+'.command.json');assert not receipt_file.exists(),'Step was already executed'
stdout=out/(settings['step']+'.stdout.log');stderr=out/(settings['step']+'.stderr.log')
receipt_file.write_text(json.dumps(receipt,indent=2)+'\\n')
with stdout.open('x') as so,stderr.open('x') as se:
 process=subprocess.run(settings['command'],cwd=root,stdin=subprocess.DEVNULL,stdout=so,stderr=se,check=False,shell=False)
receipt.update({'exit_code':process.returncode,'ended_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'deadline_units_after':{name:sha(Path(name)) for name in settings['units']},'binary_sha256_after':{name:sha(path) for name,path in binary_paths.items()},'stdout_log':str(stdout.relative_to(root)),'stderr_log':str(stderr.relative_to(root))})
receipt['deadline_units_unchanged']=receipt['deadline_units_after']==units_before
receipt_file.write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
assert receipt['deadline_units_unchanged'],'Deadline unit bytes changed during graphical work'
assert receipt['binary_sha256_after']==binaries_before,'Renderer binary bytes changed during graphical work'
sys.exit(process.returncode)
'''


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('step', choices=('setup', 'preview', 'display-reset', 'capture'))
    args = parser.parse_args()
    SOURCE = args.source
    assert len(SOURCE) == 64 and set(SOURCE) <= set("0123456789abcdef")
    root = Path.cwd()
    config = Path(os.environ.get('PROJECT3_AWS_CONFIG', root.parent / 'project_1/.local/aws/connection.json'))
    assert json.loads(config.read_text())['instance_id'] == 'i-0b8cecd23bc0a3a6b', 'Unexpected configured instance'
    commands = {
        'setup': ['python3', '.local/setup_final_display_remote.py'],
        'preview': [WRAPPER, 'python3', 'scripts/run_hedge_suite.py', '--phase', 'preview', '--build', 'build-submission', '--prefix', 'submission-checkpoint-final'],
        'display-reset': [WRAPPER, 'python3', '-c', "import subprocess;subprocess.run(['xrandr','--output','DVI-D-0','--mode','1920x1080'],check=True);x=subprocess.check_output(['xrandr','--current'],text=True);g=subprocess.check_output(['glxinfo','-B'],text=True);assert any(s.startswith('DVI-D-0 connected') and '1920x1080+0+0' in s for s in x.splitlines());assert 'A10G' in g and 'direct rendering: Yes' in g;print(x);print(g)"],
        'capture': [WRAPPER, 'python3', '.local/run_final_interaction_remote.py', '--root', REMOTE_ROOT, '--expected-source-sha256', SOURCE, '--prefix', 'submission-checkpoint-final', '--build-dir', 'build-submission', '--scene', 'scenes/gltf-showcase.json'],
    }
    code = remote_code(args.step, commands[args.step])
    remote_command = 'python3 -c ' + shlex.quote(code)
    argv = [sys.executable, 'scripts/remote.py', '--instance-type', 'g5.xlarge', 'run', remote_command]
    logdir = root / '.local/checkpoint-graphical-logs'
    logdir.mkdir(exist_ok=True)
    receipt_path = logdir / (args.step + '.local-command.json')
    assert not receipt_path.exists(), 'Step receipt already exists'
    receipt = {'step': args.step, 'local_driver': str(Path(__file__).relative_to(root)),
               'local_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'actual_remote_argv': commands[args.step], 'remote_command': remote_command,
               'remote_command_sha256': hashlib.sha256(remote_command.encode()).hexdigest(),
               'shell': False, 'started_at_utc': now(), 'status': 'running'}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with (logdir / (args.step + '.local.stdout.log')).open('x') as stdout, (logdir / (args.step + '.local.stderr.log')).open('x') as stderr:
        process = subprocess.run(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False, shell=False)
    receipt.update({'status': 'succeeded' if process.returncode == 0 else 'failed',
                    'exit_code': process.returncode, 'ended_at_utc': now()})
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('step', 'status', 'exit_code', 'ended_at_utc')}))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
