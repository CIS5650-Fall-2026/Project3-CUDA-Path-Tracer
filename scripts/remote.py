#!/usr/bin/env python3
"""Project 3 transfers/commands using the existing private workstation metadata.

The AWS account remains local. No credential or connection file is uploaded.
Start/stop the workstation with the existing Project 1 session helper.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/ubuntu/cis5650/project3'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(os.environ.get(
        'PROJECT3_AWS_CONFIG', ROOT.parent / 'project_1/.local/aws/connection.json')))
    parser.add_argument('--instance-type', default='g6.xlarge', choices=['g6.xlarge', 'g5.xlarge'])
    parser.add_argument('action', choices=['upload', 'download', 'run'])
    parser.add_argument('arguments', nargs='*')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    info = json.loads(subprocess.check_output([
        'aws', 'ec2', 'describe-instances', '--instance-ids', config['instance_id'],
        '--region', config['region'], '--output', 'json'], text=True))
    instance = info['Reservations'][0]['Instances'][0]
    tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
    if instance['InstanceType'] != args.instance_type or tags.get('Project') != 'CIS5650-Project1':
        raise RuntimeError('Unexpected workstation identity')
    if instance['State']['Name'] != 'running':
        raise RuntimeError('The existing workstation must be running')
    host = instance['PublicIpAddress']
    known = args.config.parent / 'known_hosts'
    # Require continuity with a previously trusted workstation host key even
    # when AWS assigns a different public address after a stop/start.
    trusted = {tuple(line.split()[1:3]) for line in known.read_text().splitlines()
               if line and not line.startswith('#') and len(line.split()) >= 3}
    scan = subprocess.run(['ssh-keyscan', '-T', '10', '-t', 'ed25519', host],
                          capture_output=True, text=True, check=False)
    observed = [line for line in scan.stdout.splitlines() if line and not line.startswith('#')]
    if not observed or any(tuple(line.split()[1:3]) not in trusted for line in observed):
        raise RuntimeError('SSH host key does not match the trusted workstation')
    ssh = ['ssh', '-i', config['ssh_key_path'], '-o', 'IdentitiesOnly=yes',
           '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',
           '-o', 'UserKnownHostsFile=' + str(known)]
    destination = 'ubuntu@' + host
    if args.action == 'run':
        if len(args.arguments) != 1:
            parser.error('run requires one quoted remote command')
        command = "export PATH=/usr/local/cuda-12.8/bin:$PATH; cd " + shlex.quote(REMOTE) + ' && ' + args.arguments[0]
        return subprocess.call([*ssh, destination, command])
    if args.action == 'upload':
        subprocess.run([*ssh, destination, 'mkdir -p ' + shlex.quote(REMOTE)], check=True)
        return subprocess.call(['rsync', '-az', '-e', shlex.join(ssh),
            '--exclude=.git/', '--exclude=.local/', '--exclude=build*/',
            '--exclude=results/', '--exclude=__pycache__/',
            str(ROOT) + '/', destination + ':' + REMOTE + '/'])
    if len(args.arguments) != 2:
        parser.error('download requires a relative remote path and local destination')
    relative, local = args.arguments
    if relative.startswith('/') or '..' in Path(relative).parts:
        parser.error('download path must remain inside the Project 3 directory')
    Path(local).mkdir(parents=True, exist_ok=True)
    return subprocess.call(['rsync', '-az', '-e', shlex.join(ssh),
        destination + ':' + REMOTE + '/' + relative, local])

if __name__ == '__main__':
    raise SystemExit(main())
