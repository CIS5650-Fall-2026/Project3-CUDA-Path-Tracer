import hashlib,pathlib,subprocess,time,json,datetime
ROOT=pathlib.Path('/home/ubuntu/cis5650/project3')
EXPECTED='3cb36a84ca94495bb36c54435931d74bfe12ae9d6c1f4891539d5cc0846890ed'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=pathlib.Path('/etc/X11/xorg.conf');backup=pathlib.Path('/etc/X11/xorg.conf.project3-video-backup')
assert sha(original)==sha(backup)==EXPECTED,'Original Xorg config mismatch'
process=subprocess.run(['ps','-C','cis565_path_tracer_headless','--no-headers','-o','pid'],capture_output=True,text=True)
assert not process.stdout.strip(),'Headless renderer still active'
text=original.read_text();assert text.count('PCI:49:0:0')==1
text=text.replace('PCI:49:0:0','PCI:0:30:0').replace('NVIDIA L4','NVIDIA A10G')
subprocess.run(['sudo','tee',str(original)],input=text,text=True,stdout=subprocess.DEVNULL,check=True)
subprocess.run(['sudo','systemctl','start','gdm3'],check=True)
wrapper='/home/ubuntu/cis5650/with-display'
ready=False
for attempt in range(30):
 process=subprocess.run([wrapper,'true'],capture_output=True,text=True)
 if process.returncode==0: ready=True;break
 time.sleep(1)
assert ready,'Display not ready'
xrandr=subprocess.check_output([wrapper,'xrandr'],text=True)
assert 'DVI-D-0 connected' in xrandr and '1920x1080' in xrandr
subprocess.run([wrapper,'xrandr','--output','DVI-D-0','--mode','1920x1080'],check=True)
glx=subprocess.check_output([wrapper,'glxinfo','-B'],text=True)
assert 'A10G' in glx and 'direct rendering: Yes' in glx
out=ROOT/'.local/final-display';out.mkdir(parents=True,exist_ok=True)
(out/'glxinfo.txt').write_text(glx)
(out/'xrandr.txt').write_text(subprocess.check_output([wrapper,'xrandr'],text=True))
display=subprocess.check_output([wrapper,'python3','-c','import os;print(os.environ["DISPLAY"])'],text=True).strip()
subprocess.run([wrapper,'ffmpeg','-y','-v','error','-f','x11grab','-i',display,'-frames:v','1',str(out/'desktop.png')],check=True)
(out/'setup.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_xorg_sha256':EXPECTED,'active_xorg_sha256':sha(original),'hardware':'NVIDIA A10G','resolution':'1920x1080'},indent=2)+'\n')
print('NVIDIA A10G desktop ready at1920x1080; private snapshot retained')
