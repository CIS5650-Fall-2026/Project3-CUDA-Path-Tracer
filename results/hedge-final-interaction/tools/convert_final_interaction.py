from pathlib import Path
import json,subprocess,shutil,hashlib
root=Path.cwd();result=root/'results/hedge-final-interaction';capture=result/'capture'
s=json.loads((capture/'summary.json').read_text());assert s['passed'] and s['status']=='succeeded'
movie=capture/'interaction.mp4';out=root/'img/interaction.mp4';gif=root/'img/interaction.gif'
assert not out.exists() and not gif.exists()
shutil.copyfile(movie,out);assert hashlib.sha256(movie.read_bytes()).digest()==hashlib.sha256(out.read_bytes()).digest()
ffmpeg='/opt/homebrew/bin/ffmpeg'
cmd=[ffmpeg,'-v','error','-i',str(movie),'-filter_complex','fps=4,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3','-loop','0',str(gif)]
r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
(result/'gif-conversion.stdout.log').write_text(r.stdout);(result/'gif-conversion.stderr.log').write_text(r.stderr)
ffprobe='/opt/homebrew/bin/ffprobe';probe_cmd=[ffprobe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(gif)];q=subprocess.run(probe_cmd,capture_output=True,text=True);assert q.returncode==0,q.stderr
(result/'gif-ffprobe.json').write_text(q.stdout)
d=json.loads(q.stdout);duration=float(d['format']['duration'])
frames=result/'inspected-frames';frames.mkdir(exist_ok=False)
for t in (2,12,16,23,30,36,44.5,46.5):
 argv=[ffmpeg,'-v','error','-ss',str(t),'-i',str(movie),'-frames:v','1',str(frames/('frame-'+str(t)+'.png'))];subprocess.run(argv,check=True)
receipt={'ffmpeg_command':cmd,'cwd':str(root),'gif':{'path':'img/interaction.gif','duration_seconds':duration,'fps':4},'mp4':{'path':'img/interaction.mp4'},'probe_command':probe_cmd,'frame_files':[str(p.relative_to(root)) for p in sorted(frames.glob('*.png'))]}
(root/'.local/interaction-conversion-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'actual_gif_seconds':duration,'frames_to_inspect':receipt['frame_files']}))
