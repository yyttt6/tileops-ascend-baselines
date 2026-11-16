import pathlib,sys,subprocess,time,json
P=pathlib.Path(__file__).resolve().parent
for family,label,ids in [('checktemplate','canonical-rope-rest',[42,43,44,45,46,47]),('checktemplate','canonical-fp8-edges',[5,6,7,8,9]),('checkcatlass','canonical-catlass',[95,96,97,98,99])]:
 while True:
  rc=subprocess.run([sys.executable,'-u',str(P/'run_supplement.py'),family,label,*map(str,ids)]).returncode
  if rc==75:time.sleep(30);continue
  if rc:raise RuntimeError(label)
  break
(P/'AFFECTED_CANONICAL_DONE.json').write_text(json.dumps({'status':'required differential original-checker checks complete','performance_windows_added':0},indent=2))
