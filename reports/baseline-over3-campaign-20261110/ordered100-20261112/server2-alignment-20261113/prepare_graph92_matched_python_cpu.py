from pathlib import Path
import zipfile,json,hashlib,datetime
base=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');src=base/'graph92-tbe-tik-extracted-v1/share/info/tbe-tik/python';dst=base/'graph92-python-v1/site-packages';assert not dst.exists();dst.mkdir(parents=True);sha=lambda b:hashlib.sha256(b).hexdigest();wheels={};files={}
for p in sorted(src.glob('*.whl')):
 wheels[str(p)]=sha(p.read_bytes())
 with zipfile.ZipFile(p) as z:
  for name in z.namelist():
   q=Path(name);assert not q.is_absolute() and '..' not in q.parts
   if name.endswith('/'):continue
   raw=z.read(name);out=dst/q;out.parent.mkdir(parents=True,exist_ok=True)
   if out.exists():assert out.read_bytes()==raw,('collision',name)
   else:out.write_bytes(raw)
   files[name]={'sha256':sha(raw),'bytes':len(raw)}
assert (dst/'te_fusion/parallel_compilation.py').is_file() and (dst/'tbe/common/buildcfg/buildcfg.py').is_file();v={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'private_python':str(dst),'wheels':wheels,'files':files,'no_install_scripts_or_imports_executed':True,'reason':'Actual traceback used system9.1 te_fusion against9.2 libtvm; select matching original9.2 Python before all imports in scoped diagnostic','runtime_acceptance':False};(dst.parent/'PROVENANCE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
