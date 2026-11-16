"""CPU-only exact pin+publishedpatch source deployment. No build/install/device access."""
import pathlib,json,subprocess,hashlib,tarfile,io,os,shlex,datetime,stat
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parents[2];repo=ROOT/'ascend_baselines/ops-nn';audit=P.parent/'full-audit-20261112';expected=next(x for x in json.loads((audit/'EXPECTED_DEPENDENCIES.json').read_text()) if x['name']=='ops-nn');assert expected['patch_status']=='applied'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected['commit']
patch=ROOT/'exports/tileops-ascend-baselines'/expected['patch'];assert patch.is_file()
S=P/'native-source-opsnn-v1';S.mkdir(exist_ok=True);bundle=S/'source-bundle.tar.gz';assert not bundle.exists(),'existing source bundle not overwritten'
manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'library':'ops-nn','pin':expected['commit'],'published_patch':str(patch),'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'expected_source':'full-audit-20261112/EXPECTED_DEPENDENCIES.json','files':{},'kernel_source_edited_by_this_task':False,'build_executed':False}
with tarfile.open(bundle,'w:gz',compresslevel=1) as out:
 for name,e in expected['files'].items():
  f=repo/name;assert f.exists() or f.is_symlink(),name
  mode=e['mode'];data=os.readlink(f).encode() if mode=='120000' else f.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==e['git_blob'],'source differs from pin+declaredpatch: '+name
  digest=hashlib.sha256(data).hexdigest();manifest['files'][name]={'sha256':digest,'mode':mode,'bytes':len(data),'git_blob':blob}
  info=tarfile.TarInfo('source/'+name);info.mtime=int(f.lstat().st_mtime);info.mode=0o755 if mode=='100755' else 0o644
  if mode=='120000':info.type=tarfile.SYMTYPE;info.linkname=data.decode();out.addfile(info)
  else:info.size=len(data);out.addfile(info,io.BytesIO(data))
 body=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('SOURCE_MANIFEST.json');info.size=len(body);out.addfile(info,io.BytesIO(body))
(S/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
h=hashlib.sha256()
with bundle.open('rb') as inp:
 for b in iter(lambda:inp.read(8388608),b''):h.update(b)
digest=h.hexdigest();print('EXACT_SOURCE_FILES',len(manifest['files']),'BUNDLE_BYTES',bundle.stat().st_size,'SHA',digest,flush=True)
remote='/opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1';code='''import pathlib,sys,hashlib,tarfile,json,os,uuid
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);p=root/'source-bundle.tar.gz';digest=sys.argv[2]
if p.exists():
 assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==digest
else:
 with p.open('xb') as out:
  while True:
   b=sys.stdin.buffer.read(8388608)
   if not b:break
   out.write(b)
 assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==digest
source=root/'source'
if source.exists():
 proof=json.loads((root/'COPY_VERIFIED.json').read_text());assert proof['bundle_sha256']==digest;print(json.dumps(proof));raise SystemExit
stage=root/('staging-'+uuid.uuid4().hex);stage.mkdir()
with tarfile.open(p) as t:t.extractall(stage,filter='data')
m=json.loads((stage/'SOURCE_MANIFEST.json').read_text());verified=0
for name,e in m['files'].items():
 f=stage/'source'/name;data=os.readlink(f).encode() if e['mode']=='120000' else f.read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256'],name;verified+=1
(stage/'source').rename(source);(stage/'SOURCE_MANIFEST.json').rename(root/'SOURCE_MANIFEST.json');stage.rmdir()
proof={'bundle_sha256':digest,'pin':m['pin'],'patch_sha256':m['patch_sha256'],'verified_files':verified,'all_pin_patch_git_blobs_and_destination_SHA_match':True,'build_executed':False,'NPU_or_shared_runtime_modified':False};(root/'COPY_VERIFIED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
cmd='python3 -c '+shlex.quote(code)+' '+shlex.quote(remote)+' '+digest
with bundle.open('rb') as inp:r=subprocess.run(['ssh','-o','ConnectTimeout=10','openlibing-DevEnv-661495',cmd],stdin=inp,capture_output=True,text=True,timeout=180)
assert r.returncode==0,r.stderr;proof=json.loads(r.stdout.splitlines()[-1]);proof.update({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_source':remote+'/source','local_manifest':str(S/'SOURCE_MANIFEST.json'),'build_entrypoint_review':'Do not execute unreviewed build.sh: install_deps.sh can apt/sudo/update-alternatives shared compilers. Review direct private CMake equivalent first.','native_binary_acceptance':False});(P/'OPS_NN_PINNED_SOURCE_DEPLOYMENT.json').write_text(json.dumps(proof,indent=2)+'\n');print('CPU_SOURCE_DEPLOYED',proof['verified_files'],flush=True)
