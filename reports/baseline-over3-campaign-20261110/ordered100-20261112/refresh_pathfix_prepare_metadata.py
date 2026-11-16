"""CPU-only source metadata revision for proven symlink identity fix, pending cases only."""
import pathlib,json,hashlib,datetime,sys
P=pathlib.Path(__file__).resolve().parent
allowed=set(json.loads(sys.argv[1]));proof=json.loads((P/'SGL_GEMV_PATH_BINDING_REPAIR.json').read_text())
sourcefile=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/row_gemv.py'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
cache={};done=[];rejected=[]
for i in sorted(allowed):
 f=P/'prepared-rollout'/str(i)/'prepare.json'
 if not f.exists():continue
 old=f.read_bytes();m=json.loads(old);diff=[]
 for name,value in m['sources'].items():
  if name not in cache:cache[name]=sha(name)
  if value!=cache[name]:diff.append(name)
 if not diff:continue
 if diff!=[str(sourcefile)] or m['sources'][str(sourcefile)]!=proof['original_sha256'] or cache[str(sourcefile)]!=proof['proposed_sha256']:
  rejected.append({'ordinal':i,'changed':diff});continue
 assert sha(f.parent/'payload.pt')==m['payload_sha256'] and sha(f.parent/'case.pkl')==m['descriptor_sha256']
 backup=f.with_name('prepare.before-pathfix-metadata.json');assert not backup.exists();backup.write_bytes(old)
 m['sources'][str(sourcefile)]=proof['proposed_sha256']
 tmp=f.with_name('prepare.new');tmp.write_text(json.dumps(m,indent=2));tmp.replace(f)
 done.append({'ordinal':i,'old_prepare_sha256':hashlib.sha256(old).hexdigest(),'new_prepare_sha256':sha(f),'payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256']})
out={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revised':done,'rejected':rejected,'only_source_path_check_changed':True,'input_descriptor_checker_kernel_unchanged':True}
(P/'PATHFIX_PREPARE_REVISION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
