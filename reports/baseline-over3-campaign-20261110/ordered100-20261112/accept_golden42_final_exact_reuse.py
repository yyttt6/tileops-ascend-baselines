"""Exact corrected-reference complete API reuse; no new native samples or pair claims."""
from pathlib import Path
import json,hashlib,datetime,fcntl,subprocess,sys
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
source_id=int(sys.argv[1]);targets=list(map(int,sys.argv[2:]));assert (source_id,tuple(targets)) in [(777,(309,)),(184,(372,804,1082,1255))]
source_file=D/('OPS_TRANSFORMER_INSTANCE_API_CURRENT.json' if source_id==777 else 'FLA_TAP_API_CURRENT.json')
qualified=next(r for r in json.loads(source_file.read_text())['rows'] if r['ordinal']==source_id)
assert qualified['full_checker_15raw_guard_valid']
cp=D/qualified['checkpoint'];assert sha(cp)==qualified['checkpoint_sha256'];e=json.loads(cp.read_text());assert e['guard']['performance_valid'] and e['guard']['workload_exit_code']==0 and e['guard']['phase']=='finished'
r=next(x for x in e['batch']['results'] if x['ordinal']==source_id);assert r['reference_policy']=='audited_cpu_fp64_exact42_v1' and r['status']=='raw_checked_pending_guard' and r['inputs_preserved_by_original_verifier'] and len(r['timing']['retained_profiles'])==15
base=D/'remote/runs'/e['tag']/str(source_id);correction=json.loads((base/'GOLDEN_CORRECTION.json').read_text());assert correction['manifest_sha256']==sha(D/'GOLDEN42_CERTIFICATES.json')
cert={x['ordinal']:x for x in json.loads((D/'GOLDEN42_CERTIFICATES.json').read_text())['rows']};s=cert[source_id];assert correction['certificate']==s and s['payload_sha256']==r['payload_sha256']
rows=[]
for i in targets:
 t=cert[i];assert t['key'][1:]==s['key'][1:]
 for k in ['params','payload_sha256','descriptor_sha256','evidence','resolved_original_tolerance','corrected_golden_sha256']:
  assert t[k]==s[k],(i,k)
 assert t['evidence']['kind']=='exact' and not t['evidence']['controls']
 meta=json.loads((D/'remote/prepared-rollout'/str(i)/'prepare.json').read_text())
 for k in ['key','params','payload_sha256','descriptor_sha256']:assert meta[k]==t[k]
 rows.append({'ordinal':i,'key':t['key'],'source_ordinal':source_id,'source_key':s['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':t['payload_sha256'],'descriptor_sha256':t['descriptor_sha256'],'corrected_golden_sha256':t['corrected_golden_sha256'],'source_qualified_row':qualified,'same_frozen_params_inputs_descriptor_Exact_checker_and_corrected_golden':True,'full_checker_15raw_guard_valid':True,'coverage_kind':'Audited complete original '+s['key'][0]+' API alternative; exact qualified measurement reuse','new_native_runs':0,'new_pairs':0,'new_final_closures':0})
name='RESOLVED_GOLDEN42_'+('INSTANCE309_EXACT_API_REUSE' if source_id==777 else 'SSD4_EXACT_API_REUSE')+'.json'
with (D.parents[2]/'.agent_memory/.maintenance.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not (D/name).exists(),'Preserve existing accepted reuse'
 backup=D/'action-backups'/('golden42-final-reuse-'+str(source_id)+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));backup.mkdir()
 p=D/'effective_coverage.py';raw=p.read_bytes();(backup/p.name).write_bytes(raw);(backup/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(raw).hexdigest()}))
 text=raw.decode();needle="'RESOLVED_GOLDEN42_MATH278_EXACT_API_REUSE.json'";assert needle in text;text=text.replace(needle,repr(name)+','+needle,1);compile(text,str(p),'exec')
 out={'checked_at':now(),'rows':rows,'not_published':True,'new_device_samples':0,'source_qualification_sha256':sha(source_file),'golden_certificate_sha256':sha(D/'GOLDEN42_CERTIFICATES.json')}
 (D/name).write_text(json.dumps(out,indent=2)+'\n');assert p.read_bytes()==raw;tmp=p.with_suffix('.tmp');tmp.write_text(text);tmp.replace(p)
 subprocess.run([sys.executable,str(D/'effective_coverage.py')],check=True)
print(json.dumps({'source':source_id,'qualified_reuse':targets}))
