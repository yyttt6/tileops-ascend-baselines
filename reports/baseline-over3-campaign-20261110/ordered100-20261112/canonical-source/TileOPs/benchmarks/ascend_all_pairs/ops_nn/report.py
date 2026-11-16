"""CPU-only incremental coverage and source inventory. Never turns a static match into a pass."""
import collections
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3];WS=ROOT.parent
sys.path.insert(0,str(ROOT))
from benchmarks.ascend_all_pairs.ops_nn.catalog import ROUTES,LEGACY,BOOLEAN_NAMES,FOREACH
from benchmarks.ascend_all_pairs.ops_nn.staging import CANDIDATES
from benchmarks.ascend_all_pairs.ops_nn.extension_routes import ROUTES as EXTRA_ROUTES,APIS as EXTRA_APIS
from benchmarks.ascend_all_pairs.ops_nn.preserved_runtime import accepted_records
ROUTES=ROUTES|EXTRA_ROUTES
HERE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'
VENDOR=WS/'ascend_baselines/ops-nn'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def known_case_restriction(name,c):
    fields=c.get('tensors',{});p=c['params']
    inputs={k:v for k,v in fields.items() if k!='output'}
    if name in FOREACH and any(v[1]=='int64' for v in inputs.values()):
        return 'Original Foreach route has no int64 support; narrowing arbitrary int64 inputs is not lossless.'
    if name=='DivFwdOp' and p.get('rounding_mode') is not None:
        return 'Current original ForeachDivList route implements true division only; floor/trunc contract requires another mapping.'
    if name in ('LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp') and any(v[1] in ('complex128','float64') for v in inputs.values()):
        return 'Original p=0 route has no FP64 lane; no exact FP64/complex128 truth specialization is implemented.'
    if name in ('CountNonzeroFwdOp','VarFwdOp','StdFwdOp','VarMeanFwdOp'):
        shape=fields['x'][0];dims=p.get('dim');dims=range(len(shape)) if dims is None else (dims,) if isinstance(dims,int) else dims
        width=math.prod(shape[d] for d in dims)
        if name=='CountNonzeroFwdOp' and width>2**24:
            return 'Original FP32 p=0 count cannot guarantee exact integer output above 2^24; another exact-count route is required.'
        if name!='CountNonzeroFwdOp' and width<=int(p.get('correction',1)):
            return 'Current original LayerNorm variance auxiliary route excludes nonpositive variance divisors and their nonfinite contract.'
    return None
def main():
    inventory=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json'
    if not (HERE/'inventory_snapshot.json').exists():(HERE/'inventory_snapshot.json').write_bytes(inventory.read_bytes())
    inv=json.loads((HERE/'inventory_snapshot.json').read_text())
    if not (HERE/'source_index.json').exists():
        index=[]
        for path in sorted(VENDOR.rglob('*_def.cpp')):
            if any(x in ('build','.git','third_party') for x in path.relative_to(VENDOR).parts):continue
            text=path.read_text();names=re.findall(r'OP_ADD\((\w+)\)',text)
            if not names:continue
            for name in names:index.append(dict(api=name,file=str(path.relative_to(WS)),line=next((i for i,s in enumerate(text.splitlines(),1) if 'class '+name in s),1),sha256=digest(path),ascend950_registered='ascend950' in text))
        save(HERE/'source_index.json',index)
    source_index=json.loads((HERE/'source_index.json').read_text())
    old=json.loads((ROOT/'benchmarks/ascend/classic_matrix.json').read_text())
    historical={(r['op'],r['case_id']):r for r in old['rows'] if r['library']=='ops-nn'}
    missing_factories=set(json.loads((HERE/'factory_audit.json').read_text())['missing_exact_name_registration']) if (HERE/'factory_audit.json').exists() else set()
    decisions=json.loads((HERE/'semantic_decisions.json').read_text()) if (HERE/'semantic_decisions.json').exists() else {}
    receipts={}
    for path in sorted(HERE.rglob('*.json')):
        if path.name in ('coverage.json','inventory_snapshot.json','progress.json','source_index.json'):continue
        try:r=json.loads(path.read_text())
        except Exception:continue
        if isinstance(r,dict) and 'op' in r and 'case_id' in r and r.get('library')=='ops-nn':receipts.setdefault((r['op'],r['case_id']),[]).append((r,path))
    operators=[];rows=[]
    preserved=accepted_records()
    for op in inv['operators']:
        name=op['op'];route=ROUTES.get(name);legacy=LEGACY.get(name)
        directory=route[1] if route else legacy[2] if legacy else None
        src=[r for r in source_index if directory and ('/ops-nn/'+directory+'/') in r['file']]
        if name in EXTRA_APIS:
            src=[r for r in source_index if r['api'] in EXTRA_APIS[name]]
        if name=='ArgminFwdOp':
            src=[r for r in source_index if r['api'] in ('ForeachNeg','AdaptiveMaxPool3d')]
        if name in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp'):
            src=src+[r for r in source_index if r['api']=='GatherV2']
        decision=decisions.get(name,{})
        base=dict(op=name,library='ops-nn',category=op['category'],source=src,
                  adapter=('TileOPs/benchmarks/ascend_all_pairs/ops_nn/staging.py' if route or legacy else None),
                  semantic_review=decision or {'status':'equivalent_api_identified' if route or legacy else 'semantic_review_pending'})
        if name in CANDIDATES:
            f=CANDIDATES[name];adapter=Path(f.__code__.co_filename)
            base.update(adapter=str(adapter.relative_to(WS)),adapter_line=f.__code__.co_firstlineno,adapter_sha256=digest(adapter))
        own=[]
        for c in (c for c in inv['cases'] if c['op']==name):
            row={**base,'case_id':c['case_id'],'params':c.get('tensors',{})|c['params'],'device_executed':False,'numerical_receipt':None,'attribution_receipt':None,'registered':False}
            previous=historical.get((name,c['case_id']))
            if legacy and previous and previous['status']=='verified':
                row.update(status='verified_historical',reason='Preserved existing complete-manifest and ownership-accepted result; not counted as a new execution.',numerical_receipt=previous.get('device_receipt') or previous.get('receipt'),attribution_receipt=previous.get('root_review'),device_executed=True,registered=True)
            elif route:
                row.update(status='supported_unverified',reason='Source-equivalent adapter implemented; no accepted device receipt yet.')
            else:row.update(status=decision.get('status','semantic_review_pending'),reason=decision.get('reason','Current library source review not yet concluded.'))
            restriction=known_case_restriction(name,c) if route else None
            if restriction:row.update(status='adapter_case_unsupported',reason=restriction)
            if name in ('MLAKVCacheWriteFwdOp','PagedKVCacheWriteFwdOp'):
                fields=c.get('tensors',{})
                source=fields.get('kv_c') or fields.get('k');target=fields.get('kv_cache') or fields.get('k_pages')
                if c['params'].get('fuse_rope') or (source and target and source[1]!=target[1]):
                    row.update(status='adapter_case_unsupported',reason='Exact-copy native ScatterUpdate route excludes fused RoPE and FP8 storage quantization; full input/output contract needs another original API mapping.')
            trials=receipts.get((name,c['case_id']),[])
            if trials:
                r,path=max(trials,key=lambda rp:rp[0].get('finished_at',''))
                row['trials']=[str(p.relative_to(WS)) for _,p in trials];row.update(numerical_receipt=str(path.relative_to(WS)),device_executed=r.get('stage') not in ('runtime_import','build'))
                row['candidate_api']=r.get('api');row['candidate_adaptation']=r.get('adaptation')
                row['candidate_record_basis']='latest actual trial, not a claim about an untested changed adapter'
                row['candidate_source_hashes']={'receipt':str(path.relative_to(WS)),'field':'source_sha256'}
                trial_sources={k:v for k,v in r.get('source_sha256',{}).items() if k.startswith('TileOPs/benchmarks/ascend_all_pairs/ops_nn/') and k.endswith('.py')}
                row['changed_since_latest_trial']=[k for k,v in trial_sources.items() if not (WS/k).exists() or digest(WS/k)!=v]
                row['latest_trial_runtime_sources_current']=not row['changed_since_latest_trial'] if trial_sources else None
                row['native_registration_sources']=[s for s in source_index if re.search(r'\b'+re.escape(s['api'])+r'\b',str(r.get('api','')).replace('aclnnForeach','Foreach'))]
                if r['status']=='verified':
                    audit=path.with_suffix('.audit.json')
                    if audit.exists() and json.loads(audit.read_text()).get('accepted'):
                        row.update(status='verified',reason='Complete original manifest, input preservation and own native compilation/launch accepted.',attribution_receipt=str(audit.relative_to(WS)),registered=True)
                    elif audit.exists() and json.loads(audit.read_text()).get('missing_apis'):
                        row.update(status='numerical_pass_attribution_pending',reason='Original manifest passed, but own native compilation/launch is unproven; this result is not accepted even when the wrapper ELF is owned.')
                    elif audit.exists() and json.loads(audit.read_text()).get('changed_adapter_sources'):
                        row.update(status='needs_revalidation',reason='Earlier numerical/native acceptance preserved; adapter sources changed for a documented ABI repair and this current implementation needs a new receipt.')
                    else:row.update(status='numerical_pass_attribution_pending',reason='Original manifest passed; native ownership audit still required.')
                elif r['status']=='unsupported' or r.get('error_type')=='UnsupportedCase':row.update(status='adapter_case_unsupported',reason=r.get('reason') or r.get('error'))
                elif r.get('error_type')=='AssertionError':row.update(status='numerical_failed',reason=r.get('error'))
                else:
                    deployment=(r.get('stage') in ('build','runtime_import') or
                                r.get('error_type') in ('ImportError','ModuleNotFoundError','FileNotFoundError') or
                                any(code in str(r.get('error','')) for code in ('returned 145000','returned 500002','cannot open shared object','not registered in the selected ops-nn vendor')))
                    row.update(status='deployment_blocked' if deployment else 'device_failed',reason=r.get('error') or r.get('cleanup_error') or r.get('sync_error') or r['status'])
                if row['status']=='deployment_blocked' and name in ('GeluAndMulFwdOp','GeluTanhAndMulFwdOp'):
                    row['reason']+='; that tested GeluMul call exceeded original ShapeCheck limits last_dim<=1024 or batch<=200000. Paired-half chunk layout repair is implemented but awaits its own device receipt; this is not a mathematical rejection.'
                    row['diagnostic_evidence']='.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn/next_repairs.json'
                if row['status']=='deployment_blocked' and name in ('AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp'):
                    row['reason']+='; that original2D trial required rank4. The newer singleton-depth original3D route awaits its own receipt.'
                    row['diagnostic_evidence']='.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn/next_repairs.json'
                if row['status']=='device_failed' and name=='AdaptiveMaxPool2dFwdOp' and '507035' in str(r.get('sync_error','')):
                    row['reason']='Original AdaptiveMaxPool2d own kernel raised error354 VEC SIMT stack overflow, followed by507035 cleanup/synchronization failure. New original3D layout route is implemented but untested; no kernel or schedule changes.'
                    row['diagnostic_evidence']='.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn/repair003_guard.json'
            if name in missing_factories and not trials:
                row.update(status='missing_workload_factory',reason='No exact-name entry in current benchmarks/_cases registry. Source semantic review is retained separately; no numerical test fabricated.',device_executed=False,registered=False)
            pin=preserved.get((name,c['case_id']))
            if pin and row['status'] not in ('verified','verified_historical'):
                row['staging_status']=row['status'];row['staging_reason']=row['reason'];row['staging_adapter']=row['adapter']
                row['staging_candidate_api']=row.get('candidate_api');row['staging_numerical_receipt']=row.get('numerical_receipt')
                from benchmarks.ascend_all_pairs.ops_nn.pinned_smoke004.staging import CANDIDATES as PINNED
                f=PINNED[name];adapter=Path(f.__code__.co_filename)
                row.update(status='verified_preserved',reason='Complete device outputs/input/native proof retained for byte-identical relocated runtime; current staging status is recorded separately. This is not a new numerical run.',registered=True,
                           adapter=str(adapter.relative_to(WS)),adapter_line=f.__code__.co_firstlineno,adapter_sha256=digest(adapter),
                           numerical_receipt=pin['receipt'],attribution_receipt='.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn/pinned_smoke004_acceptance.json',
                           preserved_native_audit=pin['native_audit'],preserved_runtime=True,preserved_runtime_sources_match=True,
                           candidate_api=pin['api'],candidate_record_basis='Byte-preserved deployment of the original complete receipt; source relocation is explicit in the acceptance manifest.',
                           candidate_source_hashes={'receipt':pin['receipt'],'field':'source_sha256','relocation_manifest':'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn/pinned_smoke004_acceptance.json'})
                row['source']=[s for s in source_index if re.search(r'\b'+re.escape(s['api'])+r'\b',pin['api'].replace('aclnnForeach','Foreach'))]
            own.append(row);rows.append(row)
        counts=dict(collections.Counter(r['status'] for r in own))
        accepted_count=sum(counts.get(s,0) for s in ('verified','verified_historical','verified_preserved'))
        if name in missing_factories or not own:op_status='missing_workload_factory'
        elif accepted_count==len(own):op_status='verified_historical' if set(counts)=={'verified_historical'} else 'verified'
        elif accepted_count:op_status='partially_verified'
        elif name in CANDIDATES:
            op_status='implemented_with_failures' if any(s in counts for s in ('device_failed','deployment_blocked','numerical_failed','numerical_pass_attribution_pending')) else 'implemented_awaiting_validation'
        else:op_status=decision.get('status','semantic_review_pending')
        operators.append({**base,'case_count':len(own),'case_status_counts':counts,'status':'missing_workload_factory' if name in missing_factories or not own else 'classified','coverage_status':op_status,'accepted_case_count':accepted_count,'missing_workload_factory':name in missing_factories or not own})
    coverage=dict(schema='ops-nn-all-pairs/v1',status_definitions={'verified':'new full numerical receipt + input preservation + own native attribution','verified_historical':'unchanged historical full acceptance, linked explicitly','oracle_unavailable':'factory exists but independent reference contract is explicitly unavailable','needs_revalidation':'earlier receipt retained; adapter source changed since execution','adapter_case_unsupported':'implemented route does not cover this case contract; alternative mapping remains unresolved','supported_unverified':'implemented equivalent API, not yet executed/accepted','numerical_pass_attribution_pending':'numerically passed but not accepted','numerical_failed':'full manifest numerical mismatch','deployment_blocked':'build/import/package failure','device_failed':'runtime error, not a mathematical rejection','unsupported_within_scope':'source-proven missing/equivalent contract mismatch','semantic_review_pending':'review incomplete','missing_workload_factory':'no manifest Case can be instantiated'},library='ops-nn',updated_at=datetime.now(timezone.utc).isoformat(),inventory_sha256=digest(HERE/'inventory_snapshot.json'),source_index='source_index.json',operators=operators,cases=rows,totals=dict(operators=len(operators),cases=len(rows),case_statuses=dict(collections.Counter(r['status'] for r in rows))))
    coverage['operator_status_definitions']={'verified':'Every exact manifest case has an accepted current receipt','verified_historical':'Every manifest case retains explicit historical acceptance','partially_verified':'At least one current/historical accepted case; remaining cases are classified separately','implemented_awaiting_validation':'Source adapter exists, but no current accepted case','implemented_with_failures':'Adapter exists; actual trial failures or missing attribution require review, with changed-source flags per case','unsupported_within_scope':'No equivalent original API mapping within the stated adaptation scope','oracle_unavailable':'Factory exists but independent oracle contract is undefined','missing_workload_factory':'No exact manifest factory is registered','semantic_review_pending':'Source review is incomplete'}
    coverage['status_definitions']['verified_preserved']='Existing full numerical/input/native receipt plus exact byte-preserved relocated runtime and unchanged dependencies; staging result tracked separately, no new device run'
    save(HERE/'coverage.json',coverage)
    progress=json.loads((HERE/'progress.json').read_text());progress.update(updated_at=coverage['updated_at'],implemented_operator_routes=len(ROUTES|LEGACY),coverage=coverage['totals']);save(HERE/'progress.json',progress)
    print(json.dumps(coverage['totals']))
if __name__=='__main__':main()
