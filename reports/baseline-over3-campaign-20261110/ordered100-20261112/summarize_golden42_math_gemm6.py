"""Exact seven original Gemm API cases with new golden; audited coverage only."""
from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent
SCOPE={270,271,272,273,274,284}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize():
    audits=json.loads((P/'GOLDEN42_MATH_GEMM6_RAW_AUDIT.json').read_text())
    cert=json.loads((P/'GOLDEN42_CERTIFICATES.json').read_text())
    qualpath=P/'server2-alignment-20261113/PRIVATE_GEMM10_QUALIFICATION.json'
    qual=json.loads(qualpath.read_text());assert qual['private_original_device_kernels_verified']
    rows=[];seen=set()
    for cp in sorted(P.glob('rollout-golden42-math-gemm6-*.CHECKPOINT.json')):
        event=json.loads(cp.read_text());a=audits[event['label']]
        assert a['checkpoint_sha256']==sha(cp) and a['guard_valid']
        base=P/'remote/runs'/event['tag']
        assert sha(base/'SOURCE.json')==event['batch']['source_sha256']
        loaded=json.loads((base/'LOADED_LIBRARIES.json').read_text())
        for r in event['batch']['results']:
            i=r['ordinal'];assert i in SCOPE
            if r['status']!='raw_checked_pending_guard':continue
            assert i not in seen;seen.add(i)
            assert next(x for x in a['rows'] if x['ordinal']==i)['status']=='formal_raw_and_full_output_verified'
            assert r['reference_policy']=='audited_cpu_fp64_exact42_v1'
            selected=r['sources'];proof=json.loads((base/str(i)/'GOLDEN_CORRECTION.json').read_text())
            assert proof['certificate']==next(x for x in cert['rows'] if x['ordinal']==i)
            assert proof['manifest_sha256']==sha(P/'GOLDEN42_CERTIFICATES.json')
            assert proof['payload_sha256']==r['payload_sha256']
            for name in ['ops_nn_gemm10_api.py','ops_nn_gemm10_cached_fidelity.py','math_gemm_nn_alternative_fidelity.py']:
                assert [v for k,v in selected.items() if k.endswith('/'+name)]==[sha(P/name)]
            assert [v for k,v in selected.items() if k.endswith('/PRIVATE_GEMM10_QUALIFICATION.json')]==[sha(qualpath)]
            assert [v for k,v in selected.items() if k.endswith('/frozen_gemm_adapter_037f902.py')]==[qual['original_provider_sha256']]
            libs={k:v for k,v in selected.items() if k.endswith('/op_api/lib/libcust_opapi.so')}
            assert len(libs)==1 and all(loaded.get(k)==v for k,v in libs.items())
            stems=[]
            for path,digest in qual['actual_private_device_paths_sha256'].items():
                assert selected.get(path)==digest;stems.append(Path(path).stem)
            assert r['timing']['measurement_protocol']=='ascend950-isolated-ffts-v3'
            assert len(r['timing']['retained_profiles'])==15
            from audit_golden42_gemm7_parser import parse_profile_path
            for profile in r['timing']['retained_profiles']:
                parsed=parse_profile_path(profile['profile_path'])
                assert any(any(t['name'].startswith(s) for s in stems) for t in parsed['profile_kernel_tasks'])
            rows.append({'ordinal':i,'key':r['key'],'api':r['api'],'device_us':r['device_us'],'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'payload_sha256':r['payload_sha256'],'prepare_sha256':r['prepare_sha256'],'full_checker_15raw_guard_valid':True,'golden_manifest_sha256':sha(P/'GOLDEN42_CERTIFICATES.json'),'golden_correction_receipt_sha256':sha(base/str(i)/'GOLDEN_CORRECTION.json'),'loaded_private_host_library_bound':True,'private_device_ELF_source_identity_and_all15_profile_names_bound':True,'private_binary_provenance_origin_preserved':'original H1 qualification; new H1/H2 runtime verified by this actual guarded case','new_pairs':0,'new_final_closures':0})
    out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'accepted_complete_APIs':len(rows),'staged_not_published':True}
    target=P/'RESOLVED_GOLDEN42_MATH_GEMM6.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(target)
    return out
