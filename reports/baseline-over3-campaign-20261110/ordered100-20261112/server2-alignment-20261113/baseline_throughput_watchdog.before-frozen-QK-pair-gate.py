#!/usr/bin/env python3
"""CPU-only liveness/rolling-throughput observer; never submits device work."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'reports/baseline-over3-campaign-20261110/ordered100-20261112'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def write(p,x):
 t=p.with_name(p.name+'.watchdog.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)

def active_controllers():
 result=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   args=proc.joinpath('cmdline').read_bytes().decode().split('\0');cwd=proc.joinpath('cwd').resolve()
   for arg in args[1:]:
    path=pathlib.Path(arg)
    if not path.name.startswith(('queue_','expand_','continue_ordered')) or path.suffix!='.py':continue
    resolved=(path if path.is_absolute() else cwd/path).resolve()
    if resolved.parent==P:
     result.append({'pid':int(proc.name),'entry':path.name,'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]});break
  except (OSError,ValueError,UnicodeError):pass
 return result

def primary_dispatch_controllers(controllers):
 roles=read(P/'CONTROLLER_ROLES.json',{})
 return [c for c in controllers if roles.get(c['entry'],{}).get('counts_as_primary_dispatch',True)]

def support_completion_event(controllers):
 candidates=[]
 for name,entry in [('SERVER2_OPSMATH_DEQUANT_CONFIGURE.json', 'queue_server2_opsmath_dequant_configure.py'), ('SERVER2_OPSMATH_DEQUANT_HOST.json', 'queue_server2_opsmath_dequant_host.py'), ('SERVER2_OPSMATH_DEQUANT_COMPILE.json', 'queue_server2_opsmath_dequant_compile.py'), ('SERVER2_OPSMATH_VECTORINF_CONFIGURE.json', 'queue_server2_opsmath_vectorinf_configure.py'), ('SERVER2_OPSMATH_VECTORINF_HOST.json', 'queue_server2_opsmath_vectorinf_host.py'), ('SERVER2_OPSMATH_VECTORINF_COMPILE.json', 'queue_server2_opsmath_vectorinf_compile.py'), ('SERVER2_OPSMATH_COUNTNZ_QUALIFICATION.json', 'queue_server2_opsmath_countnz_qualification.py'), ('SERVER1_COUNTNZ_QUALIFICATION.json', 'queue_server1_countnz_qualification.py'), ('SERVER2_OPSMATH_COUNTNZ_CONFIGURE.json', 'queue_server2_opsmath_countnz_configure.py'), ('SERVER2_OPSMATH_COUNTNZ_HOST.json', 'queue_server2_opsmath_countnz_host.py'), ('SERVER2_OPSMATH_COUNTNZ_COMPILE.json', 'queue_server2_opsmath_countnz_compile.py'), ('SERVER2_OPSMATH_SCANBOOL_MARKER_V2.json', 'queue_server2_opsmath_scanbool_marker_v2.py'), ('SERVER2_OPSMATH_SCANBOOL_V2_QUALIFICATION.json', 'queue_server2_opsmath_scanbool_v2_qualification.py'), ('SERVER1_SCANBOOL_V2_QUALIFICATION.json', 'queue_server1_scanbool_v2_qualification.py'), ('SERVER2_OPSMATH_SCANBOOL_HOST2.json', 'queue_server2_opsmath_scanbool_host2.py'), ('SERVER2_OPSMATH_SCANBOOL_CONFIGURE.json', 'queue_server2_opsmath_scanbool_configure.py'), ('SERVER2_OPSMATH_SCANBOOL_HOST.json', 'queue_server2_opsmath_scanbool_host.py'), ('SERVER2_OPSMATH_SCANBOOL_COMPILE.json', 'queue_server2_opsmath_scanbool_compile.py'), ('SERVER2_OPSMATH_SCANBOOL_QUALIFICATION.json', 'queue_server2_opsmath_scanbool_qualification.py'), ('SERVER1_SCANBOOL_QUALIFICATION.json', 'queue_server1_scanbool_qualification.py'), ('SERVER2_OPSMATH_LEAF4_CONFIGURE.json', 'queue_server2_opsmath_leaf4_configure.py'), ('SERVER2_OPSMATH_LEAF4_HOST.json', 'queue_server2_opsmath_leaf4_host.py'), ('SERVER2_OPSMATH_LEAF4_COMPILE.json', 'queue_server2_opsmath_leaf4_compile.py'), ('SERVER2_OPSMATH_LEAF4_QUALIFICATION.json', 'queue_server2_opsmath_leaf4_qualification.py'), ('SERVER1_LEAF4_QUALIFICATION.json', 'queue_server1_leaf4_qualification.py'), ('SERVER2_OPSMATH_REALDIV_CONFIGURE.json', 'queue_server2_opsmath_realdiv_configure.py'), ('SERVER2_OPSMATH_REALDIV_HOST.json', 'queue_server2_opsmath_realdiv_host.py'), ('SERVER2_OPSMATH_REALDIV_FLOAT32_COMPILE.json', 'queue_server2_opsmath_realdiv_float32_compile.py'), ('SERVER2_OPSMATH_REALDIV_F16_SUPPLEMENT.json', 'queue_server2_opsmath_realdiv_f16_supplement.py'), ('SERVER2_OPSMATH_REALDIV_CAPI_V2_QUALIFICATION.json', 'queue_server2_opsmath_realdiv_capi_v2_qualification.py'), ('SERVER1_REALDIV_CAPI_V2_QUALIFICATION.json', 'queue_server1_realdiv_capi_v2_qualification.py'), ('SERVER1_GRAPH92_MIN_QUALIFICATION.json', 'queue_server1_graph92_min_qualification.py'), ('SERVER2_GRAPH92_MATCHED_HEADERS_MIN_DIAGNOSTIC.json', 'queue_server2_graph92_matched_headers_min_diagnostic.py'), ('SERVER2_GRAPH92_MATCHED_TBE_MIN_DIAGNOSTIC.json', 'queue_server2_graph92_matched_tbe_min_diagnostic.py'), ('SERVER2_GRAPH92_MATERIALIZED_MIN_DIAGNOSTIC.json', 'queue_server2_graph92_materialized_min_diagnostic.py'), ('SERVER2_GRAPH92_COMPLETE_MIN_DIAGNOSTIC.json', 'queue_server2_graph92_complete_min_diagnostic.py'), ('SERVER2_GRAPH92_MIN_OPP91_DIAGNOSTIC.json', 'queue_server2_graph92_min_opp91_diagnostic.py'), ('SERVER2_OPSMATH_ADD_MARKER_INT32_COMPILE.json', 'queue_server2_opsmath_add_marker_int32_compile.py'), ('SERVER2_OPSMATH_ADD_FLOAT3_COMPILE.json', 'queue_server2_opsmath_add_float3_compile.py'), ('SERVER2_OPSMATH_ADD_COMMAND_GENERATION.json', 'queue_server2_opsmath_add_command_generation.py'), ('SERVER2_OPSMATH_ADD_HOST_GENERATION_V2.json', 'queue_server2_opsmath_add_host_generation_v2.py'), ('SERVER2_OPSMATH_ADD_HOST_GENERATION.json', 'queue_server2_opsmath_add_host_generation.py'), ('SERVER2_OPSMATH_ADD_CONFIGURE_V2.json', 'queue_server2_opsmath_add_configure_v2.py'), ('SERVER2_OPSMATH_ADD_CONFIGURE.json', 'queue_server2_opsmath_add_configure.py'), ('SERVER2_GRAPH92_MIN_DIAGNOSTIC.json', 'queue_server2_graph92_min_diagnostic.py'), ('SERVER2_SEGMENTMIN_API_GENERATION.json', 'queue_server2_native_segmentmin_api_generation.py'), ('SERVER2_FP8QUANT_FLOAT32_COMPILE.json', 'queue_server2_native_fp8quant_float32_compile.py'), ('SERVER2_FP8QUANT_GENERATION.json', 'queue_server2_native_fp8quant_generation.py'), ('SERVER2_INPLACERMS_FLOAT2_COMPILE.json', 'queue_server2_native_inplacerms_float2_compile.py'), ('SERVER2_INPLACERMS_GENERATION.json', 'queue_server2_native_inplacerms_generation.py'), ('SERVER2_LPNORM_FLOAT32_COMPILE.json', 'queue_server2_native_lpnorm_float32_compile.py'), ('SERVER2_LPNORM_GENERATION.json', 'queue_server2_native_lpnorm_generation.py'), ('SERVER2_MASKEDSCATTER_BYTE2_COMPILE.json', 'queue_server2_native_maskedscatter_byte2_compile.py'), ('SERVER2_MASKEDSCATTER_GENERATION.json', 'queue_server2_native_maskedscatter_generation.py'), ('SERVER2_GATHER_FLOAT3_COMPILE.json', 'queue_server2_native_gather_float3_compile.py'), ('SERVER2_GATHER_GENERATION.json', 'queue_server2_native_gather_generation.py'), ('SERVER2_ARGMIN_HOST_BUILD_V1.json', 'queue_server2_native_argmin_host_build_v1.py'), ('SERVER2_REFERENCE_BUILD.json', 'queue_server2_aligned_native_build.py'), ('SERVER2_NATIVE_LOAD.json', 'queue_server2_native_load.py'), ('SERVER2_DEVICE_BUILD.json', 'queue_server2_native_device_build.py'), ('SERVER2_UNARY_REFERENCE_BUILD.json', 'queue_server2_native_unary_build.py'), ('SERVER2_UNARY_DEVICE_BUILD.json', 'queue_server2_native_unary_device_build.py'), ('SERVER2_SEGMENT_REFERENCE_BUILD.json', 'queue_server2_native_segment_build.py'), ('SERVER2_SEGMENT_DEVICE_BUILD.json', 'queue_server2_native_segment_device_build.py'), ('SERVER2_SEGMENT_DEVICE_BUILD_V2.json', 'queue_server2_native_segment_device_build_v2.py'), ('SERVER2_SEGMENT_DEVICE_BUILD_V4.json', 'queue_server2_native_segment_device_build_v4.py'), ('SERVER2_ADD_REFERENCE_BUILD.json', 'queue_server2_native_add_build.py'), ('SERVER2_ADD_DEVICE_BUILD.json', 'queue_server2_native_add_device_build.py'), ('SERVER2_ADD_DEVICE_BUILD_V2.json', 'queue_server2_native_add_device_build_v2.py'), ('SERVER2_SOFTMAX_REFERENCE_BUILD.json', 'queue_server2_native_softmax_build.py'), ('SERVER2_SOFTMAX_DEVICE_BUILD.json', 'queue_server2_native_softmax_device_build.py'), ('SERVER2_GEMM10_FP32_COMPILE_V2.json', 'queue_server2_native_gemm10_fp32_compile_v2.py'), ('SERVER2_GEMM10_FP32_COMPILE.json', 'queue_server2_native_gemm10_fp32_compile.py'), ('SERVER2_GEMM10_GENERATION.json', 'queue_server2_native_gemm10_generation.py'), ('SERVER2_MAXPOOL_DEVICE_BUILD.json', 'queue_server2_native_maxpool_device_build.py'), ('SERVER2_POOLMEAN_REFERENCE_BUILD.json', 'queue_server2_native_poolmean_build.py'), ('SERVER2_LAYERNORM_REFERENCE_BUILD.json', 'queue_server2_native_layernorm_build.py'), ('SERVER2_LAYERNORM_DEVICE_BUILD.json', 'queue_server2_native_layernorm_device_build.py'), ('SERVER2_SEGMENT_DEVICE_BUILD_V3.json', 'queue_server2_native_segment_device_build_v3.py')]:
  row=read(P/'server2-alignment-20261113'/name,{})
  if row and row.get('entry',entry)==entry:candidates.append((row.get('queued_at',''),row,entry))
 if not candidates:return None
 _,state,entry=max(candidates,key=lambda item:item[0])
 live=any(c['pid']==state.get('pid') and c['entry']==entry for c in controllers)
 status=state.get('status','')
 bad_query=state.get('task_level',{}).get('query_failed') if status not in ('yield_0200_Shanghai_nightly_or_unobserved_dispatch','yield_worker_queue_or_disk','await_fresh_physical_idle','deferred_no_child') else None
 terminal=status in ('generated_pending_compile_scope_review','built_pending_identity_and_load','reference_build_failed_requires_diagnosis','stopped_requires_diagnosis','loaded_pending_numeric_qualification','load_failed_requires_diagnosis','device_built_pending_identity_and_execution','device_build_failed_requires_diagnosis')
 # A successful handled build needs no interruption once real formal dispatch has continued.
 # Failures, dead active builds and zero-throughput checks remain unsuppressed.
 if terminal and not live and not bad_query and status in ('generated_pending_compile_scope_review','built_pending_identity_and_load','loaded_pending_numeric_qualification','device_built_pending_identity_and_execution') and primary_dispatch_controllers(controllers):return None
 if bad_query or (terminal and not live) or (status in ('guarded_server2_reference_build','guarded_native_load') and not live):
  return {'kind':'server2_native_support_needs_inspection','pid':state.get('pid'),'status':status,'tag':state.get('tag'),'query_failed':bad_query,'query_error':state.get('task_level',{}).get('error') if bad_query else None}
 return None

def support_event_key(event):
 if event is None:return None
 if event.get('query_failed'):
  # One controller's task-query outage is one incident. Preserve current
  # details separately; changing queried status/EOF text is not a new failure.
  return {'kind':event.get('kind'),'pid':event.get('pid'),'query_failed':event['query_failed']}
 return event

def paired_results(now):
 records={}
 for pattern in ['SGL_*CURRENT.json','VLLM_*CURRENT.json','FLA_*CURRENT.json','TLA_*CURRENT.json','OPS_NN_*CURRENT.json','OPS_MATH_*CURRENT.json','TORCH_GEGLU3_CANDIDATE_CURRENT.json','TORCH_QK2_NATIVE_RMS_CURRENT.json','TORCH_QK2_NATIVE_RMS_ROPE_CURRENT.json','TORCH_SCAN_GROUPED_LAYOUT_CURRENT.json']:
  for f in P.glob(pattern):
   obj=read(f,{})
   for row in obj.get('rows',[]):
    if row.get('guard_checker_15raw_valid') is not True:continue
    old=row.get('original_checkpoint');new=row.get('candidate_checkpoint')
    if not old or not new or not (P/old).is_file() or not (P/new).is_file():continue
    identity=(old,new);records[identity]={'ordinal':row.get('ordinal'),'completed_at_local':max((P/old).stat().st_mtime,(P/new).stat().st_mtime),'summary':f.name}
 return records

def alarm_due(now,reason):
 schedule=P/'RECURRING_ALARMS.json'
 with schedule.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);s=read(schedule,{})
  if s.get('disabled') or not s.get('interval_seconds'):return 'disabled_no_change'
  if s.get('pending'):return 'delivery_already_pending'
  if any(m['due_at']<=now and 'delivered_at' not in m and not m.get('superseded_at') for m in s.get('milestones',[])):return 'overdue_reminder_already_armed'
  s.setdefault('milestones',[]).append({'due_at':now,'after_seconds':now-s['start_at'],'watchdog_reason':reason})
  write(schedule,s);return 'immediate_inspection_armed'

def qualified_event_growth(now):
 # Read the same audited event stream used by real half-hour inspections.
 # A fresh observer has no 60min sample window; that is unknown, never zero.
 try:
  import importlib.util
  spec=importlib.util.spec_from_file_location('baseline_qualified_event_inspection',P/'inspect_baseline_alarm.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  last60,last30=module.growth(read(P/'CURRENT_VALID_COVERAGE.json'),datetime.datetime.fromtimestamp(now,datetime.timezone.utc))
  return {'last60':last60,'last30':last30,'error':None}
 except (OSError,ValueError,AssertionError,KeyError,TypeError) as exc:
  return {'last60':None,'last30':None,'error':type(exc).__name__+': '+str(exc)}

def final_closure_events(now):
 try:
  import importlib.util
  spec=importlib.util.spec_from_file_location('baseline_qualified_final_closures',P/'qualified_final_closures.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return {'events':module.collect(P,datetime.datetime.fromtimestamp(now,datetime.timezone.utc)),'error':None}
 except (OSError,ValueError,AssertionError,KeyError,TypeError) as exc:
  return {'events':[],'error':type(exc).__name__+': '+str(exc)}

def main():
 now=time.time();out=P/'THROUGHPUT_WATCHDOG.json'
 with (P/'throughput-watchdog.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=read(out,{})
  if (P/'STOP').exists() or (P.parent/'STOP').exists() or read(P.parent/'STATE.json',{}).get('status')!='active':
   write(out,{'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paused':True,'action':'No reminder or work dispatch; user pause/stop wins'});return
  controllers=active_controllers();primary=primary_dispatch_controllers(controllers);pairs=paired_results(now);recent=[x for x in pairs.values() if now-3600<=x['completed_at_local']<=now]
  coverage=read(P/'CURRENT_VALID_COVERAGE.json',{}).get('effective_coverage');samples=[x for x in previous.get('coverage_observations',[]) if x['at']>=now-7200];samples.append({'at':now,'coverage':coverage})
  anchor=next((x for x in reversed(samples) if x['at']<=now-3600),None)
  growth=coverage-anchor['coverage'] if anchor and coverage is not None else None
  event_growth=qualified_event_growth(now)
  closure_growth=final_closure_events(now);finals=closure_growth['events']
  if event_growth['last60'] is not None:growth=event_growth['last60']
  idle_since=None if primary else previous.get('idle_since',now)
  current=read(P/'CURRENT_PROCESS.json',{})
  dispatch_event=None
  if not primary and current.get('status') in ('group_finished_next_dispatch_required','completed_group_no_active_controller'):
   dispatch_event={'kind':'group_finished_needs_successor','entry':current.get('entry'),'pid':current.get('prior_pid'),'at':current.get('checked_at'),'outcome':current.get('outcome')}
  elif not primary and current.get('status')=='active' and current.get('pid') and not any(c['pid']==current['pid'] and c['entry']==current.get('entry') for c in controllers):
   dispatch_event={'kind':'recorded_active_controller_missing','entry':current.get('entry'),'pid':current.get('pid'),'at':current.get('updated_at')}
  support_event=support_completion_event(controllers)
  new_support=support_event is not None and support_event_key(support_event)!=support_event_key(previous.get('last_handled_support_event'))
  new_dispatch=dispatch_event is not None and dispatch_event!=previous.get('last_handled_dispatch_event')
  urgent=new_dispatch or new_support
  reasons=[]
  if new_dispatch:reasons.append(dispatch_event['kind'])
  if support_event is not None:reasons.append(support_event['kind'])
  if idle_since is not None and now-idle_since>=300:reasons.append('pending_scope_without_live_controller_for_5min')
  if len(recent)==0 and not finals and growth is not None and growth<=0:reasons.append('zero_valid_API_pairs_and_newcoverage_last_hour')
  if len(recent)==0 and not finals and growth is None:reasons.append('qualified_growth_evidence_unavailable')
  if closure_growth['error']:reasons.append('qualified_final_closure_evidence_invalid')
  if growth is not None and growth<0:reasons.append('qualified_coverage_regressed')
  report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checked_at_epoch':now,'active_controllers':controllers,'primary_dispatch_controllers':primary,'idle_since':idle_since,'effective_coverage':coverage,'coverage_observations':samples,'observed_new_coverage_last_hour':growth,'coverage_growth_unknown_until_full_observation_window':growth is None,'qualified_event_growth':event_growth,'observed_new_coverage_last30min':event_growth['last30'],'audited_API_pairs_last_hour':len(recent),'audited_final_under3_closures_last_hour':sum(x['under3'] for x in finals),'audited_final_retained_over3_closures_last_hour':sum(x['retained_over3'] for x in finals),'qualified_final_closures':closure_growth,'audited_API_pair_evidence':recent,'counting':'Only source summaries with full guard/checker15raw qualification and complete checkpoint pairs; no attempts/rejections/traversal counted. New coverage separately observed.','issues':reasons,'last_alert_at':previous.get('last_alert_at'),'device_work_submitted':False}
  report['current_support_event']=support_event
  report['last_handled_dispatch_event']=dispatch_event if new_dispatch else previous.get('last_handled_dispatch_event')
  report['last_handled_support_event']=support_event if new_support else previous.get('last_handled_support_event')
  report['dispatch_event_requires_immediate_inspection']=urgent
  if reasons and (urgent or now-(previous.get('last_alert_at') or 0)>=1800):
   report['action']=alarm_due(now,','.join(reasons));report['last_alert_at']=now
  elif reasons:report['action']='Existing inspection/remediation cooldown; real process and throughput remain checked each minute'
  else:report['action']='Continue observing; no anomaly detected in monitored dimensions'
  write(out,report)
  with (P/'THROUGHPUT_WATCHDOG_HISTORY.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in report.items() if k not in ['coverage_observations','audited_API_pair_evidence']})+'\n')
  print(json.dumps({k:report[k] for k in ['checked_at','audited_API_pairs_last_hour','active_controllers','issues','action']}))
if __name__=='__main__':main()
