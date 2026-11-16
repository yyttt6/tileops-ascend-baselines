"""Confirm exact remaining coverage is explicitly held; never treats it as complete."""
from pathlib import Path
import json

def confirmed_user_held_scope(directory,unresolved):
 try:
  d=Path(directory);read=lambda p:json.loads(Path(p).read_text());state=read(d.parent/'STATE.json');work=state.get('remaining69_current_work',{})
  if state.get('current_dispatch_status')!='remaining69_allowed21_qualified_remaining48_explicit_user_holds_no_eligible_native_successor':return None
  path=(d.parent/work['report']).resolve()
  if not path.is_relative_to(d.parent.resolve()):return None
  report=read(path)
  if report['status']!=work['status'] or not report['no_remaining_eligible_case_in_this69_scope'] or not report['not_all69_qualified_or_complete']:return None
  a=d/'server2-alignment-20261113';gold=read(a/'REMAINING42_USER_DECISION_KEEP_ORIGINAL_GOLDEN_20261116.json');complex_=read(a/'COMPLEX6_USER_DECISION_KEEP_ORIGINAL_INPUT_CHECK_20261116.json')
  if gold['user_decision']!='保持原 golden，仅继续其余项' or gold['golden_changes_authorized'] is not False:return None
  if complex_['user_decision']!='保持原检查方式，六项留待处理' or complex_['compatibility_not_applied'] is not True:return None
  g=set(gold['scope']);c=set(complex_['scope']);held=g|c
  if len(g)!=42 or len(c)!=6 or g&c or held!=set(report['remaining_user_held']) or held!=set(unresolved):return None
  if set(report['initial_remaining_scope'])!=held|set(report['newly_qualified']) or held&set(report['newly_qualified']):return None
  return {'status':'explicit_user_constraints_cover_exact_remaining_scope','report':str(path),'reference_golden_held':sorted(g),'complex_input_checker_held':sorted(c),'unresolved_count':len(held),'not_qualified_or_completed':True,'permitted_native_successors_in_this_scope':0}
 except (OSError,KeyError,ValueError,TypeError,AssertionError):return None
