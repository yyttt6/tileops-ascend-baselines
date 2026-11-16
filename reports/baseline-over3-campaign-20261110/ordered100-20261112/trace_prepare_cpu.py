import pathlib,runpy,sys,os
P=pathlib.Path(__file__).resolve().parent;sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages');sys.argv=[str(P/'prepare_rollout.py'),'diagnose870','870']
def trace(frame,event,arg):
 if frame.f_code.co_filename==str(P/'prepare_rollout.py') and frame.f_code.co_name=='<module>':
  if event=='line':print('PREP_LINE',frame.f_lineno,'ids=',frame.f_locals.get('ids'),'D=',frame.f_locals.get('D'),flush=True)
  elif event in ('return','exception'):print('PREP_EVENT',event,str(arg)[:200],flush=True)
 return trace if frame.f_code.co_filename==str(P/'prepare_rollout.py') and frame.f_code.co_name=='<module>' else None
sys.settrace(trace);runpy.run_path(str(P/'prepare_rollout.py'),run_name='__main__')
