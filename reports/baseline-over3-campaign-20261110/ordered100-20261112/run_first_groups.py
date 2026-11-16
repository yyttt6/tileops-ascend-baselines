"""Strict queue order; only bounded batches, no waiting inside the NPU queue."""
import pathlib,sys,json,time,fcntl,itertools
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import controller as c
lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ledger=json.loads((P/'LEDGER.json').read_text()) if (P/'LEDGER.json').exists() else {}

def group(family,mode,ids,prefix):
 pending=list(ids)
 while pending:
  if (P/'STOP').exists() or (P.parent/'STOP').exists():raise RuntimeError('STOP')
  selected=pending[:8];label=prefix+'-'+str(selected[0])+'-'+str(selected[-1]);result=c.run(family,mode,selected,label)
  if result is None:
   print('DEFER',label,flush=True);time.sleep(30);continue
  if 'batch' not in result:
   print('GUARD_INVALID',label,flush=True);return
  done=[]
  for r in result['batch']['results']:
   i=r['ordinal'];done.append(i)
   if not prefix.startswith('old'):
    ledger[str(i)]={'result':r,'label':label,'tag':result['tag'],'guarded_seconds':result['guarded_seconds'],'guard':result['guard'],'archive_sha256':result['archive_sha256']}
  c.put(P/'LEDGER.json',ledger)
  pending=[i for i in pending if i not in done]
  if not done:raise RuntimeError('no rows')
  print('LEDGER',len(ledger),'valid',sum(x['result']['status']=='raw_checked_pending_guard' for x in ledger.values()),flush=True)
for i in range(1,5):group('deepgemm','auto',[i],'new-deepgemm')
# Original representative control qualifies streams/task topology before expansion.
group('template','isolated',[5],'old-tilekernels')
for op,rs in itertools.groupby(json.loads((P/'TASKS.json').read_text())[5:52],key=lambda r:r['op']):
 ids=[r['ordinal'] for r in rs]
 # All pure CPU preparations run separately and may still be writing their final receipt.
 while True:
  ready=c.remote('test -f '+c.R+'/prepared/'+str(ids[-1])+'/prepare.json').returncode==0
  if ready:break
  print('WAIT_CPU',op,ids,flush=True);time.sleep(10)
 group('template','auto',ids,'new-tilekernels')
c.put(P/'PHASE1_DONE.json',{'ledger_rows':len(ledger),'status':'first52traversed','next':'ATVC43 source closure/reuse plus2unmeasured, then catlass5'})
