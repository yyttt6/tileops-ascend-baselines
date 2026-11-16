import pathlib,sys,json,time,fcntl
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import controller as c
lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=json.loads((P/'LEDGER.json').read_text())
def group(family,mode,ids,prefix):
 pending=list(ids)
 while pending:
  label=prefix+'-'+str(pending[0])+'-'+str(pending[-1]);e=c.run(family,mode,pending,label)
  if e is None:print('DEFER',label,flush=True);time.sleep(30);continue
  if 'batch' not in e:raise RuntimeError('guard invalid')
  done=[]
  for r in e['batch']['results']:
   i=r['ordinal'];done.append(i)
   if not prefix.startswith('old'):
    if str(i) in ledger:
     c.put(P/('CASE'+str(i)+'.before-sdk91-retry.json'),ledger[str(i)])
    ledger[str(i)]={'result':r,'tag':e['tag'],'label':label,'guarded_seconds':e['guarded_seconds'],'guard':e['guard'],'archive_sha256':e['archive_sha256'],'new_condition':'restored original CatlassSDK9.1 with existing binary cache'}
  c.put(P/'LEDGER.json',ledger);pending=[i for i in pending if i not in done]
  if not done:raise RuntimeError('no progress')
group('catlass','isolated',[95,96,97,98,99],'new-catlass-sdk91')
for i in [6,7,8,9]:group('template','isolated',[i],'old-tilekernels')
c.put(P/'FIRST100_TRAVERSED.json',{'status':'100 traversed, full audit/throughput decision next','reviewed':len(ledger),'source_reuse':41,'candidate_legs':13})
