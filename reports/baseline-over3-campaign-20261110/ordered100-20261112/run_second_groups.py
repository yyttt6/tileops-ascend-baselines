import pathlib,sys,json,math,time,fcntl,itertools
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import controller as c
assert (P/'PHASE1_DONE.json').exists()
lock=(P/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ledger=json.loads((P/'LEDGER.json').read_text());reuse=json.loads((P/'ATVC_REUSE_REVIEW.json').read_text())
for r in reuse['rows']:ledger[str(r['ordinal'])]={'result':r,'fresh_windows':0,'reuse_evidence':'ATVC_REUSE_REVIEW.json'}
c.put(P/'LEDGER.json',ledger)
candidates={}
def group(family,mode,indices,prefix):
 pending=list(indices)
 while pending:
  chosen=pending[:8];label=prefix+'-'+str(chosen[0])+'-'+str(chosen[-1]);e=c.run(family,mode,chosen,label)
  if e is None:print('DEFER',label,flush=True);time.sleep(30);continue
  if 'batch' not in e:raise RuntimeError('guard/process invalid '+label)
  done=[]
  for r in e['batch']['results']:
   i=r['ordinal'];done.append(i);entry={'result':r,'tag':e['tag'],'label':label,'guarded_seconds':e['guarded_seconds'],'archive_sha256':e['archive_sha256'],'guard':e['guard']}
   if family=='candidate':candidates[str(i)]=entry
   elif not prefix.startswith('old'):ledger[str(i)]=entry
  c.put(P/'LEDGER.json',ledger);c.put(P/'CANDIDATE_LEDGER.json',candidates)
  pending=[i for i in pending if i not in done]
  if not done:raise RuntimeError('no progress')
  print('REVIEWED',len(ledger),'candidate legs',len(candidates),flush=True)
rows=json.loads((P/'TASKS.json').read_text());eligible=[]
for i in range(13,31):
 meta=json.loads((P/'remote/prepared'/str(i)/'prepare.json').read_text());q=meta['params']['q'][0]
 if math.prod(q)%256==0:eligible.append(i)
c.put(P/'ZERO_PAD_ELIGIBLE.json',{'eligible':eligible,'unchanged_nonzero_tail':[i for i in range(13,31) if i not in eligible],'no_identical_candidate_measurements':True})
for op,rs in itertools.groupby(eligible,key=lambda i:rows[i]['op']):group('candidate','auto',list(rs),'candidate-pad')
group('template','auto',[76,77],'new-atvc')
group('template','isolated',[95,96,97,98,99],'new-catlass')
# Same exact FP8 set as the shared-process treatment, with no estimate from unlike cases.
for i in [6,7,8,9]:group('template','isolated',[i],'old-tilekernels')
c.put(P/'FIRST100_TRAVERSED.json',{'status':'100 traversed, throughput/selection audit next','reviewed':len(ledger),'source_reuse':len(reuse['rows']),'candidate_legs':len(candidates)})
