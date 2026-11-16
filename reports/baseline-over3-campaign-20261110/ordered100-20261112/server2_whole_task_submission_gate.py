import subprocess,json
known_active={}
def tasks():
 global known_active
 result={};blocking=[]
 # A long active workflow must not disappear behind a limit of newer completed runs.
 for key,row in list(known_active.items()):
  repo=row['repo'];r=subprocess.run(['gh','run','view',str(row['databaseId']),'-R',repo,'--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
  if r.returncode:return False,{'query_failed':repo,'run':row['databaseId'],'error':r.stderr}
  current=json.loads(r.stdout)
  if current['status']=='completed':known_active.pop(key)
  else:blocking.append({'repo':repo,**current})
 if blocking:return False,{'blocking':blocking,'source':'explicit previouslyobservedrunIDs, independent of recenthistorylimit'}
 for repo in ['yyttt6/TileOPs','yyttt6/TileOPs.github.io']:
  rows=[]
  for status in ['in_progress','queued','waiting','requested','action_required']:
   r=subprocess.run(['gh','run','list','-R',repo,'--status',status,'--limit','100','--json','databaseId,status,conclusion,workflowName,createdAt'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   if r.returncode:return False,{'query_failed':repo,'status':status,'error':r.stderr}
   rows.extend(json.loads(r.stdout))
  result[repo]=rows
  for row in rows:
   name=row['workflowName'].lower();relevant=('ascend 950' in name) if repo.endswith('/TileOPs') else ('nightly' in name)
   if relevant and row['status']!='completed':
    value={'repo':repo,**row};blocking.append(value);known_active[repo+':'+str(row['databaseId'])]=value
 return not blocking,{'blocking':blocking,'runs':result}
