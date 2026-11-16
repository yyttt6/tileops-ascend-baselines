"""Recover retained original compiler events missed during CANN file rotation."""
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path('/root/dyq/workspace/.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn')

def main():
    out=ROOT/'recovered-native.jsonl'
    old=[json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    seen={(r['file'],r['line'],r['text']) for r in old};count=0
    with out.open('a') as stream:
        for path in ROOT.glob('*/cann_logs/**/*.log'):
            for number,line in enumerate(path.open(errors='replace'),1):
                if 'LogTaskDetails' not in line or 'result[success]' not in line:continue
                if 'allpairs_nn_impl/dynamic/' not in line:continue
                text=line.rstrip('\n');key=(str(path),number,text)
                if key in seen:continue
                row=dict(file=str(path),line=number,text=text,captured_at=datetime.now(timezone.utc).isoformat(),
                         origin='retained raw CANN file recovery, original event text preserved')
                stream.write(json.dumps(row)+'\n');seen.add(key);count+=1
    print(json.dumps({'recovered_original_compile_events':count}))

if __name__=='__main__':main()
