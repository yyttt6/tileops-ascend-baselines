"""Compact CPU-only receipt inspection without emitting compiler IR dumps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    directory=root/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/receipts'/a.run
    counts=Counter()
    for f in sorted(directory.glob('*.json')):
        row=json.loads(f.read_text());counts[row['status']]+=1
        if row['status']=='verified':continue
        error=row.get('error',row.get('reason',''))
        diagnostics=[line for line in error.splitlines() if re.search(r'error:|Error:|overflow|[Aa]ssert|[Mm]ismatch|[Uu]nsupported|[Nn]ot implemented',line)]
        frames=re.findall(r'File "([^"]+)", line (\d+), in (.+)',row.get('traceback',''))
        print(json.dumps({'op':row['op'],'case_id':row['case_id'],'status':row['status'],'stage':row.get('stage'),
             'candidate_calls':row.get('candidate_calls'),'error':error[:600] if len(error)<2000 else (diagnostics[-4:] or error[-600:]),
             'last_frames':frames[-3:]},ensure_ascii=False))
    print(json.dumps({'run':a.run,'status_counts':dict(counts)}))

if __name__=='__main__':main()
