"""Extract compact original-native binary launch evidence without touching logs."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla'


def main():
    matches=[]
    seen=set()
    for path in (EVIDENCE/'native_logs').rglob('*.log'):
        for line_number,line in enumerate(path.open(errors='replace'),1):
            if ('Register Kernel succ' in line or 'bin path is' in line) and 'fla_npu' in line:
                parsed=re.search(r'bin path(?: is|:)\s+(\S+\.o)',line)
                if not parsed:continue
                binary=Path(parsed.group(1))
                key=(str(path.parent),str(binary))
                if key in seen:continue
                seen.add(key)
                matches.append(dict(log=str(path.relative_to(ROOT)),line=line_number,
                    binary=str(binary.relative_to(ROOT)),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),text=line.strip()))
    (EVIDENCE/'native_launch_audit.json').write_text(json.dumps({'schema':'fla-native-launch/v1','actual_bin_loads':matches},indent=2)+'\n')
    print('owned binary loads',len(matches))


if __name__=='__main__':main()
