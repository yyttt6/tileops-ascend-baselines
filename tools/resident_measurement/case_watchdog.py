"""Only outer task budget differs from correctness watchdog; code unchanged."""
import argparse
from pathlib import Path
from watchdog import supervise
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
raise SystemExit(supervise(cmd,a.output,timeout=1250,grace=5,reap=5))
