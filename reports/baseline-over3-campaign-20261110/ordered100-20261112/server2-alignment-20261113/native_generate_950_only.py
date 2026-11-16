"""Use only original exact950 configs across original top/inner/exc classes."""
from pathlib import Path
import configparser,hashlib,json,subprocess,sys

def configs(root):
 root=Path(root);found=[];seen=set()
 for folder in [root,root/'inner',root/'exc']:
  p=folder/'aic-ascend950-ops-info.ini'
  if not p.is_file():continue
  c=configparser.ConfigParser(strict=False,interpolation=None);c.read(p);ops=set(c.sections())
  if not ops:continue
  if seen & ops:raise ValueError('Ambiguous duplicate original950 definitions: '+repr(seen & ops))
  seen.update(ops);found.append((p,sorted(ops)))
 if not found:raise FileNotFoundError('No nonempty original950 opsinfo in top/inner/exc')
 return found

def main():
 build=Path(sys.argv[1]);generator=Path(sys.argv[2]);records=[]
 for p,ops in configs(build/'autogen'):
  records.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ops':ops})
  subprocess.run([sys.executable,str(generator),str(p),'','',str(build/'tbe/ascendc'),str(build/'tbe/dynamic'),str(build/'autogen')],check=True)
 print(json.dumps({'original_single_soc950_generation':records,'generator_sha256':hashlib.sha256(generator.read_bytes()).hexdigest(),'source_or_kernel_edits':False}),flush=True)
if __name__=='__main__':main()
