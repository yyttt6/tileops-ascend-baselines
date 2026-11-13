"""Offline environment preparation. Does not import torch or start device work."""
import argparse, hashlib, json, pathlib, shutil
ORIGINAL='a1779017efc75841fc03a9561f902ec2a25e8a6de709356f1a9700a077ec803a'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(source, destination, candidate):
 source=source.resolve(strict=True);destination=destination.absolute()
 if destination.exists():raise ValueError('refuse existing destination')
 parent=destination.parent.resolve(strict=True);destination=parent/destination.name
 if destination.is_relative_to(source) or source.is_relative_to(destination):raise ValueError('overlapping environment paths')
 matches=list(source.glob('lib/python*/site-packages/torch/_inductor/fx_passes/post_grad.py'))
 if len(matches)!=1 or digest(matches[0])!=ORIGINAL:raise ValueError('original source mismatch')
 rel=matches[0].relative_to(source)
 # copy2 creates independent regular files; symlinks=False dereferences source links.
 shutil.copytree(source,destination,symlinks=False,copy_function=shutil.copy2)
 files={}
 for p in destination.rglob('*'):
  if p.is_symlink():raise ValueError('unexpected copied symlink')
  if p.is_file():
   src=source/p.relative_to(destination)
   if src.exists() and p.stat().st_dev==src.stat().st_dev and p.stat().st_ino==src.stat().st_ino:raise ValueError('shared inode')
   copied_hash=digest(p)
   if not src.is_file() or copied_hash!=digest(src):raise ValueError('copied file mismatch')
   files[str(p.relative_to(destination))]=copied_hash
 if candidate is not None:
  target=destination/rel
  if digest(target)!=ORIGINAL:raise ValueError('copy source mismatch')
  target.write_bytes(candidate.read_bytes())
  files[str(rel)]=digest(target)
 assert digest(matches[0])==ORIGINAL
 return {'prefix':str(destination),'source_prefix':str(source),'backend':str(rel),'backend_sha':digest(destination/rel),'files':files,'device_run':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--destination-parent',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--candidate-sha',required=True);a=p.parse_args()
 if digest(a.candidate)!=a.candidate_sha:raise ValueError('candidate SHA mismatch')
 # Both output roots must be new. No implicit cleanup on failure.
 for name in ['round-control-original03','round-candidate03']:
  if (a.destination_parent/name).exists():raise ValueError('environment exists; preserve and inspect')
 for name,candidate in [('round-control-original03',None),('round-candidate03',a.candidate)]:
  r=prepare(a.source,a.destination_parent/name,candidate)
  (a.destination_parent/(name+'.fingerprint.json')).write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
