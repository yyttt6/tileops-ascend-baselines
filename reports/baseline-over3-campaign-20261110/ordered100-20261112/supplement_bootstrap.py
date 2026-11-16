import sys,pathlib,runpy,importlib.abc,importlib.util
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages','/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe',str(P)]
class CapturedATVC(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='benchmarks.ascend_all_pairs.atvc':return None
  f=P/'workspace/isolated_atvc/addcmul.py'
  class Loader(importlib.abc.Loader):
   def create_module(self,spec):return None
   def exec_module(self,m):
    m.__file__=str(f);exec(compile(f.read_text(),str(f),'exec'),m.__dict__);m.PROVIDERS={op:m.addcmul for op in ['LogicalAndFwdOp','LogicalNotFwdOp','MulFwdOp','SubFwdOp','NegFwdOp']}
  return importlib.util.spec_from_loader(fullname,Loader(),is_package=True)
sys.meta_path.insert(0,CapturedATVC());sys.argv=[str(P/'supplement.py'),*sys.argv[1:]];runpy.run_path(str(P/'supplement.py'),run_name='__main__')
