"""Bind unchanged original PTO23 API to current realguard and exact relocated pins.

Never changes the old registration source/data, compiler bytes, kernel definitions,
block sizes, full-math adapter, checker, or precision. No legacy owner is claimed.
"""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,sys,types,datetime
P=Path(__file__).resolve().parent
REG=P/'native/pto-registration-current-v1'
SDK=P/'native/pto-locked-sdk-920b2-v1'
LIB=P/'native/tla-original-v1/lib'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _load_original():
 p=REG/'registration_parent.py';proof=json.loads((P/'PTO_CURRENT_BINDING_CPU_PROOF.json').read_text());assert sha(p)==proof['original_registration_sha256'];spec=importlib.util.spec_from_file_location('_pto_current_original_registration',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def map_path(p):
 p=Path(p);old=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2')
 if p.is_relative_to(old):return SDK/p.relative_to(old)
 old=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_tla_compiler/handoff/lib')
 if p.parent==old and p.name in ('libtilelang_module.so','libtvm.so','libtvm_runtime.so'):return LIB/p.name
 return p

def bind_guard(m,expected):
 assert expected.startswith('root/baseline-ordered100/rollout-pto-current-') and not expected.startswith('baseline700-')
 source=(REG/'registration_parent.py').read_text();node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='require_guard');original=ast.dump(node,include_attributes=False);count=[0]
 class Replace(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='baseline700-20261026/threehost_pto_registration23':count[0]+=1;new=ast.Name(id='EXPECTED_ACTUAL_OWNER',ctx=ast.Load());new._restore_guard_constant=copy.deepcopy(n);return new
   return n
 class Restore(ast.NodeTransformer):
  def visit(self,n):
   if hasattr(n,'_restore_guard_constant'):return copy.deepcopy(n._restore_guard_constant)
   return super().visit(n)
 node=Replace().visit(node);assert count==[1] and ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==original
 env=dict(vars(m),EXPECTED_ACTUAL_OWNER=expected);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(REG/'registration_parent.py'),'exec'),env);m.require_guard=env['require_guard'];return {'only_guard_owner_constant_rebound_to_actual_current_owner':True,'full_reverse_guard_AST_identical':True,'actual_owner':expected,'PID_start_boot_PGID_kernel_flock_checks_retained':True}

def build(case,tag):
 # Called only inside complete currentforegroundguard, before any compiler import.
 assert tag.startswith('rollout-pto-current-') and Path(tag).name==tag
 m=_load_original();expected='root/baseline-ordered100/'+tag;os.environ['PTO23_OWNER']=expected;proof=bind_guard(m,expected);m.require_guard()
 host,pin=m.current_host();assert host=='server1';assert sys.version==pin['python'];original_pin=copy.deepcopy(pin);pin['pins']={str(map_path(path)):digest for path,digest in pin['pins'].items()}
 assert len(pin['pins'])==len(original_pin['pins'])
 for path,digest in pin['pins'].items():m.checked(path,digest)
 for path,digest in m.dataset('LOCAL_PINS.json').items():m.checked(REG/path,digest)
 assert m.matches(case),'Full original contract required, not key-onlymatch'
 original_load=m.load
 def load(name,path):
  mod=original_load(name,path)
  if name=='_pto23_setup':
   original_setup=mod.setup
   def setup(*args):
    original_setup(*args)
    # Original bytes at taskprivate paths; original setup body remains unchanged.
    os.environ.update(TILELANG_LIBRARY_PATH=str(LIB),TVM_LIBRARY_PATH=str(LIB),ASCEND_HOME_PATH=str(SDK),ASCEND_TOOLKIT_HOME=str(SDK),BISHENG_HOME=str(SDK/'tools/bisheng_compiler'))
    os.environ['PATH']=str(SDK/'tools/bisheng_compiler/bin')+':'+os.environ['PATH']
   mod.setup=setup
  return mod
 m.current_host=lambda:(host,copy.deepcopy(pin));m.load=load
 runroot=REG/'runs'/tag;runroot.mkdir(parents=True,exist_ok=False);os.environ['PTO23_RUN_ROOT']=str(runroot)
 proof.update({'host':host,'pins_verified':len(pin['pins']),'original_local_source_pins_verified':len(m.dataset('LOCAL_PINS.json')),'full_original_contract_identical':True,'kernel_and_complete_adapter_unchanged':True,'SDK_exact_server2_tree_verified':'PTO_PRIVATE_DEPENDENCY_PREPARE.json','path_relocations':{p:str(map_path(p)) for p in original_pin['pins'] if str(map_path(p))!=p},'helper_sha256':sha(__file__)})
 candidate=m.make(case);sources=dict(candidate.source_sha256);sources[str(P/'pto_current_binding.py')]=sha(__file__)
 for f in ['PTO_CURRENT_BINDING_CPU_PROOF.json','PTO_PRIVATE_DEPENDENCY_PREPARE.json','PTO_SDK920B2_SERVER2_SOURCE_MANIFEST.json']:sources[str(P/f)]=sha(P/f)
 sources[str(REG/'registration_parent.py')]=sha(REG/'registration_parent.py');sources.update(pin['pins']);sources.update({str(REG/f):sha(REG/f) for f in ['DATA_HASHES.json','CONTRACTS.json','LOCAL_PINS.json','HOSTS.json']})
 return candidate,sources,proof
