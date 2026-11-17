"""Capture-only v6: traverse nested JITImpl/cache runtime imports to depth12."""
from pathlib import Path
import json,hashlib,inspect,shutil
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FrozenQuant:
 def __init__(self,case):
  import tileops.ascend
  from tileops.backend import _registry
  assert "ascend" in _registry.REGISTERED_KERNEL_TYPES.get(case._call.signature.name,{})
  import tileops.quantization as quant_ops
  OpClass=getattr(quant_ops,case._call.signature.name)
  from tileops.backend import BUILTIN
  self.op=OpClass(**case.arguments,target=BUILTIN);self.impl=case._entry.binder(self.op,case)
 def capture(self, dest):
     import torch, tilelang, torch_npu
     dest = Path(dest)
     out = dest / 'tileops-generated'
     out.mkdir(exist_ok=True)
     seen = set()
     keep_alive = []  # Runtime imports produce ephemeral Python wrappers; prevent id reuse.
     records = []
     binaries = {}
     sources = {}
 
     def walk(obj, label, depth=0):
         if obj is None or id(obj) in seen or depth > 12 or isinstance(obj, (str, int, float, bytes, Path, torch.Tensor, type)):
             return
         seen.add(id(obj))
         keep_alive.append(obj)
         record = {'label': label, 'type': type(obj).__module__ + '.' + type(obj).__qualname__, 'python_object_id':id(obj)}
         records.append(record)
         # TVM FFI may execute an in-memory runtime module with no libpath.
         # Serialize the actual imported device module; do not rebuild it.
         if hasattr(obj, 'get_write_formats') and hasattr(obj, 'get_property_mask'):
             record['runtime_kind'] = str(obj.kind)
             record['write_formats'] = [str(x) for x in obj.get_write_formats()]
             record['property_mask'] = int(obj.get_property_mask())
             if record['property_mask'] & 1:
                 choices = [x for x in record['write_formats'] if x not in ('c','cc','cpp','cu','asm','ptx')]
                 if choices:
                     fmt = choices[0]
                     f = out / ('runtime-binary-' + str(len(binaries)) + '.' + fmt)
                     obj.write_to_file(str(f), fmt)
                     assert f.is_file() and f.stat().st_size > 0
                     binaries['runtime_module:' + label] = {'retained':f.name,'sha256':sha(f),'module_kind':str(obj.kind),'format':fmt,'identity_source':'serialized actual executing runtime module; no new compilation','owner_label':label}
             if obj.kind == 'asc' and obj.is_binary_serializable() and not record['write_formats']:
                 # Official TVM serializer preserves the actual runtime code_ bytes
                 # consumed by aclrtBinaryLoadFromData. No compiler or kernel edit.
                 from tvm import get_global_func
                 packed = get_global_func('runtime.ModulePackImportsToTensor')(obj)
                 payload = packed.numpy().tobytes()
                 assert len(payload)>64 and int.from_bytes(payload[:8],'little') == len(payload)-8
                 assert b'\x7fELF' in payload, 'serialized Ascend module lacks executing ELF binary'
                 f = out / ('runtime-binary-' + str(len(binaries)) + '.tvmbytes')
                 f.write_bytes(payload)
                 binaries['runtime_module:' + label] = {'retained':f.name,'sha256':sha(f),'module_kind':'asc','format':'TVM serialized actual binary module with code_ ELF','identity_source':'runtime.ModulePackImportsToTensor; actual loaded module code_, no compilation','owner_label':label,'elf_magic_offset':payload.index(b'\x7fELF')}
             imported = list(obj.imports)
             record['imported_runtime_kinds'] = [str(child.kind) for child in imported]
             for i, child in enumerate(imported):
                 walk(child, label + '.imports' + str(i), depth + 1)
         if hasattr(obj, 'get_kernel_source'):
             text = obj.get_kernel_source()
             if text:
                 f = out / ('kernel-' + str(len(sources)) + '.cc')
                 f.write_text(text)
                 sources[f.name] = sha(f)
                 record['source_artifact'] = f.name
         for attr in ['libpath', 'lib_path', 'kernel_lib_path', 'device_binary_path', 'aibin_path']:
             value = getattr(obj, attr, None)
             if isinstance(value, (str, Path)) and Path(value).is_file():
                 source = Path(value)
                 f = out / ('binary-' + str(len(binaries)) + source.suffix)
                 shutil.copyfile(source, f)
                 binaries[str(source)] = {'retained': f.name, 'sha256': sha(f), 'owner_label': label}
         if isinstance(obj, dict):
             for i, value in enumerate(obj.values()):
                 walk(value, label + '.dict' + str(i), depth + 1)
         elif isinstance(obj, (list, tuple)):
             for i, value in enumerate(obj):
                 walk(value, label + '.item' + str(i), depth + 1)
         elif hasattr(obj, '__dict__'):
             for key, value in vars(obj).items():
                 if key in {'_run', 'compiled', 'adapter', 'artifact', 'rt_mod', '_entries_by_call', '_entries', '_target_kernels', 'entry', 'kernel', '_kernel', '_bound', '_instance', 'instance', 'op', '_callable', 'func'} or any((s in key.lower() for s in ['kernel', 'entry', 'compiled'])) or type(value).__module__.startswith('tilelang.jit'):
                     walk(value, label + '.' + key, depth + 1)
         if inspect.isfunction(obj) and obj.__closure__:
             for i, cell in enumerate(obj.__closure__):
                 walk(cell.cell_contents, label + '.closure' + str(i), depth + 1)
     walk(self.op, 'public_op')
     modules = {}
     import sys
     for name in sorted({type(self.op).__module__} | {x['type'].rpartition('.')[0] for x in records}):
         module = sys.modules.get(name)
         if module and getattr(module, '__file__', None):
             modules[name] = {'path': str(Path(module.__file__).resolve()), 'sha256': sha(module.__file__)}
     compiled_entries = []
     for rec in records:
         if rec['type'] == 'tilelang.jit.kernel.JITKernel':
             owned = [v['retained'] for v in binaries.values() if v.get('owner_label','').startswith(rec['label'] + '.') or v.get('owner_label') == rec['label']]
             compiled_entries.append({'label':rec['label'],'source':rec.get('source_artifact'),'binaries':owned,'complete':bool(rec.get('source_artifact')) and bool(owned)})
     complete = bool(sources) and bool(binaries) and bool(compiled_entries) and all(v['complete'] for v in compiled_entries)
     report = {'measured_library': 'TileOps', 'public_op_type': type(self.op).__module__ + '.' + type(self.op).__qualname__, 'actual_modules': modules, 'object_evidence': records, 'generated_sources_sha256': sources, 'generated_binaries': binaries, 'tilelang': {'version': getattr(tilelang, '__version__', None), 'path': tilelang.__file__}, 'original_kernel_source_unchanged': True, 'compiled_entries':compiled_entries,'generated_identity_complete':complete}
     (dest / 'TILEOPS_GENERATED_IDENTITY.json').write_text(json.dumps(report, indent=2) + '\n')
     return report
