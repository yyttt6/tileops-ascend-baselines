"""Locked independent template profile. Worker installs NoNPU before CPU imports."""
import pathlib,runpy,sys
P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=['/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages','/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe',str(P)]
sys.argv=[str(P/'worker.py'),*sys.argv[1:]];runpy.run_path(str(P/'worker.py'),run_name='__main__')
