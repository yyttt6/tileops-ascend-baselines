"""Read only the unchanged full32 parser and retained profile artifacts."""
from pathlib import Path
import importlib.util
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('golden42_full32',P.parents[1]/'torch-all2208-20261105/tools/kernel_task_profile_full32.py')
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def parse_profile_path(path):
    raw=P/'remote'/Path(path).relative_to('/opt/baseline-over3-20261110/ordered100-20261112')
    return parser.parse_profile(raw,1)
