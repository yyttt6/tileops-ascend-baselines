"""Private exact-key registration; parent review required before installation."""
from pathlib import Path
import importlib.util
import os

KEYS = frozenset(("moe-fwd-bfloat16",))
VENDOR = Path("/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn")

def make(case):
    if case._call.signature.name != "GroupedGemmFwdOp" or case.id not in KEYS:
        raise ValueError("Outside this exact ops-nn Grouped3 scope")
    if Path(os.environ.get("OPS_NN_VENDOR", "")).resolve() != VENDOR.resolve():
        raise RuntimeError("Source recipe.sh before importing the device runtime")
    spec = importlib.util.spec_from_file_location("grouped3_original_adapter", Path(__file__).with_name("adapter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make(case)

def install(providers):
    prior = providers.get("GroupedGemmFwdOp")
    def route(case):
        if case._call.signature.name == "GroupedGemmFwdOp" and case.id in KEYS:
            return make(case)
        if prior is not None:
            return prior(case)
        raise ValueError("No registered exact case")
    providers["GroupedGemmFwdOp"] = route
