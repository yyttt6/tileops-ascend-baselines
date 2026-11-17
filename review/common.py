"""CPU-only catalog, identity and contract helpers."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TILEOPS_COMMIT = "62c5fa4d1f80b5325c6632aa23622ec2d6c3f2c9"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Explicit serialization required: {type(value)}")


def normalize_params(op, params, expected):
    """Only documented public metadata renames; never change tensor values."""
    params = plain(params)
    if op in {"TopKMaskFwdOp", "TopPMaskFwdOp", "TopKTopPMaskFwdOp"}:
        if ("top_k" in params and "k" in params) or ("top_p" in params and "p" in params):
            raise ValueError("Ambiguous metadata alias; do not drop one input")
        params = {{"top_k": "k", "top_p": "p"}.get(k, k): v for k, v in params.items()}
    if op == "GDNFwdOp" and "validate_inputs" not in expected:
        if "validate_inputs" in params:
            if params.pop("validate_inputs") is not False:
                raise ValueError("Public GDN validate_inputs changed")
    if params != expected:
        raise ValueError("Case contract differs. Review an explicit metadata/ABI bridge; "
                         f"do not weaken the comparison. actual={params}, expected={expected}")
    return params
