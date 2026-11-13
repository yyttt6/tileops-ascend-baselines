"""Reconstruct frozen sources from this checkout without device access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
TOOLS = {'baseline_protocol.py': '4921791a16097d80c404396ac17a0c266bab708b8df3e69040c84df5b7b56c48',
         'measure_baselines.py': '39e9973489069562543aed004f7381adbe6418be7313c3d9ce3cff53caa9127a',
         'kernel_task_profile_full32.py': 'd745454235f8f5c0676f70b75b81e0dd757ccceb63ffdf5f8deb923cb8f3d753'}

def prepare(checkout, destination):
    if destination.exists():
        raise ValueError('refuse existing destination')
    source = checkout/'reports/torch-v010-20261106'
    pins = json.loads((P/'source-pins.json').read_text())
    pins.update({'frozen/tools/'+k: v for k, v in TOOLS.items()})
    for name, digest in pins.items():
        data = (source/name.removeprefix('frozen/')).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source mismatch: '+name)
    destination.mkdir(parents=True)
    for name in pins:
        output = destination/name.removeprefix('frozen/')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name.removeprefix('frozen/'), output)
    (destination/'tools/__init__.py').touch()
    return len(pins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.checkout.resolve(), args.destination))
