"""Exact reviewed routes; device imports remain lazy and provider-local."""
import hashlib
import importlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'accepted_verified_20261026.json').read_text())
RECORDS = {tuple(r['key'][1:]): r for r in DATA['records']}


def validate(record):
    from benchmarks.ascend_baselines import UnsupportedCase
    pins = {HERE / p: h for p, h in DATA['runtime_files'].items()}
    for label in ('checker', 'guard', 'review'):
        pins[Path(record[label])] = record[label + '_sha256']
    for path, expected in pins.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedCase('Reviewed runtime/receipt changed; revalidation required: ' + str(path))
    # Receipt pins include the unchanged vendor and original source dependencies.
    checker = json.loads(Path(record['checker']).read_text())
    workspace = HERE.parents[3]
    for name, expected in checker['source_sha256'].items():
        path = workspace / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedCase('Reviewed dependency changed: ' + str(path))
    vendor = Path(record['vendor']).resolve()
    if (Path(os.environ.get('OPS_NN_VENDOR', '')).resolve() != vendor or
            os.environ.get('ASCEND_CUSTOM_OPP_PATH') != str(vendor)):
        raise UnsupportedCase('Select the reviewed vendor before device imports with ops_nn/run_verified.py: ' + str(vendor))


def make(case):
    record = RECORDS[(case._call.signature.name, case.id)]
    validate(record)
    module, function = record['factory'].split(':')
    if module != 'ascend_rms_cases':
        # Same isolated per-process cache location as both verified launchers.
        from .verified_20261026 import graph
        graph.WORKSPACE = Path('/proc') / str(os.getpid()) / 'cwd'
    adapter = importlib.import_module('.verified_20261026.' + module, __package__)
    return getattr(adapter, function)(case)


def install(providers):
    for op in {key[0] for key in RECORDS}:
        previous = providers.get(op)
        def factory(case, _op=op, _previous=previous):
            if case._call.signature.name == _op and (_op, case.id) in RECORDS:
                return make(case)
            if _previous is not None:
                return _previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No accepted exact ops-nn case: ' + _op + '/' + case.id)
        factory.__name__ = 'reviewed_20261026_' + op
        providers[op] = factory
