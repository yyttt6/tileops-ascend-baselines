"""Produce a pinned candidate file offline; never mutate an installed backend."""
import argparse
import hashlib
from pathlib import Path

ORIGINAL = 'a1779017efc75841fc03a9561f902ec2a25e8a6de709356f1a9700a077ec803a'
CANDIDATE = '0be87418b543e882f4e7351a21a0a9bcf578b33a8be271c81b0e861d6da853db'
OLD = b'@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])\ndef int_noop(x):\n    return is_integer_dtype(x.dtype)'
NEW = b'@register_noop_decomp([aten.ceil, aten.floor, aten.trunc])\ndef int_noop(x):\n    return is_integer_dtype(x.dtype)\n\n\n@register_noop_decomp([aten.round])\ndef round_int_noop(x, decimals=0):\n    # Negative decimals can change integer values; retain that operation.\n    return is_integer_dtype(x.dtype) and type(decimals) is int and decimals >= 0'

def materialize(source, output):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL or raw.count(OLD) != 1:
        raise ValueError('original backend does not match the reviewed source')
    candidate = raw.replace(OLD, NEW)
    if hashlib.sha256(candidate).hexdigest() != CANDIDATE:
        raise ValueError('candidate differs from the device-tested bytes')
    with output.open('xb') as stream:
        stream.write(candidate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    materialize(args.source, args.output)
