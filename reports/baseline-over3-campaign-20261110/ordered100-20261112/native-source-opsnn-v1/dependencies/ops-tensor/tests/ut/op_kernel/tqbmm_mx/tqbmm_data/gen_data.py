#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""
Generate TQBMM MX UT test data (binary .bin files).

Usage:
    python3 gen_data.py --m 64 --n 128 --k 128 --batch 2 --dtype fp8_e4m3 [--trans_batch_a]
"""

import os
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--dtype", type=str, default="fp8_e4m3", choices=["fp8_e4m3", "fp4_e2m1"]
    )
    p.add_argument(
        "--trans_batch_a",
        action="store_true",
        help="A matrix uses [m, batch, k] layout",
    )
    return p.parse_args()


def get_mx_scale_k_len(k):
    return ((k + 63) // 64) * 2


def pack_fp4(values):
    """Pack pairs of uint8 FP4 values into bytes, low nibble first.

    Matches TTK pack_4bits and the hardware MMAD expectation (element i -> low
    nibble of byte i//2, element i+1 -> high nibble). Keep in sync with
    examples/scripts/gen_data.py.
    """
    if len(values) % 2 != 0:
        raise ValueError("FP4 element count must be even")
    result = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        result[i // 2] = (values[i] & 0xF) | ((values[i + 1] & 0xF) << 4)
    return result


def main():
    args = parse_args()
    m, n, k, batch = args.m, args.n, args.k, args.batch
    scale_k_len = get_mx_scale_k_len(k)

    # Generate random data
    if args.dtype == "fp4_e2m1":
        # FP4: 1 byte per element (packed as 2 elements per byte)
        a_data = np.random.randint(0, 16, size=m * batch * k, dtype=np.uint8)
        b_data = np.random.randint(0, 16, size=k * batch * n, dtype=np.uint8)
    else:
        a_data = np.random.randint(0, 256, size=m * batch * k, dtype=np.uint8)
        b_data = np.random.randint(0, 256, size=k * batch * n, dtype=np.uint8)

    # Scale data: E8M0, filled with 0x7f (neutral scale ~1.0)
    if args.trans_batch_a:
        scale_a = np.full(m * batch * scale_k_len, 0x7F, dtype=np.uint8)
    else:
        scale_a = np.full(batch * m * scale_k_len, 0x7F, dtype=np.uint8)
    scale_b = np.full(batch * scale_k_len * n, 0x7F, dtype=np.uint8)

    # Write binary files
    out_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dtype == "fp4_e2m1":
        a_packed = pack_fp4(a_data)
        a_packed.tofile(os.path.join(out_dir, "input_a.bin"))
        b_packed = pack_fp4(b_data)
        b_packed.tofile(os.path.join(out_dir, "input_b.bin"))
    else:
        a_data.tofile(os.path.join(out_dir, "input_a.bin"))
        b_data.tofile(os.path.join(out_dir, "input_b.bin"))

    scale_a.tofile(os.path.join(out_dir, "scale_a.bin"))
    scale_b.tofile(os.path.join(out_dir, "scale_b.bin"))

    print(f"Generated data: M={m}, N={n}, K={k}, batch={batch}, dtype={args.dtype}")
    print(
        f"  input_a.bin: {os.path.getsize(os.path.join(out_dir, 'input_a.bin'))} bytes"
    )
    print(
        f"  input_b.bin: {os.path.getsize(os.path.join(out_dir, 'input_b.bin'))} bytes"
    )
    print(
        f"  scale_a.bin: {os.path.getsize(os.path.join(out_dir, 'scale_a.bin'))} bytes"
    )
    print(
        f"  scale_b.bin: {os.path.getsize(os.path.join(out_dir, 'scale_b.bin'))} bytes"
    )


if __name__ == "__main__":
    main()
