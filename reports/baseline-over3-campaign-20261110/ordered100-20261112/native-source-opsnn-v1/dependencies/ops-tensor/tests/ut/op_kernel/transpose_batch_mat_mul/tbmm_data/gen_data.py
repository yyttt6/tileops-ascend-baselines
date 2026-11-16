# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class TbmmShapeConfig:
    m: int
    n: int
    k: int
    batch: int
    dtype: str = 'float16'
    trans_batch_a: bool = False


def gen_tbmm_data(cfg: TbmmShapeConfig, output_dir: str = './'):
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        'float16': np.float16,
        'float32': np.float32,
        'bfloat16': np.float32
    }
    np_dtype = dtype_map.get(cfg.dtype)
    if np_dtype is None:
        raise ValueError(
            f"Unsupported dtype: {cfg.dtype} (expected one of {list(dtype_map)})"
        )

    a_logical = np.random.randn(cfg.batch, cfg.m, cfg.k).astype(np_dtype)
    b = np.random.randn(cfg.batch, cfg.k, cfg.n).astype(np_dtype)

    if cfg.dtype == 'float16':
        golden_logical = np.matmul(a_logical.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    elif cfg.dtype == 'bfloat16':
        golden_logical = np.matmul(a_logical, b)
        a_logical = float32_to_bfloat16(a_logical)
        b = float32_to_bfloat16(b)
        golden_logical = float32_to_bfloat16(golden_logical)
    else:
        golden_logical = np.matmul(a_logical, b)

    # C is always stored in transposed-batch layout: [m, batch, n]
    golden_stored = golden_logical.transpose(1, 0, 2)

    # A storage layout depends on trans_batch_a
    if cfg.trans_batch_a:
        a_stored = a_logical.transpose(1, 0, 2)  # [m, batch, k]
    else:
        a_stored = a_logical  # [batch, m, k]

    a_stored.tofile(os.path.join(output_dir, 'input_a.bin'))
    b.tofile(os.path.join(output_dir, 'input_b.bin'))
    golden_stored.tofile(os.path.join(output_dir, 'golden_c.bin'))

    return a_stored, b, golden_stored


def float32_to_bfloat16(arr):
    float32_view = arr.view(np.uint32)
    bfloat16_bits = (float32_view >> 16).astype(np.uint16)
    return bfloat16_bits


def main():
    parser = argparse.ArgumentParser(description='Generate TransposeBatchMatMul test data')
    parser.add_argument('--m', type=int, required=True, help='M dimension')
    parser.add_argument('--n', type=int, required=True, help='N dimension')
    parser.add_argument('--k', type=int, required=True, help='K dimension')
    parser.add_argument('--batch', type=int, default=1, help='Batch dimension')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--trans_batch_a', action='store_true',
                        help='A stored in transposed-batch layout [m, batch, k]')
    parser.add_argument('--output_dir', type=str, default='./')

    args = parser.parse_args()
    cfg = TbmmShapeConfig(
        m=args.m, n=args.n, k=args.k, batch=args.batch,
        dtype=args.dtype, trans_batch_a=args.trans_batch_a,
    )
    gen_tbmm_data(cfg, args.output_dir)


if __name__ == '__main__':
    main()
