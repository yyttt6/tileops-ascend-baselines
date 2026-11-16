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
import numpy as np


def gen_matmul_data(m, n, k, dtype='float16', output_dir='./', batch=1):
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        'float16': np.float16,
        'float32': np.float32,
        'bfloat16': np.float32
    }

    np_dtype = dtype_map[dtype]

    if batch > 1:
        a = np.random.randn(batch, m, k).astype(np_dtype)
        b = np.random.randn(batch, k, n).astype(np_dtype)
        if dtype == 'float16':
            golden = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        elif dtype == 'bfloat16':
            golden = np.matmul(a, b)
            a = float32_to_bfloat16(a)
            b = float32_to_bfloat16(b)
            golden = float32_to_bfloat16(golden)
        else:
            golden = np.matmul(a, b)
    else:
        a = np.random.randn(m, k).astype(np_dtype)
        b = np.random.randn(k, n).astype(np_dtype)
        if dtype == 'float16':
            golden = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        elif dtype == 'bfloat16':
            golden = np.matmul(a, b)
            a = float32_to_bfloat16(a)
            b = float32_to_bfloat16(b)
            golden = float32_to_bfloat16(golden)
        else:
            golden = np.matmul(a, b)

    a.tofile(os.path.join(output_dir, 'input_a.bin'))
    b.tofile(os.path.join(output_dir, 'input_b.bin'))
    golden.tofile(os.path.join(output_dir, 'golden_c.bin'))

    return a, b, golden


def float32_to_bfloat16(arr):
    """Convert float32 array to bfloat16 binary format (uint16).
    
    bfloat16 = float32 truncated to upper 16 bits (1 sign + 8 exponent + 7 mantissa).
    """
    float32_view = arr.view(np.uint32)
    bfloat16_bits = (float32_view >> 16).astype(np.uint16)
    return bfloat16_bits


def main():
    parser = argparse.ArgumentParser(description='Generate Blaze MatMul test data')
    parser.add_argument('--m', type=int, required=True, help='M dimension')
    parser.add_argument('--n', type=int, required=True, help='N dimension')
    parser.add_argument('--k', type=int, required=True, help='K dimension')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--batch', type=int, default=1, help='Batch dimension')

    args = parser.parse_args()

    gen_matmul_data(args.m, args.n, args.k, args.dtype, args.output_dir, args.batch)

if __name__ == '__main__':
    main()